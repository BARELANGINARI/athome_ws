#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include "icecream.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace zed_vision {

constexpr float PI_F = 3.14159265358979323846f;

float wrap180(float angle)
{
  while (angle > 180.0f) {angle -= 360.0f;}
  while (angle < -180.0f) {angle += 360.0f;}
  return angle;
}

float wrap360(float angle)
{
  angle = std::fmod(angle, 360.0f);
  if (angle < 0.0f) {
    angle += 360.0f;
  }
  if (angle >= 360.0f || std::abs(angle) < 1e-6f) {
    angle = 0.0f;
  }
  return angle;
}

float quatToYaw(float x, float y, float z, float w)
{
  return std::atan2(2.0f * (w * z + x * y), 1.0f - 2.0f * (y * y + z * z));
}

bool computeBbox(
  const std::vector<sl::float2> & points,
  int width, int height, cv::Rect & bbox)
{
  float min_x = static_cast<float>(width);
  float min_y = static_cast<float>(height);
  float max_x = 0.0f;
  float max_y = 0.0f;
  bool found = false;

  for (const auto & p : points) {
    if (std::isfinite(p.x) && std::isfinite(p.y) &&
        p.x >= 0 && p.y >= 0 && p.x < width && p.y < height)
    {
      min_x = std::min(min_x, p.x);
      min_y = std::min(min_y, p.y);
      max_x = std::max(max_x, p.x);
      max_y = std::max(max_y, p.y);
      found = true;
    }
  }

  if (!found) {
    return false;
  }

  const int pad_x = static_cast<int>((max_x - min_x) * 0.05f + 5.0f);
  const int pad_y = static_cast<int>((max_y - min_y) * 0.05f + 5.0f);
  const int x0 = std::max(0, static_cast<int>(min_x) - pad_x);
  const int y0 = std::max(0, static_cast<int>(min_y) - pad_y);
  const int x1 = std::min(width - 1, static_cast<int>(max_x) + pad_x);
  const int y1 = std::min(height - 1, static_cast<int>(max_y) + pad_y);

  if (x1 <= x0 || y1 <= y0) {
    return false;
  }

  bbox = cv::Rect(x0, y0, x1 - x0, y1 - y0);
  return true;
}

float getMedian(std::vector<float> & v)
{
  const size_t n = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + n, v.end());
  return v[n];
}

bool get3dCoordinates(
  const cv::Rect & cur_box, const sl::Mat & point_cloud, sl::float4 & point)
{
  const unsigned int R_max_global = 10;
  const unsigned int obj_size = std::min(cur_box.width, cur_box.height);
  const unsigned int R_max = std::min(R_max_global, obj_size / 2);
  const int center_i = cur_box.x + cur_box.width / 2;
  const int center_j = cur_box.y + cur_box.height / 2;
  const int w = static_cast<int>(point_cloud.getWidth());
  const int h = static_cast<int>(point_cloud.getHeight());

  std::vector<float> x_vect, y_vect, z_vect;
  for (unsigned int R = 0; R < R_max; ++R) {
    for (int dy = -static_cast<int>(R); dy <= static_cast<int>(R); ++dy) {
      for (int dx = -static_cast<int>(R); dx <= static_cast<int>(R); ++dx) {
        const int i = center_i + dx;
        const int j = center_j + dy;
        sl::float4 out{NAN, NAN, NAN, NAN};
        if (i >= 0 && i < w && j >= 0 && j < h) {
          point_cloud.getValue(i, j, &out);
        }
        if (std::isfinite(out.z)) {
          x_vect.push_back(out.x);
          y_vect.push_back(out.y);
          z_vect.push_back(out.z);
        }
      }
    }
  }

  if (!x_vect.empty()) {
    point.x = getMedian(x_vect);
    point.y = getMedian(y_vect);
    point.z = getMedian(z_vect);
    return true;
  } else {
    point.x = NAN;
    point.y = NAN;
    point.z = NAN;
    return false;
  }
}

bool get3dCoordinatesWithOffset(
  const cv::Rect & rect, const sl::Mat & point_cloud,
  double & x, double & y, double & z)
{
  const int centroidX = rect.x + rect.width / 2;
  const int centroidY = rect.y + rect.height / 2;
  sl::float4 point{NAN, NAN, NAN, NAN};

  // Tahap 1: Cek titik tengah piksel secara langsung (cepat O(1))
  if (centroidX >= 0 && centroidX < static_cast<int>(point_cloud.getWidth()) &&
      centroidY >= 0 && centroidY < static_cast<int>(point_cloud.getHeight())) {
    point_cloud.getValue(centroidX, centroidY, &point);
  }

  // Tahap 2: Fallback ke median sampling R_max jika titik tengah berada di NaN/pantulan
  if (!std::isfinite(point.z)) {
    get3dCoordinates(rect, point_cloud, point);
  }

  if (!std::isfinite(point.z)) {
    return false;
  }

  // Tahap 3: Terapkan offset kalibrasi kamera ZED robot (cm → meter)
  // Referensi: robotPos.x = point.x + 13; robotPos.y = -point.y + 3; robotPos.z = point.z + 44;
  x = (point.x + 13.0) / 100.0;
  y = (-point.y + 3.0) / 100.0;  // y di-negate karena sumbu Y ZED terbalik
  z = (point.z + 44.0) / 100.0;
  return true;
}

// Ekstraksi warna dominan torso (HSV) untuk memori visual ringan
cv::Scalar extractTorsoColor(const cv::Mat & bgr, const cv::Rect & bbox)
{
  if (bgr.empty() || bbox.width <= 0 || bbox.height <= 0) {
    return cv::Scalar(0, 0, 0);
  }
  // Ambil area torso di sepertiga atas-tengah badan
  int tw = std::max(4, bbox.width / 3);
  int th = std::max(4, bbox.height / 4);
  int tx = std::clamp(bbox.x + bbox.width / 3, 0, bgr.cols - tw);
  int ty = std::clamp(bbox.y + bbox.height / 5, 0, bgr.rows - th);

  cv::Rect torso_roi(tx, ty, tw, th);
  torso_roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
  if (torso_roi.area() <= 0) return cv::Scalar(0, 0, 0);

  cv::Mat crop = bgr(torso_roi);
  cv::Mat hsv;
  cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);
  return cv::mean(hsv);
}

}  // namespace zed_vision  

struct YoloDetection
{
  cv::Rect rect;
  std::string label;  
  float score{0.0f};
  int class_id{-1};
};

class YoloDetector
{
public:
  YoloDetector();
  ~YoloDetector();

  YoloDetector(const YoloDetector &) = delete;
  YoloDetector & operator=(const YoloDetector &) = delete;

  // model_path: yolov8n_sim.onnx (COCO pretrained) atau custom YOLO11n (botol, cracker, dll).
  bool load(
    const std::string & model_path, const std::string & classes_path,
    float conf_threshold = 0.45f, float nms_threshold = 0.45f,
    int input_size = 640, const std::string & default_label = "object",
    std::string * err = nullptr);

  bool ready() const;

  void detect(
    const cv::Mat & bgr,
    const std::vector<std::string> & bag_labels,
    std::vector<YoloDetection> & out);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct YoloDetector::Impl
{
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "zed_yolo"};
  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
    OrtArenaAllocator, OrtMemTypeDefault);
  std::string input_name;
  std::string output_name;
  std::vector<std::string> names;
  std::string default_label{"object"};
  float conf_threshold{0.45f};
  float nms_threshold{0.45f};
  int input_size{640};
  bool ready{false};
};

YoloDetector::YoloDetector()
: impl_(std::make_unique<Impl>()) {}

YoloDetector::~YoloDetector() = default;

bool YoloDetector::ready() const {return impl_->ready;}

bool YoloDetector::load(
  const std::string & model_path, const std::string & classes_path,
  float conf_threshold, float nms_threshold, int input_size,
  const std::string & default_label,
  std::string * err)
{
  Impl & im = *impl_;
  im.conf_threshold = conf_threshold;
  im.nms_threshold = nms_threshold;
  im.input_size = input_size;
  im.default_label = default_label;
  im.ready = false;

  im.names.clear();
  if (!classes_path.empty()) {
    std::ifstream f(classes_path);
    std::string line;
    while (std::getline(f, line)) {
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      if (!line.empty()) {
        im.names.push_back(line);
      }
    }
  }

  try {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(2);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    im.session = std::make_unique<Ort::Session>(im.env, model_path.c_str(), opts);

    Ort::AllocatorWithDefaultOptions alloc;
    im.input_name = im.session->GetInputNameAllocated(0, alloc).get();
    im.output_name = im.session->GetOutputNameAllocated(0, alloc).get();
  } catch (const std::exception & e) {
    if (err) {*err = e.what();}
    im.session.reset();
    return false;
  }
  im.ready = true;
  return true;
}

void YoloDetector::detect(
  const cv::Mat & bgr,
  const std::vector<std::string> & bag_labels,
  std::vector<YoloDetection> & out)
{
  out.clear();
  Impl & im = *impl_;
  if (!im.ready || !im.session || bgr.empty()) {
    return;
  }

  const std::unordered_set<std::string> want_bag(bag_labels.begin(), bag_labels.end());

  cv::Mat blob = cv::dnn::blobFromImage(
    bgr, 1.0 / 255.0, cv::Size(im.input_size, im.input_size), cv::Scalar(), true, false);

  std::array<int64_t, 4> input_shape{1, 3, im.input_size, im.input_size};
  Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
    im.mem_info, blob.ptr<float>(), blob.total(), input_shape.data(), input_shape.size());

  const char * in_names[] = {im.input_name.c_str()};
  const char * out_names[] = {im.output_name.c_str()};
  std::vector<Ort::Value> outputs;
  try {
    outputs = im.session->Run(
      Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);
  } catch (...) {
    return;
  }

  if (outputs.empty() || !outputs.front().IsTensor()) {
    return;
  }

  const auto shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
  const float * data = outputs.front().GetTensorData<float>();

  int channels = (shape.size() == 3) ? static_cast<int>(shape[1]) : 84;
  int num_dets = (shape.size() == 3) ? static_cast<int>(shape[2]) : 8400;
  int num_classes = channels - 4;
  if (num_classes <= 0) {
    return;
  }

  cv::Mat det_2d(channels, num_dets, CV_32F, const_cast<float *>(data));
  cv::Mat transposed = det_2d.t();

  const float x_factor = static_cast<float>(bgr.cols) / static_cast<float>(im.input_size);
  const float y_factor = static_cast<float>(bgr.rows) / static_cast<float>(im.input_size);

  static const std::vector<std::string> kCoco80 = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
  };

  auto get_label = [&](int id) -> std::string {
    if (id >= 0 && id < static_cast<int>(im.names.size())) {
      return im.names[static_cast<size_t>(id)];
    }
    if (num_classes == 80 && id >= 0 && id < 80) {
      return kCoco80[static_cast<size_t>(id)];
    }
    return "";
  };

  auto is_wanted = [&](int id) {
    if (num_classes == 1) return true;
    if (want_bag.empty()) return true;
    const std::string n = get_label(id);
    if (!n.empty() && want_bag.count(n) > 0) return true;
    if (num_classes == 80) {
      if (want_bag.count("bag") > 0 && (id == 24 || id == 26 || id == 28)) return true;
      if (want_bag.count("chair") > 0 && (id == 56 || id == 57)) return true;
    }
    return false;
  };

  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  std::vector<int> class_ids;

  for (int i = 0; i < num_dets; ++i) {
    const float * row = transposed.ptr<float>(i);
    float score = 0.0f;
    int cls_id = 0;

    if (num_classes == 1) {
      score = row[4];
      cls_id = 0;
    } else {
      cv::Mat scores(1, num_classes, CV_32F, const_cast<float *>(row + 4));
      cv::Point class_id_pt;
      double max_score = 0.0;
      cv::minMaxLoc(scores, nullptr, &max_score, nullptr, &class_id_pt);
      score = static_cast<float>(max_score);
      cls_id = class_id_pt.x;
    }

    if (score < im.conf_threshold || !is_wanted(cls_id)) {
      continue;
    }

    const float cx = row[0];
    const float cy = row[1];
    const float w = row[2];
    const float h = row[3];

    int left = std::max(0, std::min(static_cast<int>((cx - 0.5f * w) * x_factor), bgr.cols - 1));
    int top = std::max(0, std::min(static_cast<int>((cy - 0.5f * h) * y_factor), bgr.rows - 1));
    int width = std::max(1, std::min(static_cast<int>(w * x_factor), bgr.cols - left));
    int height = std::max(1, std::min(static_cast<int>(h * y_factor), bgr.rows - top));

    if (width <= 4 || height <= 4) {
      continue;
    }

    boxes.emplace_back(left, top, width, height);
    confidences.push_back(score);
    class_ids.push_back(cls_id);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, im.conf_threshold, im.nms_threshold, indices);

  out.reserve(indices.size());
  for (int idx : indices) {
    YoloDetection d;
    d.rect = boxes[static_cast<size_t>(idx)];
    d.score = confidences[static_cast<size_t>(idx)];
    d.class_id = class_ids[static_cast<size_t>(idx)];
    if (num_classes == 1) {
      d.label = im.default_label;
    } else {
      d.label = get_label(d.class_id);
      if (d.label.empty()) d.label = im.default_label;
    }
    out.push_back(d);
  }
}

static std::vector<std::string> loadLabels(const std::string & filepath)
{
  std::vector<std::string> labels;
  std::ifstream f(filepath);
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (!line.empty()) {
      labels.push_back(line);
    }
  }
  return labels;
}

class ZedNode : public rclcpp::Node
{
public:
  ZedNode();
  ~ZedNode() override;

private:
  void publishYaw();
  void loop();
  void processBodiesAndYolo(const cv::Mat & bgr, const sl::Mat & point_cloud);
  void showPreview(const cv::Mat & bgr);

  // ---------- Deteksi Gestur (Opsi 1 & Opsi 2) ----------
  // Murni hanya menggunakan Bahu (Shoulder) dan Pergelangan Tangan (Wrist):
  // Opsi 1: Menghitung kemiringan vektor bahu -> pergelangan (dx, dy).
  //         dy > 0 (menunjuk bawah) & |dx| >= 0.5 * dy (diagonal serong).
  // Opsi 2: Memeriksa kedua lengan (kiri: bahu 5, wrist 7 | kanan: bahu 12, wrist 14).
  std::string classifyArmRaw(
    const std::vector<sl::float2> & kpt, int shoulder, int wrist)
  {
    constexpr float kThreshold = 0.5f;
    if (shoulder >= static_cast<int>(kpt.size()) || wrist >= static_cast<int>(kpt.size())) {
      return "";
    }
    const auto & s = kpt[shoulder];
    const auto & w = kpt[wrist];
    if (!std::isfinite(s.x) || !std::isfinite(s.y) || !std::isfinite(w.x) || !std::isfinite(w.y) ||
        s.x < 0.f || s.y < 0.f || w.x < 0.f || w.y < 0.f) {
      return "";
    }

    const float dx = w.x - s.x;
    const float dy = w.y - s.y;
    if (dy <= 0.0f) { return ""; }
    if (std::fabs(dx) < kThreshold * dy) { return ""; }

    return dx < 0.0f ? "DOWN_LEFT" : "DOWN_RIGHT";
  }

  std::string detectGesture(const std::vector<sl::float2> & kpt)
  {
    if (kpt.size() < 15) return "NONE";

    // Cek kedua lengan: kiri (5, 7) dan kanan (12, 14)
    const std::string left = classifyArmRaw(kpt, 5, 7);
    const std::string right = classifyArmRaw(kpt, 12, 14);

    if (right == "DOWN_RIGHT" || left == "DOWN_RIGHT") {
      return "RIGHT";
    }
    if (left == "DOWN_LEFT" || right == "DOWN_LEFT") {
      return "LEFT";
    }
    return "NONE";
  }

  // ---------- 1. Klasifikasi Warna Pakaian (Baju Torso) ----------
  std::string classifyClothingColor(
    const cv::Mat & bgr, const cv::Rect & bbox,
    const std::vector<sl::float2> & kpt = {})
  {
    if (bgr.empty() || bbox.width <= 0 || bbox.height <= 0) return "casual attire";

    // Tentukan perbatasan torso/baju: strictly DARI PINGGANG KE ATAS (waist-up, leher ke pinggang).
    // Bagian bawah (celana/kaki) sama sekali tidak diikutsertakan agar tidak salah terdeteksi.
    int torso_y1 = -1;
    int torso_y2 = -1;
    int torso_x1 = -1;
    int torso_x2 = -1;

    // Batas bawah baju: PINGGANG (Waist / Hip Keypoints)
    float waist_y = -1.0f;
    if (kpt.size() > 22 && std::isfinite(kpt[18].y) && std::isfinite(kpt[22].y)) {
      waist_y = (kpt[18].y + kpt[22].y) * 0.5f;
    } else if (kpt.size() > 0 && std::isfinite(kpt[0].y)) {
      waist_y = kpt[0].y;
    }

    // Batas atas baju: BAHU / LEHER (Shoulder / Neck Keypoints)
    float shoulder_y = -1.0f;
    if (kpt.size() > 12 && std::isfinite(kpt[5].y) && std::isfinite(kpt[12].y)) {
      shoulder_y = (kpt[5].y + kpt[12].y) * 0.5f;
    } else if (kpt.size() > 3 && std::isfinite(kpt[3].y)) {
      shoulder_y = kpt[3].y;
    }

    if (waist_y > 0.0f && shoulder_y > 0.0f && waist_y > (shoulder_y + 10.0f)) {
      torso_y1 = static_cast<int>(shoulder_y);
      torso_y2 = static_cast<int>(waist_y);
      torso_x1 = bbox.x + static_cast<int>(bbox.width * 0.15);
      torso_x2 = bbox.x + static_cast<int>(bbox.width * 0.85);
    } else {
      // Fallback proporsi tubuh:
      // Dari bahu/dada (20% tinggi dari atas bbox) sampai batas pinggang (50% tinggi bbox).
      // Bagian >50% adalah celana/kaki, yang diabaikan.
      torso_y1 = bbox.y + static_cast<int>(bbox.height * 0.20);
      torso_y2 = bbox.y + static_cast<int>(bbox.height * 0.50);
      torso_x1 = bbox.x + static_cast<int>(bbox.width * 0.15);
      torso_x2 = bbox.x + static_cast<int>(bbox.width * 0.85);
    }

    torso_y1 = std::clamp(torso_y1, 0, bgr.rows);
    torso_y2 = std::clamp(torso_y2, 0, bgr.rows);
    torso_x1 = std::clamp(torso_x1, 0, bgr.cols);
    torso_x2 = std::clamp(torso_x2, 0, bgr.cols);

    if (torso_y2 <= torso_y1 || torso_x2 <= torso_x1) {
      return "casual attire";
    }

    cv::Rect torso_roi(torso_x1, torso_y1, torso_x2 - torso_x1, torso_y2 - torso_y1);
    torso_roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
    if (torso_roi.area() <= 0) return "casual attire";

    // 1. Coba inferensi via Model AI ONNX jika tersedia
    if (color_session_ && !color_labels_.empty()) {
      cv::Mat crop = bgr(torso_roi);
      cv::Mat resized;
      cv::resize(crop, resized, cv::Size(224, 224));
      cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

      cv::Mat blob = cv::dnn::blobFromImage(
        resized, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), false, false);

      std::array<int64_t, 4> input_shape{1, 3, 224, 224};
      Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        ort_attr_mem_info_, blob.ptr<float>(), blob.total(),
        input_shape.data(), input_shape.size());

      const char * in_names[] = {color_in_name_.c_str()};
      const char * out_names[] = {color_out_name_.c_str()};

      try {
        auto outputs = color_session_->Run(
          Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);
        if (!outputs.empty()) {
          const auto shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
          int num_classes = (shape.size() >= 2) ? static_cast<int>(shape[1]) : static_cast<int>(color_labels_.size());
          const float * logits = outputs.front().GetTensorData<float>();

          float max_l = -1e9f;
          for (int i = 0; i < num_classes; ++i) {
            if (logits[i] > max_l) max_l = logits[i];
          }
          float sum_exp = 0.0f;
          std::vector<float> probs(num_classes);
          for (int i = 0; i < num_classes; ++i) {
            probs[i] = std::exp(logits[i] - max_l);
            sum_exp += probs[i];
          }
          int best_idx = 0;
          float best_p = 0.0f;
          for (int i = 0; i < num_classes; ++i) {
            probs[i] /= sum_exp;
            if (probs[i] > best_p) {
              best_p = probs[i];
              best_idx = i;
            }
          }
          if (best_p > 0.30f && best_idx < static_cast<int>(color_labels_.size())) {
            std::string res = color_labels_[best_idx];
            std::transform(res.begin(), res.end(), res.begin(), ::tolower);
            if (res.find("shirt") == std::string::npos && res.find("clothes") == std::string::npos) {
              res += " shirt";
            }
            return res;
          }
        }
      } catch (...) {
        // Fallback ke HSV jika terjadi kegagalan inferensi runtime
      }
    }

    // 2. Fallback: Analisis Ruang Warna HSV pada torso_roi (pinggang ke atas)
    cv::Mat crop = bgr(torso_roi);
    cv::Mat hsv;
    cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);
    cv::Scalar mean_hsv = cv::mean(hsv);

    double h = mean_hsv[0]; // 0 - 180
    double s = mean_hsv[1]; // 0 - 255
    double v = mean_hsv[2]; // 0 - 255

    std::string color = "dark";
    if (v < 50.0) {
      color = "black";
    } else if (s < 35.0 && v > 180.0) {
      color = "white";
    } else if (s < 45.0) {
      color = "gray";
    } else if ((h >= 0.0 && h <= 10.0) || (h >= 170.0 && h <= 180.0)) {
      color = "red";
    } else if (h > 10.0 && h <= 25.0) {
      color = "orange";
    } else if (h > 25.0 && h <= 35.0) {
      color = "yellow";
    } else if (h > 35.0 && h <= 85.0) {
      color = "green";
    } else if (h > 85.0 && h <= 130.0) {
      color = "blue";
    } else if (h > 130.0 && h <= 155.0) {
      color = "purple";
    } else if (h > 155.0 && h < 170.0) {
      color = "pink";
    }
    return color + " shirt";
  }

  // ---------- 2. Klasifikasi Warna Rambut ----------
  std::string classifyHairColor(const cv::Mat & bgr, const cv::Rect & bbox)
  {
    if (bgr.empty() || bbox.width <= 0 || bbox.height <= 0) return "dark hair";

    // 1. Coba inferensi via Model AI ONNX jika tersedia
    if (hair_session_ && !hair_labels_.empty()) {
      int head_y1 = std::clamp(bbox.y, 0, bgr.rows);
      int head_y2 = std::clamp(static_cast<int>(bbox.y + bbox.height * 0.20), 0, bgr.rows);
      int head_x1 = std::clamp(static_cast<int>(bbox.x + bbox.width / 4), 0, bgr.cols);
      int head_x2 = std::clamp(static_cast<int>(bbox.x + 3 * bbox.width / 4), 0, bgr.cols);

      if (head_y2 > head_y1 && head_x2 > head_x1) {
        cv::Mat crop = bgr(cv::Rect(head_x1, head_y1, head_x2 - head_x1, head_y2 - head_y1));
        cv::Mat resized;
        cv::resize(crop, resized, cv::Size(224, 224));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

        cv::Mat blob = cv::dnn::blobFromImage(
          resized, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), false, false);

        std::array<int64_t, 4> input_shape{1, 3, 224, 224};
        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
          ort_attr_mem_info_, blob.ptr<float>(), blob.total(),
          input_shape.data(), input_shape.size());

        const char * in_names[] = {hair_in_name_.c_str()};
        const char * out_names[] = {hair_out_name_.c_str()};

        try {
          auto outputs = hair_session_->Run(
            Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);
          if (!outputs.empty()) {
            const auto shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
            int num_classes = (shape.size() >= 2) ? static_cast<int>(shape[1]) : static_cast<int>(hair_labels_.size());
            const float * logits = outputs.front().GetTensorData<float>();

            float max_l = -1e9f;
            for (int i = 0; i < num_classes; ++i) {
              if (logits[i] > max_l) max_l = logits[i];
            }
            float sum_exp = 0.0f;
            std::vector<float> probs(num_classes);
            for (int i = 0; i < num_classes; ++i) {
              probs[i] = std::exp(logits[i] - max_l);
              sum_exp += probs[i];
            }
            int best_idx = 0;
            float best_p = 0.0f;
            for (int i = 0; i < num_classes; ++i) {
              probs[i] /= sum_exp;
              if (probs[i] > best_p) {
                best_p = probs[i];
                best_idx = i;
              }
            }
            if (best_p > 0.35f && best_idx < static_cast<int>(hair_labels_.size())) {
              std::string res = hair_labels_[best_idx];
              std::transform(res.begin(), res.end(), res.begin(), ::tolower);
              return res;
            }
          }
        } catch (...) {
          // Fallback ke HSV jika terjadi kegagalan inferensi runtime
        }
      }
    }

    // 2. Fallback: Analisis Ruang Warna HSV
    int hw = std::max(4, static_cast<int>(bbox.width * 0.35));
    int hh = std::max(4, static_cast<int>(bbox.height * 0.15));
    int hx = std::clamp(bbox.x + static_cast<int>(bbox.width * 0.325), 0, bgr.cols - hw);
    int hy = std::clamp(bbox.y, 0, bgr.rows - hh);

    cv::Rect head_roi(hx, hy, hw, hh);
    head_roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
    if (head_roi.area() <= 0) return "dark hair";

    cv::Mat crop = bgr(head_roi);
    cv::Mat hsv;
    cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);
    cv::Scalar mean_hsv = cv::mean(hsv);

    double h = mean_hsv[0];
    double s = mean_hsv[1];
    double v = mean_hsv[2];

    if (v < 55.0) return "black hair";
    if (s < 40.0 && v > 150.0) return "gray hair";
    if (h >= 15.0 && h <= 35.0 && v > 130.0) return "blonde hair";
    if (h >= 8.0 && h <= 30.0) return "brown hair";
    return "dark hair";
  }

  // ---------- 3. Klasifikasi Gender ----------
  std::string classifyGender(const std::vector<sl::float2> & kpt, const cv::Mat & bgr, const cv::Rect & bbox)
  {
    // 1. Coba inferensi via Model AI ONNX jika tersedia
    if (gender_session_ && !bgr.empty() && bbox.width > 0 && bbox.height > 0) {
      int x1 = std::clamp(bbox.x, 0, bgr.cols);
      int y1 = std::clamp(bbox.y, 0, bgr.rows);
      int x2 = std::clamp(bbox.x + bbox.width, 0, bgr.cols);
      int upper_y2 = std::clamp(static_cast<int>(bbox.y + bbox.height * 0.55), 0, bgr.rows);

      if (upper_y2 > y1 && x2 > x1) {
        cv::Mat crop = bgr(cv::Rect(x1, y1, x2 - x1, upper_y2 - y1));
        cv::Mat resized;
        cv::resize(crop, resized, cv::Size(224, 224));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

        cv::Mat blob = cv::dnn::blobFromImage(
          resized, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), false, false);

        std::array<int64_t, 4> input_shape{1, 3, 224, 224};
        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
          ort_attr_mem_info_, blob.ptr<float>(), blob.total(),
          input_shape.data(), input_shape.size());

        const char * in_names[] = {gender_in_name_.c_str()};
        const char * out_names[] = {gender_out_name_.c_str()};

        try {
          auto outputs = gender_session_->Run(
            Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);
          if (!outputs.empty()) {
            const float * logits = outputs.front().GetTensorData<float>();
            float l0 = logits[0]; // female
            float l1 = logits[1]; // male
            float m = std::max(l0, l1);
            float exp0 = std::exp(l0 - m);
            float exp1 = std::exp(l1 - m);
            float p0 = exp0 / (exp0 + exp1);
            float p1 = exp1 / (exp0 + exp1);

            if (p0 > 0.65f) return "female";
            if (p1 > 0.65f) return "male";
          }
        } catch (...) {
          // Fallback ke heuristik jika terjadi kesalahan inferensi
        }
      }
    }

    // 2. Fallback: Analisis Proporsi Bahu / Pinggul dari Keypoint ZED
    if (kpt.size() >= 23) {
      const auto & l_sh = kpt[5];
      const auto & r_sh = kpt[12];
      const auto & l_hip = kpt[18];
      const auto & r_hip = kpt[22];

      if (std::isfinite(l_sh.x) && std::isfinite(r_sh.x) &&
          std::isfinite(l_hip.x) && std::isfinite(r_hip.x)) {
        float shoulder_w = std::abs(l_sh.x - r_sh.x);
        float hip_w = std::abs(l_hip.x - r_hip.x);
        if (hip_w > 5.0f) {
          float ratio = shoulder_w / hip_w;
          if (ratio > 1.30f) return "male";
          if (ratio < 1.10f) return "female";
        }
      }
    }
    if (bbox.height > 0 && bbox.width > 0) {
      float aspect = static_cast<float>(bbox.height) / static_cast<float>(bbox.width);
      if (aspect > 2.85f) return "male";
    }
    return "male";
  }

  sl::Camera zed_;
  sl::Mat zed_image_;
  sl::Mat zed_point_cloud_;
  sl::Bodies bodies_;
  sl::BodyTrackingRuntimeParameters body_runtime_parameters_;

  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr yaw_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr bag_dets_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr drink_dets_publisher_; // publish /zed/drink_detections (botol, cracker, drinks)
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr persons_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr gesture_publisher_;  // publish gesture RIGHT/LEFT/NONE
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr attributes_publisher_; // publish /zed/person_attributes
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr empty_seat_publisher_; // publish /zed/empty_seat (left, center, right)
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr chair_dets_publisher_; // publish /zed/chair_detections
  rclcpp::TimerBase::SharedPtr timer_;

  bool publish_yaw_{true};
  bool yaw_360_{true};
  double camera_fps_{30.0};
  bool enable_body_tracking_{true};
  bool debug_window_{false};
  bool mirror_{true};
  bool preview_inited_{false};
  bool preview_dead_{false};
  int preview_fps_{0};
  rclcpp::Time preview_fps_t_{0, 0, RCL_ROS_TIME};

  // ONNX Runtime untuk Atribut Person (Gender, Clothing, Hair)
  Ort::Env ort_attr_env_{ORT_LOGGING_LEVEL_WARNING, "zed_attributes"};
  Ort::MemoryInfo ort_attr_mem_info_ = Ort::MemoryInfo::CreateCpu(
    OrtArenaAllocator, OrtMemTypeDefault);

  std::unique_ptr<Ort::Session> gender_session_;
  std::string gender_in_name_;
  std::string gender_out_name_;

  std::unique_ptr<Ort::Session> color_session_;
  std::string color_in_name_;
  std::string color_out_name_;
  std::vector<std::string> color_labels_;

  std::unique_ptr<Ort::Session> hair_session_;
  std::string hair_in_name_;
  std::string hair_out_name_;
  std::vector<std::string> hair_labels_;

  struct PrevPerson {
    int id{-1};
    cv::Rect bbox;
    bool has_pos{false};
    double x_m{0.0};
    double y_m{0.0};
    double range_m{0.0};
    bool is_sitting{false};
    std::vector<cv::Point2f> kpt;
    std::string clothing{""};
    std::string hair{""};
    std::string gender{""};
  };
  struct PrevBag {
    cv::Rect bbox;
    bool has_pos{false};
    double x_m{0.0};
    double y_m{0.0};
    double range_m{0.0};
  };
  struct PrevItem {
    std::string label;
    cv::Rect bbox;
    bool has_pos{false};
    double x_m{0.0};
    double y_m{0.0};
    double range_m{0.0};
  };
  struct PrevChair {
    cv::Rect bbox;
    bool has_pos{false};
    double x_m{0.0};
    double y_m{0.0};
    double range_m{0.0};
    bool occupied{false};
    std::string side{"center"};
  };
  std::vector<PrevPerson> preview_persons_;
  std::vector<PrevBag> preview_bags_;
  std::vector<PrevItem> preview_drinks_;
  std::vector<PrevChair> preview_chairs_;

  YoloDetector yolo_;
  YoloDetector botol_yolo_;
  YoloDetector cracker_yolo_;
  YoloDetector pringles_yolo_;
  YoloDetector chair_yolo_;

  bool enable_yolo_{true};
  bool enable_bag_{true};
  bool enable_drinks_{true};
  bool enable_chair_{true};
  std::vector<std::string> bag_labels_;
  std::vector<std::string> chair_labels_;
  double yolo_min_area_{1500.0};
  double drink_min_area_{100.0};
  int yolo_stride_{2};
  int yolo_frame_{0};
  int drink_frame_{0};
  int chair_frame_{0};
  std::vector<YoloDetection> last_bag_dets_;
  std::vector<YoloDetection> last_drink_dets_;
  std::vector<YoloDetection> last_chair_dets_;

  // Memori Operator (Stabilisasi ID & Re-ID saat hilang-timbul)
  struct OperatorMemory {
    bool active{false};
    int stable_id{0};               // ID konsisten yang selalu dipublish untuk orang awal
    double x{0.0};
    double y{0.0};
    double vx{0.0};                 // Estimasi kecepatan X (m/s)
    double vy{0.0};                 // Estimasi kecepatan Y (m/s)
    cv::Scalar torso_hsv{0, 0, 0};   // Karakteristik warna pakaian
    bool has_color{false};
    rclcpp::Time last_seen{0, 0, RCL_ROS_TIME};
  } op_mem_;

  std::string task_mode_{"all"};
  bool is_help_me_carry_{true};
  bool is_receptionist_{true};

  std::string guest_name_{""};
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr person_name_sub_;
};

ZedNode::ZedNode()
: Node("zed")
{
  declare_parameter("task", "all");
  declare_parameter("camera_fps", 30.0);
  declare_parameter("publish_yaw", true);
  declare_parameter("yaw_360", true);
  declare_parameter("enable_body_tracking", true);
  declare_parameter("depth_minimum_distance", 30.0);
  declare_parameter("enable_yolo", true);
  declare_parameter("debug_window", false);
  declare_parameter("mirror", true);
  declare_parameter("yolo_model", std::string("models/yolov8n_sim.onnx"));
  declare_parameter("yolo_classes", std::string("models/coco_names.txt"));
  declare_parameter("yolo_conf", 0.25);
  declare_parameter("yolo_nms", 0.45);
  declare_parameter("yolo_size", 640);
  declare_parameter("yolo_stride", 2);
  declare_parameter("bag_labels",std::vector<std::string>{"bag", "backpack", "handbag", "suitcase"});
  declare_parameter("yolo_min_area", 1500.0);
  declare_parameter("drink_min_area", 100.0);
  declare_parameter("drink_conf", 0.20);
  declare_parameter("botol_model", std::string("models/water.onnx"));
  declare_parameter("cracker_model", std::string("models/cheezit.onnx"));
  declare_parameter("pringles_model", std::string("models/pringles_ chips _can.onnx"));
  declare_parameter("enable_drinks", true);
  declare_parameter("chair_model", std::string(""));
  declare_parameter("enable_chair", true);
  declare_parameter("chair_labels", std::vector<std::string>{"chair", "couch", "bench"});
  declare_parameter("enable_bag", true);

  // Parameter Model AI untuk Atribut Orang (Gender, Warna Pakaian, Warna Rambut)
  declare_parameter("gender_model", std::string("models/gender_cls.onnx"));
  declare_parameter("color_model", std::string("models/color_model.onnx"));
  declare_parameter("color_labels", std::string("models/color_names.txt"));
  declare_parameter("hair_model", std::string("models/hair_model.onnx"));
  declare_parameter("hair_labels", std::string("models/hair_names.txt"));

  task_mode_ = get_parameter("task").as_string();
  std::string t = task_mode_;
  std::transform(t.begin(), t.end(), t.begin(), ::tolower);
  if (t == "help_me_carry" || t == "hmc") {
    is_help_me_carry_ = true;
    is_receptionist_ = false;
  } else if (t == "receptionist" || t == "rec") {
    is_help_me_carry_ = false;
    is_receptionist_ = true;
  } else {
    is_help_me_carry_ = true;
    is_receptionist_ = true;
  }
  enable_bag_ = get_parameter("enable_bag").as_bool();
  std::cout << "Vision Task Mode: " << task_mode_ << ", enable_bag: " << (enable_bag_ ? "true" : "false") << std::endl;

  const double camera_fps = get_parameter("camera_fps").as_double();
  camera_fps_ = camera_fps;

  publish_yaw_ = get_parameter("publish_yaw").as_bool();
  yaw_360_ = get_parameter("yaw_360").as_bool();
  enable_body_tracking_ = get_parameter("enable_body_tracking").as_bool();

  if (camera_fps <= 0.0) {
    throw std::runtime_error("camera_fps must be greater than zero");
  }

  yaw_publisher_ = create_publisher<std_msgs::msg::Float64>("/zed/yaw_deg", 10);
  bag_dets_publisher_ = create_publisher<std_msgs::msg::String>("/zed/bag_detections", 10);
  drink_dets_publisher_ = create_publisher<std_msgs::msg::String>("/zed/drink_detections", 10);
  persons_publisher_ = create_publisher<std_msgs::msg::String>("/zed/person_label", 10);
  gesture_publisher_ = create_publisher<std_msgs::msg::String>("/zed/gesture", 10);
  attributes_publisher_ = create_publisher<std_msgs::msg::String>("/zed/person_attributes", 10);
  empty_seat_publisher_ = create_publisher<std_msgs::msg::String>("/zed/empty_seat", 10);
  chair_dets_publisher_ = create_publisher<std_msgs::msg::String>("/zed/chair_detections", 10);

  person_name_sub_ = create_subscription<std_msgs::msg::String>(
    "/zed/set_person_name", 10,
    [this](const std_msgs::msg::String::SharedPtr msg) {
      if (!msg) return;
      std::string n = msg->data;
      while (!n.empty() && (n.back() == ' ' || n.back() == '\r' || n.back() == '\n' || n.back() == '\t')) {
        n.pop_back();
      }
      while (!n.empty() && (n.front() == ' ' || n.front() == '\t')) {
        n.erase(n.begin());
      }
      guest_name_ = n;
      if (!guest_name_.empty()) {
        std::cout << "Guest name set: " << guest_name_ << std::endl;
      } else {
        std::cout << "Guest name reset" << std::endl;
      }
    });

  sl::InitParameters init_parameters;
  init_parameters.depth_mode = sl::DEPTH_MODE::ULTRA;
  init_parameters.coordinate_units = sl::UNIT::CENTIMETER;
  init_parameters.camera_resolution = sl::RESOLUTION::VGA;
  init_parameters.camera_fps = static_cast<int>(camera_fps);
  init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
  init_parameters.depth_minimum_distance =
    static_cast<float>(get_parameter("depth_minimum_distance").as_double());

  std::cout << "### STARTING CAMERA ###" << std::endl;

  sl::ERROR_CODE state = sl::ERROR_CODE::FAILURE;
  for (int attempt = 1; attempt <= 5; ++attempt) {
    state = zed_.open(init_parameters);
    if (state == sl::ERROR_CODE::SUCCESS) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  if (state != sl::ERROR_CODE::SUCCESS) {
    std::cerr << "ZED open failed setelah 5 percobaan: " << sl::toString(state) << std::endl;
    throw std::runtime_error(std::string("ZED open failed: ") + sl::toString(state).c_str());
  }

  sl::PositionalTrackingParameters positional_parameters;
  const auto tracking_state = zed_.enablePositionalTracking(positional_parameters);
  if (tracking_state != sl::ERROR_CODE::SUCCESS) {
    zed_.close();
    throw std::runtime_error("Enable positional tracking failed");
  }

  sl::BodyTrackingParameters body_parameters;
  body_parameters.detection_model = sl::BODY_TRACKING_MODEL::HUMAN_BODY_MEDIUM;
  body_parameters.body_format = sl::BODY_FORMAT::BODY_34;
  body_parameters.enable_tracking = true;
  body_parameters.enable_body_fitting = true;

  if (enable_body_tracking_) {
    const auto body_state = zed_.enableBodyTracking(body_parameters);
    if (body_state != sl::ERROR_CODE::SUCCESS) {
      zed_.close();
      throw std::runtime_error("Enable body tracking failed");
    }
  }

  body_runtime_parameters_.detection_confidence_threshold = 25;

  enable_yolo_   = get_parameter("enable_yolo").as_bool();
  debug_window_  = get_parameter("debug_window").as_bool();
  mirror_        = get_parameter("mirror").as_bool();
  bag_labels_    = get_parameter("bag_labels").as_string_array();
  yolo_min_area_ = get_parameter("yolo_min_area").as_double();
  drink_min_area_ = get_parameter("drink_min_area").as_double();
  yolo_stride_   = std::max(1, static_cast<int>(get_parameter("yolo_stride").as_int()));

  auto resolve = [](const std::string & p) {
    if (p.empty()) return std::string("");
    if (std::filesystem::path(p).is_absolute()) return p;
    try {
      std::string sp = ament_index_cpp::get_package_share_directory("athome_vision") + "/" + p;
      if (std::filesystem::exists(sp)) return sp;
    } catch (...) {}
    if (std::filesystem::exists(p)) return p;
    if (std::filesystem::exists("src/athome_vision/" + p)) return std::string("src/athome_vision/") + p;
    const std::string ws_src = "/home/dell/Documents/athome_ws/src/athome_vision/" + p;
    if (std::filesystem::exists(ws_src)) return ws_src;
    return std::string("share/athome_vision/") + p;
  };

  if (enable_yolo_) {
    std::string model_param = get_parameter("yolo_model").as_string();
    if (model_param.empty()) {
      enable_yolo_ = false;
    } else {
      std::string model_path = resolve(model_param);
      if (!std::filesystem::exists(model_path)) {
        std::cerr << "YOLO model not found, YOLO disabled: " << model_path << std::endl;
        enable_yolo_ = false;
      } else {
        std::string classes_path = resolve(get_parameter("yolo_classes").as_string());
        float conf = static_cast<float>(get_parameter("yolo_conf").as_double());
        float nms  = static_cast<float>(get_parameter("yolo_nms").as_double());
        int size   = static_cast<int>(get_parameter("yolo_size").as_int());

        std::string err;
        if (!yolo_.load(model_path, classes_path, conf, nms, size, "bag", &err)) {
          std::cerr << "YOLO load error: " << err << std::endl;
          enable_yolo_ = false;
        } else {
          std::cout << "YOLO loaded successfully: " << model_path << std::endl;
        }
      }
    }
  }

  // Load Model-model YOLO untuk Deteksi Minuman / Makanan (Botol & Cracker)
  enable_drinks_ = get_parameter("enable_drinks").as_bool();
  if (enable_drinks_) {
    float drink_conf = static_cast<float>(get_parameter("drink_conf").as_double());
    float nms  = static_cast<float>(get_parameter("yolo_nms").as_double());
    int size   = static_cast<int>(get_parameter("yolo_size").as_int());

    const std::string botol_param = get_parameter("botol_model").as_string();
    if (!botol_param.empty()) {
      std::string botol_path = resolve(botol_param);
      if (!std::filesystem::exists(botol_path)) {
        std::string fallback = resolve("models/botol.onnx");
        if (std::filesystem::exists(fallback)) botol_path = fallback;
      }
      if (std::filesystem::exists(botol_path)) {
        std::string err;
        if (botol_yolo_.load(botol_path, "", drink_conf, nms, size, "Water", &err)) {
          std::cout << "Water/Botol ONNX model loaded successfully: " << botol_path << std::endl;
        } else {
          std::cerr << "Water/Botol ONNX load error: " << err << std::endl;
        }
      } else {
        std::cerr << "Water/Botol ONNX model not found: " << botol_path << std::endl;
      }
    }

    const std::string cracker_param = get_parameter("cracker_model").as_string();
    if (!cracker_param.empty()) {
      std::string cracker_path = resolve(cracker_param);
      if (!std::filesystem::exists(cracker_path)) {
        std::string fallback = resolve("models/cheezit.onnx");
        if (!std::filesystem::exists(fallback)) fallback = resolve("models/best_cracker.onnx");
        if (std::filesystem::exists(fallback)) cracker_path = fallback;
      }
      if (std::filesystem::exists(cracker_path)) {
        std::string err;
        if (cracker_yolo_.load(cracker_path, "", drink_conf, nms, size, "Cheez-It", &err)) {
          std::cout << "Cheez-It/Cracker ONNX model loaded successfully: " << cracker_path << std::endl;
        } else {
          std::cerr << "Cheez-It/Cracker ONNX load error: " << err << std::endl;
        }
      } else {
        std::cerr << "Cheez-It/Cracker ONNX model not found: " << cracker_path << std::endl;
      }
    }

    const std::string pringles_param = get_parameter("pringles_model").as_string();
    if (!pringles_param.empty()) {
      std::string pringles_path = resolve(pringles_param);
      if (!std::filesystem::exists(pringles_path)) {
        std::string fallback = resolve("models/pringles_ chips _can.onnx");
        if (!std::filesystem::exists(fallback)) fallback = resolve("models/pringles.onnx");
        if (std::filesystem::exists(fallback)) pringles_path = fallback;
      }
      if (std::filesystem::exists(pringles_path)) {
        std::string err;
        if (pringles_yolo_.load(pringles_path, "", drink_conf, nms, size, "Pringles", &err)) {
          std::cout << "Pringles ONNX model loaded successfully: " << pringles_path << std::endl;
        } else {
          std::cerr << "Pringles ONNX load error: " << err << std::endl;
        }
      } else {
        std::cerr << "Pringles ONNX model not found: " << pringles_path << std::endl;
      }
    }
  }

  // Load Model YOLO untuk Deteksi Kursi (jika ada model custom kursi, jika tidak fallback ke YOLO COCO)
  enable_chair_ = get_parameter("enable_chair").as_bool();
  chair_labels_ = get_parameter("chair_labels").as_string_array();
  if (enable_chair_) {
    const std::string chair_param = get_parameter("chair_model").as_string();
    if (!chair_param.empty()) {
      std::string chair_path = resolve(chair_param);
      if (std::filesystem::exists(chair_path)) {
        float conf = static_cast<float>(get_parameter("yolo_conf").as_double());
        float nms  = static_cast<float>(get_parameter("yolo_nms").as_double());
        int size   = static_cast<int>(get_parameter("yolo_size").as_int());
        std::string err;
        if (chair_yolo_.load(chair_path, "", conf, nms, size, "chair", &err)) {
          std::cout << "Custom Chair ONNX model loaded successfully: " << chair_path << std::endl;
        } else {
          std::cerr << "Custom Chair ONNX load error: " << err << std::endl;
        }
      }
    }
  }

  // Load Model-model AI untuk Atribut Orang (Gender, Color, Hair)
  Ort::SessionOptions attr_opts;
  attr_opts.SetIntraOpNumThreads(2);
  attr_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  Ort::AllocatorWithDefaultOptions alloc;

  // 1. Gender Classification Model
  const std::string gender_param = get_parameter("gender_model").as_string();
  if (!gender_param.empty()) {
    std::string gender_path = resolve(gender_param);
    if (std::filesystem::exists(gender_path)) {
      try {
        gender_session_ = std::make_unique<Ort::Session>(ort_attr_env_, gender_path.c_str(), attr_opts);
        gender_in_name_ = gender_session_->GetInputNameAllocated(0, alloc).get();
        gender_out_name_ = gender_session_->GetOutputNameAllocated(0, alloc).get();
        std::cout << "Gender ONNX model loaded successfully: " << gender_path << std::endl;
      } catch (const std::exception & e) {
        std::cerr << "Gender ONNX model load error: " << e.what() << std::endl;
        gender_session_.reset();
      }
    } else {
      std::cout << "Gender ONNX model not found, fallback to heuristic: " << gender_path << std::endl;
    }
  }

  // 2. Clothing Color Classification Model & Labels
  const std::string color_labels_param = get_parameter("color_labels").as_string();
  if (!color_labels_param.empty()) {
    std::string color_labels_path = resolve(color_labels_param);
    if (std::filesystem::exists(color_labels_path)) {
      color_labels_ = loadLabels(color_labels_path);
    }
  }
  const std::string color_param = get_parameter("color_model").as_string();
  if (!color_param.empty()) {
    std::string color_path = resolve(color_param);
    if (std::filesystem::exists(color_path)) {
      try {
        color_session_ = std::make_unique<Ort::Session>(ort_attr_env_, color_path.c_str(), attr_opts);
        color_in_name_ = color_session_->GetInputNameAllocated(0, alloc).get();
        color_out_name_ = color_session_->GetOutputNameAllocated(0, alloc).get();
        std::cout << "Clothing color ONNX model loaded successfully: " << color_path << " (" << color_labels_.size() << " labels)" << std::endl;
      } catch (const std::exception & e) {
        std::cerr << "Clothing color ONNX model load error: " << e.what() << std::endl;
        color_session_.reset();
      }
    } else {
      std::cout << "Clothing color ONNX model not found, fallback to HSV: " << color_path << std::endl;
    }
  }

  // 3. Hair Classification Model & Labels
  const std::string hair_labels_param = get_parameter("hair_labels").as_string();
  if (!hair_labels_param.empty()) {
    std::string hair_labels_path = resolve(hair_labels_param);
    if (std::filesystem::exists(hair_labels_path)) {
      hair_labels_ = loadLabels(hair_labels_path);
    }
  }
  const std::string hair_param = get_parameter("hair_model").as_string();
  if (!hair_param.empty()) {
    std::string hair_path = resolve(hair_param);
    if (std::filesystem::exists(hair_path)) {
      try {
        hair_session_ = std::make_unique<Ort::Session>(ort_attr_env_, hair_path.c_str(), attr_opts);
        hair_in_name_ = hair_session_->GetInputNameAllocated(0, alloc).get();
        hair_out_name_ = hair_session_->GetOutputNameAllocated(0, alloc).get();
        std::cout << "Hair ONNX model loaded successfully: " << hair_path << " (" << hair_labels_.size() << " labels)" << std::endl;
      } catch (const std::exception & e) {
        std::cerr << "Hair ONNX model load error: " << e.what() << std::endl;
        hair_session_.reset();
      }
    } else {
      std::cout << "Hair ONNX model not found, fallback to HSV: " << hair_path << std::endl;
    }
  }

  const auto period = std::chrono::duration<double>(1.0 / camera_fps);
  timer_ = create_wall_timer(period, std::bind(&ZedNode::loop, this));

  std::cout << "################Start capturing################" << std::endl;
}

ZedNode::~ZedNode()
{
  zed_.close();
}

void ZedNode::publishYaw()
{
  if (!publish_yaw_) return;
  sl::SensorsData sensors_data;
  if (zed_.getSensorsData(sensors_data, sl::TIME_REFERENCE::IMAGE) != sl::ERROR_CODE::SUCCESS ||
      !sensors_data.imu.is_available)
  {
    return;
  }
  const auto q = sensors_data.imu.pose.getOrientation();
  float yaw = zed_vision::quatToYaw(q.x, q.y, q.z, q.w);
  const float yaw_deg = yaw * 180.0f / zed_vision::PI_F;

  std_msgs::msg::Float64 message;
  message.data = static_cast<double>(
    yaw_360_ ? zed_vision::wrap360(yaw_deg) : zed_vision::wrap180(yaw_deg));
  yaw_publisher_->publish(message);
}

void ZedNode::processBodiesAndYolo(const cv::Mat & bgr, const sl::Mat & point_cloud)
{
  using namespace zed_vision;

  preview_bags_.clear();
  if (enable_bag_) {
    std::vector<YoloDetection> bag_dets = last_bag_dets_;
    if (enable_yolo_ && yolo_.ready()) {
      if (yolo_frame_++ % std::max(1, yolo_stride_) == 0) {
        yolo_.detect(bgr, bag_labels_, bag_dets);
        last_bag_dets_ = bag_dets;
      }
    }

    std::string bag_str;
    for (const auto & d : bag_dets) {
      if (static_cast<double>(d.rect.area()) < yolo_min_area_) continue;
      double x = 0.0, y = 0.0, z = 0.0;
      const bool ok = get3dCoordinatesWithOffset(d.rect, point_cloud, x, y, z);
      const double dist = ok ? std::hypot(x, y) : 0.0;
      preview_bags_.push_back({d.rect, ok, x, y, dist});
      if (is_help_me_carry_ && ok) {
        IC(d.label, x, y, dist);
      }

      if (!bag_str.empty()) bag_str += ";";
      bag_str += cv::format("bag:%d,%d,%d,%d|", d.rect.x, d.rect.y, d.rect.width, d.rect.height);
      bag_str += ok ? cv::format("%.2f,%.2f,%.2f|%.2f", x, y, z, d.score) : cv::format("nan,nan,nan|%.2f", d.score);
    }
    std_msgs::msg::String bag_msg;
    bag_msg.data = bag_str;
    bag_dets_publisher_->publish(bag_msg);
  }

  // Deteksi Minuman & Makanan (Botol & Cracker) untuk Receptionist
  std::vector<YoloDetection> drink_dets = last_drink_dets_;
  if (enable_drinks_) {
    if (drink_frame_++ % std::max(1, yolo_stride_) == 0) {
      drink_dets.clear();
      if (botol_yolo_.ready()) {
        std::vector<YoloDetection> b_dets;
        botol_yolo_.detect(bgr, {}, b_dets);
        drink_dets.insert(drink_dets.end(), b_dets.begin(), b_dets.end());
      }
      if (cracker_yolo_.ready()) {
        std::vector<YoloDetection> c_dets;
        cracker_yolo_.detect(bgr, {}, c_dets);
        drink_dets.insert(drink_dets.end(), c_dets.begin(), c_dets.end());
      }
      if (pringles_yolo_.ready()) {
        std::vector<YoloDetection> p_dets;
        pringles_yolo_.detect(bgr, {}, p_dets);
        drink_dets.insert(drink_dets.end(), p_dets.begin(), p_dets.end());
      }
      last_drink_dets_ = drink_dets;
    }
  }

  preview_drinks_.clear();
  std::string drink_str;
  for (const auto & d : drink_dets) {
    if (static_cast<double>(d.rect.area()) < drink_min_area_) continue;
    double x = 0.0, y = 0.0, z = 0.0;
    const bool ok = get3dCoordinatesWithOffset(d.rect, point_cloud, x, y, z);
    const double dist = ok ? std::hypot(x, y) : 0.0;
    preview_drinks_.push_back({d.label, d.rect, ok, x, y, dist});
    if (is_receptionist_ && ok) {
      IC(d.label, x, y, dist);
    }

    if (!drink_str.empty()) drink_str += ";";
    drink_str += cv::format("%s:%d,%d,%d,%d|", d.label.c_str(), d.rect.x, d.rect.y, d.rect.width, d.rect.height);
    drink_str += ok ? cv::format("%.2f,%.2f,%.2f|%.2f", x, y, z, d.score) : cv::format("nan,nan,nan|%.2f", d.score);
  }
  std_msgs::msg::String drink_msg;
  drink_msg.data = drink_str;
  drink_dets_publisher_->publish(drink_msg);

  preview_persons_.clear();
  std::string person_str = "NO_PERSON";
  if (bodies_.body_list.empty()) {
    std_msgs::msg::String m;
    m.data = "NO_PERSON";
    persons_publisher_->publish(m);
    // Tidak ada orang → gesture NONE
    std_msgs::msg::String g;
    g.data = "NONE";
    gesture_publisher_->publish(g);
    // Atribut kosong
    std_msgs::msg::String attr_msg;
    attr_msg.data = "";
    attributes_publisher_->publish(attr_msg);
  } else {
    struct PersonEntry {
      const sl::BodyData * body{nullptr};
      int id{-1};
      cv::Rect bbox;
      std::string label{"operator"};
      double x{0.0}, y{0.0}, z{0.0};
      bool has_depth{false};
      double range_m{std::numeric_limits<double>::infinity()};
      double dist{std::numeric_limits<double>::infinity()};
    };
    std::vector<PersonEntry> entries;
    for (const auto & body : bodies_.body_list) {
      if (body.keypoint_2d.size() < 34) continue;
      cv::Rect bbox;
      if (!computeBbox(body.keypoint_2d, bgr.cols, bgr.rows, bbox)) continue;

      PersonEntry e;
      e.body = &body;
      e.id = body.id;
      e.bbox = bbox;
      e.has_depth = get3dCoordinatesWithOffset(bbox, point_cloud, e.x, e.y, e.z);
      e.range_m = (e.has_depth && std::isfinite(e.x) && std::isfinite(e.y))
                  ? std::hypot(e.x, e.y) : (1e6 - static_cast<double>(bbox.area()));
      e.dist = e.range_m;
      entries.push_back(e);
    }

    std::sort(entries.begin(), entries.end(), [](const PersonEntry & a, const PersonEntry & b) {
      return a.range_m < b.range_m;
    });

    const auto current_time = now();

    // Logika Memori Operator: Menjaga ID orang awal tetap stabil meskipun hilang-timbul
    int op_match_idx = -1;
    if (op_mem_.active) {
      const double dt = (current_time - op_mem_.last_seen).seconds();
      // Toleransi hilang-timbul hingga 4.0 detik
      if (dt <= 4.0) {
        // Prediksi posisi berdasarkan kecepatan (Kalman / Constant Velocity model)
        const double pred_x = op_mem_.x + op_mem_.vx * std::min(dt, 2.0);
        const double pred_y = op_mem_.y + op_mem_.vy * std::min(dt, 2.0);
        // Radius pencarian melebar bertahap seiring durasi hilang
        const double max_match_dist = 1.0 + 0.8 * dt;

        double best_cost = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < entries.size(); ++i) {
          const auto & cand = entries[i];
          if (!cand.has_depth) continue;

          const double dist_diff = std::hypot(cand.x - pred_x, cand.y - pred_y);
          if (dist_diff > max_match_dist) continue;

          // Hitung penalti warna pakaian torso jika tersedia
          double color_penalty = 0.0;
          if (op_mem_.has_color) {
            const cv::Scalar cand_hsv = zed_vision::extractTorsoColor(bgr, cand.bbox);
            // Selisih nilai Hue & Saturation
            const double dh = std::abs(cand_hsv[0] - op_mem_.torso_hsv[0]);
            const double ds = std::abs(cand_hsv[1] - op_mem_.torso_hsv[1]);
            color_penalty = (dh / 180.0) * 0.5 + (ds / 255.0) * 0.3;
          }

          const double total_cost = dist_diff + color_penalty;
          if (total_cost < best_cost) {
            best_cost = total_cost;
            op_match_idx = static_cast<int>(i);
          }
        }
      } else {
        // Timeout > 4 detik: reset memori agar bisa mengunci operator baru
        op_mem_.active = false;
        op_mem_.has_color = false;
      }
    }

    // Jika belum ada operator yang dikunci atau match ditemukan
    if (!op_mem_.active && !entries.empty()) {
      // Kunci orang terdekat pertama kali sebagai operator
      op_match_idx = 0;
      op_mem_.active = true;
      op_mem_.stable_id = 0;  // Tetapkan ID stabil selalu 0
      op_mem_.vx = 0.0;
      op_mem_.vy = 0.0;
      if (entries[0].has_depth) {
        op_mem_.x = entries[0].x;
        op_mem_.y = entries[0].y;
      }
      op_mem_.torso_hsv = zed_vision::extractTorsoColor(bgr, entries[0].bbox);
      op_mem_.has_color = true;
      op_mem_.last_seen = current_time;
      if (is_help_me_carry_) {
        IC(op_mem_.stable_id, op_mem_.x, op_mem_.y);
      }
    }

    // Jika kandidat operator cocok dengan memori, update posisinya & jadikan urutan pertama (operator)
    if (op_match_idx >= 0 && op_match_idx < static_cast<int>(entries.size())) {
      if (op_match_idx > 0) {
        std::swap(entries[0], entries[static_cast<size_t>(op_match_idx)]);
      }
      auto & op = entries.front();
      op.id = op_mem_.stable_id;  // Paksa ID ZED kembali ke ID stabil orang awal!
      op.label = "operator";

      if (op.has_depth) {
        const double dt = std::max(0.01, (current_time - op_mem_.last_seen).seconds());
        if (dt < 1.0) {
          // Update estimasi kecepatan dengan low-pass filter
          op_mem_.vx = 0.7 * op_mem_.vx + 0.3 * ((op.x - op_mem_.x) / dt);
          op_mem_.vy = 0.7 * op_mem_.vy + 0.3 * ((op.y - op_mem_.y) / dt);
        }
        op_mem_.x = op.x;
        op_mem_.y = op.y;
      }
      op_mem_.last_seen = current_time;
      // Perbarui warna torso secara adaptif (moving average)
      const cv::Scalar new_hsv = zed_vision::extractTorsoColor(bgr, op.bbox);
      op_mem_.torso_hsv = 0.85 * op_mem_.torso_hsv + 0.15 * new_hsv;
      op_mem_.has_color = true;
    }

    if (!entries.empty()) {
      entries.front().label = "operator";
      if (is_help_me_carry_ && entries.front().has_depth) {
        const auto & person = entries.front();
        IC(person.id, person.label, person.x, person.y, person.dist);
      }
    }

    std::ostringstream ss;
    bool first_p = true;
    std::string gesture_result = "NONE";  // gesture dari operator terdekat
    for (const auto & e : entries) {
      PrevPerson pp{e.id, e.bbox, e.has_depth, e.x, e.y, e.range_m, false, {}};
      const auto & points = e.body->keypoint_2d;
      for (const auto & pt : points) {
        pp.kpt.emplace_back(std::isfinite(pt.x) ? pt.x : NAN, std::isfinite(pt.y) ? pt.y : NAN);
      }

      // Klasifikasi 3 Atribut Visual Tamu (Baju dari pinggang ke atas, Rambut, Gender)
      // Khusus Receptionist: Hanya dijalankan SETELAH tamu memberi tahu namanya (guest_name_ tidak kosong).
      // Pada Help Me Carry: HANYA mendeteksi person dan bag saja tanpa ada klasifikasi atribut.
      const bool should_classify_attributes = (is_receptionist_ && !guest_name_.empty());
      std::string shirt = "";
      std::string hair = "";
      std::string gender = "";

      if (should_classify_attributes) {
        shirt = classifyClothingColor(bgr, e.bbox, points);
        hair = classifyHairColor(bgr, e.bbox);
        gender = classifyGender(points, bgr, e.bbox);
      }

      // Deteksi gesture & publish atribut dari orang terdekat (entry pertama)
      if (first_p) {
        gesture_result = detectGesture(points);
        if (is_help_me_carry_ && gesture_result != "NONE") {
          IC(e.id, gesture_result);
        }

        // Publish ke /zed/person_attributes untuk task receptionist
        std_msgs::msg::String attr_msg;
        if (should_classify_attributes) {
          attr_msg.data = gender + ", " + shirt + ", " + hair;
        } else {
          attr_msg.data = "";
        }
        attributes_publisher_->publish(attr_msg);

        if (is_receptionist_) {
          std::string disp_name = (is_help_me_carry_) ? ("ID " + std::to_string(e.id)) : (guest_name_.empty() ? "Guest" : guest_name_);
          IC(disp_name, e.x, e.y, e.range_m);
          if (should_classify_attributes) {
            IC(disp_name, gender, shirt, hair);
          }
        }
      }

      pp.clothing = shirt;
      pp.hair = hair;
      pp.gender = gender;
      preview_persons_.push_back(pp);

      if (!first_p) ss << ";";
      std::string label;
      if (is_receptionist_ && !is_help_me_carry_) {
        // Task Receptionist: Tanpa ID, gunakan nama tamu dari memori HRI jika ada
        label = guest_name_.empty() ? "guest" : guest_name_;
      } else {
        // Task Help Me Carry: Gunakan ID operator
        label = first_p ? "operator" : "person";
      }
      first_p = false;
      ss << label << ":" << e.bbox.x << "," << e.bbox.y << "," << e.bbox.width << "," << e.bbox.height << "|";
      if (e.has_depth) ss << std::fixed << std::setprecision(2) << e.x << "," << e.y << "," << e.z;
      else ss << "nan,nan,nan";
      ss << "|id=" << e.id << "|kpt=";

      for (size_t k = 0; k < 34; ++k) {
        if (k > 0) ss << ",";
        if (k < points.size() && std::isfinite(points[k].x) && std::isfinite(points[k].y)) {
          ss << std::fixed << std::setprecision(2) << points[k].x << "," << points[k].y;
        } else {
          ss << "-1,-1";
        }
      }
    }

    // Publish gesture ke /zed/gesture
    std_msgs::msg::String gesture_msg;
    gesture_msg.data = gesture_result;
    gesture_publisher_->publish(gesture_msg);

    person_str = first_p ? "NO_PERSON" : ss.str();
    std_msgs::msg::String m;
    m.data = person_str;
    persons_publisher_->publish(m);
  }

  // Deteksi Kursi & Klasifikasi Kosong vs Penuh untuk Receptionist
  std::vector<YoloDetection> chair_dets = last_chair_dets_;
  if (enable_chair_) {
    if (chair_frame_++ % std::max(1, yolo_stride_) == 0) {
      chair_dets.clear();
      if (chair_yolo_.ready()) {
        chair_yolo_.detect(bgr, {}, chair_dets);
      } else if (enable_yolo_ && yolo_.ready()) {
        yolo_.detect(bgr, chair_labels_, chair_dets);
      }
      last_chair_dets_ = chair_dets;
    }
  }

  preview_chairs_.clear();
  std::string chair_str;
  std::string best_empty_side = "none";
  double best_empty_dist = 1e9;

  for (const auto & d : chair_dets) {
    if (static_cast<double>(d.rect.area()) < yolo_min_area_) continue;
    double x = 0.0, y = 0.0, z = 0.0;
    const bool ok = get3dCoordinatesWithOffset(d.rect, point_cloud, x, y, z);
    const double dist = ok ? std::hypot(x, y) : 0.0;

    // Tentukan apakah kursi terisi (occupied) atau kosong (empty)
    bool is_occupied = false;
    for (auto & p : preview_persons_) {
      cv::Rect inter = d.rect & p.bbox;
      if (inter.area() > 0) {
        const double chair_area = static_cast<double>(d.rect.area());
        const double overlap_ratio = static_cast<double>(inter.area()) / chair_area;
        const double person_overlap = static_cast<double>(inter.area()) / static_cast<double>(p.bbox.area());
        if (overlap_ratio > 0.18 || person_overlap > 0.20) {
          is_occupied = true;
          p.is_sitting = true;
          break;
        }
      }
      // Cek apakah titik tengah badan atau titik kunci orang berada di dalam kotak kursi
      const int p_cx = p.bbox.x + p.bbox.width / 2;
      const int p_cy = p.bbox.y + p.bbox.height / 2;
      if (d.rect.contains(cv::Point(p_cx, p_cy))) {
        is_occupied = true;
        p.is_sitting = true;
        break;
      }
      for (const auto & kp : p.kpt) {
        if (std::isfinite(kp.x) && std::isfinite(kp.y)) {
          if (d.rect.contains(cv::Point(static_cast<int>(kp.x), static_cast<int>(kp.y)))) {
            is_occupied = true;
            p.is_sitting = true;
            break;
          }
        }
      }
      // Cek jarak 3D jika keduanya memiliki data depth
      if (ok && p.has_pos) {
        const double d3d = std::hypot(x - p.x_m, y - p.y_m);
        if (d3d < 0.65) {
          is_occupied = true;
          p.is_sitting = true;
          break;
        }
      }
      if (is_occupied) {
        p.is_sitting = true;
        break;
      }
    }

    // Tentukan sisi horizontal (left, center, right)
    std::string side = "center";
    if (ok) {
      if (y > 0.20) {
        side = "left";
      } else if (y < -0.20) {
        side = "right";
      }
    } else {
      const int chair_cx = d.rect.x + d.rect.width / 2;
      const int W = bgr.cols > 0 ? bgr.cols : 640;
      if (chair_cx < W / 3) {
        side = mirror_ ? "right" : "left";
      } else if (chair_cx > 2 * W / 3) {
        side = mirror_ ? "left" : "right";
      }
    }

    preview_chairs_.push_back({d.rect, ok, x, y, dist, is_occupied, side});

    if (!is_occupied) {
      if (ok && dist > 0.1 && dist < best_empty_dist) {
        best_empty_dist = dist;
        best_empty_side = side;
      } else if (best_empty_side == "none") {
        best_empty_side = side;
      }
    }

    if (is_receptionist_) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "\033[1;35m[CHAIR]\033[0m Kursi %s di sisi %s (3D: x=%.2f m, y=%.2f m, dist=%.2f m)",
        is_occupied ? "TERISI (OCCUPIED)" : "KOSONG (EMPTY)", side.c_str(), x, y, dist);
    }

    if (!chair_str.empty()) chair_str += ";";
    chair_str += cv::format("chair:%d,%d,%d,%d|", d.rect.x, d.rect.y, d.rect.width, d.rect.height);
    chair_str += ok ? cv::format("%.2f,%.2f,%.2f|", x, y, z) : "nan,nan,nan|";
    chair_str += is_occupied ? "occupied|" : "empty|";
    chair_str += side;
  }

  // Publish /zed/chair_detections
  std_msgs::msg::String chair_msg;
  chair_msg.data = chair_str;
  chair_dets_publisher_->publish(chair_msg);

  // Publish /zed/empty_seat untuk Receptionist State Machine
  std_msgs::msg::String empty_seat_msg;
  empty_seat_msg.data = best_empty_side;
  empty_seat_publisher_->publish(empty_seat_msg);

  showPreview(bgr);
}

void ZedNode::showPreview(const cv::Mat & bgr)
{
  if (!debug_window_ || preview_dead_ || bgr.empty()) {return;}

  try {
    cv::Mat canvas;
    if (mirror_) {
      cv::flip(bgr, canvas, 1);
    } else {
      canvas = bgr.clone();
    }
    const int W = canvas.cols;
    const int H = canvas.rows;

    const bool flip = mirror_;
    auto flip_rect = [&](const cv::Rect & in) {
        cv::Rect r = in;
        if (flip) {r.x = W - r.x - r.width;}
        return r;
    };
    auto flip_x = [&](int x) {return flip ? (W - x) : x;};

    // Helper banner label dengan background solid (gaya visualize zed_ws)
    auto draw_banner = [&](const std::string & text, int x, int y, const cv::Scalar & border_col, const cv::Scalar & bg_col, const cv::Scalar & text_col, double font_scale = 0.55, int thickness = 1) {
      int base_line = 0;
      cv::Size sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &base_line);
      int bx = std::clamp(x, 0, std::max(0, W - sz.width - 6));
      int by = std::clamp(y - sz.height - 4, 0, std::max(0, H - sz.height - 6));
      cv::Rect bg_rect(bx, by, sz.width + 6, sz.height + 6);
      bg_rect &= cv::Rect(0, 0, W, H);
      if (bg_rect.width > 0 && bg_rect.height > 0) {
        cv::rectangle(canvas, bg_rect, bg_col, -1);
        cv::rectangle(canvas, bg_rect, border_col, 1, cv::LINE_AA);
      }
      cv::putText(canvas, text, cv::Point(bx + 3, by + sz.height + 1), cv::FONT_HERSHEY_SIMPLEX, font_scale, text_col, thickness, cv::LINE_AA);
    };

    for (size_t i = 0; i < std::min<size_t>(4, preview_persons_.size()); ++i) {
      const auto & p = preview_persons_[i];
      if (p.has_pos && p.range_m > 4.5) {continue;}
      cv::Rect r = flip_rect(p.bbox);
      r &= cv::Rect(0, 0, W, H);
      if (r.width <= 0 || r.height <= 0) {continue;}

      const cv::Scalar bbox_col = p.is_sitting ? cv::Scalar(0, 200, 255) : cv::Scalar(0, 255, 0); // Emas jika duduk di kursi, Hijau jika berdiri
      cv::rectangle(canvas, r, bbox_col, 2);

      std::string label_txt;
      if (is_receptionist_ && !is_help_me_carry_) {
        // Receptionist: Tanpa ID! Gunakan nama tamu dari memori HRI jika ada
        label_txt = (i == 0) ? (guest_name_.empty() ? "Guest" : guest_name_) : "Person";
      } else {
        // Help Me Carry: Gunakan ID operator
        label_txt = "ID: " + std::to_string(p.id);
      }
      if (p.is_sitting) {
        label_txt += " [Seated]";
      }

      draw_banner(label_txt, r.x, r.y, bbox_col, p.is_sitting ? cv::Scalar(0, 80, 110) : cv::Scalar(0, 70, 0), cv::Scalar(255, 255, 255), 0.58, 2);

      if (!p.gender.empty() || !p.clothing.empty() || !p.hair.empty()) {
        std::string attr_txt;
        if (!p.gender.empty()) attr_txt += p.gender;
        if (!p.clothing.empty()) attr_txt += (attr_txt.empty() ? "" : " | ") + p.clothing;
        if (!p.hair.empty()) attr_txt += (attr_txt.empty() ? "" : " | ") + p.hair;
        draw_banner(attr_txt, r.x, r.y + r.height + 22, cv::Scalar(0, 180, 180), cv::Scalar(20, 30, 30), cv::Scalar(0, 255, 255), 0.48, 1);
      }

      const int center_x = r.x + r.width / 2;
      const int center_y = r.y + r.height / 2;
      cv::circle(canvas, cv::Point(center_x, center_y), 5, cv::Scalar(0, 255, 255), -1);
      cv::drawMarker(canvas, cv::Point(center_x, center_y), cv::Scalar(0, 255, 255), cv::MARKER_CROSS, 14, 2);
      if (i == 0) {
        cv::line(canvas, cv::Point(center_x, center_y), cv::Point(W / 2, H), cv::Scalar(0, 69, 255), 2);
      }

      for (size_t k = 0; k < p.kpt.size(); ++k) {
        const cv::Point2f & pt = p.kpt[k];
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y)) {continue;}
        const bool big = (k == 5 || k == 7 || k == 12 || k == 14);
        const int kx = flip_x(static_cast<int>(std::lround(pt.x)));
        const int ky = static_cast<int>(std::lround(pt.y));
        if (kx < 0 || ky < 0 || kx >= W || ky >= H) {continue;}
        cv::circle(
          canvas, cv::Point(kx, ky), big ? 5 : 3,
          big ? cv::Scalar(0, 200, 255) : cv::Scalar(0, 255, 0), -1);
      }
    }

    if (enable_bag_) {
      for (size_t i = 0; i < std::min<size_t>(3, preview_bags_.size()); ++i) {
        const auto & b = preview_bags_[i];
        cv::Rect r = flip_rect(b.bbox);
        r &= cv::Rect(0, 0, W, H);
        if (r.width > 0 && r.height > 0) {
          cv::rectangle(canvas, r, cv::Scalar(255, 0, 255), 2);
          const int bag_cx = r.x + r.width / 2;
          const int bag_cy = r.y + r.height / 2;
          cv::circle(canvas, cv::Point(bag_cx, bag_cy), 4, cv::Scalar(255, 0, 255), -1);
          cv::line(canvas, cv::Point(bag_cx, bag_cy), cv::Point(W / 2, H), cv::Scalar(255, 0, 255), 2);

          std::string bag_txt = "Bag";
          if (b.has_pos) {
            bag_txt += cv::format(" (%.2fm, %.2fm)", b.x_m, b.y_m);
          }
          draw_banner(bag_txt, r.x, r.y, cv::Scalar(255, 0, 255), cv::Scalar(125, 40, 81), cv::Scalar(253, 168, 208), 0.52, 1);
        }
      }
    }

    for (size_t i = 0; i < std::min<size_t>(4, preview_drinks_.size()); ++i) {
      const auto & d = preview_drinks_[i];
      cv::Rect r = flip_rect(d.bbox);
      r &= cv::Rect(0, 0, W, H);
      if (r.width > 0 && r.height > 0) {
        cv::rectangle(canvas, r, cv::Scalar(255, 255, 0), 2);
        const int item_cx = r.x + r.width / 2;
        const int item_cy = r.y + r.height / 2;
        cv::circle(canvas, cv::Point(item_cx, item_cy), 4, cv::Scalar(255, 255, 0), -1);
        std::string txt = d.label;
        if (d.has_pos) {
          txt += cv::format(" (%.2fm, %.2fm)", d.x_m, d.y_m);
        }
        draw_banner(txt, r.x, r.y, cv::Scalar(255, 255, 0), cv::Scalar(80, 80, 0), cv::Scalar(255, 255, 255), 0.52, 1);
      }
    }

    for (size_t i = 0; i < std::min<size_t>(4, preview_chairs_.size()); ++i) {
      const auto & c = preview_chairs_[i];
      cv::Rect r = flip_rect(c.bbox);
      r &= cv::Rect(0, 0, W, H);
      if (r.width > 0 && r.height > 0) {
        const cv::Scalar color = c.occupied ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0); // Merah: Penuh, Hijau: Kosong
        cv::rectangle(canvas, r, color, 2);
        const int chair_cx = r.x + r.width / 2;
        const int chair_cy = r.y + r.height / 2;
        cv::circle(canvas, cv::Point(chair_cx, chair_cy), 4, color, -1);
        std::string txt = (c.occupied ? "Chair (Occupied)" : "Chair (Empty): ") + c.side;
        if (c.has_pos) {
          txt += cv::format(" (%.2fm, %.2fm)", c.x_m, c.y_m);
        }
        const cv::Scalar chair_bg = c.occupied ? cv::Scalar(0, 0, 110) : cv::Scalar(0, 80, 0);
        draw_banner(txt, r.x, r.y, color, chair_bg, cv::Scalar(255, 255, 255), 0.52, 2);
      }
    }

    const auto now_t = now();
    if ((now_t - preview_fps_t_).seconds() >= 1.0) {
      preview_fps_t_ = now_t;
      preview_fps_ = 0;
    }

    int hud_y = 52;
    auto draw_hud = [&](const std::string & title, double x, double y, double dist, const cv::Scalar & col) {
      cv::putText(canvas, title, cv::Point(10, hud_y), cv::FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv::LINE_AA);
      cv::putText(canvas, "x : " + cv::format("%.2f m", x), cv::Point(10, hud_y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv::LINE_AA);
      cv::putText(canvas, "y : " + cv::format("%.2f m", y), cv::Point(10, hud_y + 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv::LINE_AA);
      cv::putText(canvas, "distance : " + cv::format("%.2f m", dist), cv::Point(10, hud_y + 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv::LINE_AA);
      hud_y += 86;
    };

    const PrevPerson p0 = (!preview_persons_.empty() && preview_persons_[0].has_pos) ? preview_persons_[0] : PrevPerson{};
    std::string person_title;
    if (is_receptionist_ && !is_help_me_carry_) {
      person_title = (guest_name_.empty() ? "Guest :" : (guest_name_ + " :"));
    } else {
      person_title = "Operator [ID " + std::to_string(p0.id) + "] :";
    }
    draw_hud(person_title, p0.x_m, p0.y_m, p0.range_m, cv::Scalar(0, 255, 0));

    if (enable_bag_) {
      const PrevBag b0 = (!preview_bags_.empty() && preview_bags_[0].has_pos) ? preview_bags_[0] : PrevBag{};
      draw_hud("Bag :", b0.x_m, b0.y_m, b0.range_m, cv::Scalar(255, 0, 255));
    }

    if (!preview_drinks_.empty() && preview_drinks_[0].has_pos) {
      draw_hud(preview_drinks_[0].label + " :", preview_drinks_[0].x_m, preview_drinks_[0].y_m, preview_drinks_[0].range_m, cv::Scalar(255, 255, 0));
    }

    for (const auto & c : preview_chairs_) {
      if (!c.occupied && c.has_pos) {
        draw_hud("Empty Chair [" + c.side + "] :", c.x_m, c.y_m, c.range_m, cv::Scalar(0, 255, 0));
        break;
      }
    }

    const std::string win = "POV";
    if (!preview_inited_) {
      cv::namedWindow(win, cv::WINDOW_AUTOSIZE);
      preview_inited_ = true;
    }
    cv::imshow(win, canvas);
    const int key = cv::waitKey(1);
    if (key == 'q' || key == 27) {
      rclcpp::shutdown();
    }
  } catch (const std::exception & e) {
    if (!preview_dead_) {
      try {cv::destroyAllWindows();} catch (...) {}
      preview_dead_ = true;
    }
  }
}

void ZedNode::loop()
{
  try {
    using namespace zed_vision;
    if (zed_.grab() != sl::ERROR_CODE::SUCCESS) {
      return;
    }

    publishYaw();

    if (zed_.retrieveImage(zed_image_, sl::VIEW::LEFT, sl::MEM::CPU) !=
      sl::ERROR_CODE::SUCCESS)
    {
      return;
    }

    cv::Mat bgra(
      zed_image_.getHeight(), zed_image_.getWidth(), CV_8UC4,
      zed_image_.getPtr<sl::uchar1>(sl::MEM::CPU),
      zed_image_.getStepBytes(sl::MEM::CPU));
    cv::Mat bgr;
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
    if (bgr.empty()) {
      return;
    }

    if (zed_.retrieveMeasure(zed_point_cloud_, sl::MEASURE::XYZ, sl::MEM::CPU) !=
      sl::ERROR_CODE::SUCCESS)
    {
      return;
    }

    if (enable_body_tracking_) {
      if (zed_.retrieveBodies(bodies_, body_runtime_parameters_) != sl::ERROR_CODE::SUCCESS) {
        return;
      }
    } else {
      bodies_.body_list.clear();
    }

    processBodiesAndYolo(bgr, zed_point_cloud_);
  } catch (...) {
  }
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<ZedNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("zed_main"), "Fatal Exception ZED: %s", e.what());
    std::cerr << "\033[91m[ZED Fatal Exception] " << e.what() << "\033[0m\n";
    rclcpp::shutdown();
    return 1;
  } catch (...) {
    RCLCPP_FATAL(rclcpp::get_logger("zed_main"), "Unknown Fatal Exception in ZED node!");
    std::cerr << "\033[91m[ZED Fatal Exception] Unknown error occurred!\033[0m\n";
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
