// Program utama zed ala maen_boss/zed_ws/src/main.cpp: satu file berisi
// semua logika deteksi (struct + vision murni + YOLO tas + node + main).
//   open() -> grab() -> retrieveImage(LEFT) + retrieveMeasure(XYZ) ->
//   YOLO tas (bag, model pretrained yolov8n_sim.onnx, filter tas saja) +
//   body tracking person (bbox + 34 keypoints) -> publish.
// zed HANYA publish deteksi mentah + yaw:
//   /zed/bag_detections  "bag:x,y,w,h|x,y,z|conf;..." ("" bila kosong)
//   /zed/person_keypoints "person:x,y,w,h|x,y,z|yaw,rel|kpt=...;..." ("NO_PERSON")
//   /zed/yaw_deg         IMU (dipakai nucleo)
// Penentuan pose/gesture/operator dilakukan di luggage_task, bukan di sini.
#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>

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
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

// --- Struct data (ala zed_ws/include/def.h) --------------------------------
// (x,y) = sudut kiri-atas bbox, (w,h) = lebar & tinggi (px).
// x/y/z 3D = posisi relatif robot (meter): x depan, y kiri, z tinggi.
struct bbox_t
{
  unsigned int x{0}, y{0}, w{0}, h{0};
  float x_3d{0}, y_3d{0}, z_3d{0};
};

constexpr double DEFAULT_FRONT_YAW_OFFSET_DEG = 180.0;
constexpr double DEFAULT_ORIENTATION_THRESHOLD_DEG = 30.0;

namespace zed_vision {

constexpr float PI_F = 3.14159265358979323846f;

float wrap180(float angle)
{
  while (angle > 180.0f) {angle -= 360.0f;}
  while (angle < -180.0f) {angle += 360.0f;}
  return angle;
}

void quatToEuler(
  float x, float y, float z, float w,
  float & roll, float & pitch, float & yaw)
{
  const float sinr = 2.0f * (w * x + y * z);
  const float cosr = 1.0f - 2.0f * (x * x + y * y);
  roll = std::atan2(sinr, cosr);

  const float sinp = 2.0f * (w * y - z * x);
  if (std::fabs(sinp) >= 1.0f) {
    pitch = std::copysign(PI_F / 2.0f, sinp);
  } else {
    pitch = std::asin(sinp);
  }

  const float siny = 2.0f * (w * z + x * y);
  const float cosy = 1.0f - 2.0f * (y * y + z * z);
  yaw = std::atan2(siny, cosy);
}

bool validPoint(
  const std::vector<sl::float2> & points,
  int index, int width, int height)
{
  if (index < 0 || index >= static_cast<int>(points.size())) {
    return false;
  }
  const auto & p = points[index];
  return std::isfinite(p.x) &&
    std::isfinite(p.y) &&
    p.x >= 0 && p.y >= 0 &&
    p.x < width && p.y < height;
}

// Hitung bbox dari semua keypoint valid. Return false bila tak ada titik valid.
bool computeBbox(
  const std::vector<sl::float2> & points,
  int width, int height, cv::Rect & bbox)
{
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float max_y = std::numeric_limits<float>::lowest();
  bool found = false;

  for (size_t i = 0; i < points.size(); ++i) {
    if (!validPoint(points, static_cast<int>(i), width, height)) {
      continue;
    }
    const auto & p = points[i];
    min_x = std::min(min_x, p.x);
    min_y = std::min(min_y, p.y);
    max_x = std::max(max_x, p.x);
    max_y = std::max(max_y, p.y);
    found = true;
  }

  if (!found) {
    return false;
  }

  const float pad_x = (max_x - min_x) * 0.05f + 5.0f;
  const float pad_y = (max_y - min_y) * 0.05f + 5.0f;

  const int x0 = std::max(0, static_cast<int>(std::floor(min_x - pad_x)));
  const int y0 = std::max(0, static_cast<int>(std::floor(min_y - pad_y)));
  const int x1 = std::min(width - 1, static_cast<int>(std::ceil(max_x + pad_x)));
  const int y1 = std::min(height - 1, static_cast<int>(std::ceil(max_y + pad_y)));

  if (x1 <= x0 || y1 <= y0) {
    return false;
  }

  bbox = cv::Rect(x0, y0, x1 - x0, y1 - y0);
  return true;
}

float getBodyYawDeg(const sl::BodyData & body)
{
  const auto & q = body.global_root_orientation;
  float roll = 0.0f;
  float pitch = 0.0f;
  float yaw = 0.0f;
  quatToEuler(q.x, q.y, q.z, q.w, roll, pitch, yaw);
  return yaw * 180.0f / PI_F;
}

std::string classifyOrientation(
  const sl::BodyData & body,
  float front_yaw_offset_deg, float orientation_threshold_deg,
  float & yaw_deg, float & relative_yaw_deg)
{
  yaw_deg = getBodyYawDeg(body);
  relative_yaw_deg = wrap180(yaw_deg - front_yaw_offset_deg);

  const float angle = std::fabs(relative_yaw_deg);
  if (angle <= orientation_threshold_deg) {
    return "FRONT";
  }
  if (angle >= 180.0f - orientation_threshold_deg) {
    return "BACK";
  }
  if (relative_yaw_deg > 0.0f) {
    return "SIDE_LEFT";
  }
  return "SIDE_RIGHT";
}

// --- Port dari maen_boss/zed_ws/src/main.cpp ------------------------------
float getMedian(std::vector<float> & v)
{
  const size_t n = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + n, v.end());
  return v[n];
}

bool get3dCoordinates(
  const bbox_t & box, const sl::Mat & point_cloud, sl::float4 & point)
{
  // Sampling cincin R di sekitar tengah bbox, median per sumbu.
  // Sama seperti get_3d_coordinates() di zed_ws (tahan depth NaN/lubang).
  const unsigned int R_max_global = 10;
  const unsigned int obj_size = std::min(box.w, box.h);
  const unsigned int R_max = std::min(R_max_global, obj_size / 2);
  const int center_i = static_cast<int>(box.x + box.w * 0.5f);
  const int center_j = static_cast<int>(box.y + box.h * 0.5f);

  std::vector<float> x_vect, y_vect, z_vect;
  for (unsigned int R = 0; R < R_max; ++R) {
    for (int y = -static_cast<int>(R); y <= static_cast<int>(R); ++y) {
      for (int x = -static_cast<int>(R); x <= static_cast<int>(R); ++x) {
        const int i = center_i + x;
        const int j = center_j + y;
        sl::float4 out;
        if (i >= 0 && i < static_cast<int>(point_cloud.getWidth()) &&
          j >= 0 && j < static_cast<int>(point_cloud.getHeight()))
        {
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

  if (x_vect.empty()) {
    point.x = NAN;
    point.y = NAN;
    point.z = NAN;
    return false;
  }
  point.x = getMedian(x_vect);
  point.y = getMedian(y_vect);
  point.z = getMedian(z_vect);
  return true;
}

bool yoloBboxToPose(
  const cv::Rect & rect, const sl::Mat & point_cloud,
  double & x, double & y, double & z)
{
  // Centroid dulu, fallback median bbox bila NaN.
  const int cx = rect.x + rect.width / 2;
  const int cy = rect.y + rect.height / 2;
  sl::float4 point;
  point_cloud.getValue(cx, cy, &point);
  if (!std::isfinite(point.z)) {
    bbox_t bb;
    bb.x = static_cast<unsigned int>(std::max(0, rect.x));
    bb.y = static_cast<unsigned int>(std::max(0, rect.y));
    bb.w = static_cast<unsigned int>(rect.width);
    bb.h = static_cast<unsigned int>(rect.height);
    if (!get3dCoordinates(bb, point_cloud, point)) {
      return false;
    }
  }
  if (!std::isfinite(point.z)) {
    return false;
  }
  // Sama kayak maen_boss awal (units CENTIMETER -> meter):
  // x depan = point.x + 13, y kiri = -point.y + 3, z atas = point.z + 44.
  constexpr double OFF_X_CM = 13.0;
  constexpr double OFF_Y_CM = 3.0;
  constexpr double OFF_Z_CM = 44.0;
  x = (point.x + OFF_X_CM) / 100.0;
  y = (-point.y + OFF_Y_CM) / 100.0;
  z = (point.z + OFF_Z_CM) / 100.0;
  return true;
}

}  // namespace zed_vision

// --- Detektor YOLO tas via ONNX Runtime CPU ("tinggal pake") ----------------
// Model: yolov8n_sim.onnx (YOLOv8n COCO pretrained, diekspor + disimplify
// sekali via src/zed/models/download_yolo.sh, sudah termasuk di repo).
// Filter hanya tas: bag, backpack, handbag, suitcase -> label "bag".
// Person via ZED SDK body tracking (bukan YOLO).
struct YoloDetection
{
  cv::Rect rect;
  std::string label;  // "bag" (dinormalisasi dari backpack/handbag/suitcase)
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

  // model_path: yolov8n_sim.onnx (COCO pretrained).
  // classes_path: nama kelas (satu per baris).
  // return false bila file tak ada / session gagal dibuat (node tetap jalan,
  // publish kosong agar luggage_task tidak hang).
  bool load(
    const std::string & model_path, const std::string & classes_path,
    float conf_threshold = 0.45f, float nms_threshold = 0.45f,
    int input_size = 640, std::string * err = nullptr);

  bool ready() const;

  // Deteksi tiap frame BGR. Mengisi out (bbox dlm koordinat gambar asli).
  // Hanya kelas tas (bag_labels) yang dikembalikan; sisanya dibuang.
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
  std::vector<int64_t> input_shape;  // [1,3,H,W] model (statis utk kita)
  std::vector<std::string> names;
  float conf_threshold{0.45f};
  float nms_threshold{0.45f};
  int input_size{640};  // dipakai bila input model dinamis
  bool ready{false};
  std::vector<float> input_tensor;  // buffer NCHW reusable
};

YoloDetector::YoloDetector()
: impl_(std::make_unique<Impl>()) {}

YoloDetector::~YoloDetector() = default;

bool YoloDetector::ready() const {return impl_->ready;}

bool YoloDetector::load(
  const std::string & model_path, const std::string & classes_path,
  float conf_threshold, float nms_threshold, int input_size,
  std::string * err)
{
  Impl & im = *impl_;
  im.conf_threshold = conf_threshold;
  im.nms_threshold = nms_threshold;
  im.input_size = input_size;
  im.ready = false;

  im.names.clear();
  {
    std::ifstream f(classes_path);
    if (!f.is_open()) {
      if (err) {*err = "classes file not found: " + classes_path;}
      return false;
    }
    std::string line;
    while (std::getline(f, line)) {
      // trim CR/spasi (classes.txt dari Windows/download)
      while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
        line.pop_back();
      }
      size_t s = 0;
      while (s < line.size() && (line[s] == ' ' || line[s] == '\t')) {++s;}
      if (s >= line.size()) {continue;}
      im.names.push_back(line.substr(s));
    }
  }
  if (im.names.empty()) {
    if (err) {*err = "classes file empty: " + classes_path;}
    return false;
  }

  try {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(2);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    im.session = std::make_unique<Ort::Session>(im.env, model_path.c_str(), opts);

    Ort::AllocatorWithDefaultOptions alloc;
    {
      auto in = im.session->GetInputNameAllocated(0, alloc);
      im.input_name = in.get();
      auto typ = im.session->GetInputTypeInfo(0);
      im.input_shape = typ.GetTensorTypeAndShapeInfo().GetShape();
    }
    {
      auto out = im.session->GetOutputNameAllocated(0, alloc);
      im.output_name = out.get();
    }
    // Model kita statis [1,3,640,640]; hormati itu bila ada.
    if (im.input_shape.size() == 4 && im.input_shape[2] > 0) {
      im.input_size = static_cast<int>(im.input_shape[2]);
    }
    // Warmup kecil agar alokasi/isi lazy ORT terjadi di init, bukan frame 1.
    im.input_tensor.assign(
      static_cast<size_t>(3 * im.input_size * im.input_size), 0.5f);
    std::array<int64_t, 4> dims{1, 3, im.input_size, im.input_size};
    Ort::Value t = Ort::Value::CreateTensor<float>(
      im.mem_info, im.input_tensor.data(),
      im.input_tensor.size(), dims.data(), dims.size());
    const char * in_names[] = {im.input_name.c_str()};
    const char * out_names[] = {im.output_name.c_str()};
    im.session->Run(
      Ort::RunOptions{nullptr}, in_names, &t, 1, out_names, 1);
  } catch (const Ort::Exception & e) {
    if (err) {*err = std::string("onnxruntime: ") + e.what();}
    im.session.reset();
    return false;
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

  // Letterbox ala YOLO: resize keep-aspect + pad 114, RGB, /255, NCHW.
  const int IN = im.input_size;
  const double scale = std::min(
    static_cast<double>(IN) / bgr.cols, static_cast<double>(IN) / bgr.rows);
  const int nw = std::max(1, static_cast<int>(std::round(bgr.cols * scale)));
  const int nh = std::max(1, static_cast<int>(std::round(bgr.rows * scale)));
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(nw, nh));
  const int dx = (IN - nw) / 2;
  const int dy = (IN - nh) / 2;
  cv::Mat padded;
  cv::copyMakeBorder(
    resized, padded, dy, IN - nh - dy, dx, IN - nw - dx,
    cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
  cv::Mat rgb;
  cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

  im.input_tensor.resize(static_cast<size_t>(3 * IN * IN));
  float * dst = im.input_tensor.data();
  const size_t plane = static_cast<size_t>(IN * IN);
  for (int y = 0; y < IN; ++y) {
    const uint8_t * row = rgb.ptr<uint8_t>(y);
    for (int x = 0; x < IN; ++x) {
      const size_t i = static_cast<size_t>(y * IN + x);
      dst[i] = row[x * 3] / 255.0f;
      dst[plane + i] = row[x * 3 + 1] / 255.0f;
      dst[2 * plane + i] = row[x * 3 + 2] / 255.0f;
    }
  }

  std::vector<float> flat;  // [attrs, dets] tampilan 2D selalu
  int num_attrs = 0, num_det = 0;
  bool is_v5 = false;
  // Standard 80 COCO class names (digunakan bila model adalah COCO 80 kelas).
  // Di scope fungsi (bukan dalam try) agar bisa dipakai lambda get_label.
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
  try {
    std::array<int64_t, 4> dims{1, 3, IN, IN};
    Ort::Value t = Ort::Value::CreateTensor<float>(
      im.mem_info, im.input_tensor.data(),
      im.input_tensor.size(), dims.data(), dims.size());
    const char * in_names[] = {im.input_name.c_str()};
    const char * out_names[] = {im.output_name.c_str()};
    auto outputs = im.session->Run(
      Ort::RunOptions{nullptr}, in_names, &t, 1, out_names, 1);
    if (outputs.empty() || !outputs.front().IsTensor()) {
      return;
    }
    auto info = outputs.front().GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> shape = info.GetShape();
    const float * data = outputs.front().GetTensorData<float>();

    // Bentuk didukung: [1,84,8400] v8, [1,25200,85] v5 (atau tanpa batch).
    int d1 = -1, d2 = -1;
    if (shape.size() == 3 && shape[0] == 1) {
      d1 = static_cast<int>(shape[1]);
      d2 = static_cast<int>(shape[2]);
    } else if (shape.size() == 2) {
      d1 = static_cast<int>(shape[0]);
      d2 = static_cast<int>(shape[1]);
    } else {
      return;
    }
    if (d1 >= 5) {
      // v8: baris=atribut [cx,cy,w,h + skor], kolom=deteksi
      num_attrs = d1;
      num_det = d2;
      is_v5 = false;
      flat.assign(data, data + static_cast<size_t>(d1) * d2);
    } else if (d2 >= 6) {
      // v5: baris=deteksi [cx,cy,w,h,obj + skor] -> transpose ke [85,N]
      num_attrs = d2;
      num_det = d1;
      is_v5 = true;
      flat.resize(static_cast<size_t>(d2) * d1);
      for (int r = 0; r < d1; ++r) {
        for (int c = 0; c < d2; ++c) {
          flat[static_cast<size_t>(c) * d1 + r] =
            data[static_cast<size_t>(r) * d2 + c];
        }
      }
    } else {
      return;  // head tak dikenal
    }
  } catch (...) {
    return;
  }

  auto at = [&](int attr, int det) -> float {
          return flat[static_cast<size_t>(attr) * num_det + det];
        };

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;

  const int base = is_v5 ? 5 : 4;  // v5: idx4 = objectness
  const int num_model_classes = num_attrs - base;
  if (num_model_classes <= 0) {
    return;
  }

  auto get_label = [&](int class_idx) -> std::string {
    if (num_model_classes == 80 && class_idx >= 0 && class_idx < 80) {
      const std::string & raw = kCoco80[static_cast<size_t>(class_idx)];
      if (raw == "backpack" || raw == "handbag" || raw == "suitcase" || raw == "bag") {
        return "bag";
      }
      return raw;
    }
    if (class_idx >= 0 && class_idx < static_cast<int>(im.names.size())) {
      return im.names[static_cast<size_t>(class_idx)];
    }
    return std::to_string(class_idx);
  };

  for (int i = 0; i < num_det; ++i) {
    const float cx = at(0, i), cy = at(1, i), w = at(2, i), h = at(3, i);
    int best_id = -1;
    float best_score = 0.0f;
    for (int c = 0; c < num_model_classes; ++c) {
      const float s = at(base + c, i);
      if (s > best_score) {
        best_score = s;
        best_id = c;
      }
    }
    if (is_v5) {
      best_score *= at(4, i);  // objectness * class score
    }
    if (best_score < im.conf_threshold) {continue;}
    if (best_id < 0) {continue;}
    const std::string name = get_label(best_id);
    const bool is_bag = (want_bag.find(name) != want_bag.end() || name == "bag" ||
      want_bag.find("bag") != want_bag.end() && (name == "backpack" || name == "handbag" || name == "suitcase"));
    if (!is_bag) {
      continue;  // hanya ambil tas (bag)
    }
    // cx,cy,w,h masih koordinat input letterbox -> petakan ke gambar asli
    const float x0 = (cx - w * 0.5f - static_cast<float>(dx)) / static_cast<float>(scale);
    const float y0 = (cy - h * 0.5f - static_cast<float>(dy)) / static_cast<float>(scale);
    const float ww = w / static_cast<float>(scale);
    const float hh = h / static_cast<float>(scale);
    const int ix = std::max(0, static_cast<int>(std::round(x0)));
    const int iy = std::max(0, static_cast<int>(std::round(y0)));
    const int iw = std::min(bgr.cols - ix, static_cast<int>(std::round(ww)));
    const int ih = std::min(bgr.rows - iy, static_cast<int>(std::round(hh)));
    if (iw <= 4 || ih <= 4) {continue;}
    boxes.emplace_back(ix, iy, iw, ih);
    scores.push_back(best_score);
    class_ids.push_back(best_id);
  }

  std::vector<int> kept;
  cv::dnn::NMSBoxes(boxes, scores, im.conf_threshold, im.nms_threshold, kept);

  out.reserve(kept.size());
  for (int idx : kept) {
    YoloDetection d;
    d.rect = boxes[static_cast<size_t>(idx)];
    d.class_id = class_ids[static_cast<size_t>(idx)];
    d.score = scores[static_cast<size_t>(idx)];
    std::string lbl = get_label(d.class_id);
    if (lbl == "backpack" || lbl == "handbag" || lbl == "suitcase") {
      lbl = "bag";
    }
    d.label = lbl;
    out.push_back(d);
  }
  // deterministik: skor tertinggi dulu (tas terbaik = out pertama)
  std::sort(
    out.begin(), out.end(),
    [](const YoloDetection & a, const YoloDetection & b) {return a.score > b.score;});
}

// --- ROS wrapper tipis: loop grab -> deteksi -> publish deteksi mentah -------
class ZedNode : public rclcpp::Node
{
public:
  ZedNode();
  ~ZedNode() override;

private:
  void publishYaw();
  void loop();
  void loopOnce();
  // Satu frame: body tracking (person) + YOLO tas (bag) + depth.
  void processBodiesAndYolo(const cv::Mat & bgr, const sl::Mat & point_cloud);
  // Preview lokal via cv::imshow (opsional, default OFF). Gambar diambil
  // langsung dari frame yang sedang diproses, jadi tidak perlu bolak-balik
  // lewat /zed/image dan tidak ada selisih QoS.
  void showPreview(const cv::Mat & bgr);

  sl::Camera zed_;
  sl::Mat zed_image_;
  sl::Mat zed_point_cloud_;
  sl::Bodies bodies_;
  sl::BodyTrackingRuntimeParameters body_runtime_parameters_;

  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr yaw_publisher_;
  // Deteksi mentah (bbox px + 3D meter + nama): dibaca luggage_task.
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr bag_dets_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr persons_publisher_;
  // Preview awal (gambar kamera asli, bukan kanvas hitam): dibaca window task.
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;

  bool publish_yaw_{true};
  bool publish_image_{true};
  // image_stride: publish /zed/image tiap N frame (hemat bandwidth/CPU,
  // kurangi delay preview). 1 = tiap frame.
  int image_stride_{2};
  int image_frame_{0};
  double camera_fps_{30.0};
  // depth_mode: "PERFORMANCE" (ringan, latency kecil) atau "ULTRA"
  // (maen_boss, akurat, berat). Ganti tanpa rebuild via param/launch.
  std::string depth_mode_{"NEURAL_LIGHT"};
  // body_model: "FAST" (ringan, frame halus) atau "MEDIUM" (akurat).
  std::string body_model_{"FAST"};
  bool enable_body_tracking_{true};

  // ---- Preview window (cv::imshow) ------------------------------------
  // Default OFF supaya aman di mesin tanpa display (kontainer, SSH tanpa
  // X). Nyala lewat param `debug_window` atau launch camera.launch.py.
  bool debug_window_{false};
  bool mirror_{true};
  bool preview_inited_{false};
  // Kalau imshow melempar (headless), preview dimatikan sekali agar tidak
  // spam tiap frame - node tetap jalan & publish normal.
  bool preview_dead_{false};
  bool preview_no_person_logged_{false};
  int preview_fps_{0};
  rclcpp::Time preview_fps_t_{0, 0, RCL_ROS_TIME};

  // Data ringkas untuk digambar di preview (dibangun sambil publish).
  struct PrevPerson {
    cv::Rect bbox;
    bool has_pos{false};
    double range_m{0.0};
    float yaw_deg{0.0f};
    // 34 keypoint; point denganNaN berarti joint tidak valid.
    std::vector<cv::Point2f> kpt;
  };
  struct PrevBag {
    cv::Rect bbox;
    bool has_pos{false};
    double range_m{0.0};
    double conf{0.0};
  };
  std::vector<PrevPerson> preview_persons_;
  std::vector<PrevBag> preview_bags_;

  // YOLO tas (model pretrained yolov8n_sim.onnx, filter tas saja).
  YoloDetector yolo_;
  bool enable_yolo_{true};
  std::vector<std::string> bag_labels_;
  double yolo_min_area_{1500.0};
  bool yolo_warned_{false};
  // Stride inferensi + cache bbox (pose depth tetap tiap frame).
  int yolo_stride_{2};
  int yolo_frame_{0};
  std::vector<YoloDetection> last_bag_dets_;

  float front_yaw_offset_deg_{180.0f};
  float orientation_threshold_deg_{30.0f};
  // Diagnosa macet (detak log tiap 5 dtk).
  uint64_t loop_count_{0};
  uint64_t img_pub_count_{0};
  uint64_t grab_fail_count_{0};
  bool first_frame_logged_{false};
};

ZedNode::ZedNode()
: Node("zed")
{
  declare_parameter("camera_fps", 30.0);
  declare_parameter("publish_yaw", true);
  declare_parameter("publish_image", true);
  declare_parameter("image_stride", 2);
  declare_parameter("depth_mode", std::string("NEURAL_LIGHT"));
  declare_parameter("body_model", std::string("FAST"));
  declare_parameter("enable_body_tracking", true);
  // Jarak depth minimum (satuan ikut coordinate_units = CENTIMETER).
  // Sama kayak maen_boss awal: 30 = 30 cm.
  declare_parameter("depth_minimum_distance", 30.0);
  declare_parameter("front_yaw_offset_deg", DEFAULT_FRONT_YAW_OFFSET_DEG);
  declare_parameter("orientation_threshold_deg", DEFAULT_ORIENTATION_THRESHOLD_DEG);
  // YOLO tas (model pretrained).
  // Default menunjuk ke install share (models/ di-install via CMake).
  // "Tinggal pake": jalankan models/download_yolo.sh sekali di MiniPC,
  // lalu rebuild agar onnx + coco names ter-copy ke share.
  declare_parameter("enable_yolo", true);
  // Preview lokal (cv::imshow) dari frame yang sedang diproses. Default
  // FALSE supaya aman tanpa display. Nyalakan dengan launch
  // `ros2 launch luggage_task camera.launch.py debug_window:=true`.
  declare_parameter("debug_window", false);
  // Balik kiri-kanan supaya gerakan tangan terbaca seperti cermin (kiri
  // orang = kiri layar). Sama seperti debug_window lama di luggage_task.
  declare_parameter("mirror", true);
  declare_parameter("yolo_model", std::string("models/yolov8n_sim.onnx"));
  declare_parameter("yolo_classes", std::string("models/coco_names.txt"));
  declare_parameter("yolo_conf", 0.25);
  declare_parameter("yolo_nms", 0.45);
  declare_parameter("yolo_size", 640);
  // yolo_stride: inferensi tiap N frame (CPU hemat). Pose 3D dari depth
  // tetap dihitung tiap frame dari bbox terakhir agar tracking mulus.
  declare_parameter("yolo_stride", 2);
  declare_parameter(
    "bag_labels",
    std::vector<std::string>{"bag", "backpack", "handbag", "suitcase"});
  declare_parameter("yolo_min_area", 1500.0);

  const double camera_fps = get_parameter("camera_fps").as_double();
  camera_fps_ = camera_fps;

  publish_yaw_ = get_parameter("publish_yaw").as_bool();
  publish_image_ = get_parameter("publish_image").as_bool();
  image_stride_ = static_cast<int>(get_parameter("image_stride").as_int());
  if (image_stride_ < 1) {image_stride_ = 1;}
  depth_mode_ = get_parameter("depth_mode").as_string();
  body_model_ = get_parameter("body_model").as_string();
  enable_body_tracking_ = get_parameter("enable_body_tracking").as_bool();

  front_yaw_offset_deg_ = static_cast<float>(
    get_parameter("front_yaw_offset_deg").as_double());
  orientation_threshold_deg_ = static_cast<float>(
    get_parameter("orientation_threshold_deg").as_double());

  if (camera_fps <= 0.0) {
    throw std::runtime_error("camera_fps must be greater than zero");
  }

  yaw_publisher_ = create_publisher<std_msgs::msg::Float64>("/zed/yaw_deg", 10);
  bag_dets_publisher_ = create_publisher<std_msgs::msg::String>("/zed/bag_detections", 10);
  persons_publisher_ = create_publisher<std_msgs::msg::String>("/zed/person_keypoints", 10);
  image_publisher_ = create_publisher<sensor_msgs::msg::Image>(
    "/zed/image", rclcpp::SensorDataQoS());

  sl::InitParameters init_parameters;
  // depth_mode: "NEURAL_LR" (default, paling ringan), "NEURAL" (paling akurat),
  // "PERFORMANCE". ZED SDK 5.x sudah menandai ULTRA/PERFORMANCE sebagai
  // DEPRECATED dan memunculkan warning "Please update your configuration to
  // use NEURAL instead." Default diganti ke NEURAL_LR supaya warning hilang
  // tanpa mengorbankan ringan (body tracking + YOLO tetap jalan).
  if (depth_mode_ == "PERFORMANCE") {
    init_parameters.depth_mode = sl::DEPTH_MODE::PERFORMANCE;
  } else if (depth_mode_ == "NEURAL") {
    init_parameters.depth_mode = sl::DEPTH_MODE::NEURAL;
  } else if (depth_mode_ == "ULTRA") {
    init_parameters.depth_mode = sl::DEPTH_MODE::ULTRA;
  } else {
    // Default: NEURAL_LR. Ringan seperti PERFORMANCE, tapi jalur neural
    // yang didukung penuh SDK 5.x.
    init_parameters.depth_mode = sl::DEPTH_MODE::NEURAL_LIGHT;
  }
  init_parameters.coordinate_units = sl::UNIT::CENTIMETER;
  init_parameters.camera_resolution = sl::RESOLUTION::VGA;
  init_parameters.camera_fps = static_cast<int>(camera_fps);
  init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
  // maen_boss: init_parameters.depth_minimum_distance = 30 (cm).
  init_parameters.depth_minimum_distance =
    static_cast<float>(get_parameter("depth_minimum_distance").as_double());

  const auto state = zed_.open(init_parameters);
  if (state != sl::ERROR_CODE::SUCCESS) {
    RCLCPP_ERROR(get_logger(), "ZED open failed: %s", sl::toString(state).c_str());
    throw std::runtime_error("ZED open failed");
  }

  sl::PositionalTrackingParameters positional_parameters;
  const auto tracking_state = zed_.enablePositionalTracking(positional_parameters);
  if (tracking_state != sl::ERROR_CODE::SUCCESS) {
    zed_.close();
    RCLCPP_ERROR(
      get_logger(), "Enable positional tracking failed: %s",
      sl::toString(tracking_state).c_str());
    throw std::runtime_error("Enable positional tracking failed");
  }

  sl::BodyTrackingParameters body_parameters;
  // FAST = frame halus di MiniPC; MEDIUM = keypoint lebih akurat.
  if (body_model_ == "MEDIUM") {
    body_parameters.detection_model = sl::BODY_TRACKING_MODEL::HUMAN_BODY_MEDIUM;
  } else {
    body_parameters.detection_model = sl::BODY_TRACKING_MODEL::HUMAN_BODY_FAST;
  }
  body_parameters.body_format = sl::BODY_FORMAT::BODY_34;
  body_parameters.enable_tracking = true;
  body_parameters.enable_body_fitting = true;

  if (enable_body_tracking_) {
    const auto body_state = zed_.enableBodyTracking(body_parameters);
    if (body_state != sl::ERROR_CODE::SUCCESS) {
      zed_.close();
      RCLCPP_ERROR(
        get_logger(), "Enable body tracking failed: %s",
        sl::toString(body_state).c_str());
      throw std::runtime_error("Enable body tracking failed");
    }
  }

  body_runtime_parameters_.detection_confidence_threshold = 40;

  // --- YOLO tas (bag) ---------------------------------------------------
  enable_yolo_ = get_parameter("enable_yolo").as_bool();
  debug_window_ = get_parameter("debug_window").as_bool();
  mirror_ = get_parameter("mirror").as_bool();
  bag_labels_ = get_parameter("bag_labels").as_string_array();
  yolo_min_area_ = get_parameter("yolo_min_area").as_double();
  yolo_stride_ = static_cast<int>(get_parameter("yolo_stride").as_int());
  if (yolo_stride_ < 1) {yolo_stride_ = 1;}
  if (enable_yolo_) {
    std::string model_rel = get_parameter("yolo_model").as_string();
    std::string classes_rel = get_parameter("yolo_classes").as_string();
    auto resolve = [](const std::string & p) {
        if (std::filesystem::path(p).is_absolute()) {return p;}
        return std::string("share/zed/") + p;  // fallback bila share tak ada
      };
    std::string model_path = resolve(model_rel);
    std::string classes_path = resolve(classes_rel);
    try {
      const std::string share =
        ament_index_cpp::get_package_share_directory("zed");
      if (!std::filesystem::path(model_rel).is_absolute()) {
        model_path = share + "/" + model_rel;
      }
      if (!std::filesystem::path(classes_rel).is_absolute()) {
        classes_path = share + "/" + classes_rel;
      }
    } catch (...) {
      // paket belum ter-install (colcon build belum jalan): pakai path relatif
    }
    const float yolo_conf =
      static_cast<float>(get_parameter("yolo_conf").as_double());
    const float yolo_nms =
      static_cast<float>(get_parameter("yolo_nms").as_double());
    const int yolo_size =
      static_cast<int>(get_parameter("yolo_size").as_int());
    std::string err;
    if (!yolo_.load(model_path, classes_path, yolo_conf, yolo_nms, yolo_size, &err)) {
      RCLCPP_WARN(
        get_logger(),
        "YOLO disabled: %s (jalan tanpa deteksi tas). "
        "Unduh pretrained: src/zed/models/download_yolo.sh lalu rebuild.",
        err.c_str());
      enable_yolo_ = false;
    } else {
      RCLCPP_INFO(
        get_logger(), "YOLO loaded: %s (%s)",
        model_path.c_str(), classes_path.c_str());
    }
  }

  const auto period = std::chrono::duration<double>(1.0 / camera_fps);
  timer_ = create_wall_timer(period, std::bind(&ZedNode::loop, this));

  RCLCPP_INFO(
    get_logger(), "ZED started (body_tracking=%s/%s, depth=%s, img_stride=%d)",
    enable_body_tracking_ ? "true" : "false", body_model_.c_str(),
    depth_mode_.c_str(), image_stride_);
  RCLCPP_INFO(
    get_logger(),
    "Publish: /zed/yaw_deg /zed/bag_detections /zed/person_keypoints /zed/image");
}

ZedNode::~ZedNode()
{
  zed_.close();
}

void ZedNode::publishYaw()
{
  using zed_vision::quatToEuler;
  using zed_vision::wrap180;
  using zed_vision::PI_F;
  if (!publish_yaw_) {
    return;
  }
  sl::SensorsData sensors_data;
  if (zed_.getSensorsData(sensors_data, sl::TIME_REFERENCE::IMAGE) !=
    sl::ERROR_CODE::SUCCESS)
  {
    return;
  }
  if (!sensors_data.imu.is_available) {
    return;
  }
  const sl::Orientation q = sensors_data.imu.pose.getOrientation();
  float roll = 0.0f, pitch = 0.0f, yaw = 0.0f;
  quatToEuler(q.x, q.y, q.z, q.w, roll, pitch, yaw);
  float yaw_deg = yaw * 180.0f / PI_F;
  yaw_deg = wrap180(yaw_deg);
  std_msgs::msg::Float64 message;
  message.data = static_cast<double>(yaw_deg);
  yaw_publisher_->publish(message);
}

void ZedNode::processBodiesAndYolo(const cv::Mat & bgr, const sl::Mat & point_cloud)
{
  using namespace zed_vision;
  // --- YOLO tas: semua deteksi -> /zed/bag_detections -----------------------
  // "bag:x,y,w,h|x,y,z|conf;..." ("" bila kosong). bbox px, 3D meter.
  std::vector<YoloDetection> bag_dets = last_bag_dets_;

  if (enable_yolo_ && yolo_.ready()) {
    // Stride: inferensi DNN tiap N frame; pose depth dari bbox cache tiap frame.
    const int stride = std::max(1, yolo_stride_);
    if (yolo_frame_ % stride == 0) {
      std::vector<YoloDetection> yolo_dets;
      yolo_.detect(bgr, bag_labels_, yolo_dets);
      // detect() sudah sortir skor tertinggi dulu + filter tas saja.
      bag_dets = yolo_dets;
      last_bag_dets_ = bag_dets;
    }
    ++yolo_frame_;
  } else if (enable_yolo_ && !yolo_warned_) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 10000,
      "YOLO model belum siap, publish tas kosong.");
    yolo_warned_ = true;
  }

  preview_bags_.clear();
  {
    std::ostringstream ss;
    bool first = true;
    for (const auto & d : bag_dets) {
      if (static_cast<double>(d.rect.area()) < yolo_min_area_) {continue;}
      double x = 0.0, y = 0.0, z = 0.0;
      const bool ok = yoloBboxToPose(d.rect, point_cloud, x, y, z);
      PrevBag pb;
      pb.bbox = d.rect;
      pb.has_pos = ok;
      pb.conf = d.score;
      pb.range_m = ok ? std::hypot(x, y) : 0.0;
      preview_bags_.push_back(pb);
      if (!first) {ss << ";";}
      first = false;
      ss << "bag:" << d.rect.x << "," << d.rect.y << ","
         << d.rect.width << "," << d.rect.height << "|";
      if (ok) {ss << x << "," << y << "," << z;}
      else {ss << "nan,nan,nan";}
      ss << "|" << d.score;
    }
    std_msgs::msg::String m;
    m.data = ss.str();
    bag_dets_publisher_->publish(m);
  }

  // --- Body tracking person -> /zed/person_keypoints -------------------------
  // "person:x,y,w,h|x,y,z|yaw,rel|kpt=x1,y1,...,x34,y34;..." ("NO_PERSON").
  // 6 output utama per person: bbox x,y,w,h (px, 4) + depth x,y,z (meter, 3)
  // + yaw,rel + 34 keypoints. Urut TERDEKAT dulu: index 0 = orang terdekat
  // = yg dianggap terdeteksi oleh luggage_task.
  preview_persons_.clear();
  {
    std_msgs::msg::String m;
    if (bodies_.body_list.empty()) {
      m.data = "NO_PERSON";
    } else {
      struct PersonEntry {
        const sl::BodyData * body{nullptr};
        cv::Rect bbox;
        float yaw_deg{0.0f};
        float rel_deg{0.0f};
        double x{0.0}, y{0.0}, z{0.0};
        bool has_depth{false};
        double range_m{std::numeric_limits<double>::infinity()};
      };
      std::vector<PersonEntry> entries;
      entries.reserve(bodies_.body_list.size());
      for (const auto & body : bodies_.body_list) {
        const auto & points = body.keypoint_2d;
        if (points.size() < 34) {continue;}
        cv::Rect bbox;
        if (!computeBbox(points, bgr.cols, bgr.rows, bbox)) {continue;}
        PersonEntry e;
        e.body = &body;
        e.bbox = bbox;
        classifyOrientation(
          body, front_yaw_offset_deg_, orientation_threshold_deg_,
          e.yaw_deg, e.rel_deg);
        double x = 0.0, y = 0.0, z = 0.0;
        e.has_depth = yoloBboxToPose(bbox, point_cloud, x, y, z);
        e.x = x;
        e.y = y;
        e.z = z;
        if (e.has_depth && std::isfinite(x) && std::isfinite(y)) {
          e.range_m = std::hypot(x, y);
        } else {
          // Fallback bila depth NaN: bbox terbesar = kemungkinan terdekat.
          e.range_m = 1e6 - static_cast<double>(bbox.area());
        }
        // Ringkas untuk preview. Keypoint tidak valid diisi NaN supaya
        // showPreview() melewatinya (konvensi sama dengan publish "-1").
        PrevPerson pp;
        pp.bbox = bbox;
        pp.has_pos = e.has_depth;
        pp.range_m = e.range_m;
        pp.yaw_deg = e.yaw_deg;
        pp.kpt.reserve(points.size());
        for (size_t k = 0; k < points.size(); ++k) {
          const float kx = points[k].x;
          const float ky = points[k].y;
          const float qnan = std::numeric_limits<float>::quiet_NaN();
          pp.kpt.emplace_back(std::isfinite(kx) ? kx : qnan,
                              std::isfinite(ky) ? ky : qnan);
        }
        entries.push_back(e);
        preview_persons_.push_back(pp);
      }
      // Orang paling dekat (range 3D terkecil) jadi urutan pertama.
      std::sort(
        entries.begin(), entries.end(),
        [](const PersonEntry & a, const PersonEntry & b) {
          return a.range_m < b.range_m;
        });
      // Preview harus urutan yang sama supaya "hijau = terdekat" konsisten
      // dengan index 0 di /zed/person_keypoints.
      std::sort(
        preview_persons_.begin(), preview_persons_.end(),
        [](const PrevPerson & a, const PrevPerson & b) {
          return a.range_m < b.range_m;
        });
      std::ostringstream ss;
      bool first_p = true;
      for (const auto & e : entries) {
        const auto & points = e.body->keypoint_2d;
        if (!first_p) {ss << ";";}
        first_p = false;
        ss << "person:" << e.bbox.x << "," << e.bbox.y << ","
           << e.bbox.width << "," << e.bbox.height << "|";
        if (e.has_depth) {ss << e.x << "," << e.y << "," << e.z;}
        else {ss << "nan,nan,nan";}
        ss << "|" << std::fixed << std::setprecision(1)
           << e.yaw_deg << "," << e.rel_deg << std::defaultfloat << "|kpt=";
        // 34 keypoint mentah langsung dari ZED SDK (float, tanpa rounding int;
        // "-1" bila joint tak valid). Format tetap kompatibel dgn parse luggage_task.
        ss << std::fixed << std::setprecision(2);
        for (size_t k = 0; k < 34; ++k) {
          if (k > 0) {ss << ",";}
          const float kx = points[k].x;
          const float ky = points[k].y;
          if (std::isfinite(kx) && std::isfinite(ky)) {
            ss << kx << "," << ky;
          } else {
            ss << "-1,-1";
          }
        }
        ss << std::defaultfloat;
      }
      m.data = first_p ? "NO_PERSON" : ss.str();
    }
    persons_publisher_->publish(m);
  }

  // Preview terakhir: menggambar frame + hasil deteksi yang baru saja
  // dipublish, jadi yang tampil di window persis sama dengan isi topik.
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

    // PENTING: kalau gambar di-flip, koordinat gambar HARUS ikut di-flip
    // juga. Kalau tidak, kotak/keypoint tetap memakai koordinat asli while
    // the image sudah terbalik -> bbox muncul di sisi yang SALAH (orang
    // geser kanan, kotaknya muncul di kiri). Rumus flip horizontal: x' = W - x.
    // Untuk lebar w, tepi kirinya jadi x' = W - x - w.
    const bool flip = mirror_;
    auto flip_rect = [&](const cv::Rect & in) {
        cv::Rect r = in;
        if (flip) {r.x = W - r.x - r.width;}
        return r;
    };
    auto flip_x = [&](int x) {return flip ? (W - x) : x;};

    // Orang. Index 0 = paling dekat (hijau tebal), sisanya abu-abu.
    for (size_t i = 0; i < preview_persons_.size(); ++i) {
      const auto & p = preview_persons_[i];
      const bool is_pick = (i == 0);
      const cv::Scalar col = is_pick ? cv::Scalar(0, 255, 0) : cv::Scalar(170, 170, 170);
      cv::Rect r = flip_rect(p.bbox);
      r &= cv::Rect(0, 0, W, H);
      if (r.width > 0 && r.height > 0) {
        cv::rectangle(canvas, r, col, is_pick ? 2 : 1);
      }
      for (size_t k = 0; k < p.kpt.size(); ++k) {
        const cv::Point2f & pt = p.kpt[k];
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y)) {continue;}
        // Bahu + pergelangan (dipakai gestur) digambar lebih besar.
        const bool big = (k == 5 || k == 7 || k == 12 || k == 14);
        const int kx = flip_x(static_cast<int>(std::lround(pt.x)));
        const int ky = static_cast<int>(std::lround(pt.y));
        if (kx < 0 || ky < 0 || kx >= W || ky >= H) {continue;}
        cv::circle(
          canvas, cv::Point(kx, ky), (big && is_pick) ? 5 : 3,
          big ? cv::Scalar(0, 200, 255) : col, -1);
      }
      // Label jarak + yaw di atas bbox (ikut ter-flip supaya nempel di atas
      // kotak yang benar).
      char lb[80];
      if (p.has_pos) {
        snprintf(lb, sizeof(lb), "#%zu %.2fm yaw%.0f", i, p.range_m, p.yaw_deg);
      } else {
        snprintf(lb, sizeof(lb), "#%zu ?", i);
      }
      const int label_x = std::max(0, std::min(W - 1, flip_x(p.bbox.x)));
      cv::putText(
        canvas, lb, cv::Point(label_x, std::max(16, p.bbox.y - 6)),
        cv::FONT_HERSHEY_SIMPLEX, 0.55, col, is_pick ? 2 : 1, cv::LINE_AA);
    }

    // Tas YOLO (magenta) + jarak.
    for (size_t i = 0; i < preview_bags_.size() && i < 5; ++i) {
      const auto & b = preview_bags_[i];
      cv::Rect r = flip_rect(b.bbox);
      r &= cv::Rect(0, 0, W, H);
      if (r.width > 0 && r.height > 0) {
        cv::rectangle(canvas, r, cv::Scalar(255, 0, 255), 2);
      }
      char tb[48];
      if (b.has_pos) {
        snprintf(tb, sizeof(tb), "tas %.2fm", b.range_m);
      } else {
        snprintf(tb, sizeof(tb), "tas ?");
      }
      cv::putText(
        canvas, tb, cv::Point(std::max(0, r.x), std::max(16, r.y - 6)),
        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
    }

    // HUD: jumlah deteksi + fps + petunjuk.
    const auto now_t = now();
    if ((now_t - preview_fps_t_).seconds() >= 1.0) {
      preview_fps_t_ = now_t;
      preview_fps_ = 0;
    }
    char hud[96];
    snprintf(
      hud, sizeof(hud), "person:%zu  tas:%zu  |  q keluar",
      preview_persons_.size(), preview_bags_.size());
    cv::putText(
      canvas, hud, cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6,
      cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    if (preview_persons_.empty()) {
      const char * hint = enable_body_tracking_
        ? "belum ada orang terdeteksi" : "enable_body_tracking:=false !";
      cv::putText(
        canvas, hint, cv::Point(10, H - 14), cv::FONT_HERSHEY_SIMPLEX, 0.55,
        cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    } else {
      cv::putText(
        canvas, "hijau = person terdekat", cv::Point(10, H - 14),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    const std::string win = "zed - person (preview)";
    if (!preview_inited_) {
      cv::namedWindow(win, cv::WINDOW_AUTOSIZE);
      preview_inited_ = true;
      RCLCPP_INFO(
        get_logger(),
        "preview window aktif: \"%s\" (q keluar, ctrl+c keluar node)", win.c_str());
    }
    cv::imshow(win, canvas);
    const int key = cv::waitKey(1);
    if (key == 'q' || key == 27) {
      RCLCPP_INFO(get_logger(), "preview ditutup dari window, shutting down");
      rclcpp::shutdown();
    }
  } catch (const std::exception & e) {
    // Tanpa display (SSH tanpa X, kontainer, systemd tanpa DISPLAY):
    // cv::imshow melempar. Node HARUS tetap jalan & publish normal, jadi
    // preview dimatikan sekali saja - bukan spam tiap frame.
    if (!preview_dead_) {
      RCLCPP_ERROR(
        get_logger(),
        "preview window gagal dibuka (headless?): %s\n"
      "  node tetap jalan & publish normal, preview dinonaktifkan. "
      "Cara cek deteksi tanpa window: ros2 topic echo /zed/person_keypoints",
      e.what());
      try {cv::destroyAllWindows();} catch (...) {}
      preview_dead_ = true;
    }
  }
}

void ZedNode::loop()
{
  // Jangan biarkan SATU frame gagal membunuh timer (penyebab sunyi total:
  // tidak ada /zed/image maupun /zed/person_keypoints).
  try {
    loopOnce();
  } catch (const std::exception & e) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 5000, "loop zed exception: %s", e.what());
  } catch (...) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 5000, "loop zed exception tak dikenal");
  }
}
void ZedNode::loopOnce()
{
  using namespace zed_vision;
  ++loop_count_;
  // Pola zed_ws/main.cpp: grab() -> retrieveImage + retrieveMeasure.
  if (zed_.grab() != sl::ERROR_CODE::SUCCESS) {
    ++grab_fail_count_;
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "zed grab gagal x%lu (kamera sibuk/lepas/USB2?)",
      static_cast<unsigned long>(grab_fail_count_));
    return;
  }
  if (!first_frame_logged_) {
    first_frame_logged_ = true;
    RCLCPP_INFO(
      get_logger(), "frame pertama OK (kamera @%.0ffps, depth=%s, body=%s)",
      camera_fps_, depth_mode_.c_str(), body_model_.c_str());
  }

  publishYaw();

  if (zed_.retrieveImage(zed_image_, sl::VIEW::LEFT, sl::MEM::CPU) !=
    sl::ERROR_CODE::SUCCESS)
  {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000, "retrieveImage LEFT gagal");
    return;
  }

  // Konversi aman (dst terpisah + step): in-place 4ch->3ch rawan exception
  // cv::Exception tiap frame yang membunuh timer (layar hitam total).
  cv::Mat bgra(
    zed_image_.getHeight(), zed_image_.getWidth(), CV_8UC4,
    zed_image_.getPtr<sl::uchar1>(sl::MEM::CPU),
    zed_image_.getStepBytes(sl::MEM::CPU));
  cv::Mat bgr;
  cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
  if (bgr.empty()) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000, "konversi BGRA->BGR kosong, frame dilewat");
    return;
  }

  // Preview dipublish DULUAN, tidak menunggu depth/body (anti-macét:
  // /zed/image tetap jalan walau depth/body tracking gagal).
  // Manual (tanpa cv_bridge, hindari konflik header OpenCV5).
  // Stride: publish tiap N frame agar preview tidak delay.
  ++image_frame_;
  if (publish_image_ && (image_frame_ % image_stride_ == 0)) {
    sensor_msgs::msg::Image img_msg;
    img_msg.header.stamp = now();
    img_msg.header.frame_id = "zed_camera";
    img_msg.height = static_cast<uint32_t>(bgr.rows);
    img_msg.width = static_cast<uint32_t>(bgr.cols);
    img_msg.encoding = "bgr8";
    img_msg.is_bigendian = false;
    img_msg.step = static_cast<uint32_t>(bgr.cols * 3);
    img_msg.data.assign(bgr.datastart, bgr.dataend);
    image_publisher_->publish(img_msg);
    ++img_pub_count_;
  }

  // Depth selalu diambil (dibutuhkan tas + person 3D), walau
  // body tracking dimatikan (mode yaw-only hemat GPU).
  if (zed_.retrieveMeasure(zed_point_cloud_, sl::MEASURE::XYZ, sl::MEM::CPU) !=
    sl::ERROR_CODE::SUCCESS)
  {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "retrieveMeasure XYZ gagal (depth mati?)");
    return;
  }

  if (loop_count_ % 150 == 0) {
    RCLCPP_INFO(
      get_logger(), "detak: loop=%lu img_pub=%lu bodies=%zu",
      static_cast<unsigned long>(loop_count_),
      static_cast<unsigned long>(img_pub_count_), bodies_.body_list.size());
  }

  if (!enable_body_tracking_) {
    // Mode yaw-only: YOLO tas tetap jalan di CPU, body tracking dilewati.
    bodies_.body_list.clear();
    processBodiesAndYolo(bgr, zed_point_cloud_);
    return;
  }

  if (zed_.retrieveBodies(bodies_, body_runtime_parameters_) != sl::ERROR_CODE::SUCCESS) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "retrieveBodies gagal (model AI belum siap? person tak terlihat?)");
    return;
  }
  processBodiesAndYolo(bgr, zed_point_cloud_);
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<ZedNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("zed"), "%s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}
