#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/string.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "icecream.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace zed_vision
{

constexpr float PI_F = 3.14159265358979323846f;
constexpr double kOffsetX = 13.0;
constexpr double kOffsetY = 3.0;
constexpr double kOffsetZ = 44.0;
constexpr double kOffsetScale = 100.0;

float wrap180(float angle)
{
  while (angle > 180.0f) {angle -= 360.0f;}
  while (angle < -180.0f) {angle += 360.0f;}
  return angle;
}

float wrap360(float angle)
{
  angle = std::fmod(angle, 360.0f);
  if (angle < 0.0f) {angle += 360.0f;}
  if (angle >= 360.0f || std::abs(angle) < 1e-6f) {angle = 0.0f;}
  return angle;
}

float quatToYaw(float x, float y, float z, float w)
{
  return std::atan2(2.0f * (w * z + x * y), 1.0f - 2.0f * (y * y + z * z));
}

bool computeBbox(
  const std::vector<sl::float2> & points, int width, int height, cv::Rect & bbox)
{
  float min_x = static_cast<float>(width), min_y = static_cast<float>(height);
  float max_x = 0.0f, max_y = 0.0f;
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
  if (!found) return false;

  const int pad_x = static_cast<int>((max_x - min_x) * 0.05f + 5.0f);
  const int pad_y = static_cast<int>((max_y - min_y) * 0.05f + 5.0f);
  const int x0 = std::max(0, static_cast<int>(min_x) - pad_x);
  const int y0 = std::max(0, static_cast<int>(min_y) - pad_y);
  const int x1 = std::min(width - 1, static_cast<int>(max_x) + pad_x);
  const int y1 = std::min(height - 1, static_cast<int>(max_y) + pad_y);
  if (x1 <= x0 || y1 <= y0) return false;

  bbox = cv::Rect(x0, y0, x1 - x0, y1 - y0);
  return true;
}

float getMedian(std::vector<float> & values)
{
  const size_t n = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + n, values.end());
  return values[n];
}

bool get3dCoordinates(const cv::Rect & box, const sl::Mat & cloud, sl::float4 & point)
{
  const unsigned int max_radius = 10;
  const unsigned int obj_size = std::min(box.width, box.height);
  const unsigned int radius = std::min(max_radius, obj_size / 2);
  const int center_i = box.x + box.width / 2;
  const int center_j = box.y + box.height / 2;
  const int w = static_cast<int>(cloud.getWidth());
  const int h = static_cast<int>(cloud.getHeight());

  std::vector<float> xs, ys, zs;
  for (unsigned int r = 0; r < radius; ++r) {
    for (int dy = -static_cast<int>(r); dy <= static_cast<int>(r); ++dy) {
      for (int dx = -static_cast<int>(r); dx <= static_cast<int>(r); ++dx) {
        const int i = center_i + dx;
        const int j = center_j + dy;
        sl::float4 out{NAN, NAN, NAN, NAN};
        if (i >= 0 && i < w && j >= 0 && j < h) {
          cloud.getValue(i, j, &out);
        }
        if (std::isfinite(out.z)) {
          xs.push_back(out.x);
          ys.push_back(out.y);
          zs.push_back(out.z);
        }
      }
    }
  }

  if (xs.empty()) {
    point.x = point.y = point.z = NAN;
    return false;
  }
  point.x = getMedian(xs);
  point.y = getMedian(ys);
  point.z = getMedian(zs);
  return true;
}

bool get3dWithOffset(
  const cv::Rect & rect, const sl::Mat & cloud, double & x, double & y, double & z)
{
  const int cx = rect.x + rect.width / 2;
  const int cy = rect.y + rect.height / 2;
  sl::float4 point{NAN, NAN, NAN, NAN};

  if (cx >= 0 && cx < static_cast<int>(cloud.getWidth()) &&
      cy >= 0 && cy < static_cast<int>(cloud.getHeight()))
  {
    cloud.getValue(cx, cy, &point);
  }
  if (!std::isfinite(point.z)) {
    get3dCoordinates(rect, cloud, point);
  }
  if (!std::isfinite(point.z)) return false;

  x = (point.x + kOffsetX) / kOffsetScale;
  y = (-point.y + kOffsetY) / kOffsetScale;
  z = (point.z + kOffsetZ) / kOffsetScale;
  return true;
}

cv::Scalar extractTorsoColor(const cv::Mat & bgr, const cv::Rect & bbox)
{
  if (bgr.empty() || bbox.width <= 0 || bbox.height <= 0) return cv::Scalar(0, 0, 0);

  const int tw = std::max(4, bbox.width / 3);
  const int th = std::max(4, bbox.height / 4);
  const int tx = std::clamp(bbox.x + bbox.width / 3, 0, bgr.cols - tw);
  const int ty = std::clamp(bbox.y + bbox.height / 5, 0, bgr.rows - th);

  cv::Rect roi(tx, ty, tw, th);
  roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
  if (roi.area() <= 0) return cv::Scalar(0, 0, 0);

  cv::Mat hsv;
  cv::cvtColor(bgr(roi), hsv, cv::COLOR_BGR2HSV);
  return cv::mean(hsv);
}

}

struct YoloDetection
{
  cv::Rect rect;
  std::string label;
  float score{0.0f};
  int class_id{-1};
};

struct PrevPerson
{
  int id{-1};
  cv::Rect bbox;
  bool has_pos{false};
  double x_m{0.0}, y_m{0.0}, range_m{0.0};
  bool is_sitting{false};
  std::vector<cv::Point2f> kpt;
  std::string clothing, hair, gender, name;
};

struct PrevBag
{
  cv::Rect bbox;
  bool has_pos{false};
  double x_m{0.0}, y_m{0.0}, range_m{0.0};
};

struct PrevItem
{
  std::string label;
  cv::Rect bbox;
  bool has_pos{false};
  double x_m{0.0}, y_m{0.0}, range_m{0.0};
};

struct PrevChair
{
  cv::Rect bbox;
  bool has_pos{false};
  double x_m{0.0}, y_m{0.0}, range_m{0.0};
  bool occupied{false};
  std::string side{"center"};
};

struct ChairMemory
{
  cv::Rect bbox;
  bool has_pos{false};
  double x{0.0}, y{0.0}, z{0.0};
  double range_m{0.0};
  rclcpp::Time last_seen{0, 0, RCL_ROS_TIME};
};

static std::vector<std::string> loadLabels(const std::string & path)
{
  std::vector<std::string> labels;
  if (path.empty()) return labels;
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!line.empty()) labels.push_back(line);
  }
  return labels;
}

class OrtModel
{
public:
  OrtModel()
  : mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}

  bool load(const std::string & path, std::string * err = nullptr)
  {
    session_.reset();
    try {
      Ort::SessionOptions opts;
      opts.SetIntraOpNumThreads(2);
      opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

      try {
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = 0;
        opts.AppendExecutionProvider_CUDA(cuda_opts);
        using_cuda_ = true;
      } catch (const std::exception & e) {
        using_cuda_ = false;
      }

      session_ = std::make_unique<Ort::Session>(env(), path.c_str(), opts);
      Ort::AllocatorWithDefaultOptions alloc;
      input_ = session_->GetInputNameAllocated(0, alloc).get();
      output_ = session_->GetOutputNameAllocated(0, alloc).get();

      if (using_cuda_) {
        std::cout << "[OrtModel] CUDA Execution Provider aktif: " << path << std::endl;
      } else {
        std::cout << "[OrtModel] CPU Execution Provider aktif: " << path << std::endl;
      }
    } catch (const std::exception & e) {
      if (err) *err = e.what();
      session_.reset();
      return false;
    }
    return true;
  }

  bool ready() const {return static_cast<bool>(session_);}
  bool usingCuda() const {return using_cuda_;}

  std::vector<int64_t> inputShape()
  {
    if (!session_) return {};
    try {
      return session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    } catch (const std::exception &) {
      return {};
    }
  }

  std::vector<Ort::Value> run(
    const cv::Mat & blob, const std::array<int64_t, 4> & shape)
  {
    if (!session_ || blob.empty()) return {};

    Ort::Value input = Ort::Value::CreateTensor<float>(
      mem_, const_cast<float *>(blob.ptr<float>()), blob.total(),
      shape.data(), shape.size());
    const char * in_names[] = {input_.c_str()};
    const char * out_names[] = {output_.c_str()};
    try {
      return session_->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
    } catch (const std::exception & e) {
      std::cerr << "[OrtModel] inference failed: " << e.what() << std::endl;
      return {};
    }
  }

private:
  static Ort::Env & env()
  {
    static Ort::Env instance(ORT_LOGGING_LEVEL_WARNING, "athome_vision");
    return instance;
  }

  std::unique_ptr<Ort::Session> session_;
  std::string input_, output_;
  Ort::MemoryInfo mem_;
  bool using_cuda_{false};
};

class YoloDetector
{
public:
  bool load(
    const std::string & model_path, const std::string & classes_path,
    float conf = 0.45f, float nms = 0.45f, int size = 640,
    const std::string & default_label = "object", std::string * err = nullptr)
  {
    conf_ = conf;
    nms_ = nms;
    input_size_ = size;
    default_label_ = default_label;
    names_ = loadLabels(classes_path);
    return model_.load(model_path, err);
  }

  bool ready() const {return model_.ready();}

  void detect(
    const cv::Mat & bgr, const std::vector<std::string> & want_labels,
    std::vector<YoloDetection> & out)
  {
    out.clear();
    if (!model_.ready() || bgr.empty()) return;

    const std::unordered_set<std::string> wanted(want_labels.begin(), want_labels.end());
    cv::Mat blob = cv::dnn::blobFromImage(
      bgr, 1.0 / 255.0, cv::Size(input_size_, input_size_), cv::Scalar(), true, false);

    std::array<int64_t, 4> shape{1, 3, input_size_, input_size_};
    std::vector<Ort::Value> outputs = model_.run(blob, shape);
    if (outputs.empty() || !outputs.front().IsTensor()) return;

    const auto out_shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
    const float * data = outputs.front().GetTensorData<float>();
    const int channels = (out_shape.size() == 3) ? static_cast<int>(out_shape[1]) : 84;
    const int count = (out_shape.size() == 3) ? static_cast<int>(out_shape[2]) : 8400;
    const int classes = channels - 4;
    if (classes <= 0) return;

    cv::Mat det(channels, count, CV_32F, const_cast<float *>(data));
    cv::Mat transposed = det.t();
    const float x_factor = static_cast<float>(bgr.cols) / static_cast<float>(input_size_);
    const float y_factor = static_cast<float>(bgr.rows) / static_cast<float>(input_size_);

    static const std::vector<std::string> coco80 = {
      "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
      "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
      "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
      "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
      "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
      "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
      "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
      "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    auto labelOf = [&](int id) -> std::string {
        if (id >= 0 && id < static_cast<int>(names_.size())) return names_[static_cast<size_t>(id)];
        if (classes == 80 && id >= 0 && id < 80) return coco80[static_cast<size_t>(id)];
        return "";
      };

    auto isWanted = [&](int id) {
        if (classes == 1) return true;
        if (wanted.empty()) return true;
        const std::string name = labelOf(id);
        if (!name.empty() && wanted.count(name) > 0) return true;
        if (classes == 80) {
          if (wanted.count("bag") > 0 && (id == 24 || id == 26 || id == 28)) return true;
          if (wanted.count("chair") > 0 && (id == 56 || id == 57)) return true;
        }
        return false;
      };

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    for (int i = 0; i < count; ++i) {
      const float * row = transposed.ptr<float>(i);
      float score = 0.0f;
      int class_id = 0;
      if (classes == 1) {
        score = row[4];
      } else {
        cv::Mat scores_row(1, classes, CV_32F, const_cast<float *>(row + 4));
        cv::Point best;
        double max_score = 0.0;
        cv::minMaxLoc(scores_row, nullptr, &max_score, nullptr, &best);
        score = static_cast<float>(max_score);
        class_id = best.x;
      }
      if (score < conf_ || !isWanted(class_id)) continue;

      const float cx = row[0], cy = row[1], w = row[2], h = row[3];
      const int left = std::max(0, std::min(static_cast<int>((cx - 0.5f * w) * x_factor), bgr.cols - 1));
      const int top = std::max(0, std::min(static_cast<int>((cy - 0.5f * h) * y_factor), bgr.rows - 1));
      const int width = std::max(1, std::min(static_cast<int>(w * x_factor), bgr.cols - left));
      const int height = std::max(1, std::min(static_cast<int>(h * y_factor), bgr.rows - top));
      if (width <= 4 || height <= 4) continue;

      boxes.emplace_back(left, top, width, height);
      scores.push_back(score);
      class_ids.push_back(class_id);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, conf_, nms_, keep);
    for (int idx : keep) {
      YoloDetection det_result;
      det_result.rect = boxes[static_cast<size_t>(idx)];
      det_result.score = scores[static_cast<size_t>(idx)];
      det_result.class_id = class_ids[static_cast<size_t>(idx)];
      det_result.label = (classes == 1) ? default_label_ : labelOf(det_result.class_id);
      if (det_result.label.empty()) det_result.label = default_label_;
      out.push_back(det_result);
    }
  }

private:
  OrtModel model_;
  std::vector<std::string> names_;
  std::string default_label_{"object"};
  float conf_{0.45f};
  float nms_{0.45f};
  int input_size_{640};
};

class AttributeAnalyzer
{
public:
  bool loadGender(const std::string & path, std::string * err = nullptr)
  {
    const bool ok = gender_.load(path, err);
    if (ok) configureGender();
    return ok;
  }

  bool loadColor(
    const std::string & path, const std::string & labels_path, std::string * err = nullptr)
  {
    color_labels_ = loadLabels(labels_path);
    return color_.load(path, err);
  }

  bool loadHair(
    const std::string & path, const std::string & labels_path, std::string * err = nullptr)
  {
    hair_labels_ = loadLabels(labels_path);
    return hair_.load(path, err);
  }

  bool loadFace(const std::string & path, std::string * err = nullptr)
  {
    return face_.load(path, err);
  }

  bool faceReady() const {return face_.ready();}

  std::string clothingColor(
    const cv::Mat & bgr, const cv::Rect & bbox, const std::vector<sl::float2> & kpt)
  {
    if (bgr.empty() || bbox.width <= 0 || bbox.height <= 0) return "casual attire";

    float waist_y = -1.0f;
    if (kpt.size() > 22 && std::isfinite(kpt[18].y) && std::isfinite(kpt[22].y)) {
      waist_y = (kpt[18].y + kpt[22].y) * 0.5f;
    } else if (!kpt.empty() && std::isfinite(kpt[0].y)) {
      waist_y = kpt[0].y;
    }

    float shoulder_y = -1.0f;
    if (kpt.size() > 12 && std::isfinite(kpt[5].y) && std::isfinite(kpt[12].y)) {
      shoulder_y = (kpt[5].y + kpt[12].y) * 0.5f;
    } else if (kpt.size() > 3 && std::isfinite(kpt[3].y)) {
      shoulder_y = kpt[3].y;
    }

    int y1, y2;
    if (waist_y > 0.0f && shoulder_y > 0.0f && waist_y > shoulder_y + 10.0f) {
      y1 = static_cast<int>(shoulder_y);
      y2 = static_cast<int>(waist_y);
    } else {
      y1 = bbox.y + static_cast<int>(bbox.height * 0.20);
      y2 = bbox.y + static_cast<int>(bbox.height * 0.50);
    }
    const int x1 = std::clamp(bbox.x + static_cast<int>(bbox.width * 0.15), 0, bgr.cols);
    const int x2 = std::clamp(bbox.x + static_cast<int>(bbox.width * 0.85), 0, bgr.cols);
    y1 = std::clamp(y1, 0, bgr.rows);
    y2 = std::clamp(y2, 0, bgr.rows);
    if (y2 <= y1 || x2 <= x1) return "casual attire";

    cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
    roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
    if (roi.area() <= 0) return "casual attire";

    if (color_.ready() && !color_labels_.empty()) {
      cv::Mat resized;
      cv::resize(bgr(roi), resized, cv::Size(224, 224));
      cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
      cv::Mat blob = cv::dnn::blobFromImage(
        resized, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), false, false);
      std::array<int64_t, 4> shape{1, 3, 224, 224};
      std::vector<Ort::Value> outputs = color_.run(blob, shape);
      if (!outputs.empty()) {
        const auto out_shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
        const int classes = (out_shape.size() >= 2)
          ? static_cast<int>(out_shape[1]) : static_cast<int>(color_labels_.size());
        std::vector<float> probs;
        softmax(outputs.front().GetTensorData<float>(), classes, probs);
        int best = 0;
        for (int i = 1; i < classes; ++i) {
          if (probs[static_cast<size_t>(i)] > probs[static_cast<size_t>(best)]) best = i;
        }
        if (probs[static_cast<size_t>(best)] > 0.30f && best < static_cast<int>(color_labels_.size())) {
          std::string result = color_labels_[static_cast<size_t>(best)];
          std::transform(result.begin(), result.end(), result.begin(), ::tolower);
          if (result.find("shirt") == std::string::npos &&
              result.find("clothes") == std::string::npos)
          {
            result += " shirt";
          }
          return result;
        }
      }
    }

    cv::Mat hsv;
    cv::cvtColor(bgr(roi), hsv, cv::COLOR_BGR2HSV);
    const cv::Scalar mean = cv::mean(hsv);
    const double h = mean[0], s = mean[1], v = mean[2];
    std::string color = "dark";
    if (v < 50.0) color = "black";
    else if (s < 35.0 && v > 180.0) color = "white";
    else if (s < 45.0) color = "gray";
    else if (h <= 10.0 || h >= 170.0) color = "red";
    else if (h <= 25.0) color = "orange";
    else if (h <= 35.0) color = "yellow";
    else if (h <= 85.0) color = "green";
    else if (h <= 130.0) color = "blue";
    else if (h <= 155.0) color = "purple";
    else if (h < 170.0) color = "pink";
    return color + " shirt";
  }

  std::string hairColor(const cv::Mat & bgr, const cv::Rect & bbox)
  {
    if (bgr.empty() || bbox.width <= 0 || bbox.height <= 0) return "dark hair";

    if (hair_.ready() && !hair_labels_.empty()) {
      const int y1 = std::clamp(bbox.y, 0, bgr.rows);
      const int y2 = std::clamp(static_cast<int>(bbox.y + bbox.height * 0.20), 0, bgr.rows);
      const int x1 = std::clamp(static_cast<int>(bbox.x + bbox.width / 4), 0, bgr.cols);
      const int x2 = std::clamp(static_cast<int>(bbox.x + 3 * bbox.width / 4), 0, bgr.cols);
      if (y2 > y1 && x2 > x1) {
        cv::Mat resized;
        cv::resize(bgr(cv::Rect(x1, y1, x2 - x1, y2 - y1)), resized, cv::Size(224, 224));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        cv::Mat blob = cv::dnn::blobFromImage(
          resized, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), false, false);
        std::array<int64_t, 4> shape{1, 3, 224, 224};
        std::vector<Ort::Value> outputs = hair_.run(blob, shape);
        if (!outputs.empty()) {
          const auto out_shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
          const int classes = (out_shape.size() >= 2)
            ? static_cast<int>(out_shape[1]) : static_cast<int>(hair_labels_.size());
          std::vector<float> probs;
          softmax(outputs.front().GetTensorData<float>(), classes, probs);
          int best = 0;
          for (int i = 1; i < classes; ++i) {
            if (probs[static_cast<size_t>(i)] > probs[static_cast<size_t>(best)]) best = i;
          }
          if (probs[static_cast<size_t>(best)] > 0.35f && best < static_cast<int>(hair_labels_.size())) {
            std::string result = hair_labels_[static_cast<size_t>(best)];
            std::transform(result.begin(), result.end(), result.begin(), ::tolower);
            return result;
          }
        }
      }
    }

    const int hw = std::max(4, static_cast<int>(bbox.width * 0.35));
    const int hh = std::max(4, static_cast<int>(bbox.height * 0.15));
    const int hx = std::clamp(bbox.x + static_cast<int>(bbox.width * 0.325), 0, bgr.cols - hw);
    const int hy = std::clamp(bbox.y, 0, bgr.rows - hh);
    cv::Rect roi(hx, hy, hw, hh);
    roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
    if (roi.area() <= 0) return "dark hair";

    cv::Mat hsv;
    cv::cvtColor(bgr(roi), hsv, cv::COLOR_BGR2HSV);
    const cv::Scalar mean = cv::mean(hsv);
    const double h = mean[0], s = mean[1], v = mean[2];
    if (v < 55.0) return "black hair";
    if (s < 40.0 && v > 150.0) return "gray hair";
    if (h >= 15.0 && h <= 35.0 && v > 130.0) return "blonde hair";
    if (h >= 8.0 && h <= 30.0) return "brown hair";
    return "dark hair";
  }

  std::string gender(
    const std::vector<sl::float2> & kpt, const cv::Mat & bgr, const cv::Rect & bbox)
  {
    if (!gender_.ready() || bgr.empty() || bbox.width <= 0 || bbox.height <= 0) {
      return "unknown";
    }

    cv::Rect roi;
    if (gender_nhwc_) {
      roi = faceRegion(bbox, kpt, bgr.cols, bgr.rows);
    } else {
      const int x1 = std::clamp(bbox.x, 0, bgr.cols);
      const int y1 = std::clamp(bbox.y, 0, bgr.rows);
      const int x2 = std::clamp(bbox.x + bbox.width, 0, bgr.cols);
      const int y2 = std::clamp(static_cast<int>(bbox.y + bbox.height * 0.55), 0, bgr.rows);
      if (y2 > y1 && x2 > x1) roi = cv::Rect(x1, y1, x2 - x1, y2 - y1);
    }

    roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
    if (roi.area() <= 0) return "unknown";

    const int size = gender_input_size_;
    cv::Mat blob = letterboxBlob(bgr, roi, size, gender_nhwc_);
    if (blob.empty()) return "unknown";

    const std::array<int64_t, 4> shape = gender_nhwc_
      ? std::array<int64_t, 4>{1, size, size, 3}
      : std::array<int64_t, 4>{1, 3, size, size};
    std::vector<Ort::Value> outputs = gender_.run(blob, shape);
    if (outputs.empty()) return "unknown";

    const auto out_shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
    const int classes = (out_shape.size() >= 2)
      ? static_cast<int>(out_shape[out_shape.size() - 1]) : 2;
    std::vector<float> probs;
    probabilities(outputs.front().GetTensorData<float>(), classes, probs);
    if (probs.size() >= 2) {
      if (probs[0] > 0.65f) return "female";
      if (probs[1] > 0.65f) return "male";
    }
    return "unknown";
  }

  std::vector<float> faceEmbedding(
    const cv::Mat & bgr, const cv::Rect & bbox, const std::vector<sl::float2> & kpt)
  {
    if (!face_.ready() || bgr.empty() || bbox.width <= 0 || bbox.height <= 0) return {};

    cv::Point2f eye_left, eye_right, nose;
    if (faceLandmarks(kpt, eye_left, eye_right, nose)) {
      std::vector<cv::Point2f> from{eye_left, eye_right, nose};
      std::vector<cv::Point2f> to{
        {38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f}};
      cv::Mat transform = cv::estimateAffinePartial2D(
        from, to, cv::noArray(), cv::LMEDS);
      if (transform.empty()) transform = cv::getAffineTransform(from, to);
      if (!transform.empty()) {
        cv::Mat aligned;
        cv::warpAffine(bgr, aligned, transform, cv::Size(112, 112),
          cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        std::vector<float> feature = embedFace(aligned);
        if (!feature.empty()) return feature;
      }
    }

    const cv::Rect roi = faceRegion(bbox, kpt, bgr.cols, bgr.rows);
    if (roi.width <= 10 || roi.height <= 10) return {};

    cv::Mat resized;
    cv::resize(bgr(roi), resized, cv::Size(112, 112));
    return embedFace(resized);
  }

  std::vector<float> extractFaceFromImage(const cv::Mat & img)
  {
    if (!face_.ready() || img.empty()) return {};
    cv::Rect roi(0, 0, img.cols, img.rows);
    if (img.cols > img.rows) {
      const int diff = img.cols - img.rows;
      roi = cv::Rect(diff / 2, 0, img.rows, img.rows);
    } else if (img.rows > img.cols) {
      const int side = img.cols;
      roi = cv::Rect(0, 0, side, std::min(img.rows, side));
    }
    cv::Mat cropped = img(roi);
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(112, 112));
    return embedFace(resized);
  }

  static float matchFace(const std::vector<float> & a, const std::vector<float> & b)
  {
    if (a.empty() || b.empty() || a.size() != b.size()) return -1.0f;
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
    return dot;
  }

  static std::string detectGesture(const std::vector<sl::float2> & kpt)
  {
    if (kpt.size() < 15) return "NONE";
    const std::string left = classifyArm(kpt, 5, 7);
    const std::string right = classifyArm(kpt, 12, 14);
    if (right == "DOWN_RIGHT" || left == "DOWN_RIGHT") return "RIGHT";
    if (left == "DOWN_LEFT" || right == "DOWN_LEFT") return "LEFT";
    return "NONE";
  }

private:
  std::vector<float> embedFace(const cv::Mat & image)
  {
    if (!face_.ready() || image.empty()) return {};

    cv::Mat blob = cv::dnn::blobFromImage(
      image, 1.0, cv::Size(112, 112), cv::Scalar(0, 0, 0), true, false);
    std::array<int64_t, 4> shape{1, 3, 112, 112};
    std::vector<Ort::Value> outputs = face_.run(blob, shape);
    if (outputs.empty()) return {};

    const float * data = outputs.front().GetTensorData<float>();
    const auto out_shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
    const int dim = (out_shape.size() >= 2) ? static_cast<int>(out_shape[1]) : 128;
    std::vector<float> feature(static_cast<size_t>(dim));
    float norm_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
      feature[static_cast<size_t>(i)] = data[i];
      norm_sq += data[i] * data[i];
    }
    const float inv_norm = 1.0f / std::sqrt(std::max(norm_sq, 1e-6f));
    for (int i = 0; i < dim; ++i) {
      feature[static_cast<size_t>(i)] *= inv_norm;
    }
    return feature;
  }

  static bool faceLandmarks(
    const std::vector<sl::float2> & kpt, cv::Point2f & eye_left,
    cv::Point2f & eye_right, cv::Point2f & nose)
  {
    if (kpt.size() <= 30) return false;
    const auto valid = [](const sl::float2 & p) {
        return std::isfinite(p.x) && std::isfinite(p.y) && p.x > 0.0f && p.y > 0.0f;
      };
    if (!valid(kpt[27]) || !valid(kpt[28]) || !valid(kpt[30])) return false;

    eye_left = cv::Point2f(kpt[28].x, kpt[28].y);
    eye_right = cv::Point2f(kpt[30].x, kpt[30].y);
    nose = cv::Point2f(kpt[27].x, kpt[27].y);
    if (eye_left.x > eye_right.x) std::swap(eye_left, eye_right);
    const float eye_dist = std::hypot(
      eye_right.x - eye_left.x, eye_right.y - eye_left.y);
    return eye_dist >= 4.0f;
  }

  void configureGender()
  {
    gender_nhwc_ = false;
    gender_input_size_ = 224;
    const auto shape = gender_.inputShape();
    if (shape.size() != 4) return;
    if (shape[3] == 3) {
      gender_nhwc_ = true;
      if (shape[1] > 0) gender_input_size_ = static_cast<int>(shape[1]);
      else if (shape[2] > 0) gender_input_size_ = static_cast<int>(shape[2]);
    } else if (shape[1] == 3) {
      gender_nhwc_ = false;
      if (shape[2] > 0) gender_input_size_ = static_cast<int>(shape[2]);
    } else if (shape[3] == -1 && shape[1] > 3) {
      gender_nhwc_ = true;
      gender_input_size_ = static_cast<int>(shape[1]);
    } else if (shape[1] == -1 && shape[3] > 3) {
      gender_nhwc_ = false;
      gender_input_size_ = static_cast<int>(shape[3]);
    }
  }

  static cv::Rect faceRegion(
    const cv::Rect & bbox, const std::vector<sl::float2> & kpt, int cols, int rows)
  {
    int x1, y1, x2, y2;
    cv::Point2f eye_left, eye_right, nose;
    if (faceLandmarks(kpt, eye_left, eye_right, nose)) {
      const float eye_dist = std::hypot(
        eye_right.x - eye_left.x, eye_right.y - eye_left.y);
      const float center_x = (eye_left.x + eye_right.x) * 0.5f;
      const float center_y = (eye_left.y + eye_right.y) * 0.5f + eye_dist * 0.15f;
      const float half = eye_dist * 1.6f;
      x1 = static_cast<int>(center_x - half);
      x2 = static_cast<int>(center_x + half);
      y1 = static_cast<int>(center_y - half);
      y2 = static_cast<int>(center_y + half);
    } else if (kpt.size() > 28 && std::isfinite(kpt[26].x) && std::isfinite(kpt[26].y) &&
        kpt[26].x > 0 && kpt[26].y > 0)
    {
      float span = bbox.width * 0.45f;
      if (kpt.size() > 3 && std::isfinite(kpt[3].y) && kpt[3].y > kpt[26].y) {
        span = std::max(span, (kpt[3].y - kpt[26].y) * 2.0f);
      }
      x1 = static_cast<int>(kpt[26].x - span * 0.5f);
      x2 = static_cast<int>(kpt[26].x + span * 0.5f);
      y1 = static_cast<int>(kpt[26].y - span * 0.65f);
      y2 = static_cast<int>(kpt[26].y + span * 0.65f);
    } else {
      y1 = bbox.y;
      y2 = bbox.y + static_cast<int>(bbox.height * 0.22);
      x1 = bbox.x + static_cast<int>(bbox.width * 0.20);
      x2 = bbox.x + static_cast<int>(bbox.width * 0.80);
    }
    x1 = std::clamp(x1, 0, cols);
    x2 = std::clamp(x2, 0, cols);
    y1 = std::clamp(y1, 0, rows);
    y2 = std::clamp(y2, 0, rows);
    if (x2 <= x1 || y2 <= y1) return {};
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
  }

  static cv::Mat letterboxBlob(
    const cv::Mat & bgr, const cv::Rect & roi, int size, bool nhwc)
  {
    if (bgr.empty() || size <= 0) return {};
    const cv::Rect safe = roi & cv::Rect(0, 0, bgr.cols, bgr.rows);
    if (safe.width <= 0 || safe.height <= 0) return {};

    const float scale = std::min(
      static_cast<float>(size) / static_cast<float>(safe.width),
      static_cast<float>(size) / static_cast<float>(safe.height));
    const int w = std::max(1, static_cast<int>(std::round(safe.width * scale)));
    const int h = std::max(1, static_cast<int>(std::round(safe.height * scale)));

    cv::Mat resized;
    cv::resize(bgr(safe), resized, cv::Size(w, h));
    cv::Mat canvas(size, size, CV_8UC3, cv::Scalar(0, 0, 0));
    resized.copyTo(canvas(cv::Rect((size - w) / 2, (size - h) / 2, w, h)));

    cv::Mat rgb;
    if (!nhwc) cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    if (nhwc) {
      cv::Mat blob(1, size * size * 3, CV_32F);
      const uchar * src = canvas.ptr<uchar>();
      float * dst = blob.ptr<float>();
      const float inv = 1.0f / 255.0f;
      for (int i = 0; i < size * size * 3; ++i) dst[i] = src[i] * inv;
      return blob;
    }
    return cv::dnn::blobFromImage(
      rgb, 1.0 / 255.0, cv::Size(size, size), cv::Scalar(0, 0, 0), false, false);
  }

  static void probabilities(
    const float * data, int count, std::vector<float> & probs)
  {
    probs.assign(static_cast<size_t>(std::max(count, 0)), 0.0f);
    if (count <= 0 || data == nullptr) return;
    float sum = 0.0f;
    bool normalized = true;
    for (int i = 0; i < count; ++i) {
      const float value = data[i];
      sum += value;
      if (!std::isfinite(value) || value < 0.0f || value > 1.0f) normalized = false;
    }
    if (normalized && std::fabs(sum - 1.0f) < 0.05f) {
      for (int i = 0; i < count; ++i) probs[static_cast<size_t>(i)] = data[i];
      return;
    }
    softmax(data, count, probs);
  }

  static std::string classifyArm(
    const std::vector<sl::float2> & kpt, int shoulder, int wrist)
  {
    constexpr float threshold = 0.5f;
    if (shoulder >= static_cast<int>(kpt.size()) || wrist >= static_cast<int>(kpt.size())) return "";
    const auto & s = kpt[static_cast<size_t>(shoulder)];
    const auto & w = kpt[static_cast<size_t>(wrist)];
    if (!std::isfinite(s.x) || !std::isfinite(s.y) || !std::isfinite(w.x) || !std::isfinite(w.y) ||
        s.x < 0.f || s.y < 0.f || w.x < 0.f || w.y < 0.f)
    {
      return "";
    }
    const float dx = w.x - s.x, dy = w.y - s.y;
    if (dy <= 0.0f || std::fabs(dx) < threshold * dy) return "";
    return dx < 0.0f ? "DOWN_LEFT" : "DOWN_RIGHT";
  }

  static void softmax(const float * logits, int count, std::vector<float> & probs)
  {
    probs.assign(static_cast<size_t>(std::max(count, 0)), 0.0f);
    if (count <= 0) return;
    float max_logit = -1e9f;
    for (int i = 0; i < count; ++i) max_logit = std::max(max_logit, logits[i]);
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
      probs[static_cast<size_t>(i)] = std::exp(logits[i] - max_logit);
      sum += probs[static_cast<size_t>(i)];
    }
    if (sum <= 0.0f) return;
    for (int i = 0; i < count; ++i) probs[static_cast<size_t>(i)] /= sum;
  }

  OrtModel gender_, color_, hair_, face_;
  std::vector<std::string> color_labels_, hair_labels_;
  bool gender_nhwc_{true};
  int gender_input_size_{224};
};

class ZedNode : public rclcpp::Node
{
public:
  ZedNode();
  ~ZedNode() override;

private:
  struct RegisteredPerson
  {
    std::string name;
    std::vector<float> feature;
    int feature_samples{0};
    cv::Scalar torso_hsv{0, 0, 0};
    bool has_color{false};
    int last_tracking_id{-1};
    rclcpp::Time last_seen{0, 0, RCL_ROS_TIME};
  };

  struct OperatorMemory
  {
    bool active{false};
    int stable_id{0};
    double x{0.0}, y{0.0}, vx{0.0}, vy{0.0};
    cv::Scalar torso_hsv{0, 0, 0};
    bool has_color{false};
    rclcpp::Time last_seen{0, 0, RCL_ROS_TIME};
  };

  void publishYaw();
  void loop();
  void processBags(const cv::Mat & bgr, const sl::Mat & cloud);
  void processDrinks(const cv::Mat & bgr, const sl::Mat & cloud);
  void processPersons(const cv::Mat & bgr, const sl::Mat & cloud);
  void processChairs(const cv::Mat & bgr, const sl::Mat & cloud);
  void showPreview(const cv::Mat & bgr);

  std::string resolvePath(const std::string & path) const;
  void loadModels();
  void loadPreRegisteredHost();
  void loadYolo(
    YoloDetector & detector, const std::string & model_param,
    const std::vector<std::string> & fallbacks, const std::string & name,
    float conf, float nms, int size);

  sl::Camera zed_;
  sl::Mat zed_image_;
  sl::Mat zed_point_cloud_;
  sl::Bodies bodies_;
  sl::BodyTrackingRuntimeParameters body_runtime_parameters_;

  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr yaw_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr bag_dets_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr drink_dets_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr persons_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr recognized_name_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr gesture_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr attributes_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr empty_seat_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr chair_dets_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;

  bool publish_yaw_{true};
  bool yaw_360_{true};
  double camera_fps_{30.0};
  bool enable_body_tracking_{true};
  bool debug_window_{false};
  bool mirror_{true};

  AttributeAnalyzer analyzer_;
  std::vector<RegisteredPerson> registered_persons_;
  std::string pending_register_name_;
  double face_threshold_{0.363};

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
  std::vector<ChairMemory> chair_memory_;

  OperatorMemory op_mem_;
  std::string task_mode_{"all"};
  bool is_help_me_carry_{true};
  bool is_receptionist_{true};
  std::string guest_name_;
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
  declare_parameter("bag_labels", std::vector<std::string>{"bag", "backpack", "handbag", "suitcase"});
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
  declare_parameter("gender_model", std::string("models/gender_model_weights.onnx"));
  declare_parameter("color_model", std::string("models/color_model.onnx"));
  declare_parameter("color_labels", std::string("models/color_names.txt"));
  declare_parameter("hair_model", std::string("models/hair_model.onnx"));
  declare_parameter("hair_labels", std::string("models/hair_names.txt"));
  declare_parameter("face_model", std::string("models/face_recognition_sface.onnx"));
  declare_parameter("face_threshold", 0.363);
  declare_parameter("host_name", std::string("Alex"));
  declare_parameter("host_photo", std::string("models/host.jpg"));

  task_mode_ = get_parameter("task").as_string();
  std::string task = task_mode_;
  std::transform(task.begin(), task.end(), task.begin(), ::tolower);
  if (task == "help_me_carry" || task == "hmc") {
    is_help_me_carry_ = true;
    is_receptionist_ = false;
  } else if (task == "receptionist" || task == "rec") {
    is_help_me_carry_ = false;
    is_receptionist_ = true;
  }
  enable_bag_ = get_parameter("enable_bag").as_bool();
  RCLCPP_INFO(
    get_logger(), "Vision Task Mode: %s, enable_bag: %s",
    task_mode_.c_str(), enable_bag_ ? "true" : "false");

  camera_fps_ = get_parameter("camera_fps").as_double();
  publish_yaw_ = get_parameter("publish_yaw").as_bool();
  yaw_360_ = get_parameter("yaw_360").as_bool();
  enable_body_tracking_ = get_parameter("enable_body_tracking").as_bool();
  if (camera_fps_ <= 0.0) throw std::runtime_error("camera_fps must be greater than zero");

  yaw_publisher_ = create_publisher<std_msgs::msg::Float64>("/zed/yaw_deg", 10);
  bag_dets_publisher_ = create_publisher<std_msgs::msg::String>("/zed/bag_detections", 10);
  drink_dets_publisher_ = create_publisher<std_msgs::msg::String>("/zed/drink_detections", 10);
  persons_publisher_ = create_publisher<std_msgs::msg::String>("/zed/person_label", 10);
  recognized_name_publisher_ =
    create_publisher<std_msgs::msg::String>("/zed/recognized_name", 10);
  gesture_publisher_ = create_publisher<std_msgs::msg::String>("/zed/gesture", 10);
  attributes_publisher_ = create_publisher<std_msgs::msg::String>("/zed/person_attributes", 10);
  empty_seat_publisher_ = create_publisher<std_msgs::msg::String>("/zed/empty_seat", 10);
  chair_dets_publisher_ = create_publisher<std_msgs::msg::String>("/zed/chair_detections", 10);

  person_name_sub_ = create_subscription<std_msgs::msg::String>(
    "/zed/set_person_name", 10,
    [this](const std_msgs::msg::String::SharedPtr msg) {
      std::string name = msg->data;
      while (!name.empty() &&
        (name.back() == ' ' || name.back() == '\r' || name.back() == '\n' || name.back() == '\t'))
      {
        name.pop_back();
      }
      while (!name.empty() && (name.front() == ' ' || name.front() == '\t')) name.erase(name.begin());
      guest_name_ = name;
      pending_register_name_ = name;
      if (!name.empty()) {
        RCLCPP_INFO(get_logger(), "Guest name set: \"%s\"", name.c_str());
      } else {
        RCLCPP_INFO(get_logger(), "Guest name reset");
      }
    });

  sl::InitParameters init;
  init.depth_mode = sl::DEPTH_MODE::ULTRA;
  init.coordinate_units = sl::UNIT::CENTIMETER;
  init.camera_resolution = sl::RESOLUTION::VGA;
  init.camera_fps = static_cast<int>(camera_fps_);
  init.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
  init.depth_minimum_distance =
    static_cast<float>(get_parameter("depth_minimum_distance").as_double());

  RCLCPP_INFO(get_logger(), "################Start capturing################");
  sl::ERROR_CODE state = sl::ERROR_CODE::FAILURE;
  for (int attempt = 1; attempt <= 5; ++attempt) {
    state = zed_.open(init);
    if (state == sl::ERROR_CODE::SUCCESS) break;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  if (state != sl::ERROR_CODE::SUCCESS) {
    RCLCPP_ERROR(get_logger(), "ZED open failed: %s", sl::toString(state).c_str());
    throw std::runtime_error(std::string("ZED open failed: ") + sl::toString(state).c_str());
  }

  if (zed_.enablePositionalTracking(sl::PositionalTrackingParameters()) !=
      sl::ERROR_CODE::SUCCESS)
  {
    zed_.close();
    throw std::runtime_error("Enable positional tracking failed");
  }

  sl::BodyTrackingParameters body_params;
  body_params.detection_model = sl::BODY_TRACKING_MODEL::HUMAN_BODY_MEDIUM;
  body_params.body_format = sl::BODY_FORMAT::BODY_34;
  body_params.enable_tracking = true;
  body_params.enable_body_fitting = true;
  if (enable_body_tracking_ && zed_.enableBodyTracking(body_params) != sl::ERROR_CODE::SUCCESS) {
    zed_.close();
    throw std::runtime_error("Enable body tracking failed");
  }
  body_runtime_parameters_.detection_confidence_threshold = 25;

  enable_yolo_ = get_parameter("enable_yolo").as_bool();
  debug_window_ = get_parameter("debug_window").as_bool();
  mirror_ = get_parameter("mirror").as_bool();
  bag_labels_ = get_parameter("bag_labels").as_string_array();
  chair_labels_ = get_parameter("chair_labels").as_string_array();
  yolo_min_area_ = get_parameter("yolo_min_area").as_double();
  drink_min_area_ = get_parameter("drink_min_area").as_double();
  yolo_stride_ = std::max(1, static_cast<int>(get_parameter("yolo_stride").as_int()));
  enable_chair_ = get_parameter("enable_chair").as_bool();
  enable_drinks_ = get_parameter("enable_drinks").as_bool();
  face_threshold_ = get_parameter("face_threshold").as_double();

  loadModels();

  timer_ = create_wall_timer(
    std::chrono::duration<double>(1.0 / camera_fps_), std::bind(&ZedNode::loop, this));
}

ZedNode::~ZedNode()
{
  zed_.close();
}

std::string ZedNode::resolvePath(const std::string & path) const
{
  if (path.empty()) return "";
  if (std::filesystem::path(path).is_absolute()) return path;
  try {
    const std::string share =
      ament_index_cpp::get_package_share_directory("athome_vision") + "/" + path;
    if (std::filesystem::exists(share)) return share;
  } catch (...) {
  }
  if (std::filesystem::exists(path)) return path;
  if (std::filesystem::exists("src/athome_vision/" + path)) return "src/athome_vision/" + path;
  const std::string ws = "/home/dell/Documents/athome_ws/src/athome_vision/" + path;
  if (std::filesystem::exists(ws)) return ws;
  return "share/athome_vision/" + path;
}

void ZedNode::loadYolo(
  YoloDetector & detector, const std::string & model_param,
  const std::vector<std::string> & fallbacks, const std::string & name,
  float conf, float nms, int size)
{
  if (model_param.empty()) return;
  std::string path = resolvePath(model_param);
  if (!std::filesystem::exists(path)) {
    for (const auto & fallback : fallbacks) {
      const std::string candidate = resolvePath(fallback);
      if (std::filesystem::exists(candidate)) {
        path = candidate;
        break;
      }
    }
  }
  if (!std::filesystem::exists(path)) {
    RCLCPP_WARN(get_logger(), "%s model not found: %s", name.c_str(), path.c_str());
    return;
  }
  std::string err;
  if (detector.load(path, "", conf, nms, size, name, &err)) {
    RCLCPP_INFO(get_logger(), "%s model loaded: %s", name.c_str(), path.c_str());
  } else {
    RCLCPP_ERROR(get_logger(), "%s model error: %s", name.c_str(), err.c_str());
  }
}

void ZedNode::loadModels()
{
  const float conf = static_cast<float>(get_parameter("yolo_conf").as_double());
  const float nms = static_cast<float>(get_parameter("yolo_nms").as_double());
  const float drink_conf = static_cast<float>(get_parameter("drink_conf").as_double());
  const int size = static_cast<int>(get_parameter("yolo_size").as_int());

  if (enable_yolo_) {
    const std::string model_param = get_parameter("yolo_model").as_string();
    const std::string model_path = resolvePath(model_param);
    if (model_param.empty() || !std::filesystem::exists(model_path)) {
      RCLCPP_WARN(get_logger(), "YOLO model not found: %s", model_path.c_str());
      enable_yolo_ = false;
    } else {
      std::string err;
      const std::string classes = resolvePath(get_parameter("yolo_classes").as_string());
      if (!yolo_.load(model_path, classes, conf, nms, size, "bag", &err)) {
        RCLCPP_ERROR(get_logger(), "YOLO load error: %s", err.c_str());
        enable_yolo_ = false;
      } else {
        RCLCPP_INFO(get_logger(), "YOLO loaded: %s", model_path.c_str());
      }
    }
  }

  if (enable_drinks_) {
    loadYolo(botol_yolo_, get_parameter("botol_model").as_string(),
      {"models/botol.onnx"}, "Water", drink_conf, nms, size);
    loadYolo(cracker_yolo_, get_parameter("cracker_model").as_string(),
      {"models/cheezit.onnx", "models/best_cracker.onnx"}, "Cheez-It", drink_conf, nms, size);
    loadYolo(pringles_yolo_, get_parameter("pringles_model").as_string(),
      {"models/pringles_ chips _can.onnx", "models/pringles.onnx"}, "Pringles",
      drink_conf, nms, size);
  }

  if (enable_chair_) {
    loadYolo(chair_yolo_, get_parameter("chair_model").as_string(),
      {}, "chair", conf, nms, size);
  }

  const std::string color_labels =
    resolvePath(get_parameter("color_labels").as_string());
  const std::string hair_labels =
    resolvePath(get_parameter("hair_labels").as_string());
  const std::string color_path = resolvePath(get_parameter("color_model").as_string());
  const std::string hair_path = resolvePath(get_parameter("hair_model").as_string());
  const std::string face_path = resolvePath(get_parameter("face_model").as_string());

  std::vector<std::string> gender_candidates;
  auto addGenderCandidate = [&](const std::string & path) {
      if (path.empty()) return;
      if (std::find(gender_candidates.begin(), gender_candidates.end(), path) !=
          gender_candidates.end()) return;
      gender_candidates.push_back(path);
    };
  addGenderCandidate(resolvePath(get_parameter("gender_model").as_string()));
  addGenderCandidate(resolvePath("models/gender_model_weights.onnx"));

  bool gender_loaded = false;
  std::string gender_err;
  for (const auto & path : gender_candidates) {
    if (!std::filesystem::exists(path)) continue;
    if (analyzer_.loadGender(path, &gender_err)) {
      RCLCPP_INFO(get_logger(), "Gender model loaded: %s", path.c_str());
      gender_loaded = true;
      break;
    }
  }
  if (!gender_loaded) {
    RCLCPP_WARN(get_logger(), "Gender model unavailable, gender will be unknown: %s",
      gender_err.c_str());
  }
  if (!color_path.empty()) {
    std::string err;
    if (!analyzer_.loadColor(color_path, color_labels, &err)) {
      RCLCPP_WARN(get_logger(), "Color model unavailable, using HSV: %s", err.c_str());
    }
  }
  if (!hair_path.empty()) {
    std::string err;
    if (!analyzer_.loadHair(hair_path, hair_labels, &err)) {
      RCLCPP_WARN(get_logger(), "Hair model unavailable, using HSV: %s", err.c_str());
    }
  }
  if (!face_path.empty()) {
    std::string err;
    if (analyzer_.loadFace(face_path, &err)) {
      RCLCPP_INFO(get_logger(), "Face SFace loaded: %s (threshold %.3f)",
        face_path.c_str(), face_threshold_);
      loadPreRegisteredHost();
    } else {
      RCLCPP_WARN(get_logger(), "Face model unavailable: %s", err.c_str());
    }
  }
}

void ZedNode::loadPreRegisteredHost()
{
  std::string host_name = get_parameter("host_name").as_string();
  std::string host_photo = get_parameter("host_photo").as_string();
  if (host_name.empty() || host_photo.empty()) return;

  std::string full_path = resolvePath(host_photo);
  if (!std::filesystem::exists(full_path)) {
    std::string fallback = resolvePath("models/" + host_photo);
    if (std::filesystem::exists(fallback)) {
      full_path = fallback;
    } else {
      RCLCPP_INFO(get_logger(),
        "[Host Pre-Register] Info: File foto host tidak ditemukan di '%s'. Pre-registrasi dari foto dilewati.",
        host_photo.c_str());
      return;
    }
  }

  cv::Mat img = cv::imread(full_path);
  if (img.empty()) {
    RCLCPP_WARN(get_logger(),
      "[Host Pre-Register] Gagal membaca gambar foto host: %s", full_path.c_str());
    return;
  }

  std::vector<float> feat = analyzer_.extractFaceFromImage(img);
  if (feat.empty()) {
    RCLCPP_WARN(get_logger(),
      "[Host Pre-Register] Gagal mengekstrak fitur wajah host dari: %s", full_path.c_str());
    return;
  }

  RegisteredPerson host_person;
  host_person.name = host_name;
  host_person.feature = feat;
  host_person.feature_samples = 30; // Pre-registered with maximum confidence
  host_person.has_color = false;
  host_person.last_tracking_id = -1;
  host_person.last_seen = now();

  auto it = std::remove_if(registered_persons_.begin(), registered_persons_.end(),
    [&host_name](const RegisteredPerson & p) { return p.name == host_name; });
  registered_persons_.erase(it, registered_persons_.end());

  registered_persons_.push_back(host_person);
  RCLCPP_INFO(get_logger(),
    "\033[1;32m[Host Pre-Register] Berhasil mendaftarkan Host '%s' dari file foto: %s (%dx%d px)\033[0m",
    host_name.c_str(), full_path.c_str(), img.cols, img.rows);
}

void ZedNode::publishYaw()
{
  if (!publish_yaw_) return;
  sl::SensorsData sensors;
  if (zed_.getSensorsData(sensors, sl::TIME_REFERENCE::IMAGE) != sl::ERROR_CODE::SUCCESS ||
      !sensors.imu.is_available)
  {
    return;
  }
  const auto q = sensors.imu.pose.getOrientation();
  const float yaw = zed_vision::quatToYaw(q.x, q.y, q.z, q.w) * 180.0f / zed_vision::PI_F;
  std_msgs::msg::Float64 msg;
  msg.data = static_cast<double>(
    yaw_360_ ? zed_vision::wrap360(yaw) : zed_vision::wrap180(yaw));
  yaw_publisher_->publish(msg);
}

void ZedNode::processBags(const cv::Mat & bgr, const sl::Mat & cloud)
{
  preview_bags_.clear();
  if (!enable_bag_) return;

  std::vector<YoloDetection> dets = last_bag_dets_;
  if (enable_yolo_ && yolo_.ready() && yolo_frame_++ % yolo_stride_ == 0) {
    yolo_.detect(bgr, bag_labels_, dets);
    last_bag_dets_ = dets;
  }

  std::string data;
  for (const auto & det : dets) {
    if (static_cast<double>(det.rect.area()) < yolo_min_area_) continue;
    double x = 0.0, y = 0.0, z = 0.0;
    const bool ok = zed_vision::get3dWithOffset(det.rect, cloud, x, y, z);
    const double dist = ok ? std::hypot(x, y) : 0.0;
    preview_bags_.push_back({det.rect, ok, x, y, dist});
    if (is_help_me_carry_ && ok) {
      IC(det.label, x, y, dist);
    }
    if (!data.empty()) data += ";";
    data += cv::format("bag:%d,%d,%d,%d|", det.rect.x, det.rect.y, det.rect.width, det.rect.height);
    data += ok ? cv::format("%.2f,%.2f,%.2f|%.2f", x, y, z, det.score)
               : cv::format("nan,nan,nan|%.2f", det.score);
  }

  std_msgs::msg::String msg;
  msg.data = data;
  bag_dets_publisher_->publish(msg);
}

void ZedNode::processDrinks(const cv::Mat & bgr, const sl::Mat & cloud)
{
  std::vector<YoloDetection> dets = last_drink_dets_;
  if (enable_drinks_ && drink_frame_++ % yolo_stride_ == 0) {
    dets.clear();
    for (YoloDetector * detector : {&botol_yolo_, &cracker_yolo_, &pringles_yolo_}) {
      if (!detector->ready()) continue;
      std::vector<YoloDetection> found;
      detector->detect(bgr, {}, found);
      dets.insert(dets.end(), found.begin(), found.end());
    }
    last_drink_dets_ = dets;
  }

  preview_drinks_.clear();
  std::string data;
  for (const auto & det : dets) {
    if (static_cast<double>(det.rect.area()) < drink_min_area_) continue;
    double x = 0.0, y = 0.0, z = 0.0;
    const bool ok = zed_vision::get3dWithOffset(det.rect, cloud, x, y, z);
    const double dist = ok ? std::hypot(x, y) : 0.0;
    preview_drinks_.push_back({det.label, det.rect, ok, x, y, dist});
    if (is_receptionist_ && ok) {
      IC(det.label, x, y, dist);
    }
    if (!data.empty()) data += ";";
    data += cv::format("%s:%d,%d,%d,%d|", det.label.c_str(),
      det.rect.x, det.rect.y, det.rect.width, det.rect.height);
    data += ok ? cv::format("%.2f,%.2f,%.2f|%.2f", x, y, z, det.score)
               : cv::format("nan,nan,nan|%.2f", det.score);
  }

  std_msgs::msg::String msg;
  msg.data = data;
  drink_dets_publisher_->publish(msg);
}

void ZedNode::processPersons(const cv::Mat & bgr, const sl::Mat & cloud)
{
  preview_persons_.clear();

  if (bodies_.body_list.empty()) {
    std_msgs::msg::String person_msg;
    person_msg.data = "NO_PERSON";
    persons_publisher_->publish(person_msg);
    std_msgs::msg::String gesture_msg;
    gesture_msg.data = "NONE";
    gesture_publisher_->publish(gesture_msg);
    std_msgs::msg::String attr_msg;
    attr_msg.data = "";
    attributes_publisher_->publish(attr_msg);
    std_msgs::msg::String name_msg;
    name_msg.data = "";
    recognized_name_publisher_->publish(name_msg);
    return;
  }

  struct PersonEntry
  {
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
    if (!zed_vision::computeBbox(body.keypoint_2d, bgr.cols, bgr.rows, bbox)) continue;
    PersonEntry entry;
    entry.body = &body;
    entry.id = body.id;
    entry.bbox = bbox;
    entry.has_depth = zed_vision::get3dWithOffset(bbox, cloud, entry.x, entry.y, entry.z);
    entry.range_m = (entry.has_depth && std::isfinite(entry.x) && std::isfinite(entry.y))
      ? std::hypot(entry.x, entry.y)
      : (1e6 - static_cast<double>(bbox.area()));
    entry.dist = entry.range_m;
    entries.push_back(entry);
  }

  std::sort(entries.begin(), entries.end(), [](const PersonEntry & a, const PersonEntry & b) {
    return a.range_m < b.range_m;
  });

  const auto current_time = now();
  int op_match_idx = -1;
  if (op_mem_.active) {
    const double dt = (current_time - op_mem_.last_seen).seconds();
    if (dt <= 4.0) {
      const double pred_x = op_mem_.x + op_mem_.vx * std::min(dt, 2.0);
      const double pred_y = op_mem_.y + op_mem_.vy * std::min(dt, 2.0);
      const double max_match_dist = 1.0 + 0.8 * dt;
      double best_cost = std::numeric_limits<double>::infinity();
      for (size_t i = 0; i < entries.size(); ++i) {
        const auto & cand = entries[i];
        if (!cand.has_depth) continue;
        const double dist_diff = std::hypot(cand.x - pred_x, cand.y - pred_y);
        if (dist_diff > max_match_dist) continue;
        double color_penalty = 0.0;
        if (op_mem_.has_color) {
          const cv::Scalar cand_hsv = zed_vision::extractTorsoColor(bgr, cand.bbox);
          const double dh = std::abs(cand_hsv[0] - op_mem_.torso_hsv[0]);
          const double ds = std::abs(cand_hsv[1] - op_mem_.torso_hsv[1]);
          color_penalty = (dh / 180.0) * 0.5 + (ds / 255.0) * 0.3;
        }
        const double cost = dist_diff + color_penalty;
        if (cost < best_cost) {
          best_cost = cost;
          op_match_idx = static_cast<int>(i);
        }
      }
    } else {
      op_mem_.active = false;
      op_mem_.has_color = false;
    }
  }

  if (!op_mem_.active && !entries.empty()) {
    op_match_idx = 0;
    op_mem_.active = true;
    op_mem_.stable_id = 0;
    op_mem_.vx = op_mem_.vy = 0.0;
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

  if (op_match_idx >= 0 && op_match_idx < static_cast<int>(entries.size())) {
    if (op_match_idx > 0) std::swap(entries[0], entries[static_cast<size_t>(op_match_idx)]);
    auto & op = entries.front();
    op.id = op_mem_.stable_id;
    op.label = "operator";
    if (op.has_depth) {
      const double dt = std::max(0.01, (current_time - op_mem_.last_seen).seconds());
      if (dt < 1.0) {
        op_mem_.vx = 0.7 * op_mem_.vx + 0.3 * ((op.x - op_mem_.x) / dt);
        op_mem_.vy = 0.7 * op_mem_.vy + 0.3 * ((op.y - op_mem_.y) / dt);
      }
      op_mem_.x = op.x;
      op_mem_.y = op.y;
    }
    op_mem_.last_seen = current_time;
    const cv::Scalar hsv = zed_vision::extractTorsoColor(bgr, op.bbox);
    op_mem_.torso_hsv = 0.85 * op_mem_.torso_hsv + 0.15 * hsv;
    op_mem_.has_color = true;
  }

  if (!entries.empty()) {
    entries.front().label = "operator";
    if (is_help_me_carry_ && entries.front().has_depth) {
      const auto & person = entries.front();
      IC(person.id, person.label, person.x, person.y, person.dist);
    }
  }

  if (!pending_register_name_.empty() && !entries.empty() && is_receptionist_) {
    const PersonEntry * target = &entries.front();
    std::vector<float> feature;
    for (const auto & candidate : entries) {
      feature = analyzer_.faceEmbedding(
        bgr, candidate.bbox, candidate.body->keypoint_2d);
      if (!feature.empty()) {
        target = &candidate;
        break;
      }
    }
    const cv::Scalar torso_hsv = zed_vision::extractTorsoColor(bgr, target->bbox);
    bool updated = false;
    for (auto & person : registered_persons_) {
      if (person.name == pending_register_name_) {
        person.last_tracking_id = target->id;
        person.last_seen = current_time;
        if (!person.has_color && target->bbox.width > 20) {
          person.torso_hsv = torso_hsv;
          person.has_color = true;
        }
        if (!feature.empty()) {
          if (person.feature.empty()) {
            person.feature = feature;
            person.feature_samples = 1;
          } else if (person.feature_samples < 25) {
            float norm_sq = 0.0f;
            for (size_t k = 0; k < feature.size(); ++k) {
              person.feature[k] = person.feature[k] * 0.85f + feature[k] * 0.15f;
              norm_sq += person.feature[k] * person.feature[k];
            }
            const float inv_norm = 1.0f / std::sqrt(std::max(norm_sq, 1e-6f));
            for (size_t k = 0; k < feature.size(); ++k) {
              person.feature[k] *= inv_norm;
            }
            person.feature_samples++;
          }
        }
        updated = true;
        if (person.feature_samples >= 3) {
          RCLCPP_INFO(get_logger(),
            "Locked face & clothing for \"%s\" (body id %d, %d face samples)",
            pending_register_name_.c_str(), target->id, person.feature_samples);
          pending_register_name_.clear();
        }
        break;
      }
    }
    if (!updated) {
      RegisteredPerson new_person;
      new_person.name = pending_register_name_;
      new_person.feature = feature;
      new_person.feature_samples = feature.empty() ? 0 : 1;
      new_person.torso_hsv = torso_hsv;
      new_person.has_color = (target->bbox.width > 20);
      new_person.last_tracking_id = target->id;
      new_person.last_seen = current_time;
      registered_persons_.push_back(new_person);
      if (new_person.feature_samples >= 3) {
        pending_register_name_.clear();
      }
    }
  } else if (!pending_register_name_.empty() && is_receptionist_ && entries.empty()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
      "Guest \"%s\" pending registration: no person detected",
      pending_register_name_.c_str());
  }

  // Ekstraksi fitur visual untuk semua orang yang terdeteksi
  struct EntryFeature {
    std::vector<float> face_feat;
    cv::Scalar torso_hsv{0, 0, 0};
    bool has_torso{false};
  };
  std::vector<EntryFeature> entry_feats(entries.size());
  for (size_t i = 0; i < entries.size(); ++i) {
    if (analyzer_.faceReady()) {
      entry_feats[i].face_feat = analyzer_.faceEmbedding(
        bgr, entries[i].bbox, entries[i].body->keypoint_2d);
    }
    entry_feats[i].torso_hsv = zed_vision::extractTorsoColor(bgr, entries[i].bbox);
    entry_feats[i].has_torso = (entries[i].bbox.width > 20 && entries[i].bbox.height > 40);
  }

  // Hitung matriks kecocokan (affinity) antara entry terdeteksi dan profil terdaftar
  struct PersonMatch {
    size_t entry_idx;
    size_t profile_idx;
    double score;
  };
  std::vector<PersonMatch> matches;

  for (size_t i = 0; i < entries.size(); ++i) {
    for (size_t j = 0; j < registered_persons_.size(); ++j) {
      const auto & prof = registered_persons_[j];
      const auto & feat = entry_feats[i];
      double score = 0.0;

      // 1. Face Cosine Similarity (SFace 128-D)
      if (!prof.feature.empty() && !feat.face_feat.empty()) {
        const float sim = AttributeAnalyzer::matchFace(feat.face_feat, prof.feature);
        if (sim >= static_cast<float>(face_threshold_)) {
          score += 2.5 + (sim - face_threshold_) * 5.0;
        } else if (sim >= 0.25f) {
          score += (sim - face_threshold_) * 1.5;
        } else {
          score -= 2.5; // Penalti kuat jika wajah berbeda
        }
      }

      // 2. Torso Clothing Color Similarity (HSV)
      if (prof.has_color && feat.has_torso) {
        double dh = std::min(std::abs(feat.torso_hsv[0] - prof.torso_hsv[0]),
                             180.0 - std::abs(feat.torso_hsv[0] - prof.torso_hsv[0]));
        double ds = std::abs(feat.torso_hsv[1] - prof.torso_hsv[1]);
        double dv = std::abs(feat.torso_hsv[2] - prof.torso_hsv[2]);
        double color_dist = (dh / 90.0) * 0.6 + (ds / 255.0) * 0.2 + (dv / 255.0) * 0.2;
        if (color_dist < 0.30) {
          score += 1.5 * (1.0 - color_dist / 0.30);
        } else if (color_dist > 0.55) {
          score -= 1.8; // Penalti kuat jika warna pakaian jelas berbeda
        }
      }

      // 3. Tracking ID continuity (ZED Tracker ID)
      if (prof.last_tracking_id == entries[i].id &&
          (current_time - prof.last_seen).seconds() < 5.0)
      {
        score += 2.0;
      }

      if (score > 0.5) {
        matches.push_back({i, j, score});
      }
    }
  }

  // Urutkan skor tertinggi ke terendah
  std::sort(matches.begin(), matches.end(), [](const PersonMatch & a, const PersonMatch & b) {
    return a.score > b.score;
  });

  // Pencocokan Eksklusif 1-ke-1 (Orang A dan B tidak bisa tertukar/saling rebut)
  std::vector<std::string> recognized_names(entries.size(), "");
  std::vector<bool> profile_assigned(registered_persons_.size(), false);

  for (const auto & m : matches) {
    if (recognized_names[m.entry_idx].empty() && !profile_assigned[m.profile_idx]) {
      recognized_names[m.entry_idx] = registered_persons_[m.profile_idx].name;
      profile_assigned[m.profile_idx] = true;

      auto & prof = registered_persons_[m.profile_idx];
      prof.last_tracking_id = entries[m.entry_idx].id;
      prof.last_seen = current_time;

      // Akumulasikan embedding wajah jika kualitas embedding bagus
      if (!entry_feats[m.entry_idx].face_feat.empty() && prof.feature_samples < 30) {
        float norm_sq = 0.0f;
        for (size_t k = 0; k < prof.feature.size(); ++k) {
          prof.feature[k] = prof.feature[k] * 0.90f + entry_feats[m.entry_idx].face_feat[k] * 0.10f;
          norm_sq += prof.feature[k] * prof.feature[k];
        }
        const float inv_norm = 1.0f / std::sqrt(std::max(norm_sq, 1e-6f));
        for (size_t k = 0; k < prof.feature.size(); ++k) {
          prof.feature[k] *= inv_norm;
        }
        prof.feature_samples++;
      }
    }
  }

  std::ostringstream ss;
  bool first_person = true;
  std::string gesture_result = "NONE";
  std::string first_recognized_name;

  for (size_t idx = 0; idx < entries.size(); ++idx) {
    const auto & entry = entries[idx];
    const auto & points = entry.body->keypoint_2d;
    std::string recognized_name = recognized_names[idx];

    // Jika sedang mendaftarkan nama tamu baru di pintu
    if (recognized_name.empty() && is_receptionist_ && !pending_register_name_.empty() && first_person) {
      recognized_name = pending_register_name_;
    }

    if (first_recognized_name.empty() && !recognized_name.empty()) {
      first_recognized_name = recognized_name;
    }

    PrevPerson person;
    person.id = entry.id;
    person.bbox = entry.bbox;
    person.has_pos = entry.has_depth;
    person.x_m = entry.x;
    person.y_m = entry.y;
    person.range_m = entry.range_m;
    person.name = recognized_name;
    for (const auto & point : points) {
      person.kpt.emplace_back(
        std::isfinite(point.x) ? point.x : NAN,
        std::isfinite(point.y) ? point.y : NAN);
    }

    const bool classify = is_receptionist_ && (!pending_register_name_.empty() || !recognized_name.empty());
    if (classify) {
      person.clothing = analyzer_.clothingColor(bgr, entry.bbox, points);
      person.hair = analyzer_.hairColor(bgr, entry.bbox);
      person.gender = analyzer_.gender(points, bgr, entry.bbox);
    }

    if (first_person) {
      gesture_result = AttributeAnalyzer::detectGesture(points);
      if (is_help_me_carry_ && gesture_result != "NONE") {
        IC(entry.id, gesture_result);
      }

      std_msgs::msg::String attr_msg;
      attr_msg.data = classify
        ? person.gender + ", " + person.clothing + ", " + person.hair
        : "";
      attributes_publisher_->publish(attr_msg);

      if (is_receptionist_) {
        std::string display;
        if (is_help_me_carry_) {
          display = "ID " + std::to_string(entry.id);
        } else if (!recognized_name.empty()) {
          display = recognized_name;
        } else {
          display = "Guest";
        }
        IC(display, entry.x, entry.y, entry.range_m);
        if (classify) {
          IC(display, person.gender, person.clothing, person.hair);
        }
      }
    }

    preview_persons_.push_back(person);

    if (!first_person) ss << ";";
    std::string label;
    if (is_receptionist_ && !is_help_me_carry_) {
      if (!recognized_name.empty()) {
        label = recognized_name;
      } else {
        label = first_person ? "guest" : "person";
      }
    } else {
      label = first_person ? "operator" : "person";
    }
    first_person = false;
    ss << label << ":" << entry.bbox.x << "," << entry.bbox.y << ","
       << entry.bbox.width << "," << entry.bbox.height << "|";
    if (entry.has_depth) {
      ss << std::fixed << std::setprecision(2) << entry.x << "," << entry.y << "," << entry.z;
    } else {
      ss << "nan,nan,nan";
    }
    ss << "|id=" << entry.id << "|kpt=";
    for (size_t k = 0; k < 34; ++k) {
      if (k > 0) ss << ",";
      if (k < points.size() && std::isfinite(points[k].x) && std::isfinite(points[k].y)) {
        ss << std::fixed << std::setprecision(2) << points[k].x << "," << points[k].y;
      } else {
        ss << "-1,-1";
      }
    }
  }

  std_msgs::msg::String gesture_msg;
  gesture_msg.data = gesture_result;
  gesture_publisher_->publish(gesture_msg);

  std_msgs::msg::String person_msg;
  person_msg.data = first_person ? "NO_PERSON" : ss.str();
  persons_publisher_->publish(person_msg);

  std_msgs::msg::String name_msg;
  name_msg.data = first_recognized_name;
  recognized_name_publisher_->publish(name_msg);
}

void ZedNode::processChairs(const cv::Mat & bgr, const sl::Mat & cloud)
{
  std::vector<YoloDetection> dets = last_chair_dets_;
  bool fresh_detections = false;
  if (enable_chair_ && chair_frame_++ % yolo_stride_ == 0) {
    dets.clear();
    if (chair_yolo_.ready()) {
      chair_yolo_.detect(bgr, {}, dets);
    } else if (enable_yolo_ && yolo_.ready()) {
      yolo_.detect(bgr, chair_labels_, dets);
    }
    last_chair_dets_ = dets;
    fresh_detections = true;
  }

  if (!enable_chair_) chair_memory_.clear();

  const auto current_time = now();

  auto same_chair = [](const ChairMemory & memory, const YoloDetection & det,
      bool has_pos, double x, double y) {
      const cv::Rect inter = memory.bbox & det.rect;
      const double inter_area = static_cast<double>(inter.area());
      const double union_area =
        static_cast<double>(memory.bbox.area()) + det.rect.area() - inter_area;
      const double iou = union_area > 0.0 ? inter_area / union_area : 0.0;
      const double mcx = memory.bbox.x + memory.bbox.width * 0.5;
      const double mcy = memory.bbox.y + memory.bbox.height * 0.5;
      const double dcx = det.rect.x + det.rect.width * 0.5;
      const double dcy = det.rect.y + det.rect.height * 0.5;
      const double scale = static_cast<double>(std::max(
        std::max(memory.bbox.width, memory.bbox.height),
        std::max(det.rect.width, det.rect.height)));
      const double center_dist = std::hypot(mcx - dcx, mcy - dcy);
      const bool depth_ok = !has_pos || !memory.has_pos ||
        std::hypot(x - memory.x, y - memory.y) < 1.0;
      return depth_ok && (iou > 0.20 || center_dist < scale * 0.75);
    };

  if (fresh_detections) {
    for (const auto & det : dets) {
      if (static_cast<double>(det.rect.area()) < yolo_min_area_) continue;
      double x = 0.0, y = 0.0, z = 0.0;
      const bool ok = zed_vision::get3dWithOffset(det.rect, cloud, x, y, z);

      int match = -1;
      double best_dist = 1e9;
      for (size_t i = 0; i < chair_memory_.size(); ++i) {
        const auto & memory = chair_memory_[i];
        if (!same_chair(memory, det, ok, x, y)) continue;
        const double mcx = memory.bbox.x + memory.bbox.width * 0.5;
        const double mcy = memory.bbox.y + memory.bbox.height * 0.5;
        const double dcx = det.rect.x + det.rect.width * 0.5;
        const double dcy = det.rect.y + det.rect.height * 0.5;
        const double center_dist = std::hypot(mcx - dcx, mcy - dcy);
        if (center_dist < best_dist) {
          best_dist = center_dist;
          match = static_cast<int>(i);
        }
      }

      if (match >= 0) {
        auto & memory = chair_memory_[static_cast<size_t>(match)];
        memory.bbox = det.rect;
        memory.last_seen = current_time;
        if (ok) {
          memory.has_pos = true;
          memory.x = x;
          memory.y = y;
          memory.z = z;
          memory.range_m = std::hypot(x, y);
        }
      } else {
        ChairMemory memory;
        memory.bbox = det.rect;
        memory.last_seen = current_time;
        if (ok) {
          memory.has_pos = true;
          memory.x = x;
          memory.y = y;
          memory.z = z;
          memory.range_m = std::hypot(x, y);
        }
        chair_memory_.push_back(memory);
      }
    }
  }

  auto compute_match_score = [&](const ChairMemory & chair, const PrevPerson & person) -> double {
      // 1. WAJIB: Ada irisan 2D antara bounding box kursi dan orang
      const cv::Rect inter = chair.bbox & person.bbox;
      if (inter.area() <= 0) return 0.0;

      const double chair_area = static_cast<double>(chair.bbox.area());
      const double person_area = static_cast<double>(person.bbox.area());
      if (chair_area <= 0.0 || person_area <= 0.0) return 0.0;

      const double overlap_chair = static_cast<double>(inter.area()) / chair_area;
      const double overlap_person = static_cast<double>(inter.area()) / person_area;

      // 2. Cek konsistensi kedalaman 3D (jika keduanya memiliki posisi 3D valid)
      if (chair.has_pos && person.has_pos) {
        const double dist_3d = std::hypot(chair.x - person.x_m, chair.y - person.y_m);
        // Jika jarak 3D > 1.2m, orang ini di latar depan/belakang, bukan duduk di kursi ini
        if (dist_3d > 1.20) return 0.0;
      }

      // 3. Keselarasan horizontal (X-axis)
      const int inter_x = std::max(0,
        std::min(chair.bbox.x + chair.bbox.width, person.bbox.x + person.bbox.width) -
        std::max(chair.bbox.x, person.bbox.x));
      const int min_w = std::min(chair.bbox.width, person.bbox.width);
      const double x_ratio = min_w > 0 ? static_cast<double>(inter_x) / min_w : 0.0;

      // 4. Pengecekan posisi tubuh orang terhadap kursi
      const int cx = person.bbox.x + person.bbox.width / 2;
      const int cy = person.bbox.y + person.bbox.height / 2;
      const int hip_y = person.bbox.y + static_cast<int>(person.bbox.height * 0.65);
      const bool hip_inside = chair.bbox.contains(cv::Point(cx, hip_y));
      const bool center_inside = chair.bbox.contains(cv::Point(cx, cy));

      // Pengecekan titik tengah kursi di dalam tubuh orang
      const int chair_cx = chair.bbox.x + chair.bbox.width / 2;
      const int chair_cy = chair.bbox.y + chair.bbox.height / 2;
      const bool chair_center_inside = person.bbox.contains(cv::Point(chair_cx, chair_cy));

      // Keypoints tubuh ZED
      bool kpt_inside = false;
      for (const auto & kp : person.kpt) {
        if (std::isfinite(kp.x) && std::isfinite(kp.y) &&
            chair.bbox.contains(cv::Point(static_cast<int>(kp.x), static_cast<int>(kp.y))))
        {
          kpt_inside = true;
          break;
        }
      }

      // Syarat kelayakan:
      // Harus memiliki overlap 2D yang nyata (> 12% dari kursi atau orang)
      // ATAU memiliki keselarasan horizontal yang kuat (x_ratio > 0.40) dengan hip/center/keypoint berada di dalam kursi
      const bool qualified = (overlap_chair > 0.12 || overlap_person > 0.12) ||
                             (x_ratio > 0.40 && (hip_inside || center_inside || chair_center_inside || kpt_inside));

      if (!qualified) return 0.0;

      // Skor kecocokan: semakin tinggi overlap dan keselarasan horizontal, semakin besar skornya
      double score = overlap_chair * 2.0 + overlap_person + x_ratio;
      if (hip_inside || center_inside) score += 1.0;
      if (kpt_inside) score += 0.5;
      if (chair.has_pos && person.has_pos) {
        const double dist_3d = std::hypot(chair.x - person.x_m, chair.y - person.y_m);
        score += std::max(0.0, 1.20 - dist_3d);
      }
      return score;
    };

  struct MatchPair {
    size_t chair_idx;
    size_t person_idx;
    double score;
  };
  std::vector<MatchPair> match_candidates;
  for (size_t c = 0; c < chair_memory_.size(); ++c) {
    for (size_t p = 0; p < preview_persons_.size(); ++p) {
      const double score = compute_match_score(chair_memory_[c], preview_persons_[p]);
      if (score > 0.0) {
        match_candidates.push_back({c, p, score});
      }
    }
  }

  // Urutkan pasangan pencocokan dari skor tertinggi ke terendah
  std::sort(match_candidates.begin(), match_candidates.end(),
    [](const MatchPair & a, const MatchPair & b) {
      return a.score > b.score;
    });

  // Pencocokan eksklusif 1-ke-1: 1 orang hanya bisa menempati MAKSIMAL 1 kursi!
  std::vector<bool> chair_occupied(chair_memory_.size(), false);
  std::vector<bool> person_assigned(preview_persons_.size(), false);

  for (const auto & match : match_candidates) {
    if (!chair_occupied[match.chair_idx] && !person_assigned[match.person_idx]) {
      chair_occupied[match.chair_idx] = true;
      person_assigned[match.person_idx] = true;
      preview_persons_[match.person_idx].is_sitting = true;
    }
  }

  auto side_of = [&](const cv::Rect & chair_bbox, bool has_pos, double y) {
      if (has_pos) {
        if (y > 0.20) return std::string("left");
        if (y < -0.20) return std::string("right");
        return std::string("center");
      }
      const int cx = chair_bbox.x + chair_bbox.width / 2;
      const int width = bgr.cols > 0 ? bgr.cols : 640;
      if (cx < width / 3) return std::string(mirror_ ? "right" : "left");
      if (cx > 2 * width / 3) return std::string(mirror_ ? "left" : "right");
      return std::string("center");
    };

  preview_chairs_.clear();
  std::string data;
  std::string best_empty_side = "none";
  double best_empty_dist = 1e9;
  constexpr double memory_timeout = 10.0;

  for (size_t i = 0; i < chair_memory_.size(); ) {
    ChairMemory & chair = chair_memory_[i];
    const bool occupied = chair_occupied[i];
    if (occupied) {
      chair.last_seen = current_time;
    } else if ((current_time - chair.last_seen).seconds() > memory_timeout) {
      chair_memory_.erase(chair_memory_.begin() + i);
      chair_occupied.erase(chair_occupied.begin() + i);
      continue;
    }

    const bool ok = chair.has_pos;
    const double dist = ok ? chair.range_m : 0.0;
    const std::string side = side_of(chair.bbox, ok, chair.y);

    preview_chairs_.push_back(
      {chair.bbox, ok, chair.x, chair.y, dist, occupied, side});

    if (!occupied) {
      if (ok && dist > 0.1 && dist < best_empty_dist) {
        best_empty_dist = dist;
        best_empty_side = side;
      } else if (best_empty_side == "none") {
        best_empty_side = side;
      }
    }

    if (is_receptionist_) {
      static auto last_chair_log = std::chrono::steady_clock::now();
      const auto chair_now = std::chrono::steady_clock::now();
      if (std::chrono::duration_cast<std::chrono::milliseconds>(
          chair_now - last_chair_log).count() > 2000)
      {
        last_chair_log = chair_now;
        IC(occupied ? "OCCUPIED" : "EMPTY", side, chair.x, chair.y, dist);
      }
    }

    if (!data.empty()) data += ";";
    data += cv::format(
      "chair:%d,%d,%d,%d|", chair.bbox.x, chair.bbox.y,
      chair.bbox.width, chair.bbox.height);
    data += ok
      ? cv::format("%.2f,%.2f,%.2f|", chair.x, chair.y, chair.z)
      : "nan,nan,nan|";
    data += occupied ? "occupied|" : "empty|";
    data += side;
    ++i;
  }

  std_msgs::msg::String chair_msg;
  chair_msg.data = data;
  chair_dets_publisher_->publish(chair_msg);

  std_msgs::msg::String seat_msg;
  seat_msg.data = best_empty_side;
  empty_seat_publisher_->publish(seat_msg);
}

void ZedNode::showPreview(const cv::Mat & bgr)
{
  if (!debug_window_ || bgr.empty()) return;

  try {
    cv::Mat canvas;
    if (mirror_) cv::flip(bgr, canvas, 1);
    else canvas = bgr.clone();
    const int W = canvas.cols;
    const int H = canvas.rows;
    const bool flip = mirror_;

    auto flipRect = [&](const cv::Rect & in) {
        cv::Rect r = in;
        if (flip) r.x = W - r.x - r.width;
        return r;
      };
    auto banner = [&](const std::string & text, int x, int y,
        const cv::Scalar & border, const cv::Scalar & bg,
        const cv::Scalar & fg, double scale = 0.55, int thickness = 1) {
        int baseline = 0;
        const cv::Size size = cv::getTextSize(
          text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
        const int bx = std::clamp(x, 0, std::max(0, W - size.width - 6));
        const int by = std::clamp(y - size.height - 4, 0, std::max(0, H - size.height - 6));
        cv::Rect bg_rect(bx, by, size.width + 6, size.height + 6);
        bg_rect &= cv::Rect(0, 0, W, H);
        if (bg_rect.width > 0 && bg_rect.height > 0) {
          cv::rectangle(canvas, bg_rect, bg, -1);
          cv::rectangle(canvas, bg_rect, border, 1, cv::LINE_AA);
        }
        cv::putText(canvas, text, cv::Point(bx + 3, by + size.height + 1),
          cv::FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv::LINE_AA);
      };

    for (size_t i = 0; i < std::min<size_t>(4, preview_persons_.size()); ++i) {
      const auto & p = preview_persons_[i];
      if (p.has_pos && p.range_m > 4.5) continue;
      cv::Rect r = flipRect(p.bbox);
      r &= cv::Rect(0, 0, W, H);
      if (r.width <= 0 || r.height <= 0) continue;

      const cv::Scalar color = p.is_sitting ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
      cv::rectangle(canvas, r, color, 2);
      std::string label;
      if (is_receptionist_ && !is_help_me_carry_) {
        if (!p.name.empty()) {
          label = p.name;
        } else {
          label = (i == 0 ? "Guest" : "Person");
        }
      } else {
        label = "ID: " + std::to_string(p.id);
      }
      if (p.is_sitting) label += " [Seated]";
      banner(label, r.x, r.y, color,
        p.is_sitting ? cv::Scalar(0, 0, 180) : cv::Scalar(0, 140, 0),
        cv::Scalar(255, 255, 255), 0.58, 2);

      if (!p.gender.empty() || !p.clothing.empty() || !p.hair.empty()) {
        std::string attrs = p.gender;
        if (!p.clothing.empty()) attrs += (attrs.empty() ? "" : " | ") + p.clothing;
        if (!p.hair.empty()) attrs += (attrs.empty() ? "" : " | ") + p.hair;
        banner(attrs, r.x, r.y + r.height + 22,
          cv::Scalar(0, 180, 180), cv::Scalar(20, 30, 30), cv::Scalar(0, 255, 255), 0.48, 1);
      }

      // Visualisasi titik tengah dan keypoint dinonaktifkan sementara (hanya bounding box)
      /*
      const int cx = r.x + r.width / 2;
      const int cy = r.y + r.height / 2;
      cv::circle(canvas, cv::Point(cx, cy), 5, cv::Scalar(0, 255, 255), -1);
      cv::drawMarker(canvas, cv::Point(cx, cy), cv::Scalar(0, 255, 255), cv::MARKER_CROSS, 14, 2);
      if (i == 0) cv::line(canvas, cv::Point(cx, cy), cv::Point(W / 2, H), cv::Scalar(0, 69, 255), 2);

      for (size_t k = 0; k < p.kpt.size(); ++k) {
        const auto & pt = p.kpt[k];
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y)) continue;
        const bool big = (k == 5 || k == 7 || k == 12 || k == 14);
        const int kx = flip ? (W - static_cast<int>(std::lround(pt.x)))
                            : static_cast<int>(std::lround(pt.x));
        const int ky = static_cast<int>(std::lround(pt.y));
        if (kx < 0 || ky < 0 || kx >= W || ky >= H) continue;
        cv::circle(canvas, cv::Point(kx, ky), big ? 5 : 3,
          big ? cv::Scalar(0, 200, 255) : cv::Scalar(0, 255, 0), -1);
      }
      */
    }

    if (enable_bag_) {
      for (size_t i = 0; i < std::min<size_t>(3, preview_bags_.size()); ++i) {
        const auto & b = preview_bags_[i];
        cv::Rect r = flipRect(b.bbox);
        r &= cv::Rect(0, 0, W, H);
        if (r.width <= 0 || r.height <= 0) continue;
        cv::rectangle(canvas, r, cv::Scalar(255, 0, 255), 2);
        const int cx = r.x + r.width / 2;
        const int cy = r.y + r.height / 2;
        cv::circle(canvas, cv::Point(cx, cy), 4, cv::Scalar(255, 0, 255), -1);
        cv::line(canvas, cv::Point(cx, cy), cv::Point(W / 2, H), cv::Scalar(255, 0, 255), 2);
        std::string text = "Bag";
        if (b.has_pos) text += cv::format(" (%.2fm, %.2fm)", b.x_m, b.y_m);
        banner(text, r.x, r.y, cv::Scalar(255, 0, 255),
          cv::Scalar(125, 40, 81), cv::Scalar(253, 168, 208), 0.52, 1);
      }
    }

    for (size_t i = 0; i < std::min<size_t>(4, preview_drinks_.size()); ++i) {
      const auto & d = preview_drinks_[i];
      cv::Rect r = flipRect(d.bbox);
      r &= cv::Rect(0, 0, W, H);
      if (r.width <= 0 || r.height <= 0) continue;
      cv::rectangle(canvas, r, cv::Scalar(255, 255, 0), 2);
      const int cx = r.x + r.width / 2;
      const int cy = r.y + r.height / 2;
      cv::circle(canvas, cv::Point(cx, cy), 4, cv::Scalar(255, 255, 0), -1);
      std::string text = d.label;
      if (d.has_pos) text += cv::format(" (%.2fm, %.2fm)", d.x_m, d.y_m);
      banner(text, r.x, r.y, cv::Scalar(255, 255, 0),
        cv::Scalar(80, 80, 0), cv::Scalar(255, 255, 255), 0.52, 1);
    }

    for (size_t i = 0; i < std::min<size_t>(4, preview_chairs_.size()); ++i) {
      const auto & c = preview_chairs_[i];
      cv::Rect r = flipRect(c.bbox);
      r &= cv::Rect(0, 0, W, H);
      if (r.width <= 0 || r.height <= 0) continue;
      const cv::Scalar color = c.occupied ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
      cv::rectangle(canvas, r, color, 2);
      const int cx = r.x + r.width / 2;
      const int cy = r.y + r.height / 2;
      cv::circle(canvas, cv::Point(cx, cy), 4, color, -1);
      std::string text = (c.occupied ? "Chair (Occupied): " : "Chair (Empty): ") + c.side;
      if (c.has_pos) text += cv::format(" (%.2fm, %.2fm)", c.x_m, c.y_m);
      banner(text, r.x, r.y, color,
        c.occupied ? cv::Scalar(0, 0, 180) : cv::Scalar(0, 140, 0),
        cv::Scalar(255, 255, 255), 0.52, 2);
    }

    int hud_y = 52;
    auto hud = [&](const std::string & title, double x, double y, double dist,
        const cv::Scalar & color) {
        cv::putText(canvas, title, cv::Point(10, hud_y),
          cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv::LINE_AA);
        cv::putText(canvas, "x : " + cv::format("%.2f m", x), cv::Point(10, hud_y + 20),
          cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
        cv::putText(canvas, "y : " + cv::format("%.2f m", y), cv::Point(10, hud_y + 40),
          cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
        cv::putText(canvas, "distance : " + cv::format("%.2f m", dist),
          cv::Point(10, hud_y + 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
        hud_y += 86;
      };

    const PrevPerson p0 = (!preview_persons_.empty() && preview_persons_[0].has_pos)
      ? preview_persons_[0] : PrevPerson{};
    std::string person_title;
    if (is_receptionist_ && !is_help_me_carry_) {
      if (!p0.name.empty()) {
        person_title = p0.name + " :";
      } else {
        person_title = "Guest :";
      }
    } else {
      person_title = "Operator [ID " + std::to_string(p0.id) + "] :";
    }
    hud(person_title, p0.x_m, p0.y_m, p0.range_m, cv::Scalar(0, 255, 0));

    if (enable_bag_) {
      const PrevBag b0 = (!preview_bags_.empty() && preview_bags_[0].has_pos)
        ? preview_bags_[0] : PrevBag{};
      hud("Bag :", b0.x_m, b0.y_m, b0.range_m, cv::Scalar(255, 0, 255));
    }
    if (!preview_drinks_.empty() && preview_drinks_[0].has_pos) {
      hud(preview_drinks_[0].label + " :", preview_drinks_[0].x_m,
        preview_drinks_[0].y_m, preview_drinks_[0].range_m, cv::Scalar(255, 255, 0));
    }
    for (const auto & c : preview_chairs_) {
      if (!c.has_pos) continue;
      hud((c.occupied ? "Occupied Chair [" : "Empty Chair [") + c.side + "] :",
        c.x_m, c.y_m, c.range_m,
        c.occupied ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0));
    }

    static bool window_ready = false;
    if (!window_ready) {
      cv::namedWindow("POV", cv::WINDOW_AUTOSIZE);
      window_ready = true;
    }
    cv::imshow("POV", canvas);
    const int key = cv::waitKey(1);
    if (key == 'q' || key == 27) rclcpp::shutdown();
  } catch (const std::exception &) {
    try {cv::destroyAllWindows();} catch (...) {}
    debug_window_ = false;
  }
}

void ZedNode::loop()
{
  try {
    if (zed_.grab() != sl::ERROR_CODE::SUCCESS) return;
    publishYaw();

    if (zed_.retrieveImage(zed_image_, sl::VIEW::LEFT, sl::MEM::CPU) != sl::ERROR_CODE::SUCCESS) {
      return;
    }
    cv::Mat bgra(
      zed_image_.getHeight(), zed_image_.getWidth(), CV_8UC4,
      zed_image_.getPtr<sl::uchar1>(sl::MEM::CPU),
      zed_image_.getStepBytes(sl::MEM::CPU));
    cv::Mat bgr;
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
    if (bgr.empty()) return;

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

    processBags(bgr, zed_point_cloud_);
    processDrinks(bgr, zed_point_cloud_);
      processPersons(bgr, zed_point_cloud_);
    processChairs(bgr, zed_point_cloud_);
    showPreview(bgr);
  } catch (const std::exception & e) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
      "Vision loop exception: %s", e.what());
  }
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<ZedNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    std::cout << "\033[91m[ZED Fatal Exception]\033[0m " << e.what() << std::endl;
    rclcpp::shutdown();
    return 1;
  } catch (...) {
    std::cout << "\033[91m[ZED Fatal Exception]\033[0m Unknown error occurred!" << std::endl;
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
