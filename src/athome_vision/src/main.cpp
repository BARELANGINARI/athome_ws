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
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
  const cv::Rect & box, const sl::Mat & point_cloud, sl::float4 & point)
{
  const int r = std::min(10, std::min(box.width, box.height) / 2);
  const int cx = box.x + box.width / 2;
  const int cy = box.y + box.height / 2;
  const int w = static_cast<int>(point_cloud.getWidth());
  const int h = static_cast<int>(point_cloud.getHeight());

  std::vector<float> x_vect, y_vect, z_vect;
  for (int dy = -r; dy <= r; ++dy) {
    for (int dx = -r; dx <= r; ++dx) {
      const int i = cx + dx, j = cy + dy;
      if (i >= 0 && i < w && j >= 0 && j < h) {
        sl::float4 out;
        point_cloud.getValue(i, j, &out);
        if (std::isfinite(out.z)) {
          x_vect.push_back(out.x);
          y_vect.push_back(out.y);
          z_vect.push_back(out.z);
        }
      }
    }
  }

  if (x_vect.empty()) {
    point.z = NAN;
    return false;
  }
  point.x = getMedian(x_vect);
  point.y = getMedian(y_vect);
  point.z = getMedian(z_vect);
  return true;
}

bool get3dCoordinatesWithOffset(
  const cv::Rect & rect, const sl::Mat & point_cloud,
  double & x, double & y, double & z)
{
  const int cx = rect.x + rect.width / 2;
  const int cy = rect.y + rect.height / 2;
  sl::float4 point{0, 0, NAN, 0};

  if (cx >= 0 && cx < static_cast<int>(point_cloud.getWidth()) &&
      cy >= 0 && cy < static_cast<int>(point_cloud.getHeight())) {
    point_cloud.getValue(cx, cy, &point);
  }

  if (!std::isfinite(point.z) && !get3dCoordinates(rect, point_cloud, point)) {
    return false;
  }

  if (!std::isfinite(point.z)) {
    return false;
  }

  x = (point.x + 13.0) / 100.0;
  y = (-point.y + 3.0) / 100.0; 
  z = (point.z + 44.0) / 100.0;
  return true;
}

bool positionToRobot(
  const sl::float3 & position, double & x, double & y, double & z)
{
  if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
      !std::isfinite(position.z))
  {
    return false;
  }
  if (std::fabs(position.x) < 1e-6f && std::fabs(position.y) < 1e-6f &&
      std::fabs(position.z) < 1e-6f)
  {
    return false;
  }

  x = (position.x + 13.0) / 100.0;
  y = (-position.y + 3.0) / 100.0;
  z = (position.z + 44.0) / 100.0;
  return true;
}

const char * trackingStateName(sl::OBJECT_TRACKING_STATE state)
{
  switch (state) {
    case sl::OBJECT_TRACKING_STATE::OK: return "OK";
    case sl::OBJECT_TRACKING_STATE::SEARCHING: return "SEARCHING";
    case sl::OBJECT_TRACKING_STATE::TERMINATE: return "TERMINATE";
    case sl::OBJECT_TRACKING_STATE::OFF: return "OFF";
    default: return "UNKNOWN";
  }
}

}  

struct YoloDetection
{
  cv::Rect rect;
  std::string label;  
  float score{0.0f};
};

class YoloDetector
{
public:
  YoloDetector();
  ~YoloDetector();

  YoloDetector(const YoloDetector &) = delete;
  YoloDetector & operator=(const YoloDetector &) = delete;

  // model_path: yolov8n_sim.onnx (COCO pretrained).
  bool load(
    const std::string & model_path, const std::string & classes_path,
    float conf_threshold = 0.45f, float nms_threshold = 0.45f,
    int input_size = 640, std::string * err = nullptr);

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
  std::string * err)
{
  Impl & im = *impl_;
  im.conf_threshold = conf_threshold;
  im.nms_threshold = nms_threshold;
  im.input_size = input_size;
  im.ready = false;

  im.names.clear();
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

  auto is_bag = [&](int id) {
    if (num_classes == 1) return true;
    if (num_classes == 80 && (id == 24 || id == 26 || id == 28)) return true;
    const std::string n = get_label(id);
    if (!n.empty()) {
      return n == "backpack" || n == "handbag" || n == "suitcase" || want_bag.count(n) > 0;
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

    if (score < im.conf_threshold || !is_bag(cls_id)) {
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
    d.label = "bag";
    out.push_back(d);
  }
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

  sl::Camera zed_;
  sl::Mat zed_image_;
  sl::Mat zed_point_cloud_;
  sl::Bodies bodies_;
  sl::BodyTrackingRuntimeParameters body_runtime_parameters_;

  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr yaw_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr bag_dets_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr persons_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr gesture_publisher_; 
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;

  bool publish_yaw_{true};
  bool publish_image_{true};
  int image_stride_{2};
  int image_frame_{0};
  bool enable_body_tracking_{true};
  bool debug_window_{false};
  bool mirror_{true};
  bool preview_inited_{false};
  bool preview_dead_{false};

  struct PrevPerson {
    cv::Rect bbox;
    bool has_pos{false};
    double x_m{0.0};
    double y_m{0.0};
    double range_m{0.0};
    std::vector<cv::Point2f> kpt;
  };
  struct PrevBag {
    cv::Rect bbox;
    bool has_pos{false};
    double x_m{0.0};
    double y_m{0.0};
    double range_m{0.0};
  };
  std::vector<PrevPerson> preview_persons_;
  std::vector<PrevBag> preview_bags_;
  std::unordered_map<int, cv::Rect> last_body_bbox_;

  YoloDetector yolo_;
  bool enable_yolo_{true};
  std::vector<std::string> bag_labels_;
  double yolo_min_area_{1500.0};
  int yolo_stride_{2};
  int yolo_frame_{0};
  std::vector<YoloDetection> last_bag_dets_;
};

ZedNode::ZedNode()
: Node("zed", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true))
{
  declare_parameter("camera_fps", 30.0);
  declare_parameter("publish_yaw", true);
  declare_parameter("publish_image", true);
  declare_parameter("image_stride", 2);
  declare_parameter("enable_body_tracking", true);
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

  const double camera_fps = get_parameter("camera_fps").as_double();

  publish_yaw_ = get_parameter("publish_yaw").as_bool();
  publish_image_ = get_parameter("publish_image").as_bool();
  image_stride_ = static_cast<int>(get_parameter("image_stride").as_int());
  if (image_stride_ < 1) {image_stride_ = 1;}
  enable_body_tracking_ = get_parameter("enable_body_tracking").as_bool();

  if (camera_fps <= 0.0) {
    throw std::runtime_error("camera_fps must be greater than zero");
  }

  yaw_publisher_ = create_publisher<std_msgs::msg::Float64>("/zed/yaw_deg", 10);
  bag_dets_publisher_ = create_publisher<std_msgs::msg::String>("/zed/bag_detections", 10);
  persons_publisher_ = create_publisher<std_msgs::msg::String>("/zed/person_label", 10);
  gesture_publisher_ = create_publisher<std_msgs::msg::String>("/zed/gesture", 10);
  image_publisher_ = create_publisher<sensor_msgs::msg::Image>(
    "/zed/image", rclcpp::SensorDataQoS());

  sl::InitParameters init_parameters;
  init_parameters.camera_resolution = sl::RESOLUTION::VGA;
  init_parameters.camera_fps = static_cast<int>(camera_fps);
  init_parameters.coordinate_units = sl::UNIT::CENTIMETER;
  init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;

  const auto state = zed_.open(init_parameters);
  if (state != sl::ERROR_CODE::SUCCESS) {
    IC(sl::toString(state).c_str());
    throw std::runtime_error("ZED open failed");
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
  body_parameters.prediction_timeout_s = 1.0f;

  if (enable_body_tracking_) {
    const auto body_state = zed_.enableBodyTracking(body_parameters);
    if (body_state != sl::ERROR_CODE::SUCCESS) {
      zed_.close();
      throw std::runtime_error("Enable body tracking failed");
    }
  }

  body_runtime_parameters_.detection_confidence_threshold = 40;

  enable_yolo_   = get_parameter("enable_yolo").as_bool();
  debug_window_  = get_parameter("debug_window").as_bool();
  mirror_        = get_parameter("mirror").as_bool();
  bag_labels_    = get_parameter("bag_labels").as_string_array();
  yolo_min_area_ = get_parameter("yolo_min_area").as_double();
  yolo_stride_   = std::max(1, static_cast<int>(get_parameter("yolo_stride").as_int()));

  if (enable_yolo_) {
    auto resolve = [](const std::string & p) {
      if (std::filesystem::path(p).is_absolute()) return p;
      try { return ament_index_cpp::get_package_share_directory("athome_vision") + "/" + p; }
      catch (...) { return std::string("share/athome_vision/") + p; }
    };

    std::string model_path   = resolve(get_parameter("yolo_model").as_string());
    std::string classes_path = resolve(get_parameter("yolo_classes").as_string());
    float conf = static_cast<float>(get_parameter("yolo_conf").as_double());
    float nms  = static_cast<float>(get_parameter("yolo_nms").as_double());
    int size   = static_cast<int>(get_parameter("yolo_size").as_int());

    std::string err;
    if (!yolo_.load(model_path, classes_path, conf, nms, size, &err)) {
      enable_yolo_ = false;
    }
  }

  const auto period = std::chrono::duration<double>(1.0 / camera_fps);
  timer_ = create_wall_timer(period, std::bind(&ZedNode::loop, this));

  IC("Start capturing");
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

  std_msgs::msg::Float64 message;
  message.data = static_cast<double>(zed_vision::wrap180(yaw * 180.0f / zed_vision::PI_F));
  yaw_publisher_->publish(message);
}


void ZedNode::processBodiesAndYolo(const cv::Mat & bgr, const sl::Mat & point_cloud)
{
  using namespace zed_vision;

  std::vector<YoloDetection> bag_dets = last_bag_dets_;
  if (enable_yolo_ && yolo_.ready()) {
    if (yolo_frame_++ % std::max(1, yolo_stride_) == 0) {
      yolo_.detect(bgr, bag_labels_, bag_dets);
      last_bag_dets_ = bag_dets;
    }
  }

  preview_bags_.clear();
  std::string bag_str;
  for (const auto & d : bag_dets) {
    if (static_cast<double>(d.rect.area()) < yolo_min_area_) continue;
    double x = 0.0, y = 0.0, z = 0.0;
    const bool ok = get3dCoordinatesWithOffset(d.rect, point_cloud, x, y, z);
    const double dist = ok ? std::hypot(x, y) : 0.0;
    preview_bags_.push_back({d.rect, ok, x, y, dist});
    if (ok) {
      IC(d.label, x, y, dist);
    }

    if (!bag_str.empty()) bag_str += ";";
    bag_str += cv::format("bag:%d,%d,%d,%d|", d.rect.x, d.rect.y, d.rect.width, d.rect.height);
    bag_str += ok ? cv::format("%.2f,%.2f,%.2f|%.2f", x, y, z, d.score) : cv::format("nan,nan,nan|%.2f", d.score);
  }
  std_msgs::msg::String bag_msg;
  bag_msg.data = bag_str;
  bag_dets_publisher_->publish(bag_msg);

  preview_persons_.clear();
  std::string person_str = "NO_PERSON";
  if (bodies_.body_list.empty()) {
    std_msgs::msg::String m;
    m.data = "NO_PERSON";
    persons_publisher_->publish(m);
    std_msgs::msg::String g;
    g.data = "NONE";
    gesture_publisher_->publish(g);
  } else {
    struct PersonEntry {
      const sl::BodyData * body{nullptr};
      int id{-1};
      cv::Rect bbox;
      std::string label{"person"};
      double x{0.0}, y{0.0}, z{0.0};
      bool has_depth{false};
      double range_m{std::numeric_limits<double>::infinity()};
    };
    std::vector<PersonEntry> entries;
    std::unordered_set<int> seen_ids;
    for (const auto & body : bodies_.body_list) {
      if (body.tracking_state == sl::OBJECT_TRACKING_STATE::TERMINATE ||
          body.tracking_state == sl::OBJECT_TRACKING_STATE::OFF)
      {
        continue;
      }

      PersonEntry e;
      e.body = &body;
      e.id = body.id;
      if (e.id >= 0) {seen_ids.insert(e.id);}

      cv::Rect bbox;
      const bool bbox_ok = body.keypoint_2d.size() >= 34 &&
        computeBbox(body.keypoint_2d, bgr.cols, bgr.rows, bbox);
      if (bbox_ok) {
        e.bbox = bbox;
        if (e.id >= 0) {last_body_bbox_[e.id] = bbox;}
      } else if (e.id >= 0 && last_body_bbox_.count(e.id) > 0) {
        e.bbox = last_body_bbox_[e.id];
      }

      if (bbox_ok) {
        e.has_depth = get3dCoordinatesWithOffset(e.bbox, point_cloud, e.x, e.y, e.z);
      }

      // Saat target terhalang, ZED memberi posisi prediksi (SEARCHING).
      // Pakai body.position sebagai fallback agar ID tetap terlihat oleh follower.
      const bool tracked_ok = body.tracking_state == sl::OBJECT_TRACKING_STATE::OK;
      if (!e.has_depth || !tracked_ok) {
        double px = 0.0, py = 0.0, pz = 0.0;
        if (zed_vision::positionToRobot(body.position, px, py, pz)) {
          e.x = px;
          e.y = py;
          e.z = pz;
          e.has_depth = true;
        }
      }

      e.range_m = (e.has_depth && std::isfinite(e.x) && std::isfinite(e.y))
                  ? std::hypot(e.x, e.y) : (1e6 - static_cast<double>(e.bbox.area()));
      entries.push_back(e);
    }

    for (auto it = last_body_bbox_.begin(); it != last_body_bbox_.end();) {
      if (seen_ids.count(it->first) == 0) {
        it = last_body_bbox_.erase(it);
      } else {
        ++it;
      }
    }

    std::sort(entries.begin(), entries.end(), [](const PersonEntry & a, const PersonEntry & b) {
      return a.range_m < b.range_m;
    });

    if (!entries.empty()) {
      entries.front().label = "person";
      if (entries.front().has_depth) {
        const auto & person = entries.front();
        IC(person.label, person.x, person.y, person.range_m);
      }
    }

    std::ostringstream ss;
    bool first_p = true;
    std::string gesture_result = "NONE";
    for (const auto & e : entries) {
      PrevPerson pp{e.bbox, e.has_depth, e.x, e.y, e.range_m, {}};
      const auto & points = e.body->keypoint_2d;
      for (const auto & pt : points) {
        pp.kpt.emplace_back(std::isfinite(pt.x) ? pt.x : NAN, std::isfinite(pt.y) ? pt.y : NAN);
      }
      preview_persons_.push_back(pp);

      if (first_p && e.body->tracking_state == sl::OBJECT_TRACKING_STATE::OK) {
        gesture_result = detectGesture(points);
        if (gesture_result != "NONE") {
          IC(gesture_result);
        }
      }

      if (!first_p) ss << ";";
      const std::string label = "person";
      first_p = false;
      ss << label << ":" << e.bbox.x << "," << e.bbox.y << "," << e.bbox.width << "," << e.bbox.height << "|";
      if (e.has_depth) ss << std::fixed << std::setprecision(2) << e.x << "," << e.y << "," << e.z;
      else ss << "nan,nan,nan";
      ss << "|id=" << e.id
         << "|state=" << zed_vision::trackingStateName(e.body->tracking_state)
         << "|kpt=";

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

    person_str = first_p ? "NO_PERSON" : ss.str();
    std_msgs::msg::String m;
    m.data = person_str;
    persons_publisher_->publish(m);
  }


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

    if (!preview_persons_.empty() && preview_persons_[0].range_m <= 3.0) {
      const auto & p = preview_persons_[0];
      cv::Rect r = flip_rect(p.bbox);
      r &= cv::Rect(0, 0, W, H);
      if (r.width > 0 && r.height > 0) {
        cv::rectangle(canvas, r, cv::Scalar(0, 255, 0), 2);
        cv::putText(
          canvas, "PERSON",
          cv::Point(r.x, std::max(20, r.y - 6)),
          cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        const int center_x = r.x + r.width / 2;
        const int center_y = r.y + r.height / 2;
        cv::circle(canvas, cv::Point(center_x, center_y), 5, cv::Scalar(0, 255, 255), -1);
        cv::drawMarker(canvas, cv::Point(center_x, center_y), cv::Scalar(0, 255, 255), cv::MARKER_CROSS, 14, 2);
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
      }
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
    draw_hud("Person :", p0.x_m, p0.y_m, p0.range_m, cv::Scalar(0, 255, 0));

    const PrevBag b0 = (!preview_bags_.empty() && preview_bags_[0].has_pos) ? preview_bags_[0] : PrevBag{};
    draw_hud("Bag :", b0.x_m, b0.y_m, b0.range_m, cv::Scalar(255, 0, 255));

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
  } catch (...) {
  }
  rclcpp::shutdown();
  return 0;
}
