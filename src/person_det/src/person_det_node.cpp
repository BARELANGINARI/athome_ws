#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "msgs/msg/person_detected.hpp"
#include "opencv2/opencv.hpp"
#include "person_det/person_detection.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/string.hpp"

#if HAVE_ZED
#include <sl/Camera.hpp>
#endif

namespace fs = std::filesystem;
using namespace std::chrono_literals;

class PersonDetNode : public rclcpp::Node {
public:
  PersonDetNode()
  : Node("person_det_node")
  {
    declare_parameter("camera_type", "auto");
    declare_parameter("camera_fps", 30.0);
    declare_parameter("publish_yaw", true);
    declare_parameter("yaw_topic", "/zed/yaw_deg");
    declare_parameter("imu_topic", "/zed/imu/data");
    declare_parameter("camera_index", 0);
    declare_parameter("conf_threshold", 0.6);
    declare_parameter("debug_window", true);
    declare_parameter("models_dir", "");
    declare_parameter("reset_memory", true);

    cam_type_ = get_parameter("camera_type").as_string();
    camera_fps_ = get_parameter("camera_fps").as_double();
    publish_yaw_ = get_parameter("publish_yaw").as_bool();
    yaw_topic_ = get_parameter("yaw_topic").as_string();
    imu_topic_ = get_parameter("imu_topic").as_string();
    cam_idx_ = get_parameter("camera_index").as_int();
    conf_thresh_ = static_cast<float>(get_parameter("conf_threshold").as_double());
    debug_window_ = get_parameter("debug_window").as_bool();
    bool reset_memory = get_parameter("reset_memory").as_bool();

    std::string models_dir = get_parameter("models_dir").as_string();
    if (models_dir.empty()) {
      try {
        models_dir = (fs::path(ament_index_cpp::get_package_share_directory("person_det")) / "models").string();
      } catch (...) {
        models_dir = "models";
      }
    }

    RCLCPP_INFO(get_logger(), "Loading models from: %s", models_dir.c_str());
    if (!detector_.init(models_dir)) {
      RCLCPP_ERROR(get_logger(), "Failed to initialize PersonDetector models!");
    } else {
      RCLCPP_INFO(get_logger(), "PersonDetector ONNX models loaded successfully.");
    }

    mem_path_ = (fs::path(models_dir) / "face_memory.json").string();
    if (reset_memory) {
      std::ofstream f(mem_path_, std::ios::trunc);
      if (f.is_open()) {
        f << "{\n}\n";
      }
      detector_.loadFaceMemory(mem_path_);
      stored_names_.clear();
      RCLCPP_INFO(get_logger(), "Face memory reset to empty for fresh session.");
    }

    sound_sub_ = create_subscription<std_msgs::msg::String>(
      "/hri/sound_event", 10,
      std::bind(&PersonDetNode::soundCallback, this, std::placeholders::_1));

    name_sub_ = create_subscription<std_msgs::msg::String>(
      "/hri/guest_name", 10,
      std::bind(&PersonDetNode::nameCallback, this, std::placeholders::_1));

    det_pub_ = create_publisher<msgs::msg::PersonDetected>(
      "/vision/detected_person", 10);

    yaw_pub_ = create_publisher<std_msgs::msg::Float64>(
      yaw_topic_, 10);

    imu_pub_ = create_publisher<sensor_msgs::msg::Imu>(
      imu_topic_, 10);

    last_reconnect_attempt_ = std::chrono::steady_clock::now();
    openCamera(cam_idx_);

    timer_ = create_wall_timer(
      33ms, std::bind(&PersonDetNode::timerCallback, this));

    RCLCPP_INFO(get_logger(), "PersonDetNode ready. Waiting for doorbell event...");
  }

  ~PersonDetNode() override
  {
#if HAVE_ZED
    if (using_zed_ && zed_.isOpened()) {
      zed_.close();
    }
#endif
    if (cap_.isOpened()) {
      cap_.release();
    }
    if (debug_window_) {
      cv::destroyAllWindows();
    }
  }

private:
  bool openCamera(int target_idx)
  {
#if HAVE_ZED
    if (cam_type_ == "auto" || cam_type_ == "zed") {
      if (zed_.isOpened()) {
        zed_.close();
      }
      sl::InitParameters init_params;
      init_params.camera_resolution = sl::RESOLUTION::VGA; // 672x376 native matching camera.launch.py
      init_params.camera_fps = static_cast<int>(camera_fps_);
      init_params.depth_mode = sl::DEPTH_MODE::NONE; // Fast RGB & IMU mode without heavy depth
      init_params.coordinate_units = sl::UNIT::METER;
      init_params.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;

      RCLCPP_INFO(get_logger(), "Attempting to open ZED camera via ZED SDK...");
      auto state = zed_.open(init_params);
      if (state == sl::ERROR_CODE::SUCCESS) {
        using_zed_ = true;
        active_cam_idx_ = -1;
        RCLCPP_INFO(get_logger(), "Kamera ZED berhasil dibuka via ZED SDK! IMU yaw publishing to '%s' AKTIF.", yaw_topic_.c_str());
        return true;
      }
      RCLCPP_WARN(get_logger(), "ZED SDK open returned %s. Falling back to OpenCV capture...", sl::toString(state).c_str());
      using_zed_ = false;
    }
#endif

    if (cap_.isOpened()) {
      cap_.release();
    }

    // Try target index first
    RCLCPP_INFO(get_logger(), "Attempting to open camera index %d...", target_idx);
    cap_.open(target_idx, cv::CAP_V4L2);
    if (cap_.isOpened()) {
      cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
      cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
      active_cam_idx_ = target_idx;
      RCLCPP_INFO(get_logger(), "Camera index %d opened successfully.", active_cam_idx_);
      return true;
    }

    RCLCPP_WARN(get_logger(), "Camera index %d could not be opened. Scanning available /dev/video* devices...", target_idx);

    // Fallback scan 0..9
    for (int idx = 0; idx <= 9; ++idx) {
      if (idx == target_idx) continue;
      cap_.open(idx, cv::CAP_V4L2);
      if (cap_.isOpened()) {
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        active_cam_idx_ = idx;
        RCLCPP_INFO(get_logger(), "Auto-fallback: Successfully opened camera at index %d!", active_cam_idx_);
        return true;
      }
    }

    active_cam_idx_ = -1;
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
      "No camera device available right now. Node will automatically retry every 2s until camera is plugged in.");
    return false;
  }

  void soundCallback(const std_msgs::msg::String::SharedPtr msg)
  {
    if (msg->data == "doorbell" && !detecting_) {
      detecting_ = true;
      RCLCPP_INFO(get_logger(), "Doorbell heard -> detection START");
    }
  }

  void nameCallback(const std_msgs::msg::String::SharedPtr msg)
  {
    guest_name_ = msg->data;
    stored_names_.erase(msg->data);
    RCLCPP_INFO(get_logger(), "Guest name received: '%s'", guest_name_.c_str());
  }

  void timerCallback()
  {
    cv::Mat frame;

#if HAVE_ZED
    if (using_zed_) {
      if (!zed_.isOpened()) {
        auto now = std::chrono::steady_clock::now();
        if (now - last_reconnect_attempt_ > 2s) {
          last_reconnect_attempt_ = now;
          openCamera(cam_idx_);
        }
        return;
      }

      // 1. Publish IMU Yaw and full IMU data (independent of video grab result)
      if (publish_yaw_) {
        sl::SensorsData sensors_data;
        if (zed_.getSensorsData(sensors_data, sl::TIME_REFERENCE::CURRENT) == sl::ERROR_CODE::SUCCESS &&
            sensors_data.imu.is_available)
        {
          const auto q = sensors_data.imu.pose.getOrientation();
          float yaw_rad = std::atan2(2.0f * (q.w * q.z + q.x * q.y), 1.0f - 2.0f * (q.y * q.y + q.z * q.z));
          float yaw_deg = yaw_rad * 180.0f / 3.141592653589793f;
          while (yaw_deg < 0.0f) {
            yaw_deg += 360.0f;
          }
          while (yaw_deg >= 360.0f) {
            yaw_deg -= 360.0f;
          }

          std_msgs::msg::Float64 yaw_msg;
          yaw_msg.data = static_cast<double>(yaw_deg);
          yaw_pub_->publish(yaw_msg);

          sensor_msgs::msg::Imu imu_msg;
          imu_msg.header.stamp = now();
          imu_msg.header.frame_id = "zed_camera_link";
          imu_msg.orientation.x = q.x;
          imu_msg.orientation.y = q.y;
          imu_msg.orientation.z = q.z;
          imu_msg.orientation.w = q.w;
          const auto la = sensors_data.imu.linear_acceleration;
          imu_msg.linear_acceleration.x = la.x;
          imu_msg.linear_acceleration.y = la.y;
          imu_msg.linear_acceleration.z = la.z;
          const auto av = sensors_data.imu.angular_velocity;
          imu_msg.angular_velocity.x = av.x * M_PI / 180.0;
          imu_msg.angular_velocity.y = av.y * M_PI / 180.0;
          imu_msg.angular_velocity.z = av.z * M_PI / 180.0;
          imu_pub_->publish(imu_msg);

          RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
            "Publishing IMU yaw: %.2f deg to '%s'", yaw_deg, yaw_topic_.c_str());
        }
      }

      // 2. Retrieve left eye image
      if (zed_.grab() == sl::ERROR_CODE::SUCCESS) {
        sl::Mat zed_image;
        if (zed_.retrieveImage(zed_image, sl::VIEW::LEFT, sl::MEM::CPU) == sl::ERROR_CODE::SUCCESS) {
          cv::Mat bgra(
            zed_image.getHeight(), zed_image.getWidth(), CV_8UC4,
            zed_image.getPtr<sl::uchar1>(sl::MEM::CPU),
            zed_image.getStepBytes(sl::MEM::CPU));
          cv::cvtColor(bgra, frame, cv::COLOR_BGRA2BGR);
        }
      }
    } else
#endif
    {
      if (!cap_.isOpened()) {
        auto now = std::chrono::steady_clock::now();
        if (now - last_reconnect_attempt_ > 2s) {
          last_reconnect_attempt_ = now;
          openCamera(cam_idx_);
        }
        return;
      }

      // Periodically check if ZED becomes available to upgrade to ZED SDK + IMU
      if (cam_type_ == "auto" || cam_type_ == "zed") {
        auto now = std::chrono::steady_clock::now();
        if (now - last_reconnect_attempt_ > 2s) {
          last_reconnect_attempt_ = now;
          openCamera(cam_idx_);
        }
      }

      if (!cap_.read(frame) || frame.empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 4000,
          "Failed to read frame from camera %d. Re-checking camera...", active_cam_idx_);
        cap_.release();
        return;
      }

      // If stereo side-by-side frame (e.g. ZED 2i where width >= 2 * height), crop left perspective
      if (frame.cols >= frame.rows * 2) {
        frame = frame(cv::Rect(0, 0, frame.cols / 2, frame.rows)).clone();
      }
    }

    if (frame.empty()) {
      return;
    }

    if (detecting_) {
      auto candidates = detector_.detect(frame, conf_thresh_);

      const person_det::PersonCandidate * nearest = nullptr;
      int max_area = -1;

      for (const auto & cand : candidates) {
        if (cand.area > max_area) {
          max_area = cand.area;
          nearest = &cand;
        }
      }

      // If guest name was received and nearest person face embedding is available, save to memory
      if (nearest != nullptr && !guest_name_.empty() && !nearest->embedding.empty()) {
        if (stored_names_.find(guest_name_) == stored_names_.end()) {
          detector_.addFaceEmbedding(guest_name_, nearest->embedding);
          detector_.saveFaceMemory(mem_path_);
          stored_names_.insert(guest_name_);
          RCLCPP_INFO(get_logger(), "Saved face embedding for '%s'", guest_name_.c_str());
        }
      }

      // Draw all candidates
      for (const auto & cand : candidates) {
        bool is_n = (nearest != nullptr && &cand == nearest);
        detector_.drawCandidate(frame, cand, is_n);
      }

      // Publish nearest person
      if (nearest != nullptr) {
        msgs::msg::PersonDetected det;
        det.name = nearest->matched_name;
        det.gender = nearest->gender;
        det.shirt_color = nearest->clothing;
        det_pub_->publish(det);
      }
    }

    if (debug_window_) {
      cv::Mat preview_mat;
      if (frame.cols != 672 || frame.rows != 376) {
        cv::resize(frame, preview_mat, cv::Size(672, 376));
      } else {
        preview_mat = frame;
      }
      cv::imshow("Person Detection", preview_mat);
      cv::waitKey(1);
    }
  }

  std::string cam_type_{"auto"};
  double camera_fps_{30.0};
  bool publish_yaw_{true};
  std::string yaw_topic_{"/zed/yaw_deg"};
  std::string imu_topic_{"/zed/imu/data"};
  int cam_idx_{0};
  int active_cam_idx_{-1};
  bool using_zed_{false};
  std::chrono::steady_clock::time_point last_reconnect_attempt_;
  float conf_thresh_{0.6f};
  bool debug_window_{true};
  bool detecting_{false};
  std::string guest_name_{""};
  std::string mem_path_{""};
  std::unordered_set<std::string> stored_names_;

  person_det::PersonDetector detector_;
  cv::VideoCapture cap_;
#if HAVE_ZED
  sl::Camera zed_;
#endif

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sound_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr name_sub_;
  rclcpp::Publisher<msgs::msg::PersonDetected>::SharedPtr det_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr yaw_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PersonDetNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
