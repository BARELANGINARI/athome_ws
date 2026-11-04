#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "msgs/msg/person_detected.hpp"
#include "opencv2/opencv.hpp"
#include "person_det/person_detection.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

namespace fs = std::filesystem;
using namespace std::chrono_literals;

class PersonDetNode : public rclcpp::Node {
public:
  PersonDetNode()
  : Node("person_det_node")
  {
    declare_parameter("camera_index", 0);
    declare_parameter("conf_threshold", 0.6);
    declare_parameter("debug_window", true);
    declare_parameter("models_dir", "");

    cam_idx_ = get_parameter("camera_index").as_int();
    conf_thresh_ = static_cast<float>(get_parameter("conf_threshold").as_double());
    debug_window_ = get_parameter("debug_window").as_bool();

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

    sound_sub_ = create_subscription<std_msgs::msg::String>(
      "/hri/sound_event", 10,
      std::bind(&PersonDetNode::soundCallback, this, std::placeholders::_1));

    name_sub_ = create_subscription<std_msgs::msg::String>(
      "/hri/guest_name", 10,
      std::bind(&PersonDetNode::nameCallback, this, std::placeholders::_1));

    det_pub_ = create_publisher<msgs::msg::PersonDetected>(
      "/vision/detected_person", 10);

    last_reconnect_attempt_ = std::chrono::steady_clock::now();
    openCamera(cam_idx_);

    timer_ = create_wall_timer(
      100ms, std::bind(&PersonDetNode::timerCallback, this));

    RCLCPP_INFO(get_logger(), "PersonDetNode ready. Waiting for doorbell event...");
  }

  ~PersonDetNode() override
  {
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
    if (!cap_.isOpened()) {
      auto now = std::chrono::steady_clock::now();
      if (now - last_reconnect_attempt_ > 2s) {
        last_reconnect_attempt_ = now;
        openCamera(cam_idx_);
      }
      return;
    }

    cv::Mat frame;
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
      cv::imshow("Person Detection", frame);
      cv::waitKey(1);
    }
  }

  int cam_idx_{0};
  int active_cam_idx_{-1};
  std::chrono::steady_clock::time_point last_reconnect_attempt_;
  float conf_thresh_{0.6f};
  bool debug_window_{true};
  bool detecting_{false};
  std::string guest_name_{""};
  std::string mem_path_{""};
  std::unordered_set<std::string> stored_names_;

  person_det::PersonDetector detector_;
  cv::VideoCapture cap_;

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sound_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr name_sub_;
  rclcpp::Publisher<msgs::msg::PersonDetected>::SharedPtr det_pub_;
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
