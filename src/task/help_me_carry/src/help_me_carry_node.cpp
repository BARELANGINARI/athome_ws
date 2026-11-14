#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "nav2_msgs/srv/clear_entire_costmap.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2/time.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

#include "icecream.hpp"
#include "person_kalman_filter.hpp"

namespace
{
constexpr const char * kNoPerson = "NO_PERSON";

struct Person
{
  int id{-1};
  bool is_operator{false};
  double x{0.0};
  double y{0.0};
  bool is_back{false};
  double bbox_x{0.0};
  double bbox_y{0.0};
  double bbox_w{0.0};
  double bbox_h{0.0};
};

struct Bag
{
  double x{0.0};
  double y{0.0};
};

std::string trim(const std::string & s)
{
  const char * ws = " \t\r\n";
  const auto first = s.find_first_not_of(ws);
  if (first == std::string::npos) {return "";}
  const auto last = s.find_last_not_of(ws);
  return s.substr(first, last - first + 1);
}

std::vector<std::string> split_char(const std::string & s, char delim)
{
  std::vector<std::string> out;
  std::string item;
  for (const char c : s) {
    if (c == delim) {
      out.push_back(item);
      item.clear();
    } else {
      item.push_back(c);
    }
  }
  out.push_back(item);
  return out;
}

geometry_msgs::msg::Quaternion yaw_to_quat(double yaw)
{
  geometry_msgs::msg::Quaternion q;
  q.x = 0.0;
  q.y = 0.0;
  q.z = std::sin(yaw / 2.0);
  q.w = std::cos(yaw / 2.0);
  return q;
}

inline double normalize_angle(double angle)
{
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

std::vector<Bag> parse_bags(const std::string & data)
{
  std::vector<Bag> bags;
  for (const auto & chunk : split_char(data, ';')) {
    const auto fields = split_char(chunk, '|');
    if (fields.size() < 2) continue;
    const auto xyz = split_char(fields[1], ',');
    if (xyz.size() >= 2) {
      try {
        const double bx = std::stod(xyz[0]);
        const double by = std::stod(xyz[1]);
        if (std::isfinite(bx) && std::isfinite(by)) bags.push_back({bx, by});
      } catch (...) {}
    }
  }
  return bags;
}

std::string find_bt_xml()
{
  try {
    const auto share = ament_index_cpp::get_package_share_directory("help_me_carry");
    const auto candidate = std::filesystem::path(share) / "behavior_trees" / "follow_person.xml";
    if (std::filesystem::is_regular_file(candidate)) {
      return candidate.string();
    }
  } catch (...) {}
  return "";
}

std::vector<Person> parse_persons(const std::string & data)
{
  std::vector<Person> out;
  if (data.empty() || data == kNoPerson) {return out;}

  std::size_t start = 0;
  while (start <= data.size()) {
    const std::size_t end = data.find(';', start);
    const std::string chunk = trim(
      data.substr(start, end == std::string::npos ? std::string::npos : end - start));
    start = (end == std::string::npos) ? data.size() + 1 : end + 1;

    std::string prefix;
    if (chunk.rfind("operator:", 0) == 0) {
      prefix = "operator:";
    } else if (chunk.rfind("person:", 0) == 0) {
      prefix = "person:";
    } else {
      continue;
    }
    const auto fields = split_char(chunk, '|');
    if (fields.size() < 2) {continue;}

    Person p;
    p.is_operator = (prefix == "operator:");
    p.id = -1;
    p.is_back = false;

    // fields[0] berisi "<label>:<bbox_x>,<bbox_y>,<bbox_w>,<bbox_h>"
    if (fields[0].size() > prefix.size()) {
      const auto bbox_str = fields[0].substr(prefix.size());
      const auto bbox_vals = split_char(bbox_str, ',');
      if (bbox_vals.size() >= 4) {
        try {
          p.bbox_x = std::stod(bbox_vals[0]);
          p.bbox_y = std::stod(bbox_vals[1]);
          p.bbox_w = std::stod(bbox_vals[2]);
          p.bbox_h = std::stod(bbox_vals[3]);
        } catch (...) {}
      }
    }
    for (size_t fi = 2; fi < fields.size(); ++fi) {
      const auto & f = fields[fi];
      if (f.rfind("id=", 0) == 0) {
        try {
          p.id = std::stoi(f.substr(3));
        } catch (...) {}
      } else if (f.rfind("back=", 0) == 0) {
        const auto val = f.substr(5);
        p.is_back = (val == "1" || val == "true");
      }
    }
    try {
      const auto xyz = split_char(fields[1], ',');
      if (xyz.size() < 2) {continue;}
      p.x = std::stod(xyz[0]);
      p.y = std::stod(xyz[1]);
      if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
        continue;
      }
    } catch (const std::exception &) {
      continue;
    }
    out.push_back(p);
  }
  return out;
}
}  // namespace

class HelpMeCarryNode : public rclcpp::Node
{
public:
  using NavigateToPose = nav2_msgs::action::NavigateToPose;
  using GoalHandle = rclcpp_action::ClientGoalHandle<NavigateToPose>;
  using Trigger = std_srvs::srv::Trigger;

  explicit HelpMeCarryNode(const std::string & node_name = "help_me_carry_node")
  : Node(node_name)
  {
    // Topics & Frames
    person_topic_ = declare_parameter<std::string>("person_topic", "/vision/person_label");
    person_back_topic_ = declare_parameter<std::string>("person_back_topic", "/vision/person_back");
    gesture_topic_ = declare_parameter<std::string>("gesture_topic", "/vision/gesture");
    bag_topic_ = declare_parameter<std::string>("bag_topic", "/vision/bag_detections");
    yaw_topic_ = declare_parameter<std::string>("yaw_topic", "/vision/yaw_deg");
    cmd_vel_topic_ = declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    nav_cmd_vel_topic_ = declare_parameter<std::string>("nav_cmd_vel_topic", "/nav_cmd_vel");
    path_topic_ = declare_parameter<std::string>("path_topic", "/following_path");
    plan_topic_ = declare_parameter<std::string>("plan_topic", "/plan");
    goal_update_topic_ = declare_parameter<std::string>("goal_update_topic", "/goal_update");
    action_name_ = declare_parameter<std::string>("action_name", "navigate_to_pose");
    frame_map_ = declare_parameter<std::string>("frame_map", "map");
    frame_base_ = declare_parameter<std::string>("frame_base", "base_link");
    scan_topic_ = declare_parameter<std::string>("scan_topic", "/scan");
    target_index_ = declare_parameter<int>("target_index", 0);
    bt_xml_ = declare_parameter<std::string>("follow_bt_xml", "");

    // Distances & Timeouts
    stop_distance_ = declare_parameter<double>("follow_stop_distance", 0.5);
    start_distance_ = declare_parameter<double>("follow_start_distance", 0.7);
    bag_stop_distance_ = declare_parameter<double>("bag_stop_distance", 0.45);
    max_range_ = declare_parameter<double>("max_follow_range", 3.0);
    max_lost_time_ = declare_parameter<double>("max_lost_time", 3.0);
    occlusion_timeout_ = declare_parameter<double>("occlusion_timeout", 1.5);
    step_back_person_distance_ = declare_parameter<double>("step_back_person_distance", 2.0);
    step_back_speed_ = declare_parameter<double>("step_back_speed", 0.15);
    step_back_timeout_ = declare_parameter<double>("step_back_timeout", 15.0);

    // Speed Limits & PID Gains
    min_vx_ = declare_parameter<double>("min_vx", 0.08);
    max_vx_ = declare_parameter<double>("max_vx", 0.32);
    max_vy_ = declare_parameter<double>("max_vy", 0.22);
    max_wz_ = declare_parameter<double>("max_wz", 0.40);
    occlusion_max_vx_ = declare_parameter<double>("occlusion_max_vx", 0.22);
    kp_dist_ = declare_parameter<double>("kp_dist", 0.45);
    kp_yaw_ = declare_parameter<double>("kp_yaw", 0.75);
    kd_yaw_ = declare_parameter<double>("kd_yaw", 0.15);
    deadband_moving_ = declare_parameter<double>("deadband_moving", 0.04);
    deadband_stop_ = declare_parameter<double>("deadband_stop", 0.12);

    // Smoothing & Filtering
    alpha_v_ = declare_parameter<double>("alpha_v", 0.35);
    alpha_w_ = declare_parameter<double>("alpha_w", 0.40);
    bearing_beta_ = declare_parameter<double>("bearing_beta", 0.35);

    // Obstacle Avoidance & Safety Corridor
    enable_avoidance_ = declare_parameter<bool>("enable_avoidance", true);
    obstacle_distance_ = declare_parameter<double>("obstacle_distance", 0.85);
    footprint_radius_ = declare_parameter<double>("footprint_radius", 0.40);
    min_stop_x_ = declare_parameter<double>("min_stop_x", 0.48);
    corridor_width_ = declare_parameter<double>("corridor_width", 0.38);
    bypass_theta_offset_deg_ = declare_parameter<double>("bypass_theta_offset_deg", 10.0);
    peeking_avoid_dist_ = declare_parameter<double>("peeking_avoid_dist", 1.10);
    peeking_strafe_speed_ = declare_parameter<double>("peeking_strafe_speed", 0.22);
    peeking_timeout_ = declare_parameter<double>("peeking_timeout", 4.0);

    // Kalman Filter Tuning & 360° LiDAR Fusion
    kf_process_noise_ = declare_parameter<double>("kf_process_noise", 1.2);
    kf_measurement_noise_ = declare_parameter<double>("kf_measurement_noise", 0.15);
    kf_ = help_me_carry::PersonKalmanFilter(kf_process_noise_, kf_measurement_noise_);
    enable_lidar_fusion_ = declare_parameter<bool>("enable_lidar_fusion", true);
    kf_lidar_noise_ = declare_parameter<double>("kf_lidar_noise", 0.25);
    lidar_leg_gate_ = declare_parameter<double>("lidar_leg_gate", 0.90);

    // Homing Queueing & Search Parameters
    homing_queue_enabled_ = declare_parameter<bool>("homing_queue_enabled", true);
    homing_queue_search_enabled_ = declare_parameter<bool>("homing_queue_search_enabled", true);
    homing_queue_distance_ = declare_parameter<double>("homing_queue_distance", 1.00);
    homing_queue_resume_distance_ = declare_parameter<double>("homing_queue_resume_distance", 1.40);
    homing_queue_corridor_ = declare_parameter<double>("homing_queue_corridor", 1.00);
    homing_queue_max_detect_dist_ = declare_parameter<double>("homing_queue_max_detect_dist", 3.00);
    homing_queue_min_persons_ = declare_parameter<int>("homing_queue_min_persons", 2);
    homing_queue_inter_dist_ = declare_parameter<double>("homing_queue_inter_dist", 1.80);
    homing_queue_crawl_speed_ = declare_parameter<double>("homing_queue_crawl_speed", 0.10);
    homing_queue_wait_duration_ = declare_parameter<double>("homing_queue_wait_duration", 5.0);
    homing_queue_clear_hysteresis_ = declare_parameter<double>("homing_queue_clear_hysteresis", 0.75);
    homing_timeout_ = declare_parameter<double>("homing_timeout", 300.0);

    // Dynamic Parameter Reconfiguration Callback
    param_cb_handle_ = add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> & params) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        for (const auto & p : params) {
          if (p.get_name() == "homing_queue_enabled" || p.get_name() == "enable_queue") {
            homing_queue_enabled_ = p.as_bool();
            RCLCPP_INFO(get_logger(), "\033[1;33m[PARAM]\033[0m homing_queue_enabled diatur menjadi: %s",
              homing_queue_enabled_ ? "TRUE (Fungsi antri AKTIF)" : "FALSE (Fungsi antri MATI)");
          } else if (p.get_name() == "homing_queue_search_enabled" || p.get_name() == "enable_search") {
            homing_queue_search_enabled_ = p.as_bool();
            RCLCPP_INFO(get_logger(), "\033[1;33m[PARAM]\033[0m homing_queue_search_enabled diatur menjadi: %s",
              homing_queue_search_enabled_ ? "TRUE (Fungsi search AKTIF)" : "FALSE (Fungsi search MATI)");
          }
        }
        return result;
      });

    if (bt_xml_.empty()) {
      bt_xml_ = find_bt_xml();
    }

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_, this, false);
    action_client_ = rclcpp_action::create_client<NavigateToPose>(this, action_name_);

    goal_update_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      goal_update_topic_, rclcpp::QoS(10).reliable());
    cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>(
      cmd_vel_topic_, rclcpp::QoS(10));
    path_pub_ = create_publisher<nav_msgs::msg::Path>(
      path_topic_, rclcpp::QoS(10));
    follow_status_pub_ = create_publisher<std_msgs::msg::String>(
      "/follow/status", rclcpp::QoS(10));

    clear_global_costmap_client_ = create_client<nav2_msgs::srv::ClearEntireCostmap>(
      "/global_costmap/clear_entirely_global_costmap");
    clear_local_costmap_client_ = create_client<nav2_msgs::srv::ClearEntireCostmap>(
      "/local_costmap/clear_entirely_local_costmap");

    cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    rclcpp::SubscriptionOptions sub_options;
    sub_options.callback_group = cb_group_;

    cmd_vel_monitor_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      cmd_vel_topic_, rclcpp::QoS(1),
      [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(cmd_vel_mon_mutex_);
        monitored_cmd_vel_ = *msg;
      }, sub_options);

    nav_cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      nav_cmd_vel_topic_, rclcpp::SensorDataQoS(),
      [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
        on_nav_cmd_vel(msg);
      }, sub_options);

    person_sub_ = create_subscription<std_msgs::msg::String>(
      person_topic_, rclcpp::SensorDataQoS(),
      [this](const std_msgs::msg::String::SharedPtr msg) {
        {
          std::lock_guard<std::mutex> lock(person_mutex_);
          latest_persons_ = parse_persons(msg->data);
          latest_person_time_ = now();
        }
        on_person_detected(msg);
      },
      sub_options);

    person_back_sub_ = create_subscription<std_msgs::msg::String>(
      person_back_topic_, rclcpp::SensorDataQoS(),
      [this](const std_msgs::msg::String::SharedPtr msg) {
        if (!msg || msg->data.empty()) return;
        const auto parts = split_char(msg->data, ',');
        if (parts.size() >= 2) {
          try {
            const double bx = std::stod(trim(parts[0]));
            const double by = std::stod(trim(parts[1]));
            std::lock_guard<std::mutex> lock(person_back_mutex_);
            latest_back_x_ = bx;
            latest_back_y_ = by;
            latest_back_time_ = now();
            has_back_person_ = true;
          } catch (...) {}
        }
      },
      sub_options);

    gesture_sub_ = create_subscription<std_msgs::msg::String>(
      gesture_topic_, rclcpp::SensorDataQoS(),
      [this](const std_msgs::msg::String::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(bag_mutex_);
        if (msg->data.find("LEFT") != std::string::npos ||
            msg->data.find("RIGHT") != std::string::npos ||
            msg->data.find("KIRI") != std::string::npos ||
            msg->data.find("KANAN") != std::string::npos)
        {
          latest_gesture_ = msg->data;
        }
      },
      sub_options);

    bag_sub_ = create_subscription<std_msgs::msg::String>(
      bag_topic_, rclcpp::SensorDataQoS(),
      [this](const std_msgs::msg::String::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(bag_mutex_);
        latest_bags_ = parse_bags(msg->data);
        latest_bag_time_ = now();
      },
      sub_options);

    yaw_sub_ = create_subscription<std_msgs::msg::Float64>(
      yaw_topic_, rclcpp::SensorDataQoS(),
      [this](const std_msgs::msg::Float64::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(bag_mutex_);
        current_yaw_deg_ = msg->data;
      },
      sub_options);

    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic_, rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(scan_mutex_);
        latest_scan_ = msg;
      },
      sub_options);

    plan_sub_ = create_subscription<nav_msgs::msg::Path>(
      plan_topic_, rclcpp::QoS(10),
      [this](const nav_msgs::msg::Path::SharedPtr msg) {
        if (msg && !msg->poses.empty() && is_homing_active_) {
          std::lock_guard<std::mutex> lock(homing_plan_mutex_);
          latest_homing_plan_ = msg;
          // Teruskan path baru ke topic visualisasi (/following_path)
          path_pub_->publish(*msg);

          double path_len = 0.0;
          for (size_t i = 1; i < msg->poses.size(); ++i) {
            path_len += std::hypot(
              msg->poses[i].pose.position.x - msg->poses[i - 1].pose.position.x,
              msg->poses[i].pose.position.y - msg->poses[i - 1].pose.position.y);
          }

          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "\033[1;36m[Homing Path]\033[0m Path homing aktif diperbarui: %zu waypoints | Panjang rute = %.2fm | Tujuan=(%.2f, %.2f)",
            msg->poses.size(), path_len,
            msg->poses.back().pose.position.x, msg->poses.back().pose.position.y);
        }
      },
      sub_options);

    start_srv_ = create_service<Trigger>(
      "/follow/person",
      [this](const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> resp) {
        handle_start_follow(resp);
      },
      rclcpp::ServicesQoS(),
      cb_group_);

    start_follow_srv_alias_ = create_service<Trigger>(
      "/help_me_carry/start_follow",
      [this](const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> resp) {
        handle_start_follow(resp);
      },
      rclcpp::ServicesQoS(),
      cb_group_);

    stop_srv_ = create_service<Trigger>(
      "/follow/stop",
      [this](const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> resp) {
        handle_stop_follow(resp);
      },
      rclcpp::ServicesQoS(),
      cb_group_);

    stop_follow_srv_alias_ = create_service<Trigger>(
      "/help_me_carry/stop_follow",
      [this](const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> resp) {
        handle_stop_follow(resp);
      },
      rclcpp::ServicesQoS(),
      cb_group_);

    align_bag_srv_ = create_service<Trigger>(
      "/task/align_bag",
      [this](const std::shared_ptr<Trigger::Request> req, std::shared_ptr<Trigger::Response> resp) {
        handle_align_bag(req, resp);
      },
      rclcpp::ServicesQoS(),
      cb_group_);

    step_back_srv_ = create_service<Trigger>(
      "/task/step_back",
      [this](const std::shared_ptr<Trigger::Request> req, std::shared_ptr<Trigger::Response> resp) {
        handle_step_back(req, resp);
      },
      rclcpp::ServicesQoS(),
      cb_group_);

    step_back_srv_alias_ = create_service<Trigger>(
      "/help_me_carry/step_back",
      [this](const std::shared_ptr<Trigger::Request> req, std::shared_ptr<Trigger::Response> resp) {
        handle_step_back(req, resp);
      },
      rclcpp::ServicesQoS(),
      cb_group_);

    rotate_safe_srv_ = create_service<Trigger>(
      "/task/rotate_safe",
      [this](const std::shared_ptr<Trigger::Request> req, std::shared_ptr<Trigger::Response> resp) {
        handle_rotate_safe(req, resp);
      },
      rclcpp::ServicesQoS(),
      cb_group_);

    rotate_safe_srv_alias_ = create_service<Trigger>(
      "/help_me_carry/rotate_safe",
      [this](const std::shared_ptr<Trigger::Request> req, std::shared_ptr<Trigger::Response> resp) {
        handle_rotate_safe(req, resp);
      },
      rclcpp::ServicesQoS(),
      cb_group_);

    homing_srv_ = create_service<Trigger>(
      "/homing",
      [this](const std::shared_ptr<Trigger::Request> req, std::shared_ptr<Trigger::Response> resp) {
        handle_homming(req, resp);
      },
      rclcpp::ServicesQoS(),
      cb_group_);

    watchdog_timer_ = create_wall_timer(
      std::chrono::milliseconds(500), [this]() { check_safety_watchdog(); });

    RCLCPP_INFO(
      get_logger(),
      "help_me_carry_node siap! Jarak aman: %.2f m | XML BT: %s",
      stop_distance_, bt_xml_.c_str());
  }

private:
  void on_person_detected(const std_msgs::msg::String::SharedPtr msg)
  {
    if (!is_following_) {
      return;
    }

    // Compute dt for rotation derivative (D-term) and Kalman Filter
    double dt = 0.033;
    if (has_last_filter_time_) {
      dt = (now() - last_filter_time_).seconds();
      if (dt <= 0.0) dt = 0.033;
      if (dt > 0.5) dt = 0.5;
    }
    last_filter_time_ = now();
    has_last_filter_time_ = true;

    // Predict target position with Kalman Filter
    if (kf_.is_initialized()) {
      kf_.predict(dt);
    }

    // Check robot map transform if available (for RViz path)
    bool has_map_tf = false;
    double rx = 0.0, ry = 0.0, ryaw = 0.0;
    try {
      auto trans = tf_buffer_->lookupTransform(frame_map_, frame_base_, tf2::TimePointZero);
      rx = trans.transform.translation.x;
      ry = trans.transform.translation.y;
      const auto & rq = trans.transform.rotation;
      const double siny_cosp = 2.0 * (rq.w * rq.z + rq.x * rq.y);
      const double cosy_cosp = 1.0 - 2.0 * (rq.y * rq.y + rq.z * rq.z);
      ryaw = std::atan2(siny_cosp, cosy_cosp);
      has_map_tf = true;
    } catch (const tf2::TransformException &) {
      // Map TF not ready. Person following continues using camera coordinates.
    }

    const auto people = parse_persons(msg->data);
    struct Candidate {
      int id{-1};
      bool is_operator{false};
      double rel_x{0.0};
      double rel_y{0.0};
      double dist{0.0};
      double bearing{0.0};
    };

    std::vector<Candidate> candidates;
    for (const auto & person : people) {
      if (!std::isfinite(person.x) || !std::isfinite(person.y)) continue;
      const double dist = std::hypot(person.x, person.y);
      if (dist > max_range_ || dist < 0.15) continue;

      // Robot axis convention:
      // person.x = forward distance (X forward)
      // -person.y = lateral position (+Y = left, -Y = right)
      // bearing relative angle (rad): > 0 left, < 0 right
      const double bearing = std::atan2(-person.y, person.x);

      Candidate cand;
      cand.id = person.id;
      cand.is_operator = person.is_operator;
      cand.rel_x = person.x;
      cand.rel_y = -person.y;
      cand.dist = dist;
      cand.bearing = bearing;
      candidates.push_back(cand);
    }

    bool person_matched = false;
    bool is_predicting = false;
    double raw_dist = 0.0;
    double raw_bearing = 0.0;
    double rel_x = 0.0;
    double rel_y = 0.0;

    if (!candidates.empty()) {
      auto target_it = candidates.end();

      // 1. If KF is active and target ID is locked:
      if (kf_.is_initialized() && locked_person_id_ >= 0) {
        // Find candidate with matching ID
        auto locked_it = std::find_if(candidates.begin(), candidates.end(), [this](const Candidate & c) {
          return c.id == locked_person_id_;
        });

        if (locked_it != candidates.end()) {
          // Validate spatial consistency via Kalman gate
          const double dist_to_kf = kf_.distance_to(locked_it->rel_x, locked_it->rel_y);
          const double gate_radius = kf_.get_gate_radius(1.2, 2.5);
          if (dist_to_kf <= gate_radius) {
            target_it = locked_it;
          } else {
            RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 1000,
              "[FOLLOW] ID %d detected outside KF gate (dist=%.2f m > gate=%.2f m). Seeking re-lock...",
              locked_person_id_, dist_to_kf, gate_radius);
          }
        }

        // 2. If locked ID not found (occluded or changed by ZED):
        // Find candidate nearest to Kalman Filter estimate (Spatial Association)
        if (target_it == candidates.end()) {
          double min_kf_dist = 999.0;
          auto best_cand = candidates.end();
          for (auto it = candidates.begin(); it != candidates.end(); ++it) {
            const double d = kf_.distance_to(it->rel_x, it->rel_y);
            if (d < min_kf_dist) {
              min_kf_dist = d;
              best_cand = it;
            }
          }

          // Check if nearest candidate is within Kalman gate radius
          const double re_gate_radius = kf_.get_gate_radius(0.85, 2.0);
          if (best_cand != candidates.end() && min_kf_dist <= re_gate_radius) {
            RCLCPP_INFO(
              get_logger(),
              "\033[93m[FOLLOW] Re-lock target via Kalman Gate: old ID %d -> new ID %d (KF dist=%.2f m <= gate=%.2f m)\033[0m",
              locked_person_id_, best_cand->id, min_kf_dist, re_gate_radius);
            target_it = best_cand;
            locked_person_id_ = target_it->id;
          }
        }
      }

      // 3. If no target locked or lost for too long (> max_lost_time_):
      const double time_since_seen = now().seconds() - last_seen_sec_;
      if (target_it == candidates.end() && (!kf_.is_initialized() || locked_person_id_ < 0 || time_since_seen > max_lost_time_)) {
        target_it = std::find_if(candidates.begin(), candidates.end(), [](const Candidate & c) {
          return c.is_operator;
        });
        if (target_it == candidates.end()) {
          auto closest_it = std::min_element(candidates.begin(), candidates.end(), [](const Candidate & a, const Candidate & b) {
            return a.dist < b.dist;
          });
          if (closest_it != candidates.end() && closest_it->dist <= 2.5) {
            target_it = closest_it;
          }
        }

        if (target_it != candidates.end()) {
          RCLCPP_INFO(
            get_logger(),
            "\033[92m[FOLLOW] Target lock initialized: [ID %d] dist=%.2f m | bearing=%+.1f deg\033[0m",
            target_it->id, target_it->dist, target_it->bearing * 180.0 / M_PI);
          locked_person_id_ = target_it->id;
          kf_.init(target_it->rel_x, target_it->rel_y);
        }
      }

      // 4. Update measurement if target verified by ZED camera
      if (target_it != candidates.end()) {
        kf_.updateZED(target_it->rel_x, target_it->rel_y);
        person_matched = true;
        is_predicting = false;
        rel_x = kf_.x();
        rel_y = kf_.y();
        raw_dist = std::hypot(rel_x, rel_y);
        raw_bearing = std::atan2(rel_y, rel_x);
        last_seen_sec_ = now().seconds();
        std_msgs::msg::String status_msg;
        status_msg.data = "TRACKING_ZED";
        follow_status_pub_->publish(status_msg);
      }
    }

    const bool camera_sees_person = person_matched;

    // =========================================================================
    // OBSTACLE AVOIDANCE & PEEKING LOGIC:
    // Jika kamera tidak melihat orang DAN LiDAR mendeteksi obstacle di depannya,
    // maka robot menghindar ke ruang kosong (kiri/kanan) dan rotasi 10 derajat
    // berlawanan (jika ke kiri rotasi ke kanan 10° dan sebaliknya).
    // =========================================================================
    if (!camera_sees_person && enable_avoidance_) {
      sensor_msgs::msg::LaserScan::SharedPtr scan;
      {
        std::lock_guard<std::mutex> lock(scan_mutex_);
        scan = latest_scan_;
      }

      if (scan && !scan->ranges.empty()) {
        double laser_tx = 0.0, laser_ty = 0.0, laser_yaw = 3.1415926535;
        try {
          auto tf_laser = tf_buffer_->lookupTransform(
            frame_base_, scan->header.frame_id, tf2::TimePointZero);
          laser_tx = tf_laser.transform.translation.x;
          laser_ty = tf_laser.transform.translation.y;
          const auto & rq = tf_laser.transform.rotation;
          const double siny = 2.0 * (rq.w * rq.z + rq.x * rq.y);
          const double cosy = 1.0 - 2.0 * (rq.y * rq.y + rq.z * rq.z);
          laser_yaw = std::atan2(siny, cosy);
        } catch (const tf2::TransformException &) {
          laser_yaw = 3.1415926535;
        }

        bool front_obstacle_detected = false;
        double min_front_dist = 999.0;
        double left_clearance = 0.0;
        double right_clearance = 0.0;
        int left_count = 0;
        int right_count = 0;

        const int num_readings = static_cast<int>(scan->ranges.size());
        for (int i = 0; i < num_readings; ++i) {
          const float r = scan->ranges[i];
          if (!std::isfinite(r) || r < scan->range_min || r > scan->range_max) {
            continue;
          }

          const double angle_laser = scan->angle_min + i * scan->angle_increment;
          const double angle_base = normalize_angle(angle_laser + laser_yaw);

          const double obs_x = laser_tx + r * std::cos(angle_base);
          const double obs_y = laser_ty + r * std::sin(angle_base);

          // Filter pantulan bodi robot
          if (std::hypot(obs_x, obs_y) < footprint_radius_) {
            continue;
          }

          // Deteksi obstacle di koridor depan
          if (obs_x > footprint_radius_ && obs_x < peeking_avoid_dist_ &&
              std::abs(obs_y) < (corridor_width_ * 1.25))
          {
            front_obstacle_detected = true;
            if (obs_x < min_front_dist) {
              min_front_dist = obs_x;
            }
          }

          // Hitung ruang kosong di sisi kiri (+10° s/d +75°) dan kanan (-75° s/d -10°)
          if (angle_base > 0.17 && angle_base < 1.30) {
            left_clearance += std::min(r, 2.5f);
            left_count++;
          } else if (angle_base < -0.17 && angle_base > -1.30) {
            right_clearance += std::min(r, 2.5f);
            right_count++;
          }
        }

        if (front_obstacle_detected) {
          const double avg_left = (left_count > 0) ? (left_clearance / left_count) : 0.0;
          const double avg_right = (right_count > 0) ? (right_clearance / right_count) : 0.0;
          const bool strafe_left = (avg_left >= avg_right);

          if (!is_peeking_avoidance_) {
            is_peeking_avoidance_ = true;
            peeking_start_time_ = now();
            peeking_start_yaw_ = get_robot_heading();
            peeking_dodge_left_ = strafe_left;

            const double rot_rad = bypass_theta_offset_deg_ * M_PI / 180.0;
            // Jika menghindar ke kiri -> rotasi ke kanan 10° (-rot_rad)
            // Jika menghindar ke kanan -> rotasi ke kiri 10° (+rot_rad)
            peeking_target_yaw_ = peeking_dodge_left_
              ? normalize_angle(peeking_start_yaw_ - rot_rad)
              : normalize_angle(peeking_start_yaw_ + rot_rad);

            RCLCPP_INFO(
              get_logger(),
              "\033[1;33m[AVOID OBSTACLE]\033[0m Kamera tidak melihat orang & Ada obstacle depan (%.2fm). Menghindar ke %s & rotasi %s %.1f°...",
              min_front_dist,
              peeking_dodge_left_ ? "KIRI" : "KANAN",
              peeking_dodge_left_ ? "KANAN" : "KIRI",
              bypass_theta_offset_deg_);
          }

          if ((now() - peeking_start_time_).seconds() <= peeking_timeout_) {
            const double cur_yaw = get_robot_heading();
            const double yaw_err = normalize_angle(peeking_target_yaw_ - cur_yaw);
            const double target_wz = std::clamp(1.5 * yaw_err, -max_wz_, max_wz_);
            const double target_vy = peeking_dodge_left_ ? peeking_strafe_speed_ : -peeking_strafe_speed_;
            const double target_vx = 0.0; // Berhenti maju agar tidak menabrak rintangan depan

            current_cmd_vx_ = (1.0 - alpha_v_) * current_cmd_vx_ + alpha_v_ * target_vx;
            current_cmd_vy_ = (1.0 - alpha_v_) * current_cmd_vy_ + alpha_v_ * target_vy;
            current_cmd_wz_ = (1.0 - alpha_w_) * current_cmd_wz_ + alpha_w_ * target_wz;

            if (std::abs(current_cmd_vx_) < 0.015) current_cmd_vx_ = 0.0;
            if (std::abs(current_cmd_vy_) < 0.015) current_cmd_vy_ = 0.0;
            if (std::abs(current_cmd_wz_) < 0.02) current_cmd_wz_ = 0.0;

            geometry_msgs::msg::Twist twist;
            twist.linear.x = current_cmd_vx_;
            twist.linear.y = current_cmd_vy_;
            twist.angular.z = current_cmd_wz_;
            cmd_vel_pub_->publish(twist);

            std_msgs::msg::String status_msg;
            status_msg.data = "AVOID_OBSTACLE_PEEKING";
            follow_status_pub_->publish(status_msg);

            RCLCPP_INFO_THROTTLE(
              get_logger(), *get_clock(), 500,
              "\033[93m[PEEKING]\033[0m Strafe %s (vy=%.2f) | Rot %s 10° (wz=%.2f, err=%.1f°) | Obs=%.2fm",
              peeking_dodge_left_ ? "KIRI" : "KANAN", twist.linear.y,
              peeking_dodge_left_ ? "KANAN" : "KIRI", twist.angular.z,
              yaw_err * 180.0 / M_PI, min_front_dist);

            return; // Selesai siklus ini, robot sedang aktif melakukan manuver peeking
          } else {
            is_peeking_avoidance_ = false;
          }
        } else {
          is_peeking_avoidance_ = false;
        }
      }
    } else {
      is_peeking_avoidance_ = false;
    }

    // 5. 360° LiDAR EKF Fusion: If ZED loses sight of target (operator walked to the side or behind):
    if (!person_matched && enable_lidar_fusion_ && kf_.is_initialized()) {
      sensor_msgs::msg::LaserScan::SharedPtr scan;
      {
        std::lock_guard<std::mutex> lock(scan_mutex_);
        scan = latest_scan_;
      }
      if (scan && !scan->ranges.empty()) {
        double laser_tx = 0.0, laser_ty = 0.0, laser_yaw = 3.1415926535;
        try {
          auto tf_laser = tf_buffer_->lookupTransform(
            frame_base_, scan->header.frame_id, tf2::TimePointZero);
          laser_tx = tf_laser.transform.translation.x;
          laser_ty = tf_laser.transform.translation.y;
          const auto & rq = tf_laser.transform.rotation;
          const double siny = 2.0 * (rq.w * rq.z + rq.x * rq.y);
          const double cosy = 1.0 - 2.0 * (rq.y * rq.y + rq.z * rq.z);
          laser_yaw = std::atan2(siny, cosy);
        } catch (const tf2::TransformException &) {
          laser_yaw = 3.1415926535;
        }

        const auto clusters = extract_leg_clusters(scan, laser_tx, laser_ty, laser_yaw);
        double best_gate_dist = 999.0;
        const LegCluster * best_cluster = nullptr;
        const double gate_rad = std::max(lidar_leg_gate_, kf_.get_gate_radius(0.6, 2.0));

        for (const auto & cl : clusters) {
          const double d = kf_.distance_to(cl.x, cl.y);
          if (d <= gate_rad && d < best_gate_dist) {
            best_gate_dist = d;
            best_cluster = &cl;
          }
        }

        if (best_cluster != nullptr) {
          // Update EKF via LiDAR measurement
          kf_.updateLiDAR(best_cluster->x, best_cluster->y, kf_lidar_noise_ * kf_lidar_noise_);
          person_matched = true;
          is_predicting = false;
          rel_x = kf_.x();
          rel_y = kf_.y();
          raw_dist = std::hypot(rel_x, rel_y);
          raw_bearing = std::atan2(rel_y, rel_x);
          last_seen_sec_ = now().seconds();

          std_msgs::msg::String status_msg;
          status_msg.data = "TRACKING_LIDAR_360";
          follow_status_pub_->publish(status_msg);

          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 1500,
            "\033[96m[FOLLOW 360°] Tracking via LiDAR Leg: dist=%.2f m | bearing=%+.1f deg (gate=%.2f m)\033[0m",
            raw_dist, raw_bearing * 180.0 / M_PI, best_gate_dist);
        }
      }
    }

    // 6. Handle total occlusion (neither ZED nor LiDAR found target):
    if (!person_matched) {
      const double elapsed = now().seconds() - last_seen_sec_;
      // DEAD-RECKONING: Track Kalman Filter prediction during short occlusion!
      if (kf_.is_initialized() && elapsed <= occlusion_timeout_) {
        person_matched = true;
        is_predicting = true;
        rel_x = kf_.x();
        rel_y = kf_.y();
        raw_dist = std::hypot(rel_x, rel_y);
        raw_bearing = std::atan2(rel_y, rel_x);
        std_msgs::msg::String status_msg;
        status_msg.data = "PREDICTING";
        follow_status_pub_->publish(status_msg);
      } else {
        // Target lost beyond timeout: decelerate and stop safely
        current_cmd_vx_ *= 0.70;
        current_cmd_vy_ *= 0.70;
        current_cmd_wz_ *= 0.70;
        if (std::abs(current_cmd_vx_) < 0.02) current_cmd_vx_ = 0.0;
        if (std::abs(current_cmd_vy_) < 0.02) current_cmd_vy_ = 0.0;
        if (std::abs(current_cmd_wz_) < 0.02) current_cmd_wz_ = 0.0;

        geometry_msgs::msg::Twist coast_twist;
        coast_twist.linear.x = current_cmd_vx_;
        coast_twist.linear.y = current_cmd_vy_;
        coast_twist.angular.z = current_cmd_wz_;
        cmd_vel_pub_->publish(coast_twist);

        if (elapsed > 2.0) {
          is_moving_ = false;
        }
        return;
      }
    }

    // =====================================================================
    // LOW-PASS FILTER ON BEARING (REMOVES CAMERA JITTER)
    // =====================================================================
    if (!has_prev_bearing_) {
      filtered_bearing_ = raw_bearing;
      prev_bearing_ = raw_bearing;
      has_prev_bearing_ = true;
    } else {
      filtered_bearing_ = (1.0 - bearing_beta_) * filtered_bearing_ + bearing_beta_ * raw_bearing;
    }

    const double d_bearing = (filtered_bearing_ - prev_bearing_) / dt;
    prev_bearing_ = filtered_bearing_;

    const double direct_dist = raw_dist;
    const double direct_bearing = filtered_bearing_;

    // =====================================================================
    // 1. UPDATE PATH TO RVIZ FOR VISUALIZATION
    // =====================================================================
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = now();

    if (has_map_tf) {
      path_msg.header.frame_id = frame_map_;
      geometry_msgs::msg::PoseStamped p_robot;
      p_robot.header = path_msg.header;
      p_robot.pose.position.x = rx;
      p_robot.pose.position.y = ry;
      p_robot.pose.position.z = 0.0;
      p_robot.pose.orientation = yaw_to_quat(ryaw);
      path_msg.poses.push_back(p_robot);

      const double target_yaw = ryaw + direct_bearing;
      geometry_msgs::msg::PoseStamped p_target;
      p_target.header = path_msg.header;
      p_target.pose.position.x = rx + direct_dist * std::cos(target_yaw);
      p_target.pose.position.y = ry + direct_dist * std::sin(target_yaw);
      p_target.pose.position.z = 0.0;
      p_target.pose.orientation = yaw_to_quat(target_yaw);
      path_msg.poses.push_back(p_target);
    } else {
      path_msg.header.frame_id = frame_base_;
      geometry_msgs::msg::PoseStamped p_robot;
      p_robot.header = path_msg.header;
      p_robot.pose.position.x = 0.0;
      p_robot.pose.position.y = 0.0;
      p_robot.pose.position.z = 0.0;
      p_robot.pose.orientation.w = 1.0;
      path_msg.poses.push_back(p_robot);

      geometry_msgs::msg::PoseStamped p_target;
      p_target.header = path_msg.header;
      p_target.pose.position.x = rel_x;
      p_target.pose.position.y = rel_y;
      p_target.pose.position.z = 0.0;
      p_target.pose.orientation = yaw_to_quat(direct_bearing);
      path_msg.poses.push_back(p_target);
    }
    path_pub_->publish(path_msg);

    // =====================================================================
    // 2. DIRECT CONTINUOUS MOTION CONTROL: DRIVE FORWARD FACING OPERATOR
    // =====================================================================
    // Forward distance hysteresis:
    if (!is_moving_) {
      // Stationary: start forward if person distance exceeds start_distance (0.7m)
      if (direct_dist > start_distance_) {
        is_moving_ = true;
      }
    } else {
      // Moving: continue until person distance < stop_distance (0.5m)
      if (direct_dist < stop_distance_) {
        is_moving_ = false;
      }
    }

    double target_vx = 0.0;
    double target_vy = 0.0;
    double target_wz = 0.0;
    std::string status_str = "STOP";

    // ---------------------------------------------------------------------
    // A. ANGULAR ROTATION CONTROL: LOCK HEADING TO OPERATOR (ANTI-HUNTING)
    // ---------------------------------------------------------------------
    // Continuous smooth deadband:
    const double deadband = is_moving_ ? deadband_moving_ : deadband_stop_;
    double eff_bearing = 0.0;
    if (direct_bearing > deadband) {
      eff_bearing = direct_bearing - deadband;
    } else if (direct_bearing < -deadband) {
      eff_bearing = direct_bearing + deadband;
    }

    // PD Controller for yaw:
    target_wz = std::clamp(kp_yaw_ * eff_bearing + kd_yaw_ * d_bearing, -max_wz_, max_wz_);

    // ---------------------------------------------------------------------
    // B. LINEAR TRANSLATION CONTROL: SIMULTANEOUS FORWARD AND ROTATION
    // ---------------------------------------------------------------------
    if (is_moving_) {
      // Forward speed proportional to distance error
      double v_desired = kp_dist_ * (direct_dist - stop_distance_);
      v_desired = std::clamp(v_desired, min_vx_, max_vx_);

      // Limit speed during dead-reckoning occlusion
      if (is_predicting) {
        v_desired = std::min(v_desired, occlusion_max_vx_);
      }

      // Target vx simultaneously with turning towards operator
      if (std::abs(direct_bearing) > 1.10) {
        // Operator is far to the side (>63 deg) or behind robot: in-place rotate first!
        target_vx = 0.0;
      } else {
        target_vx = v_desired * std::max(0.15, std::cos(direct_bearing));
      }
      target_vy = 0.0;
      status_str = is_predicting ? "KALMAN PREDICTING (OCCLUDED)" : "FOLLOWING OPERATOR";

      // --- ROBUST LIDAR OBSTACLE & WALL AVOIDANCE ---
      bool obstacle_detected = false;
      if (enable_avoidance_) {
        sensor_msgs::msg::LaserScan::SharedPtr scan;
        {
          std::lock_guard<std::mutex> lock(scan_mutex_);
          scan = latest_scan_;
        }

        if (scan && !scan->ranges.empty()) {
          // Lookup transform from base_link to lidar sensor frame
          // Default yaw = 3.14159 rad because Lidar is mounted backwards (180 deg)
          double laser_tx = 0.0;
          double laser_ty = 0.0;
          double laser_yaw = 3.1415926535;
          try {
            auto tf_laser = tf_buffer_->lookupTransform(
              frame_base_, scan->header.frame_id, tf2::TimePointZero);
            laser_tx = tf_laser.transform.translation.x;
            laser_ty = tf_laser.transform.translation.y;
            const auto & rq = tf_laser.transform.rotation;
            const double siny = 2.0 * (rq.w * rq.z + rq.x * rq.y);
            const double cosy = 1.0 - 2.0 * (rq.y * rq.y + rq.z * rq.z);
            laser_yaw = std::atan2(siny, cosy);
          } catch (const tf2::TransformException &) {
            laser_yaw = 3.1415926535;
          }

          double rep_y = 0.0;
          double min_front_x = 999.0;
          double left_clearance = 0.0;
          double right_clearance = 0.0;
          int left_count = 0;
          int right_count = 0;

          const int num_readings = static_cast<int>(scan->ranges.size());
          for (int i = 0; i < num_readings; ++i) {
            const float r = scan->ranges[i];
            if (!std::isfinite(r) || r < scan->range_min || r > scan->range_max) {
              continue;
            }

            // Calculate beam angle in robot frame (base_link)
            const double angle_laser = scan->angle_min + i * scan->angle_increment;
            double angle_base = angle_laser + laser_yaw;
            angle_base = std::atan2(std::sin(angle_base), std::cos(angle_base));

            // Obstacle coordinate in base_link
            const double obs_x = laser_tx + r * std::cos(angle_base);
            const double obs_y = laser_ty + r * std::sin(angle_base);

            // 1. FILTER ROBOT FOOTPRINT: Ignore robot body reflections
            if (std::hypot(obs_x, obs_y) < footprint_radius_) {
              continue;
            }

            // 2. Evaluate ONLY front sector in base_link (-75 deg to +75 deg)
            if (angle_base < -1.30 || angle_base > 1.30) {
              continue;
            }

            // 3. Evaluate left vs right front clearance
            if (angle_base > 0.15 && angle_base < 1.20) {
              left_clearance += std::min(r, 2.0f);
              left_count++;
            } else if (angle_base < -0.15 && angle_base > -1.20) {
              right_clearance += std::min(r, 2.0f);
              right_count++;
            }

            // 4. FILTER OPERATOR: Do not treat operator's legs as obstacle
            if (std::abs(angle_base - direct_bearing) < 0.30 && std::abs(r - direct_dist) < 0.40) {
              continue;
            }

            // 5. Danger corridor in front of robot
            if (obs_x > footprint_radius_ && obs_x < obstacle_distance_ && std::abs(obs_y) < corridor_width_) {
              obstacle_detected = true;
              if (obs_x < min_front_x) {
                min_front_x = obs_x;
              }
            }

            // Lateral repulsive force (Y) for obstacle avoidance
            if (obs_x > footprint_radius_ * 0.8 && r < obstacle_distance_) {
              const double weight = (1.0 / std::max(0.20f, r)) - (1.0 / obstacle_distance_);
              rep_y -= weight * std::sin(angle_base);
            }
          }

          if (obstacle_detected) {
            // 1. HARD SAFETY STOP / BRAKING:
            if (min_front_x <= min_stop_x_) {
              target_vx = 0.0;
            } else if (min_front_x < obstacle_distance_) {
              // Progressive braking approaching obstacle
              double brake_ratio = (min_front_x - min_stop_x_) / (obstacle_distance_ - min_stop_x_);
              target_vx *= std::clamp(brake_ratio, 0.0, 1.0);
            }

            // 2. TANGENTIAL STRAFE VIA OMNI (Avoid towards open space):
            if (std::abs(rep_y) < 0.4) {
              double avg_left = (left_count > 0) ? (left_clearance / left_count) : 0.0;
              double avg_right = (right_count > 0) ? (right_clearance / right_count) : 0.0;
              rep_y = (avg_left >= avg_right) ? 1.5 : -1.5;
            }

            double avoid_vy = std::clamp(0.22 * rep_y, -max_vy_, max_vy_);
            target_vy = avoid_vy;
            status_str = "AVOIDING WALL/OBSTACLE";
          }
        }
      }

      target_vx = std::clamp(target_vx, 0.0, max_vx_);
      target_vy = std::clamp(target_vy, -max_vy_, max_vy_);
    } else {
      target_vx = 0.0;
      target_vy = 0.0;
      status_str = (std::abs(target_wz) > 0.03) ? "ALIGNING HEADING (STATIONARY)" : "STOPPED (Distance < 0.5m)";
    }

    // ---------------------------------------------------------------------
    // 3. LOW-PASS FILTER / SMOOTHING (Mencegah Hentakan & Menjaga Responsif)
    // ---------------------------------------------------------------------
    current_cmd_vx_ = (1.0 - alpha_v_) * current_cmd_vx_ + alpha_v_ * target_vx;
    current_cmd_vy_ = (1.0 - alpha_v_) * current_cmd_vy_ + alpha_v_ * target_vy;
    current_cmd_wz_ = (1.0 - alpha_w_) * current_cmd_wz_ + alpha_w_ * target_wz;

    // Bersihkan noise angka sangat kecil mendekati nol
    if (std::abs(current_cmd_vx_) < 0.015) current_cmd_vx_ = 0.0;
    if (std::abs(current_cmd_vy_) < 0.015) current_cmd_vy_ = 0.0;
    if (std::abs(current_cmd_wz_) < 0.02) current_cmd_wz_ = 0.0;

    geometry_msgs::msg::Twist twist;
    twist.linear.x = current_cmd_vx_;
    twist.linear.y = current_cmd_vy_;
    twist.angular.z = current_cmd_wz_;

    // Kirim langsung ke base driver (/cmd_vel)
    cmd_vel_pub_->publish(twist);

    // Print status debug teratur di terminal (2 Hz)
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 500,
      "\033[94m[DEBUG] Operator [ID %d]: dist=%.2f m | bearing=%+.1f deg | Status: %s | vx=%.2f | vy=%.2f | wz=%.2f\033[0m",
      locked_person_id_, direct_dist, direct_bearing * 180.0 / M_PI,
      status_str.c_str(), twist.linear.x, twist.linear.y, twist.angular.z);
  }

  void send_nav2_goal(const geometry_msgs::msg::PoseStamped & target_pose)
  {
    if (!is_following_ || is_nav_active_) {
      return;
    }

    if ((now() - last_nav_fail_time_).seconds() < 1.0) {
      return;
    }

    if (!action_client_->wait_for_action_server(std::chrono::milliseconds(500))) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "\033[91m[FOLLOW] Action server Nav2 ('%s') belum siap!\033[0m", action_name_.c_str());
      return;
    }

    NavigateToPose::Goal goal;
    goal.behavior_tree = bt_xml_;
    goal.pose = target_pose;

    auto send_opts = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    send_opts.goal_response_callback = [this](const GoalHandle::SharedPtr & handle) {
      if (!handle) {
        RCLCPP_WARN(get_logger(), "\033[91m[FOLLOW] Goal following ditolak oleh Nav2!\033[0m");
        is_nav_active_ = false;
        last_nav_fail_time_ = now();
      } else {
        goal_handle_ = handle;
        is_nav_active_ = true;
        RCLCPP_INFO(get_logger(), "\033[92m[FOLLOW] Goal following diterima Nav2 (Avoidance Costmap aktif).\033[0m");
      }
    };
    send_opts.result_callback = [this](const GoalHandle::WrappedResult & res) {
      is_nav_active_ = false;
      goal_handle_.reset();
      if (res.code == rclcpp_action::ResultCode::SUCCEEDED) {
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1500,
          "\033[92m[FOLLOW] Arrived near operator. Waiting for operator to move...\033[0m");
      } else if (res.code == rclcpp_action::ResultCode::CANCELED) {
        RCLCPP_INFO(get_logger(), "\033[93m[FOLLOW] Nav2 following goal canceled.\033[0m");
      } else if (res.code == rclcpp_action::ResultCode::ABORTED) {
        last_nav_fail_time_ = now();
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "\033[91m[FOLLOW] Nav2 aborted (obstacle/unreachable). Retrying shortly...\033[0m");
      }
    };

    is_nav_active_ = true;
    action_client_->async_send_goal(goal, send_opts);
  }

  void handle_start_follow(std::shared_ptr<Trigger::Response> resp)
  {
    if (is_following_) {
      resp->success = true;
      resp->message = "Following is already active.";
      return;
    }

    if (!origin_recorded_) {
      try {
        auto trans = tf_buffer_->lookupTransform(frame_map_, frame_base_, tf2::TimePointZero);
        origin_x_ = trans.transform.translation.x;
        origin_y_ = trans.transform.translation.y;
        origin_qz_ = trans.transform.rotation.z;
        origin_qw_ = trans.transform.rotation.w;
        origin_recorded_ = true;
        RCLCPP_INFO(
          get_logger(),
          "[FOLLOW] Starting pose recorded for homing: (%.2f, %.2f)",
          origin_x_, origin_y_);
      } catch (const tf2::TransformException &) {
        RCLCPP_WARN(get_logger(), "[FOLLOW] Map TF not ready yet, fallback origin (0.0, 0.0) active.");
      }
    }

    {
      std::lock_guard<std::mutex> lock(bag_mutex_);
      latest_gesture_ = "NONE";
    }

    kf_.reset();
    has_last_filter_time_ = false;
    locked_person_id_ = -1;
    is_following_ = true;
    is_nav_active_ = false;
    is_moving_ = false;
    current_cmd_vx_ = 0.0;
    current_cmd_vy_ = 0.0;
    current_cmd_wz_ = 0.0;
    prev_bearing_ = 0.0;
    filtered_bearing_ = 0.0;
    has_prev_bearing_ = false;
    last_seen_sec_ = now().seconds();

    resp->success = true;
    resp->message = "Person following active (Start distance: >0.7m, Stop distance: <0.5m).";
    RCLCPP_INFO(get_logger(), "%s", resp->message.c_str());
    IC(resp->success, resp->message);
  }

  void handle_stop_follow(std::shared_ptr<Trigger::Response> resp)
  {
    is_following_ = false;
    is_homing_active_ = false;
    is_nav_active_ = false;
    is_moving_ = false;
    is_peeking_avoidance_ = false;
    current_cmd_vx_ = 0.0;
    current_cmd_vy_ = 0.0;
    current_cmd_wz_ = 0.0;
    locked_person_id_ = -1;
    prev_bearing_ = 0.0;
    filtered_bearing_ = 0.0;
    has_prev_bearing_ = false;
    kf_.reset();
    has_last_filter_time_ = false;

    if (goal_handle_) {
      action_client_->async_cancel_goal(goal_handle_);
      goal_handle_.reset();
    }

    if (homing_goal_handle_) {
      action_client_->async_cancel_goal(homing_goal_handle_);
      homing_goal_handle_.reset();
    }

    geometry_msgs::msg::Twist stop_twist;
    cmd_vel_pub_->publish(stop_twist);

    {
      std::lock_guard<std::mutex> lock(bag_mutex_);
      latest_gesture_ = "NONE";
    }

    resp->success = true;
    resp->message = "Person following stopped.";
    RCLCPP_INFO(get_logger(), "%s", resp->message.c_str());
    IC(resp->success, resp->message);
  }

  void check_safety_watchdog()
  {
    if (!is_following_ || max_lost_time_ <= 0.0) {
      return;
    }

    if (is_peeking_avoidance_) {
      const double elapsed = now().seconds() - last_seen_sec_;
      if (elapsed <= peeking_timeout_ + 1.0) {
        return; // Berikan kesempatan bagi manuver peeking obstacle untuk menemukan orang kembali
      }
    }

    const double elapsed = now().seconds() - last_seen_sec_;
    if (elapsed > max_lost_time_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 4000,
        "Watchdog: Operator not visible for %.1f s (max: %.1f s)! Stopping robot.",
        elapsed, max_lost_time_);
      if (is_nav_active_ && goal_handle_) {
        action_client_->async_cancel_goal(goal_handle_);
        goal_handle_.reset();
        is_nav_active_ = false;
      }
      // Reset filter so re-acquisition is not blocked by gating
      kf_.reset();
      has_last_filter_time_ = false;
      locked_person_id_ = -1;
      is_moving_ = false;
      current_cmd_vx_ = 0.0;
      current_cmd_vy_ = 0.0;
      current_cmd_wz_ = 0.0;

      geometry_msgs::msg::Twist stop_twist;
      cmd_vel_pub_->publish(stop_twist);

      std_msgs::msg::String status_msg;
      status_msg.data = "LOST";
      follow_status_pub_->publish(status_msg);
    }
  }

  void handle_align_bag(
    const std::shared_ptr<Trigger::Request> /*req*/,
    std::shared_ptr<Trigger::Response> resp)
  {
    // 1. Stop person following if active
    if (is_following_) {
      auto dummy = std::make_shared<Trigger::Response>();
      handle_stop_follow(dummy);
    }

    // 2. Wait until operator gives valid gesture (RIGHT or LEFT)
    RCLCPP_INFO(get_logger(), "[Align Bag] Waiting for operator pointing gesture (LEFT / RIGHT)...");
    const auto wait_start = now();
    const double gesture_timeout = 20.0;
    std::string gest = "NONE";
    bool is_right = false;
    bool is_left = false;

    rclcpp::Rate loop_rate(10);
    while (rclcpp::ok() && (now() - wait_start).seconds() < gesture_timeout) {
      {
        std::lock_guard<std::mutex> lock(bag_mutex_);
        gest = latest_gesture_;
      }
      is_right = (gest.find("RIGHT") != std::string::npos || gest.find("KANAN") != std::string::npos);
      is_left  = (gest.find("LEFT") != std::string::npos || gest.find("KIRI") != std::string::npos);

      if (is_right || is_left) {
        RCLCPP_INFO(get_logger(), "[Align Bag] Gesture detected: %s", gest.c_str());
        break;
      }
      loop_rate.sleep();
    }

    if (!is_right && !is_left) {
      resp->success = false;
      resp->message = "Timeout waiting for operator gesture (latest: " + gest + ")";
      RCLCPP_WARN(get_logger(), "[Align Bag] %s", resp->message.c_str());
      return;
    }

    // 3. Fetch bag detections from ZED
    std::vector<Bag> bags;
    const auto bag_start = now();
    while (rclcpp::ok() && (now() - bag_start).seconds() < 3.0) {
      {
        std::lock_guard<std::mutex> lock(bag_mutex_);
        bags = latest_bags_;
      }
      if (!bags.empty()) {
        break;
      }
      loop_rate.sleep();
    }

    // 4. Filter bag on selected side (rel_y < 0: right, rel_y > 0: left)
    Bag target_bag{0.0, 0.0};
    double min_d = 1e9;
    for (const auto & b : bags) {
      const double rel_y = -b.y; // Robot convention: +Y = left, -Y = right
      if ((is_right && rel_y < 0.0) || (is_left && rel_y > 0.0)) {
        const double d = std::hypot(b.x, rel_y);
        if (d < min_d) { min_d = d; target_bag = {b.x, rel_y}; }
      }
    }

    if (min_d > 1e8) {
      if (!bags.empty()) {
        target_bag = {bags.front().x, -bags.front().y};
        RCLCPP_WARN(get_logger(), "[Align Bag] Bag on specific side not found, targeting nearest bag.");
      } else {
        resp->success = false;
        resp->message = std::string("Bag on ") + (is_right ? "right" : "left") + " side not detected by ZED.";
        return;
      }
    }

    // 4. Calculate rotation angle using vision yaw (/vision/yaw_deg)
    auto get_vision_yaw = [&]() {
      std::lock_guard<std::mutex> lock(bag_mutex_);
      return current_yaw_deg_ * M_PI / 180.0;
    };

    const double target_yaw = normalize_angle(get_vision_yaw() + std::atan2(target_bag.y, target_bag.x));
    const auto t0 = now();
    geometry_msgs::msg::Twist twist;

    while (rclcpp::ok() && (now() - t0).seconds() < 5.0) {
      const double err = normalize_angle(target_yaw - get_vision_yaw());
      if (std::abs(err) <= 0.04) break;

      double w = std::clamp(1.2 * err, -0.45, 0.45);
      if (std::abs(w) < 0.12) w = (err > 0 ? 0.12 : -0.12);
      twist.angular.z = w;
      cmd_vel_pub_->publish(twist);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // 5. Stop rotation
    twist.angular.z = 0.0;
    cmd_vel_pub_->publish(twist);
    {
      std::lock_guard<std::mutex>log(bag_mutex_);
      latest_gesture_ = "NONE";
    }
    
    resp->success = true;

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // 6. Approach bag
    RCLCPP_INFO(get_logger(), "[Align Bag] Aligned with bag. Approaching to distance %.2f m...", bag_stop_distance_);
    const auto t_approach = now();
    const double max_approach_sec = 10.0;
    double last_known_dist = target_bag.x;

    while (rclcpp::ok() && (now() - t_approach).seconds() < max_approach_sec) {
      std::vector<Bag> current_bags;
      {
        std::lock_guard<std::mutex> lock(bag_mutex_);
        current_bags = latest_bags_;
      }

      Bag chosen_bag{0.0, 0.0};
      double min_d = 1e9;
      bool found_front_bag = false;

      for (const auto & b : current_bags) {
        if (b.x <= 0.05) continue;
        const double rel_y = -b.y; // +Y = left, -Y = right
        const double d = std::hypot(b.x, rel_y);
        // Prioritize bags in front of robot
        if (std::abs(rel_y) < 0.8 && d < min_d) {
          min_d = d;
          chosen_bag = {b.x, rel_y};
          found_front_bag = true;
        }
      }

      if (found_front_bag) {
        last_known_dist = min_d;
        if (chosen_bag.x <= bag_stop_distance_ || min_d <= bag_stop_distance_) {
          RCLCPP_INFO(get_logger(), "[Align Bag] In front of bag! Distance: %.2f m (target: %.2f m).", min_d, bag_stop_distance_);
          break;
        }

        const double d_err = min_d - bag_stop_distance_;
        double vx = std::clamp(0.40 * d_err, 0.08, 0.25);
        const double angle_err = std::atan2(chosen_bag.y, chosen_bag.x);
        double wz = std::clamp(0.80 * angle_err, -0.25, 0.25);

        twist.linear.x = vx;
        twist.linear.y = 0.0;
        twist.angular.z = wz;
        cmd_vel_pub_->publish(twist);
      } else {
        if (last_known_dist <= bag_stop_distance_ + 0.15) {
          RCLCPP_INFO(get_logger(), "[Align Bag] Bag close (%.2f m) near ZED FOV limit. Finished approach.", last_known_dist);
          break;
        }
        twist.linear.x = 0.05;
        twist.angular.z = 0.0;
        cmd_vel_pub_->publish(twist);
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // 7. Stop total
    twist.linear.x = 0.0;
    twist.linear.y = 0.0;
    twist.angular.z = 0.0;
    cmd_vel_pub_->publish(twist);

    resp->success = true;
    resp->message = std::string("Robot aligned and approached ") +
                    (is_right ? "right" : "left") + " bag, then stopped.";
    RCLCPP_INFO(get_logger(), "%s", resp->message.c_str());
    IC(resp->success, resp->message);
  }

  void handle_step_back(
    const std::shared_ptr<Trigger::Request> /*req*/,
    std::shared_ptr<Trigger::Response> resp)
  {
    RCLCPP_INFO(get_logger(), "\033[1;36m[Step Back]\033[0m Menerima perintah service mundur...");

    // 1. Stop person following jika sedang aktif
    if (is_following_) {
      auto dummy = std::make_shared<Trigger::Response>();
      handle_stop_follow(dummy);
    }

    // 2. Clear deteksi tas lama sebelum mulai mundur
    {
      std::lock_guard<std::mutex> lock(bag_mutex_);
      latest_bags_.clear();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    const auto t_start = now();
    bool bag_detected = false;
    bool person_dist_reached = false;
    bool obstacle_detected = false;
    double last_person_dist = 0.0;
    double last_bag_dist = 0.0;

    RCLCPP_INFO(
      get_logger(),
      "[Step Back] Robot mulai bergerak mundur (speed: -%.2f m/s, target person dist: %.2f m, timeout: %.1f s)",
      std::abs(step_back_speed_), step_back_person_distance_, step_back_timeout_);

    while (rclcpp::ok()) {
      const double elapsed = (now() - t_start).seconds();
      if (elapsed > step_back_timeout_) {
        RCLCPP_WARN(get_logger(), "[Step Back] Timeout (%.1f s) tercapai.", step_back_timeout_);
        break;
      }

      // Check 1: Deteksi Tas dari Vision YOLO (/vision/bag_detections)
      {
        std::lock_guard<std::mutex> lock(bag_mutex_);
        if ((now() - latest_bag_time_).seconds() < 1.0) {
          for (const auto & b : latest_bags_) {
            if (b.x > 0.05) {
              bag_detected = true;
              last_bag_dist = std::hypot(b.x, b.y);
              break;
            }
          }
        }
      }
      if (bag_detected) {
        RCLCPP_INFO(
          get_logger(),
          "\033[1;32m[Step Back]\033[0m Tas terdeteksi pada jarak %.2f m! Berhenti mundur.",
          last_bag_dist);
        break;
      }

      // Check 2: Deteksi jarak orang target dari Vision (/vision/person_label) >= step_back_person_distance_ (2.0 m)
      {
        std::lock_guard<std::mutex> lock(person_mutex_);
        if ((now() - latest_person_time_).seconds() < 1.0) {
          const Person * target_person = nullptr;
          double min_target_dist = 1e9;

          // Cari orang target di depan robot (utamakan yang berlabel operator, atau orang terdekat di depan)
          for (const auto & p : latest_persons_) {
            if (p.x <= 0.10) continue;
            const double d = std::hypot(p.x, p.y);

            if (p.is_operator) {
              target_person = &p;
              min_target_dist = d;
              break;
            }

            if (d < min_target_dist) {
              min_target_dist = d;
              target_person = &p;
            }
          }

          // Hanya evaluasi jarak orang target (bukan orang di latar belakang / background)
          if (target_person != nullptr && min_target_dist < 1e8) {
            if (min_target_dist >= step_back_person_distance_) {
              person_dist_reached = true;
              last_person_dist = min_target_dist;
            }
          }
        }
      }
      if (person_dist_reached) {
        RCLCPP_INFO(
          get_logger(),
          "\033[1;32m[Step Back]\033[0m Jarak orang target mencapai %.2f m (>= %.2f m)! Berhenti mundur.",
          last_person_dist, step_back_person_distance_);
        break;
      }

      // Check 3: Safety check obstacle di belakang robot via LiDAR (/scan)
      if (enable_avoidance_) {
        sensor_msgs::msg::LaserScan::SharedPtr scan;
        {
          std::lock_guard<std::mutex> lock(scan_mutex_);
          scan = latest_scan_;
        }
        if (scan && !scan->ranges.empty()) {
          double laser_tx = 0.0, laser_ty = 0.0, laser_yaw = 3.1415926535;
          try {
            auto tf_laser = tf_buffer_->lookupTransform(
              frame_base_, scan->header.frame_id, tf2::TimePointZero);
            laser_tx = tf_laser.transform.translation.x;
            laser_ty = tf_laser.transform.translation.y;
            const auto & rq = tf_laser.transform.rotation;
            const double siny = 2.0 * (rq.w * rq.z + rq.x * rq.y);
            const double cosy = 1.0 - 2.0 * (rq.y * rq.y + rq.z * rq.z);
            laser_yaw = std::atan2(siny, cosy);
          } catch (const tf2::TransformException &) {
            laser_yaw = 3.1415926535;
          }

          const int num_readings = static_cast<int>(scan->ranges.size());
          for (int i = 0; i < num_readings; ++i) {
            const float r = scan->ranges[i];
            if (!std::isfinite(r) || r < scan->range_min || r > scan->range_max) continue;
            const double angle_laser = scan->angle_min + i * scan->angle_increment;
            double angle_base = angle_laser + laser_yaw;
            angle_base = std::atan2(std::sin(angle_base), std::cos(angle_base));
            const double obs_x = laser_tx + r * std::cos(angle_base);
            const double obs_y = laser_ty + r * std::sin(angle_base);

            // Obstacle tepat di belakang robot (zona bahaya tabrakan mundur)
            if (obs_x < -footprint_radius_ && obs_x > -(footprint_radius_ + 0.15) &&
                std::abs(obs_y) < corridor_width_) {
              obstacle_detected = true;
              break;
            }
          }
        }
      }
      if (obstacle_detected) {
        RCLCPP_WARN(
          get_logger(),
          "\033[1;33m[Step Back]\033[0m Terdeteksi rintangan di belakang robot! Berhenti mundur demi keselamatan.");
        break;
      }

      // Kirim perintah mundur
      geometry_msgs::msg::Twist twist;
      twist.linear.x = -std::abs(step_back_speed_);
      twist.linear.y = 0.0;
      twist.angular.z = 0.0;
      cmd_vel_pub_->publish(twist);

      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // Stop total pergerakan robot
    geometry_msgs::msg::Twist stop_twist;
    cmd_vel_pub_->publish(stop_twist);

    char dist_buf[32];
    if (bag_detected) {
      std::snprintf(dist_buf, sizeof(dist_buf), "%.2f", last_bag_dist);
      resp->success = true;
      resp->message = std::string("Mundur selesai: tas terdeteksi pada jarak ") + dist_buf + " m.";
    } else if (person_dist_reached) {
      std::snprintf(dist_buf, sizeof(dist_buf), "%.2f", last_person_dist);
      char target_buf[32];
      std::snprintf(target_buf, sizeof(target_buf), "%.2f", step_back_person_distance_);
      resp->success = true;
      resp->message = std::string("Mundur selesai: jarak orang mencapai ") + dist_buf +
                      " m (target: " + target_buf + " m).";
    } else if (obstacle_detected) {
      resp->success = false;
      resp->message = "Mundur dihentikan: terdeteksi rintangan di belakang robot.";
    } else {
      char timeout_buf[32];
      std::snprintf(timeout_buf, sizeof(timeout_buf), "%.1f", step_back_timeout_);
      resp->success = false;
      resp->message = std::string("Mundur timeout (") + timeout_buf +
                      " s) sebelum mendeteksi tas atau jarak orang 2m.";
    }

    RCLCPP_INFO(get_logger(), "[Step Back] %s (success=%s)", resp->message.c_str(), resp->success ? "true" : "false");
    IC(resp->success, resp->message);
  }

  double get_robot_heading()
  {
    try {
      auto trans = tf_buffer_->lookupTransform(frame_map_, frame_base_, tf2::TimePointZero);
      const auto & rq = trans.transform.rotation;
      const double siny_cosp = 2.0 * (rq.w * rq.z + rq.x * rq.y);
      const double cosy_cosp = 1.0 - 2.0 * (rq.y * rq.y + rq.z * rq.z);
      return std::atan2(siny_cosp, cosy_cosp);
    } catch (const tf2::TransformException &) {
      try {
        auto trans = tf_buffer_->lookupTransform("odom", frame_base_, tf2::TimePointZero);
        const auto & rq = trans.transform.rotation;
        const double siny_cosp = 2.0 * (rq.w * rq.z + rq.x * rq.y);
        const double cosy_cosp = 1.0 - 2.0 * (rq.y * rq.y + rq.z * rq.z);
        return std::atan2(siny_cosp, cosy_cosp);
      } catch (const tf2::TransformException &) {
        std::lock_guard<std::mutex> lock(bag_mutex_);
        return current_yaw_deg_ * M_PI / 180.0;
      }
    }
  }

  struct LegCluster
  {
    double x{0.0};
    double y{0.0};
    double dist{0.0};
    double span{0.0};
    int count{0};
  };

  std::vector<LegCluster> extract_leg_clusters(
    const sensor_msgs::msg::LaserScan::SharedPtr & scan,
    double laser_tx, double laser_ty, double laser_yaw)
  {
    std::vector<LegCluster> clusters;
    if (!scan || scan->ranges.empty()) return clusters;

    struct Pt { double x; double y; };
    std::vector<Pt> current_pts;

    const int num_readings = static_cast<int>(scan->ranges.size());
    for (int i = 0; i < num_readings; ++i) {
      const float r = scan->ranges[i];
      if (!std::isfinite(r) || r < 0.15f || r > max_range_) {
        if (current_pts.size() >= 3) {
          double sum_x = 0.0, sum_y = 0.0;
          for (const auto & p : current_pts) {
            sum_x += p.x;
            sum_y += p.y;
          }
          const double cx = sum_x / current_pts.size();
          const double cy = sum_y / current_pts.size();
          const double span = std::hypot(current_pts.front().x - current_pts.back().x,
                                         current_pts.front().y - current_pts.back().y);
          if (span >= 0.04 && span <= 0.40) {
            clusters.push_back({cx, cy, std::hypot(cx, cy), span, static_cast<int>(current_pts.size())});
          }
        }
        current_pts.clear();
        continue;
      }

      const double angle = scan->angle_min + i * scan->angle_increment;
      const double cos_a = std::cos(angle + laser_yaw);
      const double sin_a = std::sin(angle + laser_yaw);
      const double lx = laser_tx + r * cos_a;
      const double ly = laser_ty + r * sin_a;

      if (!current_pts.empty()) {
        const double d_prev = std::hypot(lx - current_pts.back().x, ly - current_pts.back().y);
        if (d_prev > 0.15) {
          if (current_pts.size() >= 3) {
            double sum_x = 0.0, sum_y = 0.0;
            for (const auto & p : current_pts) {
              sum_x += p.x;
              sum_y += p.y;
            }
            const double cx = sum_x / current_pts.size();
            const double cy = sum_y / current_pts.size();
            const double span = std::hypot(current_pts.front().x - current_pts.back().x,
                                           current_pts.front().y - current_pts.back().y);
            if (span >= 0.04 && span <= 0.40) {
              clusters.push_back({cx, cy, std::hypot(cx, cy), span, static_cast<int>(current_pts.size())});
            }
          }
          current_pts.clear();
        }
      }
      current_pts.push_back({lx, ly});
    }

    if (current_pts.size() >= 3) {
      double sum_x = 0.0, sum_y = 0.0;
      for (const auto & p : current_pts) {
        sum_x += p.x;
        sum_y += p.y;
      }
      const double cx = sum_x / current_pts.size();
      const double cy = sum_y / current_pts.size();
      const double span = std::hypot(current_pts.front().x - current_pts.back().x,
                                     current_pts.front().y - current_pts.back().y);
      if (span >= 0.04 && span <= 0.40) {
        clusters.push_back({cx, cy, std::hypot(cx, cy), span, static_cast<int>(current_pts.size())});
      }
    }

    return clusters;
  }

  struct ClearanceSummary {
    double front{999.0};
    double back{999.0};
    double left{999.0};
    double right{999.0};
    double min_any{999.0};
  };

  ClearanceSummary get_clearance_summary()
  {
    ClearanceSummary sum;
    sensor_msgs::msg::LaserScan::SharedPtr scan;
    {
      std::lock_guard<std::mutex> lock(scan_mutex_);
      scan = latest_scan_;
    }
    if (!scan || scan->ranges.empty()) return sum;

    double laser_tx = 0.0, laser_ty = 0.0, laser_yaw = 3.1415926535;
    try {
      auto tf_laser = tf_buffer_->lookupTransform(frame_base_, scan->header.frame_id, tf2::TimePointZero);
      laser_tx = tf_laser.transform.translation.x;
      laser_ty = tf_laser.transform.translation.y;
      const auto & rq = tf_laser.transform.rotation;
      const double siny = 2.0 * (rq.w * rq.z + rq.x * rq.y);
      const double cosy = 1.0 - 2.0 * (rq.y * rq.y + rq.z * rq.z);
      laser_yaw = std::atan2(siny, cosy);
    } catch (...) {}

    const int n = static_cast<int>(scan->ranges.size());
    for (int i = 0; i < n; ++i) {
      const float r = scan->ranges[i];
      if (!std::isfinite(r) || r < scan->range_min || r > scan->range_max) continue;
      const double angle_laser = scan->angle_min + i * scan->angle_increment;
      const double angle_base = normalize_angle(angle_laser + laser_yaw);
      const double px = laser_tx + r * std::cos(angle_base);
      const double py = laser_ty + r * std::sin(angle_base);
      const double d = std::hypot(px, py);
      if (d < 0.22) continue; // Abaikan pantulan rangka/sasis robot sendiri

      if (d < sum.min_any) sum.min_any = d;

      // Sektor Depan (-30 s.d. +30 deg)
      if (std::abs(angle_base) < 0.52) {
        if (d < sum.front) sum.front = d;
      }
      // Sektor Belakang (> 150 deg atau < -150 deg)
      else if (std::abs(angle_base) > 2.62) {
        if (d < sum.back) sum.back = d;
      }
      // Sektor Kiri (60 s.d. 120 deg)
      else if (angle_base > 1.05 && angle_base < 2.09) {
        if (d < sum.left) sum.left = d;
      }
      // Sektor Kanan (-120 s.d. -60 deg)
      else if (angle_base < -1.05 && angle_base > -2.09) {
        if (d < sum.right) sum.right = d;
      }
    }
    return sum;
  }

  double find_safe_heading(double & out_best_clearance)
  {
    sensor_msgs::msg::LaserScan::SharedPtr scan;
    {
      std::lock_guard<std::mutex> lock(scan_mutex_);
      scan = latest_scan_;
    }

    out_best_clearance = 0.0;
    if (!scan || scan->ranges.empty()) {
      RCLCPP_WARN(get_logger(), "\033[1;31m[Safe Rotate]\033[0m Data LaserScan /scan tidak tersedia!");
      return 0.0;
    }

    // 1. Transform laser frame ke base_link
    double laser_tx = 0.0;
    double laser_ty = 0.0;
    double laser_yaw = 3.1415926535;
    try {
      auto tf_laser = tf_buffer_->lookupTransform(frame_base_, scan->header.frame_id, tf2::TimePointZero);
      laser_tx = tf_laser.transform.translation.x;
      laser_ty = tf_laser.transform.translation.y;
      const auto & rq = tf_laser.transform.rotation;
      const double siny = 2.0 * (rq.w * rq.z + rq.x * rq.y);
      const double cosy = 1.0 - 2.0 * (rq.y * rq.y + rq.z * rq.z);
      laser_yaw = std::atan2(siny, cosy);
    } catch (const tf2::TransformException &) {
      laser_yaw = 3.1415926535;
    }

    // 2. Tentukan arah relatif robot menuju titik origin (homing)
    double angle_to_origin_rel = M_PI; // Default 180 deg (belakang robot)
    try {
      auto trans = tf_buffer_->lookupTransform(frame_map_, frame_base_, tf2::TimePointZero);
      const double rx = trans.transform.translation.x;
      const double ry = trans.transform.translation.y;
      const auto & rq = trans.transform.rotation;
      const double siny = 2.0 * (rq.w * rq.z + rq.x * rq.y);
      const double cosy = 1.0 - 2.0 * (rq.y * rq.y + rq.z * rq.z);
      const double ryaw = std::atan2(siny, cosy);
      if (origin_recorded_) {
        const double origin_map_yaw = std::atan2(origin_y_ - ry, origin_x_ - rx);
        angle_to_origin_rel = normalize_angle(origin_map_yaw - ryaw);
      }
    } catch (const tf2::TransformException &) {
      // TF map belum siap, gunakan default M_PI
    }

    // 3. Konversi semua titik scan yang valid ke koordinat base_link
    struct LaserPt { double x, y; };
    std::vector<LaserPt> pts;
    pts.reserve(scan->ranges.size());
    const int n = static_cast<int>(scan->ranges.size());
    for (int i = 0; i < n; ++i) {
      const float r = scan->ranges[i];
      if (!std::isfinite(r) || r < scan->range_min || r > scan->range_max) continue;
      const double angle_laser = scan->angle_min + i * scan->angle_increment;
      const double angle_base = normalize_angle(angle_laser + laser_yaw);
      const double px = laser_tx + r * std::cos(angle_base);
      const double py = laser_ty + r * std::sin(angle_base);
      if (std::hypot(px, py) < 0.22) continue; // Abaikan titik internal robot
      pts.push_back({px, py});
    }

    if (pts.empty()) {
      out_best_clearance = 3.0;
      return angle_to_origin_rel;
    }

    // 4. Evaluasi 36 kandidat heading (-180 sampai +180 derajat)
    const int num_candidates = 36;
    const double step = 2.0 * M_PI / num_candidates;
    const double corridor_half_width = std::max(0.32, corridor_width_);
    const double max_eval_dist = 3.5;

    double best_score = -1e9;
    double best_heading = 0.0;
    double best_clearance = 0.0;
    double front_clearance = max_eval_dist;

    for (int c = 0; c < num_candidates; ++c) {
      const double phi = normalize_angle(-M_PI + c * step);
      const double cos_p = std::cos(phi);
      const double sin_p = std::sin(phi);

      double min_fwd = max_eval_dist;
      for (const auto & p : pts) {
        // Rotasi titik ke koordinat lokal kandidat heading
        const double fwd = p.x * cos_p + p.y * sin_p;
        const double lat = -p.x * sin_p + p.y * cos_p;

        if (fwd > 0.05 && fwd < max_eval_dist && std::abs(lat) < corridor_half_width) {
          if (fwd < min_fwd) {
            min_fwd = fwd;
          }
        }
      }

      if (std::abs(phi) < 0.15) {
        front_clearance = min_fwd;
      }

      // Hitung skor:
      // Area dengan clearance < 0.60 m dianggap terhalang rintangan
      double score = 0.0;
      const double safe_threshold = 0.60; // 0.60 m
      if (min_fwd < safe_threshold) {
        score = -100.0 + min_fwd;
      } else {
        // Keselarasan dengan arah origin (1.0 jika tepat menuju origin, 0.0 jika bertolak belakang)
        const double origin_align = 0.5 * (1.0 + std::cos(phi - angle_to_origin_rel));
        // Bonus clearance: jika clearance sudah aman (>= 1.2 m), beri skor penuh
        const double cl_bonus = std::min(min_fwd, 1.5) / 1.5;
        // Prioritas utama: Kepala robot HARUS MENGHADAP KE ARAH ORIGIN (bobot 10.0),
        // clearance sebagai syarat keamanan dan penentu tie-breaker (bobot 1.0).
        score = origin_align * 10.0 + cl_bonus * 1.0;
      }

      if (score > best_score) {
        best_score = score;
        best_heading = phi;
        best_clearance = min_fwd;
      }
    }

    out_best_clearance = best_clearance;
    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[SAFE ROTATE SCAN]\033[0m Evaluasi Heading: Best Angle=%.1f° (Clearance=%.2f m, Score=%.2f) | Clearance Depan Saat Ini=%.2f m | Arah Origin=%.1f°",
      best_heading * 180.0 / M_PI, best_clearance, best_score, front_clearance, angle_to_origin_rel * 180.0 / M_PI);

    return best_heading;
  }

  bool rotate_to_safe_area(double max_duration_sec = 8.0)
  {
    double clearance = 0.0;
    double safe_rel_yaw = find_safe_heading(clearance);

    // Jika depan sudah aman (clearance >= 1.0 m) dan heading aman hanya berbeda kecil (< 15 deg), lewati
    if (std::abs(safe_rel_yaw) < 0.26 && clearance >= 1.0) {
      RCLCPP_INFO(
        get_logger(),
        "\033[1;32m[SAFE ROTATE]\033[0m Area depan robot sudah lapang (clearance: %.2f m) dan selaras. Rotasi tidak diperlukan.",
        clearance);
      return true;
    }

    const double start_yaw = get_robot_heading();
    const double target_yaw = normalize_angle(start_yaw + safe_rel_yaw);
    const auto t_start = now();

    RCLCPP_INFO(
      get_logger(),
      "\033[1;33m[SAFE ROTATE START]\033[0m Memulai rotasi aman di tempat: Start Yaw=%.1f° -> Target Yaw=%.1f° (Perlu putar %.1f°, Area Lapang: %.2fm)...",
      start_yaw * 180.0 / M_PI, target_yaw * 180.0 / M_PI, safe_rel_yaw * 180.0 / M_PI, clearance);

    geometry_msgs::msg::Twist twist;
    rclcpp::Rate rate(20);
    int loop_count = 0;

    while (rclcpp::ok()) {
      const double elapsed = (now() - t_start).seconds();
      if (elapsed > max_duration_sec) {
        RCLCPP_WARN(
          get_logger(),
          "\033[1;31m[SAFE ROTATE TIMEOUT]\033[0m Waktu rotasi melebihi %.1f s. Menghentikan putaran in-place.",
          max_duration_sec);
        break;
      }

      // Safety check rintangan yang benar-benar menempel bodi luar robot (< 0.26 m dari pusat)
      ClearanceSummary clear = get_clearance_summary();
      if (clear.min_any < 0.26) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "\033[1;31m[SAFE ROTATE EMERGENCY STOP]\033[0m Objek sangat dekat dengan bodi (jarak: %.2f m)! Rotasi dihentikan darurat.",
          clear.min_any);
        break;
      }

      const double cur_yaw = get_robot_heading();
      const double err = normalize_angle(target_yaw - cur_yaw);

      // Selesai jika deviasi sudut <= 0.08 rad (~4.5 deg)
      if (std::abs(err) <= 0.08) {
        RCLCPP_INFO(
          get_logger(),
          "\033[1;32m[SAFE ROTATE SUCCEEDED]\033[0m Rotasi in-place berhasil diselesaikan! Posisi akhir: %.1f° (sisa error: %.1f°, waktu: %.2f s).",
          cur_yaw * 180.0 / M_PI, err * 180.0 / M_PI, elapsed);
        break;
      }

      // Proportional angular speed dengan batas minimum untuk mengatasi friksi roda
      double wz = std::clamp(1.2 * err, -max_wz_, max_wz_);
      if (std::abs(wz) < 0.15) {
        wz = (err > 0.0 ? 0.15 : -0.15);
      }

      twist.linear.x = 0.0;
      twist.linear.y = 0.0;
      twist.angular.z = wz;
      cmd_vel_pub_->publish(twist);

      loop_count++;
      if (loop_count % 5 == 0) { // Print setiap ~250 ms
        RCLCPP_INFO(
          get_logger(),
          "[SAFE ROTATE RUNNING] t=%.1fs | Yaw Saat Ini: %.1f° -> Target: %.1f° | Sisa Error: %.1f° | Cmd Vel Wz: %.2f rad/s",
          elapsed, cur_yaw * 180.0 / M_PI, target_yaw * 180.0 / M_PI, err * 180.0 / M_PI, wz);
      }

      rate.sleep();
    }

    // Stop total pergerakan robot
    geometry_msgs::msg::Twist stop_twist;
    cmd_vel_pub_->publish(stop_twist);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    return true;
  }

  bool is_facing_away_or_blocked()
  {
    double cur_x = 0.0, cur_y = 0.0, cur_yaw = 0.0;
    bool has_tf = false;
    try {
      auto trans = tf_buffer_->lookupTransform(frame_map_, frame_base_, tf2::TimePointZero);
      cur_x = trans.transform.translation.x;
      cur_y = trans.transform.translation.y;
      const auto & rq = trans.transform.rotation;
      const double siny = 2.0 * (rq.w * rq.z + rq.x * rq.y);
      const double cosy = 1.0 - 2.0 * (rq.y * rq.y + rq.z * rq.z);
      cur_yaw = std::atan2(siny, cosy);
      has_tf = true;
    } catch (...) {}

    double rel_angle_origin = 0.0;
    if (has_tf && origin_recorded_) {
      const double origin_map_yaw = std::atan2(origin_y_ - cur_y, origin_x_ - cur_x);
      rel_angle_origin = normalize_angle(origin_map_yaw - cur_yaw);
    }

    ClearanceSummary clear = get_clearance_summary();

    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[HOMING DIAGNOSTIC]\033[0m Posisi: (x=%.2f, y=%.2f, yaw=%.1f°) | Selisih Sudut ke Origin: %.1f° | Jarak Sensor: Depan=%.2fm, Belakang=%.2fm, Kiri=%.2fm, Kanan=%.2fm",
      cur_x, cur_y, cur_yaw * 180.0 / M_PI, rel_angle_origin * 180.0 / M_PI,
      clear.front, clear.back, clear.left, clear.right);

    // Kriteria membelakangi atau terhalang:
    // 1. Selisih sudut ke origin > 45 derajat (robot membelakangi jalur pulang)
    // 2. ATAU jarak rintangan di depan < 0.70 m (ada meja / orang tepat di depan muka robot)
    bool facing_away = std::abs(rel_angle_origin) > 0.78;
    bool front_blocked = clear.front < 0.70;

    if (facing_away || front_blocked) {
      RCLCPP_INFO(
        get_logger(),
        "\033[1;33m[HOMING DIAGNOSTIC]\033[0m Status: %s%s -> Memerlukan rotasi di tempat ke area lapang!",
        facing_away ? "[MEMBELAKANGI ORIGIN] " : "",
        front_blocked ? "[DEPAN TERHALANG < 0.70m]" : "");
      return true;
    }

    RCLCPP_INFO(
      get_logger(),
      "\033[1;32m[HOMING DIAGNOSTIC]\033[0m Status: Posisi hadap sudah selaras (%.1f°) dan depan lapang (%.2fm). Siap maju!",
      rel_angle_origin * 180.0 / M_PI, clear.front);
    return false;
  }

  void handle_rotate_safe(
    const std::shared_ptr<Trigger::Request> /*req*/,
    std::shared_ptr<Trigger::Response> resp)
  {
    RCLCPP_INFO(get_logger(), "\033[1;36m[Rotate Safe]\033[0m Menerima perintah service rotasi aman...");
    if (is_following_) {
      auto dummy = std::make_shared<Trigger::Response>();
      handle_stop_follow(dummy);
    }

    double clearance = 0.0;
    double safe_yaw = find_safe_heading(clearance);
    bool ok = rotate_to_safe_area();

    char buf[128];
    std::snprintf(buf, sizeof(buf), "Rotasi aman selesai: heading %.1f deg, clearance %.2f m.",
                  safe_yaw * 180.0 / M_PI, clearance);
    resp->success = ok;
    resp->message = buf;
    RCLCPP_INFO(get_logger(), "[Rotate Safe] %s", resp->message.c_str());
    IC(resp->success, resp->message);
  }

  // ===========================================================================
  // HELPER: Kirim goal Nav2 ke origin (atau recorded origin jika tersedia).
  // Mengisi out_promise & out_future untuk tracking status Nav2.
  // Mengembalikan true jika goal berhasil dikirim dan diterima Nav2.
  // ===========================================================================
  bool send_homing_goal_to_nav2(
    std::shared_ptr<std::promise<rclcpp_action::ResultCode>> & out_promise,
    std::shared_future<rclcpp_action::ResultCode> & out_future)
  {
    out_promise = std::make_shared<std::promise<rclcpp_action::ResultCode>>();
    out_future  = out_promise->get_future().share();

    NavigateToPose::Goal goal;
    goal.behavior_tree =
      "/opt/ros/jazzy/share/nav2_bt_navigator/behavior_trees/"
      "navigate_to_pose_w_replanning_and_recovery.xml";
    goal.pose.header.frame_id    = frame_map_;
    goal.pose.header.stamp       = now();
    goal.pose.pose.position.x    = origin_recorded_ ? origin_x_ : 0.0;
    goal.pose.pose.position.y    = origin_recorded_ ? origin_y_ : 0.0;
    goal.pose.pose.position.z    = 0.0;
    goal.pose.pose.orientation.x = 0.0;
    goal.pose.pose.orientation.y = 0.0;
    goal.pose.pose.orientation.z = origin_recorded_ ? origin_qz_ : 0.0;
    goal.pose.pose.orientation.w = origin_recorded_ ? origin_qw_ : 1.0;

    auto shared_promise = out_promise;  // capture by value untuk lambda

    auto send_opts = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    send_opts.goal_response_callback =
      [this](const GoalHandle::SharedPtr & handle) {
        if (!handle) {
          RCLCPP_WARN(get_logger(), "[Homing] Nav2 menolak goal homing.");
        } else {
          homing_goal_handle_ = handle;
          RCLCPP_INFO(get_logger(), "[Homing] Nav2 menerima goal homing -> navigasi ke origin...");
        }
      };
    send_opts.result_callback =
      [shared_promise](const GoalHandle::WrappedResult & res) {
        shared_promise->set_value(res.code);
      };

    auto send_future = action_client_->async_send_goal(goal, send_opts);
    if (send_future.wait_for(std::chrono::seconds(4)) != std::future_status::ready ||
        !send_future.get())
    {
      RCLCPP_ERROR(get_logger(), "[Homing] Gagal mengirim goal Nav2.");
      nav_goal_active_ = false;
      return false;
    }
    nav_goal_active_ = true;
    return true;
  }

  // ===========================================================================
  // HELPER: Batalkan goal Nav2 homing yang sedang berjalan (non-blocking).
  // ===========================================================================
  void cancel_homing_nav2_goal()
  {
    nav_goal_active_ = false;
    if (homing_goal_handle_) {
      action_client_->async_cancel_goal(homing_goal_handle_);
      homing_goal_handle_.reset();
      RCLCPP_INFO(get_logger(), "[Homing] Goal Nav2 dibatalkan sementara (antri aktif).");
    }
  }

  // ===========================================================================
  // HELPER: Bersihkan costmap Nav2 (global & local) dari jejak antrian lama
  // agar global planner langsung mendapatkan jalur terpendek dan terbersih
  // ===========================================================================
  void clear_costmaps()
  {
    if (clear_global_costmap_client_ && clear_global_costmap_client_->service_is_ready()) {
      auto req = std::make_shared<nav2_msgs::srv::ClearEntireCostmap::Request>();
      clear_global_costmap_client_->async_send_request(req);
      RCLCPP_INFO(get_logger(), "[Homing Queue] Global costmap dibersihkan untuk kalkulasi path baru.");
    }
    if (clear_local_costmap_client_ && clear_local_costmap_client_->service_is_ready()) {
      auto req = std::make_shared<nav2_msgs::srv::ClearEntireCostmap::Request>();
      clear_local_costmap_client_->async_send_request(req);
      RCLCPP_INFO(get_logger(), "[Homing Queue] Local costmap dibersihkan untuk kalkulasi path baru.");
    }
  }

  // ===========================================================================
  // CALLBACK: Kecepatan dari Nav2 (/nav_cmd_vel)
  // Meneruskan perintah Nav2 ke base driver (/cmd_vel) dengan kontrol orientasi hadap antrian
  // ===========================================================================
  void on_nav_cmd_vel(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    if (!msg) return;

    {
      std::lock_guard<std::mutex> lock(cmd_vel_mon_mutex_);
      monitored_cmd_vel_ = *msg;
    }

    if (is_following_) {
      return;
    }

    // Jika sedang homing aktif dan Nav2 aktif:
    if (is_homing_active_ && nav_goal_active_) {
      // Jika robot sedang melakukan sweep search in-place (kiri/kanan 20 deg):
      // tahan perintah Nav2 agar rotasi sweep pencarian antrian tidak di-override
      if (is_queue_search_active_) {
        return;
      }

      geometry_msgs::msg::Twist out = *msg;
      // Jika antrian terdeteksi tapi TERHALANG OBSTACLE (is_queue_facing_active_):
      // Sesuai permintaan: "saat sedang dalam fungsi antri, tetap utamakan avoidance obstacle dari nav2"
      // Teruskan translasi (linear.x, linear.y) dari Nav2 agar avoidance obstacle Nav2 berjalan penuh,
      // sambil mempertahankan orientasi haluan (angular.z) tertuju ke antrian!
      if (is_queue_facing_active_) {
        std::lock_guard<std::mutex> lock(queue_facing_mutex_);
        out.angular.z = queue_facing_wz_;
        cmd_vel_pub_->publish(out);
        return;
      }

      // Jika jalur bersih dan sedang dalam mode antri manual:
      // tahan perintah Nav2 agar kecepatan antri manual yang memegang kendali
      if (is_homing_queuing_) {
        return;
      }

      // Passthrough normal untuk Nav2 homing
      cmd_vel_pub_->publish(out);
      return;
    }

    // Default passthrough untuk navigasi umum selain homing
    if (!is_homing_queuing_) {
      cmd_vel_pub_->publish(*msg);
    }
  }

  // ===========================================================================
  // HELPER: Cek apakah jalur garis lurus dari robot ke target antrian terhalang obstacle
  // menggunakan data laser scan terkini.
  // ===========================================================================
  bool is_queue_path_blocked(double target_x, double target_y)
  {
    sensor_msgs::msg::LaserScan::SharedPtr scan;
    {
      std::lock_guard<std::mutex> lock(scan_mutex_);
      scan = latest_scan_;
    }
    if (!scan || scan->ranges.empty()) {
      return false;
    }

    const double target_dist = std::hypot(target_x, target_y);
    if (target_dist < 0.60) {
      return false; // Jarak sangat dekat, tidak ada rintangan di antaranya
    }

    double laser_tx = 0.0, laser_ty = 0.0, laser_yaw = 3.1415926535;
    try {
      auto tf_laser = tf_buffer_->lookupTransform(
        frame_base_, scan->header.frame_id, tf2::TimePointZero);
      laser_tx = tf_laser.transform.translation.x;
      laser_ty = tf_laser.transform.translation.y;
      const auto & rq = tf_laser.transform.rotation;
      const double siny = 2.0 * (rq.w * rq.z + rq.x * rq.y);
      const double cosy = 1.0 - 2.0 * (rq.y * rq.y + rq.z * rq.z);
      laser_yaw = std::atan2(siny, cosy);
    } catch (const tf2::TransformException &) {
      laser_yaw = 3.1415926535;
    }

    const double ux = target_x / target_dist;
    const double uy = target_y / target_dist;

    const int num_readings = static_cast<int>(scan->ranges.size());
    int blocked_points = 0;

    for (int i = 0; i < num_readings; ++i) {
      const float r = scan->ranges[i];
      if (!std::isfinite(r) || r < scan->range_min || r > scan->range_max) continue;
      const double angle_laser = scan->angle_min + i * scan->angle_increment;
      double angle_base = angle_laser + laser_yaw;
      angle_base = std::atan2(std::sin(angle_base), std::cos(angle_base));
      const double obs_x = laser_tx + r * std::cos(angle_base);
      const double obs_y = laser_ty + r * std::sin(angle_base);

      const double proj = obs_x * ux + obs_y * uy;
      const double perp = std::abs(-obs_x * uy + obs_y * ux);

      // Titik dihitung memblokir jika:
      // 1. Berada di luar radius fisik robot (proj > 0.28m)
      // 2. Berada SEBELUM tubuh orang target (proj < target_dist - 0.45m),
      //    agar tubuh/kaki orang antrian itu sendiri TIDAK memblokir dirinya!
      // 3. Masuk dalam koridor lintasan robot (perp < 0.35m)
      if (proj > 0.28 && proj < (target_dist - 0.45) && perp < 0.35) {
        blocked_points++;
        if (blocked_points >= 4) {
          return true;
        }
      }
    }

    return false;
  }

  // ===========================================================================
  // HELPER: Cek apakah orang memotong/menghalangi lintasan path homing Nav2
  // Syarat:
  // 1. px_base > 0.30 m dan jarak orang d < homing_queue_max_detect_dist_ (< 3 meter)
  // 2. Jarak tegak lurus orang ke garis lintasan path Nav2 <= corridor_tolerance
  // ===========================================================================
  bool is_person_blocking_path(
    double px_base, double py_base,
    double corridor_tolerance,
    double & out_dist_to_path)
  {
    // Dapatkan transform posisi robot di frame map
    double rx = 0.0, ry = 0.0, ryaw = 0.0;
    bool has_tf = false;
    try {
      auto trans = tf_buffer_->lookupTransform(frame_map_, frame_base_, tf2::TimePointZero);
      rx = trans.transform.translation.x;
      ry = trans.transform.translation.y;
      const auto & rq = trans.transform.rotation;
      ryaw = std::atan2(
        2.0 * (rq.w * rq.z + rq.x * rq.y),
        1.0 - 2.0 * (rq.y * rq.y + rq.z * rq.z));
      has_tf = true;
    } catch (...) {}

    nav_msgs::msg::Path::SharedPtr plan;
    {
      std::lock_guard<std::mutex> lock(homing_plan_mutex_);
      plan = latest_homing_plan_;
    }

    // Jika path Nav2 tersedia dan TF valid: hitung jarak orang ke segmen path terdekat di depan robot
    if (plan && plan->poses.size() >= 2 && has_tf) {
      const double cos_yaw = std::cos(ryaw);
      const double sin_yaw = std::sin(ryaw);

      // Cari indeks waypoint path yang paling dekat dengan robot
      size_t start_idx = 0;
      double min_d_robot = 1e9;
      for (size_t i = 0; i < plan->poses.size(); ++i) {
        const double d = std::hypot(
          plan->poses[i].pose.position.x - rx,
          plan->poses[i].pose.position.y - ry);
        if (d < min_d_robot) {
          min_d_robot = d;
          start_idx = i;
        }
      }

      double min_dist_to_path = 1e9;
      double cum_dist = 0.0;
      bool found_segment = false;

      // Telusuri segmen path ke depan robot hingga akumulasi 3.5 meter
      for (size_t i = start_idx; i + 1 < plan->poses.size() && cum_dist < 3.5; ++i) {
        const auto & p1 = plan->poses[i].pose.position;
        const auto & p2 = plan->poses[i + 1].pose.position;

        const double seg_len = std::hypot(p2.x - p1.x, p2.y - p1.y);
        cum_dist += seg_len;

        // Transform koordinat titik path (p1, p2) dari map ke base_link
        const double x1_b =  (p1.x - rx) * cos_yaw + (p1.y - ry) * sin_yaw;
        const double y1_b = -(p1.x - rx) * sin_yaw + (p1.y - ry) * cos_yaw;
        const double x2_b =  (p2.x - rx) * cos_yaw + (p2.y - ry) * sin_yaw;
        const double y2_b = -(p2.x - rx) * sin_yaw + (p2.y - ry) * cos_yaw;

        // Abaikan segmen yang berada sepenuhnya di belakang robot
        if (x1_b < -0.30 && x2_b < -0.30) {
          continue;
        }

        // Hitung jarak tegak lurus dari posisi orang (px_base, py_base) ke segmen garis path
        const double dx = x2_b - x1_b;
        const double dy = y2_b - y1_b;
        const double l2 = dx * dx + dy * dy;
        double dist_seg = 0.0;
        if (l2 < 1e-6) {
          dist_seg = std::hypot(px_base - x1_b, py_base - y1_b);
        } else {
          const double t = std::clamp(((px_base - x1_b) * dx + (py_base - y1_b) * dy) / l2, 0.0, 1.0);
          const double proj_x = x1_b + t * dx;
          const double proj_y = y1_b + t * dy;
          dist_seg = std::hypot(px_base - proj_x, py_base - proj_y);
        }

        if (dist_seg < min_dist_to_path) {
          min_dist_to_path = dist_seg;
          found_segment = true;
        }
      }

      if (found_segment) {
        out_dist_to_path = min_dist_to_path;
        return (min_dist_to_path <= corridor_tolerance);
      }
    }

    // Fallback jika path belum dipublikasikan oleh Nav2:
    // Anggap lintasan adalah koridor lurus di depan robot (|py| <= corridor_tolerance)
    if (px_base > 0.30 && std::abs(py_base) <= corridor_tolerance) {
      out_dist_to_path = std::abs(py_base);
      return true;
    }
    return false;
  }

  // ===========================================================================
  // HANDLER: /homing service
  //
  // Alur kerja:
  //  1. Stop follow jika aktif
  //  2. Kirim goal Nav2 ke origin -> Nav2 mulai navigasi
  //  3. Loop monitoring 20 Hz:
  //       a. Cek Nav2 selesai         -> keluar
  //       b. Cek timeout global       -> keluar
  //       c. Deteksi orang di koridor depan (ZED + LiDAR):
  //            ADA ORANG:
  //              -> PAUSE Nav2 (cancel goal agar tidak override cmd_vel kita)
  //              -> jarak <= queue_distance  : STOP, tunggu orang jalan
  //              -> jarak >  queue_distance  : CRAWL, ikuti antrian pelan
  //              -> selalu koreksi angular untuk menghadap orang
  //            JALUR BERSIH (setelah antri):
  //              -> RESUME Nav2 (kirim ulang goal ke origin)
  // ===========================================================================
  void handle_homming(
    const std::shared_ptr<Trigger::Request> /*req*/,
    std::shared_ptr<Trigger::Response> resp)
  {
    RCLCPP_INFO(get_logger(), "[Homing] Service /homing dipanggil. Memulai navigasi pulang...");

    // 1. Hentikan person following jika sedang aktif
    if (is_following_) {
      auto dummy = std::make_shared<Trigger::Response>();
      handle_stop_follow(dummy);
    }

    // 2. Pastikan Nav2 action server tersedia
    if (!action_client_->wait_for_action_server(std::chrono::seconds(3))) {
      resp->success = false;
      resp->message = "Nav2 'navigate_to_pose' action server not available.";
      RCLCPP_ERROR(get_logger(), "%s", resp->message.c_str());
      return;
    }

    is_homing_active_  = true;
    is_homing_queuing_ = false;
    {
      std::lock_guard<std::mutex> lock(homing_plan_mutex_);
      latest_homing_plan_.reset();
    }

    // 3. Kirim goal pertama ke Nav2
    std::shared_ptr<std::promise<rclcpp_action::ResultCode>> nav_promise;
    std::shared_future<rclcpp_action::ResultCode> nav_future;

    if (!send_homing_goal_to_nav2(nav_promise, nav_future)) {
      is_homing_active_ = false;
      resp->success = false;
      resp->message = "Gagal mengirim goal awal homing ke Nav2.";
      return;
    }

    // -------------------------------------------------------------------------
    // State internal monitoring loop
    //   nav_goal_active  : true  = Nav2 sedang navigasi ke origin
    //                      false = Nav2 di-pause (robot sedang antri manual)
    //   is_homing_queuing_: true = robot sedang antri di belakang orang
    // -------------------------------------------------------------------------
    rclcpp::Rate   rate(20);       // 20 Hz
    const auto     t_start = now();
    bool           nav_finished    = false;
    nav_goal_active_ = true;
    is_queue_facing_active_ = false;
    is_queue_search_active_ = false;
    rclcpp_action::ResultCode nav_result = rclcpp_action::ResultCode::UNKNOWN;
    rclcpp::Time   last_corridor_person_time{0, 0, RCL_ROS_TIME};
    double         last_queue_dist = 1e9;
    double         last_queue_x    = 0.0;
    double         last_queue_y    = 0.0;
    bool           queue_completed = false;
    bool           has_reached_queue_stop = false;
    bool           queue_engaged = false;
    rclcpp::Time   queue_wait_start_time{0, 0, RCL_ROS_TIME};

    // ── Variabel State Machine Pencarian Antrian (Search 20 deg kiri/kanan) ────
    enum class QueueSearchState {
      INACTIVE,
      SWEEP_LEFT,
      HOLD_LEFT,
      SWEEP_RIGHT,
      HOLD_RIGHT,
      RETURN_CENTER
    };

    QueueSearchState search_state = QueueSearchState::INACTIVE;
    double search_center_yaw = 0.0;
    rclcpp::Time search_hold_start_time{0, 0, RCL_ROS_TIME};
    rclcpp::Time last_search_time = now();
    bool has_started_following_path = false;
    const double kSearchSweepAngle = 20.0 * M_PI / 180.0; // 20 derajat (0.349 rad)
    const double kSearchIntervalSec = 5.0; // loop setiap 5 detik sekali

    // Kecepatan gerak antri yang di-smoothing (filter transisi mulus & anti tabrak)
    double queue_cmd_vx_smooth = 0.0;
    double queue_cmd_vy_smooth = 0.0;
    double queue_cmd_wz_smooth = 0.0;

    get_parameter("homing_queue_enabled", homing_queue_enabled_);
    get_parameter("homing_queue_search_enabled", homing_queue_search_enabled_);

    RCLCPP_INFO(
      get_logger(),
      "[Homing] Loop monitoring aktif | queue_enabled=%s | search_enabled=%s | "
      "stop_dist=%.2fm | resume_dist=%.2fm | corridor=%.2fm | max_detect=%.2fm | "
      "wait_dur=%.1fs | crawl=%.2fm/s",
      homing_queue_enabled_ ? "true" : "false",
      homing_queue_search_enabled_ ? "true" : "false",
      homing_queue_distance_, homing_queue_resume_distance_,
      homing_queue_corridor_, homing_queue_max_detect_dist_,
      homing_queue_wait_duration_, homing_queue_crawl_speed_);

    while (rclcpp::ok() && is_homing_active_) {

      // ── A. Cek apakah Nav2 sudah selesai (hanya jika Nav2 aktif) ──────────
      if (nav_goal_active_ &&
          nav_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
      {
        nav_result   = nav_future.get();
        nav_finished = true;
        break;
      }

      // ── B. Cek timeout global ──────────────────────────────────────────────
      if ((now() - t_start).seconds() > homing_timeout_) {
        RCLCPP_WARN(
          get_logger(), "[Homing] Global timeout (%.1f s) tercapai.", homing_timeout_);
        break;
      }

      // ── C. Social Queue Check ──────────────────────────────────────────────
      if (!homing_queue_enabled_) {
        // Jika fungsi antri dimatikan via parameter:
        // Jika sebelumnya sedang antri, batalkan dan lanjutkan navigasi Nav2
        if (is_homing_queuing_ || queue_engaged) {
          is_homing_queuing_ = false;
          queue_engaged      = false;
          queue_completed    = true;
          is_queue_search_active_ = false;
          search_state = QueueSearchState::INACTIVE;
          send_homing_goal_to_nav2(nav_promise, nav_future);
          nav_goal_active_ = true;
        }
        rate.sleep();
        continue;
      }

      // Jika robot sudah selesai antri: hiraukan semua orang yang menghadap
      // belakang maupun berdekatan. Robot fokus penuh di path Nav2 sampai ke origin.
      if (queue_completed) {
        rate.sleep();
        continue;
      }

      // === C.1 Deteksi antrian di path homing (Opsi 4: Multi-Person Queue Chain) ==
      double closest_x    = 1e9;
      double closest_y    = 0.0;
      double closest_dist = 1e9;
      bool   person_in_corridor = false;
      bool   queue_has_back_facing = false;
      size_t detected_queue_size = 0;

      struct QueueCand {
        int id{-1};
        double px_b{0.0};
        double py_b{0.0};
        double cam_x{0.0};
        double cam_y{0.0};
        double dist{0.0};
        bool in_corridor{false};
        double dist_to_path{1e9};
        bool is_back{false};
        double bbox_x{0.0};
        double bbox_y{0.0};
        double bbox_w{0.0};
        double bbox_h{0.0};
        double bbox_cx{0.0};
        double bbox_cy{0.0};
      };
      std::vector<QueueCand> cands;

      // Ambil deteksi orang terkini dari kamera vision
      {
        std::lock_guard<std::mutex> lock(person_mutex_);
        if ((now() - latest_person_time_).seconds() < 1.0) {
          for (const auto & p : latest_persons_) {
            const double px_b = p.x;
            const double py_b = -p.y;
            const double d = std::hypot(px_b, py_b);

            // Syarat 1: Orang berada dalam jarak < homing_queue_max_detect_dist_ dan di depan robot (px_b > 0.10 m)
            if (px_b <= 0.10 || d >= homing_queue_max_detect_dist_) {
              continue;
            }

            QueueCand qc;
            qc.id = p.id;
            qc.px_b = px_b;
            qc.py_b = py_b;
            qc.cam_x = p.x;
            qc.cam_y = p.y;
            qc.dist = d;
            qc.is_back = p.is_back;
            qc.bbox_x = p.bbox_x;
            qc.bbox_y = p.bbox_y;
            qc.bbox_w = p.bbox_w;
            qc.bbox_h = p.bbox_h;
            qc.bbox_cx = p.bbox_x + p.bbox_w * 0.5;
            qc.bbox_cy = p.bbox_y + p.bbox_h * 0.5;

            // Syarat koridor path Nav2
            qc.in_corridor = is_person_blocking_path(px_b, py_b, homing_queue_corridor_, qc.dist_to_path);

            // Cek apakah orang ini terkonfirmasi menghadap belakang via /vision/person_back
            {
              std::lock_guard<std::mutex> b_lock(person_back_mutex_);
              if (has_back_person_ && (now() - latest_back_time_).seconds() < 1.5) {
                if (std::hypot(p.x - latest_back_x_, p.y - latest_back_y_) < 0.85) {
                  qc.is_back = true;
                }
              }
            }

            cands.push_back(qc);
          }
        }
      }

      // ── DETEKSI / TRACKING ANTRIAN ────────────────────────────────────────
      // Aturan antrian (sesuai permintaan user):
      // 1. Inisiasi antrian: butuh >= 2 orang menghadap belakang (back) yang posisinya (x, y) berdekatan.
      // 2. Saat menyesuaikan posisi (queue_engaged): ubah menjadi mode SINGLE BACK saja,
      //    sehingga robot fokus mensejajarkan posisi X dan Y dengan 1 orang terdekat di depannya
      //    tanpa terpengaruh jika orang kedua terhalang/teroklusi di kamera.

      if (queue_engaged) {
        // ── MODE PENYESUAIAN POSISI: SINGLE BACK TRACKING ─────────────────────
        double best_back_dist = 1e9;
        size_t best_back_idx = cands.size();

        for (size_t i = 0; i < cands.size(); ++i) {
          if (cands[i].is_back) {
            if (cands[i].dist < best_back_dist) {
              best_back_dist = cands[i].dist;
              best_back_idx = i;
            }
          }
        }

        // Toleransi jika flag is_back drop sesaat karena sudut atau jarak dekat:
        // tetap kunci orang terdekat di sekitar posisi antri sebelumnya
        if (best_back_idx == cands.size()) {
          for (size_t i = 0; i < cands.size(); ++i) {
            const double err_to_last = std::hypot(cands[i].px_b - last_queue_x, cands[i].py_b - last_queue_y);
            if (err_to_last < 0.85 && cands[i].dist < best_back_dist) {
              best_back_dist = cands[i].dist;
              best_back_idx = i;
            }
          }
        }

        if (best_back_idx < cands.size()) {
          closest_dist = cands[best_back_idx].dist;
          closest_x = cands[best_back_idx].px_b;
          closest_y = cands[best_back_idx].py_b;
          person_in_corridor = true;
          queue_has_back_facing = true;
          detected_queue_size = 1;

          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 1000,
            "\033[1;36m[Homing Queue]\033[0m Menyesuaikan posisi antrian (Single Back Mode): "
            "Target orang antri di (%.2f, %.2f) jarak=%.2fm.",
            closest_x, closest_y, closest_dist);
        } else {
          person_in_corridor = false;
        }

      } else {
        // ── MODE INISIASI: FOKUS HANYA JIKA MENDETEKSI >= 2 PERSON BACK UNIK & BERDEKATAN ───
        // Langkah 1: Kumpulkan semua kandidat yang menghadap belakang (is_back)
        std::vector<size_t> raw_back_indices;
        for (size_t i = 0; i < cands.size(); ++i) {
          if (cands[i].is_back) {
            raw_back_indices.push_back(i);
          }
        }

        // Langkah 2: Deduplikasi orang (mencegah 1 orang terdeteksi ganda dianggap 2 orang)
        // Dua deteksi dianggap orang fisik yang sama jika ID sama atau jarak pusat tubuh < 0.25m
        std::vector<size_t> back_indices;
        for (size_t idx : raw_back_indices) {
          bool is_duplicate = false;
          for (size_t kept : back_indices) {
            const double d_between = std::hypot(
              cands[idx].px_b - cands[kept].px_b,
              cands[idx].py_b - cands[kept].py_b);
            const bool same_id = (cands[idx].id != -1 && cands[idx].id == cands[kept].id);
            if (same_id || d_between < 0.25) {
              is_duplicate = true;
              break;
            }
          }
          if (!is_duplicate) {
            back_indices.push_back(idx);
          }
        }

        if (back_indices.size() < static_cast<size_t>(homing_queue_min_persons_)) {
          // Kurang dari 2 orang unik menghadap belakang — BUKAN antrian, abaikan & lanjutkan homing!
          if (!back_indices.empty()) {
            RCLCPP_INFO_THROTTLE(
              get_logger(), *get_clock(), 2000,
              "\033[1;30m[Homing Queue]\033[0m Terdeteksi %zu orang facing back (< %d orang) — bukan antrian, abaikan & lanjutkan homing.",
              back_indices.size(), homing_queue_min_persons_);
          }
        } else {
          // Terdeteksi >= 2 orang facing back unik: cari klaster orang-orang yang posisinya (x, y) berdekatan!
          const size_t m = back_indices.size();
          std::vector<bool> visited(m, false);
          std::vector<std::vector<size_t>> valid_clusters;

          for (size_t a = 0; a < m; ++a) {
            if (visited[a]) continue;

            std::vector<size_t> cluster;
            std::queue<size_t> q;
            visited[a] = true;
            q.push(a);

            while (!q.empty()) {
              const size_t curr = q.front();
              q.pop();
              cluster.push_back(back_indices[curr]);

              for (size_t b = 0; b < m; ++b) {
                if (visited[b]) continue;
                const size_t idx_curr = back_indices[curr];
                const size_t idx_b = back_indices[b];

                // Jarak euclidean (x, y) dalam koordinat robot / kamera (meter)
                const double dist_3d = std::hypot(
                  cands[idx_curr].px_b - cands[idx_b].px_b,
                  cands[idx_curr].py_b - cands[idx_b].py_b);

                // Dua orang yang berbeda berada dalam jarak antrian (0.25m s/d homing_queue_inter_dist_)
                bool is_close = (dist_3d >= 0.25 && dist_3d <= homing_queue_inter_dist_);

                if (is_close && cands[idx_curr].bbox_w > 0 && cands[idx_b].bbox_w > 0) {
                  const double dx_bbox = std::abs(cands[idx_curr].bbox_cx - cands[idx_b].bbox_cx);
                  if (dx_bbox > 800.0) {
                    is_close = false; // Terlalu jauh di ujung frame citra kamera yang berlawanan
                  }
                }

                if (is_close) {
                  visited[b] = true;
                  q.push(b);
                }
              }
            }

            // Syarat mutlak: sebuah antrian harus memiliki minimal 2 orang unik yang saling berdekatan!
            if (cluster.size() >= static_cast<size_t>(homing_queue_min_persons_)) {
              valid_clusters.push_back(cluster);
            }
          }

          if (valid_clusters.empty()) {
            // Ada 2+ orang facing back, namun posisi (x, y) mereka berjauhan — bukan antrian, abaikan!
            RCLCPP_INFO_THROTTLE(
              get_logger(), *get_clock(), 2000,
              "\033[1;30m[Homing Queue]\033[0m Terdeteksi %zu orang facing back, tetapi posisi (x, y) mereka TIDAK DEKAT (> %.2fm) — bukan antrian, abaikan.",
              back_indices.size(), homing_queue_inter_dist_);
          } else {
            // VALID ANTRIAN DITEMUKAN!
            // Pilih klaster antrian dan ambil orang paling dekat di dalam antrian tersebut sebagai target robot
            double best_cluster_dist = 1e9;
            size_t best_cand_idx = cands.size();
            size_t best_cluster_size = 0;

            for (const auto & cl : valid_clusters) {
              for (const size_t idx : cl) {
                if (cands[idx].dist < best_cluster_dist) {
                  best_cluster_dist = cands[idx].dist;
                  best_cand_idx = idx;
                  best_cluster_size = cl.size();
                }
              }
            }

            if (best_cand_idx < cands.size()) {
              closest_dist = cands[best_cand_idx].dist;
              closest_x = cands[best_cand_idx].px_b;
              closest_y = cands[best_cand_idx].py_b;
              person_in_corridor = true;
              queue_has_back_facing = true;
              detected_queue_size = best_cluster_size;
              queue_engaged = true; // Kunci antrian! Beralih ke mode penyesuaian posisi (Single Back)

              RCLCPP_INFO(
                get_logger(),
                "\033[1;32m[Homing Queue]\033[0m VALID ANTRIAN TERKONFIRMASI (%zu orang berdekatan)! "
                "Mengunci antrian dan beralih ke mode penyesuaian posisi (Single Back). Target: (%.2f, %.2f) jarak=%.2fm.",
                detected_queue_size, closest_x, closest_y, closest_dist);
            }
          }
        }
      }

      // ── LOGIKA PENCARIAN ANTRIAN (SEARCH SWEEP KIRI/KANAN +-20 DEG DI TEMPAT) ───────────
      // Dikontrol oleh parameter: homing_queue_search_enabled (true = search sweep aktif, false = mati)
      if (homing_queue_search_enabled_) {
        // 1. Jika antrian sudah terkonfirmasi saat sedang search: langsung batalkan search!
        if (queue_engaged && search_state != QueueSearchState::INACTIVE) {
          search_state = QueueSearchState::INACTIVE;
          is_queue_search_active_ = false;
          RCLCPP_INFO(
            get_logger(),
            "\033[1;32m[Homing Queue Search]\033[0m ANTRIAN DITEMUKAN SAAT SEARCH! Search dihentikan langsung & masuk fungsi antri.");
        }

        // 2. Deteksi kapan robot sudah selesai menyelaraskan theta pada path & mulai melaju di path
        if (!has_started_following_path && nav_goal_active_ && !queue_engaged && !is_homing_queuing_) {
          double cur_vx = 0.0;
          {
            std::lock_guard<std::mutex> lock(cmd_vel_mon_mutex_);
            cur_vx = monitored_cmd_vel_.linear.x;
          }
          if (cur_vx > 0.05) {
            has_started_following_path = true;
            last_search_time = now();
            RCLCPP_INFO(
              get_logger(),
              "\033[1;35m[Homing Queue Search]\033[0m Robot mulai melaju pada path (theta sejajar). Timer search aktif (interval 5.0s).");
          }
        }

        // 3. Picu pencarian setiap 5 detik sekali jika antrian belum ditemukan
        if (has_started_following_path && !queue_engaged && !queue_completed && !is_homing_queuing_ && nav_goal_active_) {
          if (search_state == QueueSearchState::INACTIVE && (now() - last_search_time).seconds() >= kSearchIntervalSec) {
            search_state = QueueSearchState::SWEEP_LEFT;
            is_queue_search_active_ = true;
            search_center_yaw = get_robot_heading();
            RCLCPP_INFO(
              get_logger(),
              "\033[1;35m[Homing Queue Search]\033[0m Interval 5s tercapai! Memulai sweep search antrian (~20 deg kiri & kanan di tempat)...");
          }
        }

        // 4. Eksekusi gerakan sweep pencarian jika search aktif
        if (search_state != QueueSearchState::INACTIVE) {
          const double cur_yaw = get_robot_heading();
          geometry_msgs::msg::Twist search_twist;

          if (search_state == QueueSearchState::SWEEP_LEFT) {
            const double target_left = normalize_angle(search_center_yaw + kSearchSweepAngle);
            const double err = normalize_angle(target_left - cur_yaw);
            if (std::abs(err) <= 0.05) {
              search_state = QueueSearchState::HOLD_LEFT;
              search_hold_start_time = now();
              cmd_vel_pub_->publish(search_twist); // stop sejenak
            } else {
              double wz = std::clamp(1.20 * err, -0.32, 0.32);
              if (std::abs(wz) < 0.12) wz = (err > 0.0) ? 0.12 : -0.12;
              search_twist.angular.z = wz;
              cmd_vel_pub_->publish(search_twist);
            }
            rate.sleep();
            continue;

          } else if (search_state == QueueSearchState::HOLD_LEFT) {
            cmd_vel_pub_->publish(search_twist); // diam sejenak untuk frame kamera
            if ((now() - search_hold_start_time).seconds() >= 0.25) {
              search_state = QueueSearchState::SWEEP_RIGHT;
            }
            rate.sleep();
            continue;

          } else if (search_state == QueueSearchState::SWEEP_RIGHT) {
            const double target_right = normalize_angle(search_center_yaw - kSearchSweepAngle);
            const double err = normalize_angle(target_right - cur_yaw);
            if (std::abs(err) <= 0.05) {
              search_state = QueueSearchState::HOLD_RIGHT;
              search_hold_start_time = now();
              cmd_vel_pub_->publish(search_twist); // stop sejenak
            } else {
              double wz = std::clamp(1.20 * err, -0.32, 0.32);
              if (std::abs(wz) < 0.12) wz = (err > 0.0) ? 0.12 : -0.12;
              search_twist.angular.z = wz;
              cmd_vel_pub_->publish(search_twist);
            }
            rate.sleep();
            continue;

          } else if (search_state == QueueSearchState::HOLD_RIGHT) {
            cmd_vel_pub_->publish(search_twist); // diam sejenak
            if ((now() - search_hold_start_time).seconds() >= 0.25) {
              search_state = QueueSearchState::RETURN_CENTER;
            }
            rate.sleep();
            continue;

          } else if (search_state == QueueSearchState::RETURN_CENTER) {
            const double err = normalize_angle(search_center_yaw - cur_yaw);
            if (std::abs(err) <= 0.06) {
              cmd_vel_pub_->publish(search_twist); // stop
              search_state = QueueSearchState::INACTIVE;
              is_queue_search_active_ = false;
              last_search_time = now();
              RCLCPP_INFO(
                get_logger(),
                "\033[1;35m[Homing Queue Search]\033[0m Sweep search selesai (belum ada antrian). Melanjutkan path Nav2.");
            } else {
              double wz = std::clamp(1.20 * err, -0.32, 0.32);
              if (std::abs(wz) < 0.12) wz = (err > 0.0) ? 0.12 : -0.12;
              search_twist.angular.z = wz;
              cmd_vel_pub_->publish(search_twist);
            }
            rate.sleep();
            continue;
          }
        }
      } else {
        // Jika parameter homing_queue_search_enabled bernilai false:
        // Pastikan gerakan sweep tidak pernah aktif
        if (search_state != QueueSearchState::INACTIVE) {
          search_state = QueueSearchState::INACTIVE;
          is_queue_search_active_ = false;
        }
      }

      // Update timestamp dan posisi terakhir jika orang terdeteksi menghalangi path
      if (person_in_corridor) {
        last_corridor_person_time = now();
        last_queue_dist = closest_dist;
        last_queue_x    = closest_x;
        last_queue_y    = closest_y;
      }

      // Hysteresis: tetap antri jika orang baru saja terdeteksi (< homing_queue_clear_hysteresis_ detik)
      // untuk mencegah goyang/flicker jika kamera drop 1 frame
      const bool recent_person =
        person_in_corridor ||
        (is_homing_queuing_ && (now() - last_corridor_person_time).seconds() < homing_queue_clear_hysteresis_);

      const double target_dist = person_in_corridor ? closest_dist : last_queue_dist;
      const double target_x    = person_in_corridor ? closest_x    : last_queue_x;
      const double target_y    = person_in_corridor ? closest_y    : last_queue_y;

      // === C.2 Keputusan: antri atau lanjutkan navigasi ======================
      if (recent_person && target_dist < 1e8) {
        // Koreksi angular: selalu menghadap dan menyelaraskan heading sejajar ke orang terdekat di antrian
        const double yaw_err = std::atan2(target_y, target_x);
        double wz_corr = 0.0;
        const double deadband_yaw = has_reached_queue_stop ? 0.08 : 0.03;
        if (std::abs(yaw_err) > deadband_yaw) {
          wz_corr = std::clamp(1.20 * yaw_err, -0.45, 0.45);
          if (std::abs(wz_corr) < 0.12) {
            wz_corr = (yaw_err > 0.0) ? 0.12 : -0.12;
          }
        }

        // ── LANGSUNG MASUK KE FUNGSI ANTRI (Pause Nav2 seketika) ──────────────
        // Sesuai permintaan user: saat mendeteksi multi back, langsung masuk ke fungsi antri
        is_queue_facing_active_ = false;
        if (nav_goal_active_) {
          cancel_homing_nav2_goal();
          nav_goal_active_   = false;
          is_homing_queuing_ = true;

          // Hentikan robot seketika dari navigasi Nav2
          geometry_msgs::msg::Twist stop_twist;
          cmd_vel_pub_->publish(stop_twist);

          RCLCPP_INFO(
            get_logger(),
            "\033[1;32m[Homing Queue]\033[0m LANGSUNG MASUK KE FUNGSI ANTRI: Nav2 di-pause! "
            "Robot masuk/mengikuti antrian (%zu orang%s): target di (%.2f, %.2f) jarak=%.2fm.",
            detected_queue_size,
            queue_has_back_facing ? " [ORANG MENGHADAP BELAKANG]" : "",
            target_x, target_y, target_dist);
        } else {
          is_homing_queuing_ = true;
        }

        // ── PENSEJAJARAN POSISI LATERAL (Y) DENGAN ORANG TERDEKAT ─────────────
        // target_y > 0 berarti orang berada di sebelah kiri robot -> robot geser kiri (+vy)
        // target_y < 0 berarti orang berada di sebelah kanan robot -> robot geser kanan (-vy)
        double vy_corr = 0.0;
        const double lateral_err = target_y;
        if (std::abs(lateral_err) > 0.04) {
          vy_corr = std::clamp(0.65 * lateral_err, -0.16, 0.16);
          if (std::abs(vy_corr) < 0.05) {
            vy_corr = (lateral_err > 0.0) ? 0.05 : -0.05;
          }
        }

        geometry_msgs::msg::Twist queue_cmd;
        queue_cmd.linear.y  = vy_corr;
        queue_cmd.angular.z = wz_corr;

        const double dist_err = target_dist - homing_queue_distance_;
        const bool is_lateral_aligned = (std::abs(lateral_err) <= 0.08);

        // ── JIKA POSE Y SUDAH SEJAJAR: LANJUT MENGIKUTI ANTRIAN ──────────────
        if (is_lateral_aligned) {
          // Pose Y sudah sejajar: fokus MENGIKUTI ANTRIAN secara dinamis.
          // Jika orang di depan maju, robot ikut maju. Jika orang berhenti, robot standby di jarak 1.0 m.
          has_reached_queue_stop = true;

          if (target_dist > homing_queue_distance_ + 0.08) {
            // Orang di depan bergerak maju -> robot ikut merayap maju mengikuti antrian
            double vx = std::clamp(0.40 * dist_err, homing_queue_crawl_speed_, 0.22);
            vx *= std::max(0.15, std::cos(yaw_err));
            queue_cmd.linear.x = vx;

            RCLCPP_INFO_THROTTLE(
              get_logger(), *get_clock(), 800,
              "\033[1;32m[Homing Queue]\033[0m MENGIKUTI ANTRIAN (Y Sejajar) — Orang maju, robot ikut maju | "
              "jarak: %.2fm (target: %.2fm) | Y=%+.2fm | vx=%.2fm/s | vy=%+.2fm/s | wz=%.2f rad/s",
              target_dist, homing_queue_distance_, lateral_err, queue_cmd.linear.x, queue_cmd.linear.y, wz_corr);

            std_msgs::msg::String st; st.data = "HOMING_QUEUING_FOLLOWING";
            follow_status_pub_->publish(st);

          } else if (target_dist < 0.70) {
            // Terlalu dekat (< 0.70m): mundur perlahan untuk menjaga jarak antri aman
            queue_cmd.linear.x = -0.06;
          } else {
            // Berada tepat di jarak antri 1.0m (0.92m - 1.08m): robot standby dalam antrian
            queue_cmd.linear.x = 0.0;

            RCLCPP_INFO_THROTTLE(
              get_logger(), *get_clock(), 1000,
              "\033[1;33m[Homing Queue]\033[0m STANDBY DALAM ANTRIAN (Y Sejajar) — Menunggu giliran antrian | "
              "jarak: %.2fm / 1.0m | Y=%+.2fm | wz=%.2f rad/s",
              target_dist, lateral_err, wz_corr);

            std_msgs::msg::String st; st.data = "HOMING_QUEUING_STANDBY";
            follow_status_pub_->publish(st);
          }

        } else {
          // ── POSE Y BELUM SEJAJAR: PRIORITASKAN MENSEJAJARKAN SUMBU Y ────────
          if (std::abs(yaw_err) > 0.45) {
            // Jika sudut hadap masih melenceng > 26 derajat, rotasi di tempat dulu agar moncong menghadap orang
            queue_cmd.linear.x = 0.0;
            queue_cmd.linear.y = 0.0;
          } else {
            // Geser lateral untuk menyamakan Y sekaligus merayap ke jarak 1.0m
            if (target_dist > homing_queue_distance_ + 0.10) {
              double vx = std::clamp(0.40 * dist_err, homing_queue_crawl_speed_, 0.20);
              vx *= std::max(0.15, std::cos(yaw_err));
              queue_cmd.linear.x = vx;
            } else if (target_dist < homing_queue_distance_ - 0.15) {
              queue_cmd.linear.x = -0.06;
            } else {
              queue_cmd.linear.x = 0.0;
            }
            queue_cmd.linear.y = vy_corr;
          }

          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 800,
            "\033[1;36m[Homing Queue]\033[0m MENSEJAJARKAN POSE Y — X=%.2fm, Y=%+.2fm | "
            "yaw_err=%+.1f deg | vx=%.2fm/s | vy=%+.2fm/s | wz=%.2f rad/s",
            target_x, lateral_err, yaw_err * 180.0 / M_PI, queue_cmd.linear.x, queue_cmd.linear.y, wz_corr);

          std_msgs::msg::String st; st.data = "HOMING_QUEUING_ALIGNING_Y";
          follow_status_pub_->publish(st);
        }

        // =====================================================================
        // ROBUST LIDAR OBSTACLE & WALL AVOIDANCE (PRIORITAS UTAMA SELAMA ANTRI)
        // Sesuai permintaan user:
        // "saat menjalankan fungsi antri pastikan avoidance selalu didahulukan,
        //  walau sedang mencoba antri jika ada obst yang dekat maka utamakan avoidance,
        //  ini dia nabrak trus padahal pas follow person aman"
        // =====================================================================
        double target_vx = queue_cmd.linear.x;
        double target_vy = queue_cmd.linear.y;
        double target_wz = wz_corr;
        bool queue_obstacle_detected = false;

        if (enable_avoidance_) {
          sensor_msgs::msg::LaserScan::SharedPtr scan;
          {
            std::lock_guard<std::mutex> lock(scan_mutex_);
            scan = latest_scan_;
          }

          if (scan && !scan->ranges.empty()) {
            double laser_tx = 0.0;
            double laser_ty = 0.0;
            double laser_yaw = 3.1415926535;
            try {
              auto tf_laser = tf_buffer_->lookupTransform(
                frame_base_, scan->header.frame_id, tf2::TimePointZero);
              laser_tx = tf_laser.transform.translation.x;
              laser_ty = tf_laser.transform.translation.y;
              const auto & rq = tf_laser.transform.rotation;
              const double siny = 2.0 * (rq.w * rq.z + rq.x * rq.y);
              const double cosy = 1.0 - 2.0 * (rq.y * rq.y + rq.z * rq.z);
              laser_yaw = std::atan2(siny, cosy);
            } catch (const tf2::TransformException &) {
              laser_yaw = 3.1415926535;
            }

            double rep_y = 0.0;
            double min_front_x = 999.0;
            double left_clearance = 0.0;
            double right_clearance = 0.0;
            int left_count = 0;
            int right_count = 0;
            double min_flank_left = 999.0;
            double min_flank_right = 999.0;

            const int num_readings = static_cast<int>(scan->ranges.size());
            for (int i = 0; i < num_readings; ++i) {
              const float r = scan->ranges[i];
              if (!std::isfinite(r) || r < scan->range_min || r > scan->range_max) {
                continue;
              }

              // Hitung sudut beam di frame robot (base_link)
              const double angle_laser = scan->angle_min + i * scan->angle_increment;
              double angle_base = angle_laser + laser_yaw;
              angle_base = std::atan2(std::sin(angle_base), std::cos(angle_base));

              // Titik koordinat obstacle di base_link
              const double obs_x = laser_tx + r * std::cos(angle_base);
              const double obs_y = laser_ty + r * std::sin(angle_base);

              // 1. FILTER FOOTPRINT ROBOT: Abaikan pantulan fisik bodi robot sendiri
              if (std::hypot(obs_x, obs_y) < footprint_radius_) {
                continue;
              }

              // 2. Evaluasi jarak sisi samping (|obs_x| < footprint_radius_ + 0.20m)
              // Mencegah robot menggeser lateral dan menyenggol dinding/meja di samping bodi
              if (std::abs(obs_x) < (footprint_radius_ + 0.20)) {
                if (obs_y > 0.0 && obs_y < min_flank_left) {
                  min_flank_left = obs_y;
                } else if (obs_y < 0.0 && -obs_y < min_flank_right) {
                  min_flank_right = -obs_y;
                }
              }

              // 3. Evaluasi clearance sektor depan robot (-75 deg s.d. +75 deg)
              if (angle_base < -1.30 || angle_base > 1.30) {
                continue;
              }

              if (angle_base > 0.15 && angle_base < 1.20) {
                left_clearance += std::min(r, 2.0f);
                left_count++;
              } else if (angle_base < -0.15 && angle_base > -1.20) {
                right_clearance += std::min(r, 2.0f);
                right_count++;
              }

              // 4. FILTER ORANG TARGET ANTRIAN:
              // Jangan perlakukan tubuh orang yang diantri sebagai rintangan asing
              if (std::abs(angle_base - yaw_err) < 0.30 && std::abs(r - target_dist) < 0.40) {
                continue;
              }

              // 5. KORIDOR BAHAYA DI DEPAN ROBOT:
              // Rintangan lain (orang lain di depan, dinding, tiang, meja) di koridor depan
              if (obs_x > footprint_radius_ && obs_x < obstacle_distance_ && std::abs(obs_y) < corridor_width_) {
                queue_obstacle_detected = true;
                if (obs_x < min_front_x) {
                  min_front_x = obs_x;
                }
              }

              // Gaya tolak lateral repulsive (Y)
              if (obs_x > footprint_radius_ * 0.8 && r < obstacle_distance_) {
                const double weight = (1.0 / std::max(0.20f, r)) - (1.0 / obstacle_distance_);
                rep_y -= weight * std::sin(angle_base);
              }
            }

            // --- PROTEKSI MUTLAK JARAK DEPAN (TERMASUK TERHADAP ORANG DEPAN) ---
            // Jika ada rintangan di depan <= min_stop_x_ (0.48m):
            // STOP MAJU MUTLAK!
            if (min_front_x <= min_stop_x_ || target_dist <= min_stop_x_) {
              target_vx = 0.0;
              if (min_front_x < (footprint_radius_ + 0.12) || target_dist < (footprint_radius_ + 0.15)) {
                target_vx = -0.06; // Terlalu mepet: mundur perlahan demi keselamatan
              }
            }

            // --- PROTEKSI GESER LATERAL KE DINDING / OBSTACLE SAMPING ---
            if (min_flank_left < (corridor_width_ + 0.05) && target_vy > 0.0) {
              target_vy = 0.0; // Dilarang geser kiri jika kiri terhalang
            }
            if (min_flank_right < (corridor_width_ + 0.05) && target_vy < 0.0) {
              target_vy = 0.0; // Dilarang geser kanan jika kanan terhalang
            }

            // --- EKSEKUSI AVOIDANCE UTAMA JIKA ADA OBSTACLE DEPAN ---
            if (queue_obstacle_detected) {
              // 1. Pengereman progresif jika mendekati obstacle
              if (min_front_x <= min_stop_x_) {
                target_vx = 0.0;
              } else if (min_front_x < obstacle_distance_) {
                double brake_ratio = (min_front_x - min_stop_x_) / (obstacle_distance_ - min_stop_x_);
                target_vx *= std::clamp(brake_ratio, 0.0, 1.0);
              }

              // 2. TANGENTIAL STRAFE VIA OMNI (Avoidance ke arah ruang terbuka):
              // Sesuai instruksi: AVOIDANCE SELALU DIDAHULUKAN
              if (std::abs(rep_y) < 0.4) {
                double avg_left = (left_count > 0) ? (left_clearance / left_count) : 0.0;
                double avg_right = (right_count > 0) ? (right_clearance / right_count) : 0.0;
                rep_y = (avg_left >= avg_right) ? 1.5 : -1.5;
              }

              double avoid_vy = std::clamp(0.22 * rep_y, -max_vy_, max_vy_);
              if (avoid_vy > 0.0 && min_flank_left < (corridor_width_ + 0.05)) {
                avoid_vy = 0.0;
              } else if (avoid_vy < 0.0 && min_flank_right < (corridor_width_ + 0.05)) {
                avoid_vy = 0.0;
              }

              target_vy = avoid_vy; // Prioritas mutlak avoidance di atas penyesuaian antrian

              RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 500,
                "\033[1;33m[Homing Queue Avoidance]\033[0m OBSTACLE TERDETEKSI (min_front=%.2fm)! "
                "MENGUTAMAKAN AVOIDANCE: vx=%.2fm/s | vy=%+.2fm/s | wz=%.2f rad/s",
                min_front_x, target_vx, target_vy, target_wz);
            }
          }
        }

        // --- FILTER & SMOOTHING (Persis seperti follow_person agar gerak mulus tanpa sentakan) ---
        target_vx = std::clamp(target_vx, -0.10, max_vx_);
        target_vy = std::clamp(target_vy, -max_vy_, max_vy_);
        target_wz = std::clamp(target_wz, -max_wz_, max_wz_);

        queue_cmd_vx_smooth = (1.0 - alpha_v_) * queue_cmd_vx_smooth + alpha_v_ * target_vx;
        queue_cmd_vy_smooth = (1.0 - alpha_v_) * queue_cmd_vy_smooth + alpha_v_ * target_vy;
        queue_cmd_wz_smooth = (1.0 - alpha_w_) * queue_cmd_wz_smooth + alpha_w_ * target_wz;

        if (std::abs(queue_cmd_vx_smooth) < 0.015) queue_cmd_vx_smooth = 0.0;
        if (std::abs(queue_cmd_vy_smooth) < 0.015) queue_cmd_vy_smooth = 0.0;
        if (std::abs(queue_cmd_wz_smooth) < 0.02)  queue_cmd_wz_smooth = 0.0;

        geometry_msgs::msg::Twist final_queue_cmd;
        final_queue_cmd.linear.x  = queue_cmd_vx_smooth;
        final_queue_cmd.linear.y  = queue_cmd_vy_smooth;
        final_queue_cmd.angular.z = queue_cmd_wz_smooth;
        cmd_vel_pub_->publish(final_queue_cmd);

      } else {
        is_queue_facing_active_ = false;

        // ── ANTRIAN SUDAH TIDAK ADA: KEMBALI / MENUJU POSISI HOMING ───────────
        if (is_homing_queuing_ || queue_engaged) {
          is_homing_queuing_ = false;
          queue_engaged      = false;
          queue_completed    = true;
          has_reached_queue_stop = false;
          last_queue_dist    = 1e9;
          queue_cmd_vx_smooth = 0.0;
          queue_cmd_vy_smooth = 0.0;
          queue_cmd_wz_smooth = 0.0;

          // Hentikan gerak antrian seketika
          geometry_msgs::msg::Twist stop_twist;
          cmd_vel_pub_->publish(stop_twist);

          // Hitung jarak robot saat ini ke titik koordinat homing (origin)
          double dist_to_homing = 1e9;
          double rx = 0.0, ry = 0.0;
          try {
            auto trans = tf_buffer_->lookupTransform(frame_map_, frame_base_, tf2::TimePointZero);
            rx = trans.transform.translation.x;
            ry = trans.transform.translation.y;
            const double ox = origin_recorded_ ? origin_x_ : 0.0;
            const double oy = origin_recorded_ ? origin_y_ : 0.0;
            dist_to_homing = std::hypot(rx - ox, ry - oy);
          } catch (...) {}

          if (dist_to_homing <= 0.50) {
            // KASUS A: Jarak robot SUDAH DEKAT dengan posisi homing (<= 0.50m)
            // Robot dinyatakan sudah tiba di posisi homing!
            RCLCPP_INFO(
              get_logger(),
              "\033[1;32m[Homing Queue]\033[0m ANTRIAN SUDAH TIDAK ADA & Jarak robot sudah dekat dengan posisi homing (%.2fm <= 0.50m). "
              "Robot tiba di posisi homing!",
              dist_to_homing);

            nav_finished = true;
            nav_result = rclcpp_action::ResultCode::SUCCEEDED;
            break;
          } else {
            // KASUS B: Robot belum di posisi homing -> buat rute baru & update path ke titik homing
            // Sesuai permintaan user:
            // "buat path ulang ke titik homing, karena bisa saja setelah sampai akhir antrian,
            //  robot dapat path yang lebih dekat, jadi update path homing setelah selesai antri"
            RCLCPP_INFO(
              get_logger(),
              "\033[1;32m[Homing Queue]\033[0m ANTRIAN SELESAI / SUDAH TIDAK ADA! "
              "Membuat path baru ke titik homing dari posisi akhir antrian (%.2f, %.2f) -> jarak ke homing: %.2fm.",
              rx, ry, dist_to_homing);

            // 1. Bersihkan costmap dari jejak rintangan orang antrian yang lama agar path lurus bersih
            clear_costmaps();

            // 2. Reset plan lama agar path homing benar-benar baru
            {
              std::lock_guard<std::mutex> lock(homing_plan_mutex_);
              latest_homing_plan_.reset();
            }

            // 3. Kirim goal baru ke Nav2 untuk mengkalkulasi rute terpendek dari posisi saat ini
            if (!send_homing_goal_to_nav2(nav_promise, nav_future)) {
              RCLCPP_ERROR(get_logger(), "[Homing] Gagal mengirim goal Nav2 baru ke posisi homing!");
              break;
            }
            nav_goal_active_ = true;

            std_msgs::msg::String st; st.data = "HOMING_PATH_UPDATED_AFTER_QUEUE";
            follow_status_pub_->publish(st);
          }
        }
      }

      rate.sleep();
    }  // end while

    // ── Cleanup ───────────────────────────────────────────────────────────────
    is_homing_active_       = false;
    is_homing_queuing_      = false;
    nav_goal_active_        = false;
    is_queue_facing_active_ = false;
    is_queue_search_active_ = false;
    {
      std::lock_guard<std::mutex> lock(queue_facing_mutex_);
      queue_facing_wz_ = 0.0;
    }
    {
      std::lock_guard<std::mutex> lock(homing_plan_mutex_);
      latest_homing_plan_.reset();
    }

    // Pastikan robot berhenti sepenuhnya setelah selesai
    geometry_msgs::msg::Twist stop_twist;
    cmd_vel_pub_->publish(stop_twist);

    if (nav_finished) {
      if (nav_result == rclcpp_action::ResultCode::SUCCEEDED) {
        resp->success = true;
        resp->message = "Robot tiba di posisi awal (origin).";
        RCLCPP_INFO(get_logger(), "[Homing] %s", resp->message.c_str());
      } else {
        resp->success = false;
        resp->message = "Navigasi homing dibatalkan atau gagal.";
        RCLCPP_WARN(get_logger(), "[Homing] %s", resp->message.c_str());
      }
    } else {
      resp->success = false;
      resp->message = "Navigasi homing timeout.";
      cancel_homing_nav2_goal();
    }
    IC(resp->success, resp->message);
  }

    std::string person_topic_;
  std::string gesture_topic_;
  std::string bag_topic_;
  std::string yaw_topic_;
  std::string cmd_vel_topic_;
  std::string path_topic_;
  std::string goal_update_topic_;
  std::string action_name_;
  std::string frame_map_;
  std::string frame_base_;
  int target_index_{0};
  double stop_distance_{0.5};
  double start_distance_{0.7};
  double bag_stop_distance_{0.45};
  double max_range_{3.0};
  double max_lost_time_{3.0};
  double occlusion_timeout_{1.5};
  std::string bt_xml_;

  double min_vx_{0.08};
  double max_vx_{0.32};
  double max_vy_{0.22};
  double max_wz_{0.40};
  double occlusion_max_vx_{0.22};
  double kp_dist_{0.45};
  double kp_yaw_{0.75};
  double kd_yaw_{0.15};
  double deadband_moving_{0.04};
  double deadband_stop_{0.12};

  double alpha_v_{0.35};
  double alpha_w_{0.40};
  double bearing_beta_{0.35};

  bool enable_avoidance_{true};
  double obstacle_distance_{0.85};
  double footprint_radius_{0.40};
  double min_stop_x_{0.48};
  double corridor_width_{0.38};
  double bypass_theta_offset_deg_{10.0};
  double peeking_avoid_dist_{1.10};
  double peeking_strafe_speed_{0.22};
  double peeking_timeout_{4.0};

  bool is_peeking_avoidance_{false};
  bool peeking_dodge_left_{true};
  double peeking_start_yaw_{0.0};
  double peeking_target_yaw_{0.0};
  rclcpp::Time peeking_start_time_{0, 0, RCL_ROS_TIME};

  double kf_process_noise_{1.2};
  double kf_measurement_noise_{0.15};
  help_me_carry::PersonKalmanFilter kf_{1.2, 0.15};
  bool enable_lidar_fusion_{true};
  double kf_lidar_noise_{0.25};
  double lidar_leg_gate_{0.90};

  int locked_person_id_{-1};
  std::atomic<bool> is_following_{false};
  std::atomic<bool> is_nav_active_{false};
  bool is_moving_{false};
  double current_cmd_vx_{0.0};
  double current_cmd_vy_{0.0};
  double current_cmd_wz_{0.0};
  double last_seen_sec_{0.0};
  rclcpp::Time last_goal_pub_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_nav_fail_time_{0, 0, RCL_ROS_TIME};

  std::mutex scan_mutex_;
  sensor_msgs::msg::LaserScan::SharedPtr latest_scan_;
  std::mutex bag_mutex_;
  std::string latest_gesture_{"NONE"};
  std::vector<Bag> latest_bags_;
  double current_yaw_deg_{0.0};
  rclcpp::Time latest_bag_time_{0, 0, RCL_ROS_TIME};

  std::mutex person_mutex_;
  std::vector<Person> latest_persons_;
  rclcpp::Time latest_person_time_{0, 0, RCL_ROS_TIME};

  double step_back_person_distance_{2.0};
  double step_back_speed_{0.15};
  double step_back_timeout_{15.0};

  rclcpp::Time last_filter_time_;
  bool has_last_filter_time_{false};

  double prev_bearing_{0.0};
  double filtered_bearing_{0.0};
  bool has_prev_bearing_{false};

  bool origin_recorded_{false};
  double origin_x_{0.0};
  double origin_y_{0.0};
  double origin_qz_{0.0};
  double origin_qw_{1.0};

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp_action::Client<NavigateToPose>::SharedPtr action_client_;
  GoalHandle::SharedPtr goal_handle_;
  GoalHandle::SharedPtr homing_goal_handle_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_update_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr follow_status_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_monitor_sub_;
  std::mutex cmd_vel_mon_mutex_;
  geometry_msgs::msg::Twist monitored_cmd_vel_{};
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr person_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr gesture_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr bag_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr yaw_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  std::string plan_topic_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr plan_sub_;
  std::mutex homing_plan_mutex_;
  nav_msgs::msg::Path::SharedPtr latest_homing_plan_;
  rclcpp::Service<Trigger>::SharedPtr start_srv_;
  rclcpp::Service<Trigger>::SharedPtr start_follow_srv_alias_;
  rclcpp::Service<Trigger>::SharedPtr stop_srv_;
  rclcpp::Service<Trigger>::SharedPtr stop_follow_srv_alias_;
  rclcpp::Service<Trigger>::SharedPtr align_bag_srv_;
  rclcpp::Service<Trigger>::SharedPtr step_back_srv_;
  rclcpp::Service<Trigger>::SharedPtr step_back_srv_alias_;
  rclcpp::Service<Trigger>::SharedPtr rotate_safe_srv_;
  rclcpp::Service<Trigger>::SharedPtr rotate_safe_srv_alias_;
  rclcpp::Service<Trigger>::SharedPtr homing_srv_;
  std::atomic<bool> is_homing_active_{false};
  std::atomic<bool> is_homing_queuing_{false};
  bool homing_queue_enabled_{true};
  bool homing_queue_search_enabled_{true};
  double homing_queue_distance_{1.00};
  double homing_queue_resume_distance_{1.40};
  double homing_queue_corridor_{1.00};
  double homing_queue_max_detect_dist_{3.00};
  int homing_queue_min_persons_{2};
  double homing_queue_inter_dist_{1.80};
  double homing_queue_crawl_speed_{0.10};
  double homing_queue_wait_duration_{5.0};
  double homing_queue_clear_hysteresis_{0.75};
  double homing_timeout_{300.0};
  std::string person_back_topic_{"/vision/person_back"};
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr person_back_sub_;
  std::mutex person_back_mutex_;
  double latest_back_x_{0.0};
  double latest_back_y_{0.0};
  rclcpp::Time latest_back_time_{0, 0, RCL_ROS_TIME};
  bool has_back_person_{false};
  std::string scan_topic_{"/scan"};
  std::string nav_cmd_vel_topic_{"/nav_cmd_vel"};
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr nav_cmd_vel_sub_;
  std::atomic<bool> nav_goal_active_{false};
  std::atomic<bool> is_queue_facing_active_{false};
  std::atomic<bool> is_queue_search_active_{false};
  std::mutex queue_facing_mutex_;
  double queue_facing_wz_{0.0};
  rclcpp::TimerBase::SharedPtr watchdog_timer_;
  rclcpp::CallbackGroup::SharedPtr cb_group_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;
  rclcpp::Client<nav2_msgs::srv::ClearEntireCostmap>::SharedPtr clear_global_costmap_client_;
  rclcpp::Client<nav2_msgs::srv::ClearEntireCostmap>::SharedPtr clear_local_costmap_client_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  std::string node_name = "help_me_carry_node";
  if (argc > 0 && argv[0]) {
    const std::string exe = std::filesystem::path(argv[0]).filename().string();
    if (exe == "tracking_controller" || exe == "follow_controller" || exe == "person_follower") {
      node_name = exe;
    }
  }
  auto node = std::make_shared<HelpMeCarryNode>(node_name);
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
