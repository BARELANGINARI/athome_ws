#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
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
#include "help_me_carry/action/follow_person.hpp"
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
    if (fields.size() >= 3) {
      const auto & f2 = fields[2];
      if (f2.rfind("id=", 0) == 0) {
        try {
          p.id = std::stoi(f2.substr(3));
        } catch (...) {}
      } else {
        try {
          p.id = std::stoi(f2);
        } catch (...) {}
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
  using FollowPerson = help_me_carry::action::FollowPerson;
  using GoalHandleFollowPerson = rclcpp_action::ServerGoalHandle<FollowPerson>;

  explicit HelpMeCarryNode(const std::string & node_name = "help_me_carry_node")
  : Node(node_name)
  {
    // Topics & Frames
    person_topic_ = declare_parameter<std::string>("person_topic", "/zed/person_label");
    gesture_topic_ = declare_parameter<std::string>("gesture_topic", "/zed/gesture");
    bag_topic_ = declare_parameter<std::string>("bag_topic", "/zed/bag_detections");
    yaw_topic_ = declare_parameter<std::string>("yaw_topic", "/zed/yaw_deg");
    cmd_vel_topic_ = declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    path_topic_ = declare_parameter<std::string>("path_topic", "/following_path");
    goal_update_topic_ = declare_parameter<std::string>("goal_update_topic", "/goal_update");
    action_name_ = declare_parameter<std::string>("action_name", "navigate_to_pose");
    frame_map_ = declare_parameter<std::string>("frame_map", "map");
    frame_base_ = declare_parameter<std::string>("frame_base", "base_link");
    target_index_ = declare_parameter<int>("target_index", 0);
    bt_xml_ = declare_parameter<std::string>("follow_bt_xml", "");

    // Distances & Timeouts
    stop_distance_ = declare_parameter<double>("follow_stop_distance", 0.5);
    start_distance_ = declare_parameter<double>("follow_start_distance", 0.7);
    bag_stop_distance_ = declare_parameter<double>("bag_stop_distance", 0.45);
    max_range_ = declare_parameter<double>("max_follow_range", 3.0);
    max_lost_time_ = declare_parameter<double>("max_lost_time", 3.0);
    occlusion_timeout_ = declare_parameter<double>("occlusion_timeout", 3.5);

    // Speed Limits & PID Gains
    min_vx_ = declare_parameter<double>("min_vx", 0.08);
    max_vx_ = declare_parameter<double>("max_vx", 0.32);
    max_vy_ = declare_parameter<double>("max_vy", 0.28);
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
    obstacle_distance_ = declare_parameter<double>("obstacle_distance", 0.75);
    footprint_radius_ = declare_parameter<double>("footprint_radius", 0.24);
    min_stop_x_ = declare_parameter<double>("min_stop_x", 0.38);
    corridor_width_ = declare_parameter<double>("corridor_width", 0.32);

    // Kalman Filter Tuning
    kf_process_noise_ = declare_parameter<double>("kf_process_noise", 1.2);
    kf_measurement_noise_ = declare_parameter<double>("kf_measurement_noise", 0.15);
    kf_ = help_me_carry::PersonKalmanFilter(kf_process_noise_, kf_measurement_noise_);

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

    cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    rclcpp::SubscriptionOptions sub_options;
    sub_options.callback_group = cb_group_;

    person_sub_ = create_subscription<std_msgs::msg::String>(
      person_topic_, rclcpp::SensorDataQoS(),
      [this](const std_msgs::msg::String::SharedPtr msg) {
        on_person_detected(msg);
      },
      sub_options);

    gesture_sub_ = create_subscription<std_msgs::msg::String>(
      gesture_topic_, rclcpp::SensorDataQoS(),
      [this](const std_msgs::msg::String::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(bag_mutex_);
        if (msg->data.find("LEFT") != std::string::npos || msg->data.find("RIGHT") != std::string::npos){
          latest_gesture_ = msg->data;
        }
      },
      sub_options);

    bag_sub_ = create_subscription<std_msgs::msg::String>(
      bag_topic_, rclcpp::SensorDataQoS(),
      [this](const std_msgs::msg::String::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(bag_mutex_);
        latest_bags_ = parse_bags(msg->data);
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
      "/scan", rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(scan_mutex_);
        latest_scan_ = msg;
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

    homing_srv_ = create_service<Trigger>(
      "/homing",
      [this](const std::shared_ptr<Trigger::Request> req, std::shared_ptr<Trigger::Response> resp) {
        handle_homming(req, resp);
      },
      rclcpp::ServicesQoS(),
      cb_group_);

    follow_action_server_ = rclcpp_action::create_server<FollowPerson>(
      this,
      "/follow/person_action",
      std::bind(&HelpMeCarryNode::handle_follow_goal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&HelpMeCarryNode::handle_follow_cancel, this, std::placeholders::_1),
      std::bind(&HelpMeCarryNode::handle_follow_accepted, this, std::placeholders::_1),
      rcl_action_server_get_default_options(),
      cb_group_);

    watchdog_timer_ = create_wall_timer(
      std::chrono::milliseconds(500), [this]() { check_safety_watchdog(); });

    RCLCPP_INFO(
      get_logger(),
      "help_me_carry_node siap! Jarak aman: %.2f m | XML BT: %s",
      stop_distance_, bt_xml_.c_str());
  }

private:
  double get_current_yaw_rad(bool has_map_tf, double ryaw)
  {
    if (has_map_tf) {
      return ryaw;
    }
    std::lock_guard<std::mutex> lock(bag_mutex_);
    return current_yaw_deg_ * M_PI / 180.0;
  }

  void on_person_detected(const std_msgs::msg::String::SharedPtr msg)
  {
    // Cek apakah ada permintaan cancel dari action client
    {
      std::lock_guard<std::mutex> lock(follow_action_mutex_);
      if (active_follow_goal_ && active_follow_goal_->is_canceling()) {
        auto dummy = std::make_shared<Trigger::Response>();
        handle_stop_follow(dummy);
        return;
      }
    }

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

      // 4. Update measurement if target verified
      if (target_it != candidates.end()) {
        kf_.update(target_it->rel_x, target_it->rel_y);
        person_matched = true;
        is_predicting = false;
        rel_x = kf_.x();
        rel_y = kf_.y();
        raw_dist = std::hypot(rel_x, rel_y);
        raw_bearing = std::atan2(rel_y, rel_x);
        last_seen_sec_ = now().seconds();
      }
    }

    // 5. Handle occluded or missing target:
    if (!person_matched) {
      const double elapsed = now().seconds() - last_seen_sec_;
      // DEAD-RECKONING: Track Kalman Filter prediction during short occlusion!
      // Batas diperpanjang jika robot sedang bermanuver menghindari rintangan dekat (<38cm)
      const double effective_timeout = is_blocked_close_ ? 4.5 : occlusion_timeout_;
      if (kf_.is_initialized() && elapsed <= effective_timeout) {
        person_matched = true;
        is_predicting = true;
        rel_x = kf_.x();
        rel_y = kf_.y();
        raw_dist = std::max(0.6, std::hypot(rel_x, rel_y));
        raw_bearing = std::atan2(rel_y, rel_x);
      } else {
        // Target lost beyond timeout: decelerate and stop safely
        is_blocked_close_ = false;
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

    // Update posisi dan sudut terakhir operator sebelum terhalang
    if (!is_blocked_close_) {
      last_known_target_bearing_ = direct_bearing;
      if (has_map_tf) {
        const double target_yaw = ryaw + direct_bearing;
        last_known_target_global_x_ = rx + direct_dist * std::cos(target_yaw);
        last_known_target_global_y_ = ry + direct_dist * std::sin(target_yaw);
        has_known_target_global_ = true;
      }
    }

    // =====================================================================
    // 2. ROBUST LIDAR OBSTACLE & SAFETY CORRIDOR EVALUATION
    // =====================================================================
    bool obstacle_detected = false;
    double min_front_x = 999.0;
    double rep_y = 0.0;
    double left_clearance = 0.0;
    double right_clearance = 0.0;
    int left_count = 0;
    int right_count = 0;

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

        const int num_readings = static_cast<int>(scan->ranges.size());
        for (int i = 0; i < num_readings; ++i) {
          const float r = scan->ranges[i];
          if (!std::isfinite(r) || r < scan->range_min || r > scan->range_max) {
            continue;
          }

          // Hitung sudut sinar di frame base_link
          const double angle_laser = scan->angle_min + i * scan->angle_increment;
          double angle_base = angle_laser + laser_yaw;
          angle_base = std::atan2(std::sin(angle_base), std::cos(angle_base));

          // Koordinat rintangan di base_link
          const double obs_x = laser_tx + r * std::cos(angle_base);
          const double obs_y = laser_ty + r * std::sin(angle_base);

          // 1. FILTER ROBOT FOOTPRINT
          if (std::hypot(obs_x, obs_y) < footprint_radius_) {
            continue;
          }

          // 2. Evaluasi HANYA sektor depan (-75 deg s/d +75 deg)
          if (angle_base < -1.30 || angle_base > 1.30) {
            continue;
          }

          // 3. Evaluasi clearance ruang kosong sisi kiri vs kanan
          if (angle_base > 0.15 && angle_base < 1.20) {
            left_clearance += std::min(r, 2.0f);
            left_count++;
          } else if (angle_base < -0.15 && angle_base > -1.20) {
            right_clearance += std::min(r, 2.0f);
            right_count++;
          }

          // 4. FILTER OPERATOR: Jangan anggap kaki operator sebagai halangan
          if (person_matched && !is_predicting &&
              std::abs(angle_base - direct_bearing) < 0.35 && std::abs(r - direct_dist) < 0.45) {
            continue;
          }

          // 5. Koridor bahaya di depan robot
          if (obs_x > footprint_radius_ && obs_x < obstacle_distance_ && std::abs(obs_y) < corridor_width_) {
            obstacle_detected = true;
            if (obs_x < min_front_x) {
              min_front_x = obs_x;
            }
          }

          // Gaya repulsif lateral (Y) untuk menghindar
          if (obs_x > footprint_radius_ * 0.8 && r < obstacle_distance_) {
            const double weight = (1.0 / std::max(0.20f, r)) - (1.0 / obstacle_distance_);
            rep_y -= weight * std::sin(angle_base);
          }
        }
      }
    }

    // =====================================================================
    // 3. DIRECT MOTION & OBSTACLE AVOIDANCE CONTROL
    // =====================================================================
    // Forward distance hysteresis:
    if (!is_moving_) {
      if (direct_dist > start_distance_) {
        is_moving_ = true;
      }
    } else {
      if (direct_dist < stop_distance_) {
        is_moving_ = false;
      }
    }

    double target_vx = 0.0;
    double target_vy = 0.0;
    double target_wz = 0.0;
    std::string status_str = "STOP";

    // Operator terverifikasi jika person_matched = true dan bukan sedang prediksi Kalman (occlusion)
    const bool operator_found = (person_matched && !is_predicting);

    if (operator_found) {
      // Jika saat manuver belok/menghindar sudah ketemu operator asli,
      // langsung hentikan manuver menghindar dan lanjut ke follow person
      if (is_blocked_close_) {
        RCLCPP_INFO(
          get_logger(),
          "\033[92m[FOLLOW] Operator asli terdeteksi kembali! Menghentikan manuver belok, melanjutkan following.\033[0m");
      }
      is_blocked_close_ = false;
    }

    // Cek apakah ada rintangan/orang yang menghadang terlalu dekat (< min_stop_x_ / 38cm)
    // Manuver belok hanya aktif saat operator sedang terhalang/belum ketemu
    const bool is_close_obstacle = (obstacle_detected && min_front_x <= min_stop_x_ && !operator_found);

    if (is_close_obstacle) {
      // -------------------------------------------------------------------
      // KASUS 1: TERHADANG TERLALU DEKAT (<38cm)
      // Stop gerak maju, strafe ke arah kosong (kiri/kanan).
      // - Kosong kanan -> strafe kanan (-vy) + rotate sedikit kiri (10 deg / +0.1745 rad)
      // - Kosong kiri  -> strafe kiri (+vy)  + rotate sedikit kanan (10 deg / -0.1745 rad)
      // -------------------------------------------------------------------
      if (!is_blocked_close_) {
        is_blocked_close_ = true;

        // Tentukan arah kosong (kiri vs kanan)
        double avg_left = (left_count > 0) ? (left_clearance / left_count) : 0.0;
        double avg_right = (right_count > 0) ? (right_clearance / right_count) : 0.0;
        if (std::abs(rep_y) >= 0.4) {
          blocked_strafe_sign_ = (rep_y > 0.0) ? 1.0 : -1.0;
        } else {
          blocked_strafe_sign_ = (avg_left >= avg_right) ? 1.0 : -1.0;
        }

        // Catat yaw awal saat mulai terhadang
        blocked_start_yaw_rad_ = get_current_yaw_rad(has_map_tf, ryaw);

        // Jika kosong dikanan (strafe_sign < 0): rotate sedikit ke kiri (+10 deg)
        // Jika kosong dikiri  (strafe_sign > 0): rotate sedikit ke kanan (-10 deg)
        const double yaw_offset_rad = (blocked_strafe_sign_ < 0.0)
          ? (10.0 * M_PI / 180.0)
          : (-10.0 * M_PI / 180.0);

        blocked_target_yaw_rad_ = blocked_start_yaw_rad_ + yaw_offset_rad;
        while (blocked_target_yaw_rad_ > M_PI) blocked_target_yaw_rad_ -= 2.0 * M_PI;
        while (blocked_target_yaw_rad_ < -M_PI) blocked_target_yaw_rad_ += 2.0 * M_PI;
      }

      target_vx = 0.0;  // Berhenti maju agar tidak menabrak rintangan

      // Kecepatan menyamping omni untuk keluar dari bayangan halangan
      target_vy = std::clamp(0.24 * blocked_strafe_sign_, -max_vy_, max_vy_);

      // Kontrol rotasi: hanya rotate 10 derajat lalu tahan sudut tersebut
      const double current_yaw = get_current_yaw_rad(has_map_tf, ryaw);
      double yaw_err = blocked_target_yaw_rad_ - current_yaw;
      while (yaw_err > M_PI) yaw_err -= 2.0 * M_PI;
      while (yaw_err < -M_PI) yaw_err += 2.0 * M_PI;

      if (std::abs(yaw_err) < 0.025) {  // Toleransi ~1.4 derajat
        target_wz = 0.0;
      } else {
        // Rotasi halus dibatasi rendah agar pandangan kamera tidak hilang
        target_wz = std::clamp(1.0 * yaw_err, -0.18, 0.18);
      }

      status_str = (blocked_strafe_sign_ < 0.0)
        ? "CLOSE OBSTACLE (<38cm): STRAFE RIGHT + ROTATE LEFT 10 DEG"
        : "CLOSE OBSTACLE (<38cm): STRAFE LEFT + ROTATE RIGHT 10 DEG";
    } else {
      // -------------------------------------------------------------------
      // KASUS 2: NORMAL FOLLOWING & RECOVERY
      // -------------------------------------------------------------------
      is_blocked_close_ = false;

      // A. KONTROL ROTASI HEADING: Mengunci hadap ke operator
      const double deadband = is_moving_ ? deadband_moving_ : deadband_stop_;
      double eff_bearing = 0.0;
      if (direct_bearing > deadband) {
        eff_bearing = direct_bearing - deadband;
      } else if (direct_bearing < -deadband) {
        eff_bearing = direct_bearing + deadband;
      }
      target_wz = std::clamp(kp_yaw_ * eff_bearing + kd_yaw_ * d_bearing, -max_wz_, max_wz_);

      // B. KONTROL TRANSLASI LINEAR
      if (is_moving_) {
        double v_desired = kp_dist_ * (direct_dist - stop_distance_);
        v_desired = std::clamp(v_desired, min_vx_, max_vx_);
        if (is_predicting) {
          v_desired = std::min(v_desired, occlusion_max_vx_);
        }
        target_vx = v_desired * std::max(0.20, std::cos(direct_bearing));
        target_vy = 0.0;
        status_str = is_predicting ? "KALMAN PREDICTING (OCCLUDED)" : "FOLLOWING OPERATOR";

        if (obstacle_detected) {
          // Pengereman progresif saat mendekati rintangan dari jarak 0.75m sampai 0.38m
          double brake_ratio = (min_front_x - min_stop_x_) / (obstacle_distance_ - min_stop_x_);
          target_vx *= std::clamp(brake_ratio, 0.0, 1.0);

          if (std::abs(rep_y) < 0.4) {
            double avg_left = (left_count > 0) ? (left_clearance / left_count) : 0.0;
            double avg_right = (right_count > 0) ? (right_clearance / right_count) : 0.0;
            rep_y = (avg_left >= avg_right) ? 1.5 : -1.5;
          }
          target_vy = std::clamp(0.22 * rep_y, -max_vy_, max_vy_);
          status_str = "AVOIDING WALL/OBSTACLE";
        }
      } else {
        target_vx = 0.0;
        target_vy = 0.0;
        status_str = (std::abs(target_wz) > 0.03) ? "ALIGNING HEADING (STATIONARY)" : "STOPPED (Distance < 0.5m)";
      }
    }

    target_vx = std::clamp(target_vx, 0.0, max_vx_);
    target_vy = std::clamp(target_vy, -max_vy_, max_vy_);

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

    // Kirim feedback berkala ke action client jika aktif
    {
      std::lock_guard<std::mutex> lock(follow_action_mutex_);
      if (active_follow_goal_ && active_follow_goal_->is_active()) {
        auto fb = std::make_shared<FollowPerson::Feedback>();
        fb->current_distance = static_cast<float>(direct_dist);
        fb->current_bearing = static_cast<float>(direct_bearing);
        fb->tracking_status = is_predicting ? "OCCLUDED_PREDICTING" : status_str;
        active_follow_goal_->publish_feedback(fb);
      }
    }

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

  rclcpp_action::GoalResponse handle_follow_goal(
    const rclcpp_action::GoalUUID & /*uuid*/,
    std::shared_ptr<const FollowPerson::Goal> goal)
  {
    RCLCPP_INFO(
      get_logger(),
      "[Action Server] Permintaan follow goal diterima (target_id: %d, distance: %.2f m)",
      goal->target_id, goal->follow_distance);
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_follow_cancel(
    const std::shared_ptr<GoalHandleFollowPerson> goal_handle)
  {
    (void)goal_handle;
    RCLCPP_INFO(get_logger(), "[Action Server] Permintaan cancel goal follow diterima.");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_follow_accepted(
    const std::shared_ptr<GoalHandleFollowPerson> goal_handle)
  {
    std::lock_guard<std::mutex> lock(follow_action_mutex_);
    if (active_follow_goal_ && active_follow_goal_->is_active()) {
      RCLCPP_INFO(get_logger(), "[Action Server] Membatalkan goal follow lama untuk goal baru.");
      auto preempt_res = std::make_shared<FollowPerson::Result>();
      preempt_res->success = false;
      preempt_res->message = "Preempted by a new follow goal.";
      preempt_res->total_time = static_cast<float>((now() - follow_action_start_time_).seconds());
      active_follow_goal_->abort(preempt_res);
    }

    active_follow_goal_ = goal_handle;
    follow_action_start_time_ = now();

    const auto goal = goal_handle->get_goal();
    if (goal->follow_distance > 0.05f) {
      stop_distance_ = goal->follow_distance;
      start_distance_ = goal->follow_distance + 0.20;
      RCLCPP_INFO(
        get_logger(),
        "[Action Server] Jarak follow diperbarui: stop_distance=%.2f m, start_distance=%.2f m",
        stop_distance_, start_distance_);
    }
    if (goal->target_id >= 0) {
      target_index_ = goal->target_id;
    }

    auto resp = std::make_shared<Trigger::Response>();
    handle_start_follow(resp);
  }

  void handle_start_follow(std::shared_ptr<Trigger::Response> resp)
  {
    if (is_following_) {
      resp->success = true;
      resp->message = "Following is already active.";
      return;
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
    is_blocked_close_ = false;
    blocked_start_yaw_rad_ = 0.0;
    blocked_target_yaw_rad_ = 0.0;
    blocked_strafe_sign_ = -1.0;
    last_known_target_bearing_ = 0.0;
    has_known_target_global_ = false;
    last_seen_sec_ = now().seconds();

    resp->success = true;
    resp->message = "Person following active (Start distance: >0.7m, Stop distance: <0.5m).";
    RCLCPP_INFO(get_logger(), "%s", resp->message.c_str());
    IC(resp->success, resp->message);
  }

  void handle_stop_follow(std::shared_ptr<Trigger::Response> resp)
  {
    is_following_ = false;
    is_nav_active_ = false;
    is_moving_ = false;
    current_cmd_vx_ = 0.0;
    current_cmd_vy_ = 0.0;
    current_cmd_wz_ = 0.0;
    locked_person_id_ = -1;
    prev_bearing_ = 0.0;
    filtered_bearing_ = 0.0;
    has_prev_bearing_ = false;
    is_blocked_close_ = false;
    blocked_start_yaw_rad_ = 0.0;
    blocked_target_yaw_rad_ = 0.0;
    blocked_strafe_sign_ = -1.0;
    last_known_target_bearing_ = 0.0;
    has_known_target_global_ = false;
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

    // Selesaikan action goal jika ada yang aktif
    {
      std::lock_guard<std::mutex> lock(follow_action_mutex_);
      if (active_follow_goal_ && active_follow_goal_->is_active()) {
        auto act_res = std::make_shared<FollowPerson::Result>();
        act_res->total_time = static_cast<float>((now() - follow_action_start_time_).seconds());
        if (active_follow_goal_->is_canceling()) {
          act_res->success = true;
          act_res->message = "Following stopped by cancel request.";
          active_follow_goal_->canceled(act_res);
        } else {
          act_res->success = true;
          act_res->message = "Following stopped via service or completion.";
          active_follow_goal_->succeed(act_res);
        }
        active_follow_goal_.reset();
      }
    }

    resp->success = true;
    resp->message = "Person following stopped.";
    RCLCPP_INFO(get_logger(), "%s", resp->message.c_str());
    IC(resp->success, resp->message);
  }

  void check_safety_watchdog()
  {
    // Cek apakah ada permintaan cancel dari action client
    {
      std::lock_guard<std::mutex> lock(follow_action_mutex_);
      if (active_follow_goal_ && active_follow_goal_->is_canceling()) {
        auto dummy = std::make_shared<Trigger::Response>();
        handle_stop_follow(dummy);
        return;
      }
    }

    if (!is_following_ || max_lost_time_ <= 0.0) {
      return;
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

      // Kirim feedback LOST ke action client jika aktif
      std::lock_guard<std::mutex> lock(follow_action_mutex_);
      if (active_follow_goal_ && active_follow_goal_->is_active()) {
        auto fb = std::make_shared<FollowPerson::Feedback>();
        fb->current_distance = -1.0f;
        fb->current_bearing = 0.0f;
        fb->tracking_status = "LOST";
        active_follow_goal_->publish_feedback(fb);
      }
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

    // 4. Calculate rotation angle using vision yaw (/zed/yaw_deg)
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

  void handle_homming(
    const std::shared_ptr<Trigger::Request> /*req*/,
    std::shared_ptr<Trigger::Response> resp)
  {
    RCLCPP_INFO(get_logger(), "[Homing] Received service call /homing...");

    // 1. Stop person following if currently active
    if (is_following_) {
      auto dummy = std::make_shared<Trigger::Response>();
      handle_stop_follow(dummy);
    }

    // 2. Ensure Nav2 action server is available
    if (!action_client_->wait_for_action_server(std::chrono::seconds(3))) {
      resp->success = false;
      resp->message = "Nav2 'navigate_to_pose' action server not available.";
      RCLCPP_ERROR(get_logger(), "%s", resp->message.c_str());
      return;
    }

    // 3. Goal origin point (x=0, y=0, w=1) on map frame with standard BT replanning & recovery
    NavigateToPose::Goal goal;
    goal.behavior_tree = "/opt/ros/jazzy/share/nav2_bt_navigator/behavior_trees/navigate_to_pose_w_replanning_and_recovery.xml";
    goal.pose.header.frame_id = frame_map_;
    goal.pose.header.stamp = now();
    goal.pose.pose.position.x = 0.0;
    goal.pose.pose.position.y = 0.0;
    goal.pose.pose.position.z = 0.0;
    goal.pose.pose.orientation.x = 0.0;
    goal.pose.pose.orientation.y = 0.0;
    goal.pose.pose.orientation.z = 0.0;
    goal.pose.pose.orientation.w = 1.0;

    auto promise = std::make_shared<std::promise<rclcpp_action::ResultCode>>();
    auto future = promise->get_future();

    auto send_opts = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    send_opts.goal_response_callback = [this](const GoalHandle::SharedPtr & handle) {
      if (!handle) {
        RCLCPP_WARN(get_logger(), "[Homing] Nav2 rejected origin goal.");
      } else {
        homing_goal_handle_ = handle;
        RCLCPP_INFO(get_logger(), "[Homing] Nav2 accepted origin goal. Navigating to start position...");
      }
    };
    send_opts.result_callback = [promise](const GoalHandle::WrappedResult & res) {
      promise->set_value(res.code);
    };

    auto send_future = action_client_->async_send_goal(goal, send_opts);
    if (send_future.wait_for(std::chrono::seconds(4)) != std::future_status::ready || !send_future.get()) {
      resp->success = false;
      resp->message = "Failed to send origin goal to Nav2.";
      return;
    }

    // 4. Wait for robot to reach origin
    if (future.wait_for(std::chrono::seconds(120)) == std::future_status::ready) {
      const auto code = future.get();
      if (code == rclcpp_action::ResultCode::SUCCEEDED) {
        resp->success = true;
        resp->message = "Robot arrived at starting position (origin).";
        RCLCPP_INFO(get_logger(), "%s", resp->message.c_str());
      } else {
        resp->success = false;
        resp->message = "Return-to-origin navigation canceled or failed.";
        RCLCPP_WARN(get_logger(), "%s", resp->message.c_str());
      }
    } else {
      resp->success = false;
      resp->message = "Return-to-origin navigation timed out.";
      if (homing_goal_handle_) {
        action_client_->async_cancel_goal(homing_goal_handle_);
      }
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
  double obstacle_distance_{0.75};
  double footprint_radius_{0.24};
  double min_stop_x_{0.38};
  double corridor_width_{0.32};

  bool is_blocked_close_{false};
  double blocked_start_yaw_rad_{0.0};
  double blocked_target_yaw_rad_{0.0};
  double blocked_strafe_sign_{-1.0};
  double last_known_target_bearing_{0.0};
  double last_known_target_global_x_{0.0};
  double last_known_target_global_y_{0.0};
  bool has_known_target_global_{false};

  double kf_process_noise_{1.2};
  double kf_measurement_noise_{0.15};
  help_me_carry::PersonKalmanFilter kf_{1.2, 0.15};

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

  rclcpp::Time last_filter_time_;
  bool has_last_filter_time_{false};

  double prev_bearing_{0.0};
  double filtered_bearing_{0.0};
  bool has_prev_bearing_{false};

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp_action::Client<NavigateToPose>::SharedPtr action_client_;
  GoalHandle::SharedPtr goal_handle_;
  GoalHandle::SharedPtr homing_goal_handle_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_update_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr person_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr gesture_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr bag_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr yaw_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Service<Trigger>::SharedPtr start_srv_;
  rclcpp::Service<Trigger>::SharedPtr start_follow_srv_alias_;
  rclcpp::Service<Trigger>::SharedPtr stop_srv_;
  rclcpp::Service<Trigger>::SharedPtr stop_follow_srv_alias_;
  rclcpp::Service<Trigger>::SharedPtr align_bag_srv_;
  rclcpp::Service<Trigger>::SharedPtr homing_srv_;
  rclcpp_action::Server<FollowPerson>::SharedPtr follow_action_server_;
  std::shared_ptr<GoalHandleFollowPerson> active_follow_goal_;
  std::mutex follow_action_mutex_;
  rclcpp::Time follow_action_start_time_{0, 0, RCL_ROS_TIME};
  rclcpp::TimerBase::SharedPtr watchdog_timer_;
  rclcpp::CallbackGroup::SharedPtr cb_group_;
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
