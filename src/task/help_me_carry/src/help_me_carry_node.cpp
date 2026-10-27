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
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2/time.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

#include "icecream.hpp"

namespace
{
constexpr const char * kNoPerson = "NO_PERSON";

struct Person
{
  int id{-1};
  std::string state{"UNKNOWN"};
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

    if (chunk.rfind("person:", 0) != 0 && chunk.rfind("operator:", 0) != 0) {
      continue;
    }
    const auto fields = split_char(chunk, '|');
    if (fields.size() < 2) {continue;}

    Person p;
    for (std::size_t i = 2; i < fields.size(); ++i) {
      if (fields[i].rfind("id=", 0) == 0) {
        try {
          p.id = std::stoi(fields[i].substr(3));
        } catch (...) {}
      } else if (fields[i].rfind("state=", 0) == 0) {
        p.state = fields[i].substr(6);
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
}

class HelpMeCarryNode : public rclcpp::Node
{
public:
  using NavigateToPose = nav2_msgs::action::NavigateToPose;
  using GoalHandle = rclcpp_action::ClientGoalHandle<NavigateToPose>;
  using Trigger = std_srvs::srv::Trigger;

  explicit HelpMeCarryNode(const std::string & node_name = "help_me_carry_node")
  : Node(node_name)
  {
    person_topic_ = declare_parameter<std::string>("person_topic", "/zed/person_label");
    gesture_topic_ = declare_parameter<std::string>("gesture_topic", "/zed/gesture");
    bag_topic_ = declare_parameter<std::string>("bag_topic", "/zed/bag_detections");
    yaw_topic_ = declare_parameter<std::string>("yaw_topic", "/zed/yaw_deg");
    action_name_ = declare_parameter<std::string>("action_name", "navigate_to_pose");
    frame_map_ = declare_parameter<std::string>("frame_map", "map");
    frame_base_ = declare_parameter<std::string>("frame_base", "base_link");
    stop_distance_ = declare_parameter<double>("follow_stop_distance", 0.5);
    start_distance_ = declare_parameter<double>("follow_start_distance", 0.7);
    max_range_ = declare_parameter<double>("max_follow_range", 3.0);
    max_lost_time_ = declare_parameter<double>("max_lost_time", 3.0);
    enable_avoidance_ = declare_parameter<bool>("enable_avoidance", true);
    obstacle_distance_ = declare_parameter<double>("obstacle_distance", 0.75);
    footprint_radius_ = declare_parameter<double>("footprint_radius", 0.25);
    max_vx_ = declare_parameter<double>("max_vx", 0.32);
    max_vy_ = declare_parameter<double>("max_vy", 0.22);
    max_wz_ = declare_parameter<double>("max_wz", 0.40);
    kp_w_ = declare_parameter<double>("kp_w", 0.75);
    kd_w_ = declare_parameter<double>("kd_w", 0.15);
    min_stop_x_ = declare_parameter<double>("min_stop_distance", 0.38);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_, this, false);
    action_client_ = rclcpp_action::create_client<NavigateToPose>(this, action_name_);

    cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>(
      "/cmd_vel", rclcpp::QoS(10));
    path_pub_ = create_publisher<nav_msgs::msg::Path>(
      "/follow/path", rclcpp::QoS(10));

    person_sub_ = create_subscription<std_msgs::msg::String>(
      person_topic_, rclcpp::SensorDataQoS(),
      [this](const std_msgs::msg::String::SharedPtr msg) {
        on_person_detected(msg);
      });

    gesture_sub_ = create_subscription<std_msgs::msg::String>(
      gesture_topic_, rclcpp::SensorDataQoS(),
      [this](const std_msgs::msg::String::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(bag_mutex_);
        latest_gesture_ = msg->data;
        if (msg->data.find("RIGHT") != std::string::npos || msg->data.find("KANAN") != std::string::npos ||
            msg->data.find("LEFT") != std::string::npos || msg->data.find("KIRI") != std::string::npos) {
          last_valid_gesture_ = msg->data;
          last_gesture_time_ = now();
        }
      });

    bag_sub_ = create_subscription<std_msgs::msg::String>(
      bag_topic_, rclcpp::SensorDataQoS(),
      [this](const std_msgs::msg::String::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(bag_mutex_);
        latest_bags_ = parse_bags(msg->data);
        if (!latest_bags_.empty()) {
          last_valid_bags_ = latest_bags_;
          last_bag_time_ = now();
        }
      });

    yaw_sub_ = create_subscription<std_msgs::msg::Float64>(
      yaw_topic_, rclcpp::SensorDataQoS(),
      [this](const std_msgs::msg::Float64::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(bag_mutex_);
        current_yaw_deg_ = msg->data;
      });

    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(scan_mutex_);
        latest_scan_ = msg;
      });

    start_srv_ = create_service<Trigger>(
      "/follow/person",
      [this](const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> resp) {
        handle_start_follow(resp);
      });

    stop_srv_ = create_service<Trigger>(
      "/follow/stop",
      [this](const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> resp) {
        handle_stop_follow(resp);
      });

    align_bag_srv_ = create_service<Trigger>(
      "/task/align_bag",
      [this](const std::shared_ptr<Trigger::Request> req, std::shared_ptr<Trigger::Response> resp) {
        handle_align_bag(req, resp);
      });

    homing_srv_ = create_service<Trigger>(
      "/homing",
      [this](const std::shared_ptr<Trigger::Request> req, std::shared_ptr<Trigger::Response> resp) {
        handle_homming(req, resp);
      });

    watchdog_timer_ = create_wall_timer(
      std::chrono::milliseconds(500), [this]() { check_safety_watchdog(); });

    IC(stop_distance_);
  }

private:
  void on_person_detected(const std_msgs::msg::String::SharedPtr msg)
  {
    if (!is_following_) {
      return;
    }

    double dt = 0.033;
    if (has_last_filter_time_) {
      dt = (now() - last_filter_time_).seconds();
      if (dt <= 0.0) dt = 0.033;
      if (dt > 0.5) dt = 0.5;
    }
    last_filter_time_ = now();
    has_last_filter_time_ = true;

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
    }

    const auto people = parse_persons(msg->data);
    struct Candidate {
      int id{-1};
      std::string state{"UNKNOWN"};
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

      const double bearing = std::atan2(-person.y, person.x);

      Candidate cand;
      cand.id = person.id;
      cand.state = person.state;
      cand.rel_x = person.x;
      cand.rel_y = -person.y;
      cand.dist = dist;
      cand.bearing = bearing;
      candidates.push_back(cand);
    }

    bool person_matched = false;
    double raw_dist = 0.0;
    double raw_bearing = 0.0;
    double rel_x = 0.0;
    double rel_y = 0.0;

    if (!candidates.empty() && locked_person_id_.load() < 0) {
      auto best = candidates.end();
      for (auto it = candidates.begin(); it != candidates.end(); ++it) {
        if (it->id < 0) {continue;}
        if (best == candidates.end() || it->dist < best->dist) {best = it;}
      }
      if (best != candidates.end()) {
        locked_person_id_.store(best->id);
        IC(best->id, best->dist, best->bearing);
      }
    }

    const int active_id = locked_person_id_.load();
    if (active_id >= 0) {
      auto target_it = std::find_if(
        candidates.begin(), candidates.end(),
        [active_id](const Candidate & c) {return c.id == active_id;});

      if (target_it != candidates.end()) {
        person_matched = true;
        raw_dist = target_it->dist;
        raw_bearing = target_it->bearing;
        rel_x = target_it->rel_x;
        rel_y = target_it->rel_y;
        last_seen_sec_ = now().seconds();
      }
    }

    if (!person_matched) {
      const double elapsed = now().seconds() - last_seen_sec_;
      if (elapsed > 0.35) {
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

        if (elapsed > 1.2) {
          is_moving_ = false;
        }
      }
      return;
    }

    if (!has_prev_bearing_) {
      filtered_bearing_ = raw_bearing;
      prev_bearing_ = raw_bearing;
      has_prev_bearing_ = true;
    } else {
      constexpr double kBeta = 0.35;
      filtered_bearing_ = (1.0 - kBeta) * filtered_bearing_ + kBeta * raw_bearing;
    }

    const double d_bearing = (filtered_bearing_ - prev_bearing_) / dt;
    prev_bearing_ = filtered_bearing_;

    const double direct_dist = raw_dist;
    const double direct_bearing = filtered_bearing_;

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

    const double deadband = is_moving_ ? 0.04 : 0.12;
    double eff_bearing = 0.0;
    if (direct_bearing > deadband) {
      eff_bearing = direct_bearing - deadband;
    } else if (direct_bearing < -deadband) {
      eff_bearing = direct_bearing + deadband;
    }

    target_wz = std::clamp(kp_w_ * eff_bearing + kd_w_ * d_bearing, -max_wz_, max_wz_);

    if (is_moving_) {
      double v_desired = 0.45 * (direct_dist - stop_distance_);
      v_desired = std::clamp(v_desired, 0.08, max_vx_);

      target_vx = v_desired * std::max(0.20, std::cos(direct_bearing));
      target_vy = 0.0;

      bool obstacle_detected = false;
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

          const int num_readings = static_cast<int>(scan->ranges.size());
          for (int i = 0; i < num_readings; ++i) {
            const float r = scan->ranges[i];
            if (!std::isfinite(r) || r < scan->range_min || r > scan->range_max) {
              continue;
            }

            const double angle_laser = scan->angle_min + i * scan->angle_increment;
            double angle_base = angle_laser + laser_yaw;
            angle_base = std::atan2(std::sin(angle_base), std::cos(angle_base));

            const double obs_x = laser_tx + r * std::cos(angle_base);
            const double obs_y = laser_ty + r * std::sin(angle_base);

            if (std::hypot(obs_x, obs_y) < footprint_radius_) {
              continue;
            }

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

            if (std::abs(angle_base - direct_bearing) < 0.30 && std::abs(r - direct_dist) < 0.40) {
              continue;
            }

            if (obs_x > footprint_radius_ && obs_x < obstacle_distance_ && std::abs(obs_y) < 0.32) {
              obstacle_detected = true;
              if (obs_x < min_front_x) {
                min_front_x = obs_x;
              }
            }

            if (obs_x > footprint_radius_ * 0.8 && r < obstacle_distance_) {
              const double weight = (1.0 / std::max(0.20f, r)) - (1.0 / obstacle_distance_);
              rep_y -= weight * std::sin(angle_base);
            }
          }

          if (obstacle_detected) {
            if (min_front_x <= min_stop_x_) {
              target_vx = 0.0;
            } else if (min_front_x < obstacle_distance_) {
              double brake_ratio = (min_front_x - min_stop_x_) / (obstacle_distance_ - min_stop_x_);
              target_vx *= std::clamp(brake_ratio, 0.0, 1.0);
            }

            if (std::abs(rep_y) < 0.4) {
              double avg_left = (left_count > 0) ? (left_clearance / left_count) : 0.0;
              double avg_right = (right_count > 0) ? (right_clearance / right_count) : 0.0;
              rep_y = (avg_left >= avg_right) ? 1.5 : -1.5;
            }

            double avoid_vy = std::clamp(max_vy_ * rep_y, -max_vy_, max_vy_);
            target_vy = avoid_vy;
          }
        }
      }

      target_vx = std::clamp(target_vx, 0.0, max_vx_);
      target_vy = std::clamp(target_vy, -max_vy_, max_vy_);
    } else {
      target_vx = 0.0;
      target_vy = 0.0;
    }

    constexpr double alpha_v = 0.35;
    constexpr double alpha_w = 0.40;

    current_cmd_vx_ = (1.0 - alpha_v) * current_cmd_vx_ + alpha_v * target_vx;
    current_cmd_vy_ = (1.0 - alpha_v) * current_cmd_vy_ + alpha_v * target_vy;
    current_cmd_wz_ = (1.0 - alpha_w) * current_cmd_wz_ + alpha_w * target_wz;

    if (std::abs(current_cmd_vx_) < 0.015) current_cmd_vx_ = 0.0;
    if (std::abs(current_cmd_vy_) < 0.015) current_cmd_vy_ = 0.0;
    if (std::abs(current_cmd_wz_) < 0.02) current_cmd_wz_ = 0.0;

    geometry_msgs::msg::Twist twist;
    twist.linear.x = current_cmd_vx_;
    twist.linear.y = current_cmd_vy_;
    twist.angular.z = current_cmd_wz_;

    cmd_vel_pub_->publish(twist);
  }

  void handle_start_follow(std::shared_ptr<Trigger::Response> resp)
  {
    if (is_following_) {
      resp->success = true;
      resp->message = "Following sudah aktif.";
      return;
    }

    has_last_filter_time_ = false;
    locked_person_id_.store(-1);
    is_following_ = true;
    is_moving_ = false;
    current_cmd_vx_ = 0.0;
    current_cmd_vy_ = 0.0;
    current_cmd_wz_ = 0.0;
    prev_bearing_ = 0.0;
    filtered_bearing_ = 0.0;
    has_prev_bearing_ = false;
    last_seen_sec_ = now().seconds();

    resp->success = true;
    resp->message = "Person following aktif (Jarak Mulai: >0.7m, Berhenti: <0.5m).";
    IC(resp->success, resp->message);
  }

  void handle_stop_follow(std::shared_ptr<Trigger::Response> resp)
  {
    is_following_ = false;
    is_moving_ = false;
    current_cmd_vx_ = 0.0;
    current_cmd_vy_ = 0.0;
    current_cmd_wz_ = 0.0;
    prev_bearing_ = 0.0;
    filtered_bearing_ = 0.0;
    has_prev_bearing_ = false;
    has_last_filter_time_ = false;
    locked_person_id_.store(-1);

    if (homing_goal_handle_) {
      action_client_->async_cancel_goal(homing_goal_handle_);
      homing_goal_handle_.reset();
    }

    geometry_msgs::msg::Twist stop_twist;
    cmd_vel_pub_->publish(stop_twist);

    resp->success = true;
    resp->message = "Person following dihentikan.";
    IC(resp->success, resp->message);
  }

  void check_safety_watchdog()
  {
    if (!is_following_ || max_lost_time_ <= 0.0) {
      return;
    }

    const double elapsed = now().seconds() - last_seen_sec_;
    if (elapsed > max_lost_time_) {
      if (is_moving_) {
        IC(locked_person_id_.load(), elapsed, max_lost_time_);
      }
      has_last_filter_time_ = false;
      is_moving_ = false;
      current_cmd_vx_ = 0.0;
      current_cmd_vy_ = 0.0;
      current_cmd_wz_ = 0.0;

      geometry_msgs::msg::Twist stop_twist;
      cmd_vel_pub_->publish(stop_twist);
    }
  }

  void handle_align_bag(
    const std::shared_ptr<Trigger::Request> /*req*/,
    std::shared_ptr<Trigger::Response> resp)
  {
    if (is_following_) {
      auto dummy = std::make_shared<Trigger::Response>();
      handle_stop_follow(dummy);
    }

    std::string gest;
    std::vector<Bag> bags;
    {
      std::lock_guard<std::mutex> lock(bag_mutex_);
      gest = latest_gesture_;
      if ((gest.find("RIGHT") == std::string::npos && gest.find("LEFT") == std::string::npos) &&
          !last_valid_gesture_.empty() && last_valid_gesture_ != "NONE") {
        const double age = (now() - last_gesture_time_).seconds();
        if (age < 30.0) {
          gest = last_valid_gesture_;
        }
      }
      bags = latest_bags_;
    }

    if (bags.empty()) {
      const auto t_wait = now();
      while (bags.empty() && (now() - t_wait).seconds() < 1.5 && rclcpp::ok()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::lock_guard<std::mutex> lock(bag_mutex_);
        bags = latest_bags_;
      }
    }

    if (bags.empty()) {
      std::lock_guard<std::mutex> lock(bag_mutex_);
      if (!last_valid_bags_.empty()) {
        const double bag_age = (now() - last_bag_time_).seconds();
        if (bag_age < 15.0) {
          bags = last_valid_bags_;
        }
      }
    }

    const bool is_right = (gest.find("RIGHT") != std::string::npos || gest.find("KANAN") != std::string::npos);
    const bool is_left  = (gest.find("LEFT") != std::string::npos || gest.find("KIRI") != std::string::npos);

    if (!is_right && !is_left && bags.empty()) {
      resp->success = false;
      resp->message = "Gesture tidak terdeteksi dan tidak ada tas terlihat oleh kamera ZED.";
      IC(resp->success, resp->message);
      return;
    }

    Bag target_bag{0.0, 0.0};
    double min_d = 1e9;
    for (const auto & b : bags) {
      const double rel_y = -b.y;
      if ((is_right && rel_y < 0.0) || (is_left && rel_y > 0.0)) {
        const double d = std::hypot(b.x, rel_y);
        if (d < min_d) { min_d = d; target_bag = {b.x, rel_y}; }
      }
    }

    if (min_d > 1e8) {
      if (!bags.empty()) {
        for (const auto & b : bags) {
          const double rel_y = -b.y;
          const double d = std::hypot(b.x, rel_y);
          if (d < min_d) { min_d = d; target_bag = {b.x, rel_y}; }
        }
      } else {
        resp->success = false;
        resp->message = std::string("Tas di sisi ") + (is_right ? "kanan" : (is_left ? "kiri" : "apapun")) + " tidak terdeteksi oleh ZED.";
        IC(resp->success, resp->message);
        return;
      }
    }

    auto get_current_yaw = [&]() -> double {
      try {
        auto trans = tf_buffer_->lookupTransform("odom", frame_base_, tf2::TimePointZero);
        const auto & rq = trans.transform.rotation;
        const double siny = 2.0 * (rq.w * rq.z + rq.x * rq.y);
        const double cosy = 1.0 - 2.0 * (rq.y * rq.y + rq.z * rq.z);
        return std::atan2(siny, cosy);
      } catch (...) {}
      try {
        auto trans = tf_buffer_->lookupTransform(frame_map_, frame_base_, tf2::TimePointZero);
        const auto & rq = trans.transform.rotation;
        const double siny = 2.0 * (rq.w * rq.z + rq.x * rq.y);
        const double cosy = 1.0 - 2.0 * (rq.y * rq.y + rq.z * rq.z);
        return std::atan2(siny, cosy);
      } catch (...) {}
      {
        std::lock_guard<std::mutex> lock(bag_mutex_);
        return current_yaw_deg_ * M_PI / 180.0;
      }
    };

    auto get_current_pos = [&]() -> std::pair<double, double> {
      try {
        auto trans = tf_buffer_->lookupTransform("odom", frame_base_, tf2::TimePointZero);
        return {trans.transform.translation.x, trans.transform.translation.y};
      } catch (...) {}
      try {
        auto trans = tf_buffer_->lookupTransform(frame_map_, frame_base_, tf2::TimePointZero);
        return {trans.transform.translation.x, trans.transform.translation.y};
      } catch (...) {}
      return {0.0, 0.0};
    };

    const double turn_angle = std::atan2(target_bag.y, target_bag.x);
    const double start_yaw = get_current_yaw();
    const double target_yaw = normalize_angle(start_yaw + turn_angle);
    const auto t_rot_start = now();
    geometry_msgs::msg::Twist twist;

    while (rclcpp::ok() && (now() - t_rot_start).seconds() < 6.0) {
      const double curr_yaw = get_current_yaw();
      const double err = normalize_angle(target_yaw - curr_yaw);
      if (std::abs(err) <= 0.05) {
        break;
      }

      double w = std::clamp(1.5 * err, -0.45, 0.45);
      if (std::abs(w) < 0.12) {
        w = (err > 0.0 ? 0.12 : -0.12);
      }
      twist.linear.x = 0.0;
      twist.linear.y = 0.0;
      twist.angular.z = w;
      cmd_vel_pub_->publish(twist);
      std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }

    twist.angular.z = 0.0;
    cmd_vel_pub_->publish(twist);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    const double initial_dist = std::hypot(target_bag.x, target_bag.y);
    const double desired_stop_dist = std::max(0.40, stop_distance_);
    const double forward_dist = initial_dist - desired_stop_dist;

    if (forward_dist > 0.05) {
      const auto start_pos = get_current_pos();
      const auto t_fwd_start = now();
      const double max_fwd_time = std::max(5.0, (forward_dist / 0.15) + 3.0);

      while (rclcpp::ok() && (now() - t_fwd_start).seconds() < max_fwd_time) {
        const auto curr_pos = get_current_pos();
        const double dist_traveled = std::hypot(curr_pos.first - start_pos.first, curr_pos.second - start_pos.second);
        const double dist_remaining = forward_dist - dist_traveled;

        if (dist_remaining <= 0.04) {
          break;
        }

        bool obstacle_too_close = false;
        {
          std::lock_guard<std::mutex> scan_lock(scan_mutex_);
          if (latest_scan_ && !latest_scan_->ranges.empty()) {
            double laser_yaw = 3.1415926535;
            try {
              auto tf_l = tf_buffer_->lookupTransform(frame_base_, latest_scan_->header.frame_id, tf2::TimePointZero);
              const auto & q = tf_l.transform.rotation;
              double siny = 2.0 * (q.w * q.z + q.x * q.y);
              double cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
              laser_yaw = std::atan2(siny, cosy);
            } catch (...) {}

            for (size_t i = 0; i < latest_scan_->ranges.size(); ++i) {
              float r = latest_scan_->ranges[i];
              if (!std::isfinite(r) || r < latest_scan_->range_min || r > latest_scan_->range_max) continue;
              if (r <= footprint_radius_) continue;
              double angle_laser = latest_scan_->angle_min + i * latest_scan_->angle_increment;
              double angle_base = normalize_angle(angle_laser + laser_yaw);
              if (std::abs(angle_base) < 0.52) {
                double ox = r * std::cos(angle_base);
                if (ox > 0.0 && ox < min_stop_x_) {
                  obstacle_too_close = true;
                  break;
                }
              }
            }
          }
        }

        if (obstacle_too_close) {
          break;
        }

        const double heading_err = normalize_angle(target_yaw - get_current_yaw());
        const double wz = std::clamp(1.0 * heading_err, -0.20, 0.20);

        const double vx = std::clamp(0.5 * dist_remaining, 0.10, 0.22);

        twist.linear.x = vx;
        twist.linear.y = 0.0;
        twist.angular.z = wz;
        cmd_vel_pub_->publish(twist);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }

    twist.linear.x = 0.0;
    twist.linear.y = 0.0;
    twist.angular.z = 0.0;
    cmd_vel_pub_->publish(twist);

    resp->success = true;
    resp->message = std::string("Robot telah berputar menghadap tas ") + (is_right ? "kanan" : (is_left ? "kiri" : "")) + " dan maju menuju tas.";
    IC(resp->success, resp->message);
  }

  void handle_homming(
    const std::shared_ptr<Trigger::Request> /*req*/,
    std::shared_ptr<Trigger::Response> resp)
  {
    if (is_following_) {
      auto dummy = std::make_shared<Trigger::Response>();
      handle_stop_follow(dummy);
    }

    if (!action_client_->wait_for_action_server(std::chrono::seconds(3))) {
      resp->success = false;
      resp->message = "Action server 'navigate_to_pose' Nav2 tidak aktif.";
      IC(resp->success, resp->message);
      return;
    }

    NavigateToPose::Goal goal;
    goal.behavior_tree = "";
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
      if (handle) {
        homing_goal_handle_ = handle;
      }
    };
    send_opts.result_callback = [promise](const GoalHandle::WrappedResult & res) {
      promise->set_value(res.code);
    };

    auto send_future = action_client_->async_send_goal(goal, send_opts);
    if (send_future.wait_for(std::chrono::seconds(4)) != std::future_status::ready || !send_future.get()) {
      resp->success = false;
      resp->message = "Gagal mengirim goal origin ke Nav2.";
      IC(resp->success, resp->message);
      return;
    }

    if (future.wait_for(std::chrono::seconds(120)) == std::future_status::ready) {
      const auto code = future.get();
      if (code == rclcpp_action::ResultCode::SUCCEEDED) {
        resp->success = true;
        resp->message = "Robot berhasil tiba kembali di posisi awal start (origin).";
      } else {
        resp->success = false;
        resp->message = "Navigasi kembali ke titik awal dibatalkan atau gagal.";
      }
    } else {
      resp->success = false;
      resp->message = "Navigasi kembali ke titik awal timeout.";
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
  std::string action_name_;
  std::string frame_map_;
  std::string frame_base_;
  double stop_distance_{0.5};
  double start_distance_{0.7};
  double max_range_{3.0};
  double max_lost_time_{3.0};
  bool enable_avoidance_{true};
  double obstacle_distance_{0.75};
  double footprint_radius_{0.25};
  double max_vx_{0.32};
  double max_vy_{0.22};
  double max_wz_{0.40};
  double kp_w_{0.75};
  double kd_w_{0.15};
  double min_stop_x_{0.38};

  std::atomic<int> locked_person_id_{-1};
  std::atomic<bool> is_following_{false};
  bool is_moving_{false};
  double current_cmd_vx_{0.0};
  double current_cmd_vy_{0.0};
  double current_cmd_wz_{0.0};
  double last_seen_sec_{0.0};

  std::mutex scan_mutex_;
  sensor_msgs::msg::LaserScan::SharedPtr latest_scan_;
  std::mutex bag_mutex_;
  std::string latest_gesture_{"NONE"};
  std::string last_valid_gesture_{"NONE"};
  rclcpp::Time last_gesture_time_{0, 0, RCL_ROS_TIME};
  std::vector<Bag> latest_bags_;
  std::vector<Bag> last_valid_bags_;
  rclcpp::Time last_bag_time_{0, 0, RCL_ROS_TIME};
  double current_yaw_deg_{0.0};

  rclcpp::Time last_filter_time_;
  bool has_last_filter_time_{false};

  double prev_bearing_{0.0};
  double filtered_bearing_{0.0};
  bool has_prev_bearing_{false};

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp_action::Client<NavigateToPose>::SharedPtr action_client_;
  GoalHandle::SharedPtr homing_goal_handle_;

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr person_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr gesture_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr bag_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr yaw_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Service<Trigger>::SharedPtr start_srv_;
  rclcpp::Service<Trigger>::SharedPtr stop_srv_;
  rclcpp::Service<Trigger>::SharedPtr align_bag_srv_;
  rclcpp::Service<Trigger>::SharedPtr homing_srv_;
  rclcpp::TimerBase::SharedPtr watchdog_timer_;
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
