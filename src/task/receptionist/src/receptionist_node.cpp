#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

namespace receptionist
{

struct Waypoint
{
  std::string name;
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
};

inline double normalize_angle(double a)
{
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

geometry_msgs::msg::Quaternion yawToQuaternion(double yaw_deg_or_rad)
{
  // Jika nilai lebih besar dari 2*pi, asumsikan input dalam satuan derajat (misal 90.0, 180.0)
  double yaw_rad = (std::abs(yaw_deg_or_rad) > 6.2831853) ? (yaw_deg_or_rad * M_PI / 180.0) : yaw_deg_or_rad;
  geometry_msgs::msg::Quaternion q;
  q.x = 0.0;
  q.y = 0.0;
  q.z = std::sin(yaw_rad * 0.5);
  q.w = std::cos(yaw_rad * 0.5);
  return q;
}

class ReceptionistNode : public rclcpp::Node
{
public:
  using NavigateToPose = nav2_msgs::action::NavigateToPose;
  using GoalHandleNav = rclcpp_action::ClientGoalHandle<NavigateToPose>;

  ReceptionistNode()
  : Node("receptionist_node")
  {
    // Declare parameters for frames and waypoints
    declare_parameter<std::string>("global_frame", "map");
    declare_parameter<std::string>("base_frame", "base_link");
    declare_parameter<std::string>("announce_topic", "/receptionist/status");
    declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    declare_parameter<std::string>("scan_topic", "/scan");
    declare_parameter<double>("nav_timeout_sec", 110.0);
    declare_parameter<double>("wall_proximity_threshold", 0.48);
    declare_parameter<double>("escape_distance", 0.35);
    declare_parameter<double>("escape_speed", 0.12);
    declare_parameter<bool>("enable_open_space_escape", true);

    // Door waypoint
    declare_parameter<double>("door.x", 0.0);
    declare_parameter<double>("door.y", 0.0);
    declare_parameter<double>("door.yaw", 180.0);

    // Beverage area waypoint
    declare_parameter<double>("beverage_area.x", 1.5);
    declare_parameter<double>("beverage_area.y", 1.5);
    declare_parameter<double>("beverage_area.yaw", 90.0);

    // Living room waypoint
    declare_parameter<double>("living_room.x", 1.5);
    declare_parameter<double>("living_room.y", 0.0);
    declare_parameter<double>("living_room.yaw", 90.0);

    // Living room center waypoint
    declare_parameter<double>("living_room_center.x", 2.0);
    declare_parameter<double>("living_room_center.y", 0.0);
    declare_parameter<double>("living_room_center.yaw", 0.0);

    global_frame_ = get_parameter("global_frame").as_string();
    base_frame_ = get_parameter("base_frame").as_string();
    nav_timeout_sec_ = get_parameter("nav_timeout_sec").as_double();
    std::string announce_topic = get_parameter("announce_topic").as_string();
    std::string cmd_vel_topic = get_parameter("cmd_vel_topic").as_string();
    std::string scan_topic = get_parameter("scan_topic").as_string();
    wall_proximity_threshold_ = get_parameter("wall_proximity_threshold").as_double();
    escape_distance_ = get_parameter("escape_distance").as_double();
    escape_speed_ = get_parameter("escape_speed").as_double();
    enable_open_space_escape_ = get_parameter("enable_open_space_escape").as_bool();

    // Callback groups
    srv_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    action_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    sensor_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    // TF buffer & listener
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    status_pub_ = create_publisher<std_msgs::msg::String>(announce_topic, 10);
    cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic, 10);

    rclcpp::SubscriptionOptions sub_opt;
    sub_opt.callback_group = sensor_cb_group_;
    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic, rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(scan_mutex_);
        latest_scan_ = msg;
      },
      sub_opt);

    // Action client Nav2
    nav_client_ = rclcpp_action::create_client<NavigateToPose>(
      this, "navigate_to_pose", action_cb_group_);

    // Services
    srv_go_door_ = create_service<std_srvs::srv::Trigger>(
      "/receptionist/go_to_door",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleGoTo("door", res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    srv_go_beverage_area_ = create_service<std_srvs::srv::Trigger>(
      "/receptionist/go_to_beverage_area",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleGoTo("beverage_area", res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    srv_go_living_room_ = create_service<std_srvs::srv::Trigger>(
      "/receptionist/go_to_living_room",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleGoTo("living_room", res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    srv_go_living_room_center_ = create_service<std_srvs::srv::Trigger>(
      "/receptionist/go_to_living_room_center",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleGoTo("living_room_center", res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    RCLCPP_INFO(get_logger(), "Receptionist Navigation Node ready. Services available:");
    RCLCPP_INFO(get_logger(), " - /receptionist/go_to_door");
    RCLCPP_INFO(get_logger(), " - /receptionist/go_to_beverage_area");
    RCLCPP_INFO(get_logger(), " - /receptionist/go_to_living_room");
    RCLCPP_INFO(get_logger(), " - /receptionist/go_to_living_room_center");
    publishStatus("Receptionist navigation node ready");
  }

  void publishStatus(const std::string & msg_str)
  {
    std_msgs::msg::String msg;
    msg.data = msg_str;
    status_pub_->publish(msg);
    RCLCPP_INFO(get_logger(), "[STATUS] %s", msg_str.c_str());
  }

  Waypoint getWaypoint(const std::string & prefix)
  {
    Waypoint wp;
    wp.name = prefix;
    wp.x = get_parameter(prefix + ".x").as_double();
    wp.y = get_parameter(prefix + ".y").as_double();
    wp.yaw = get_parameter(prefix + ".yaw").as_double();
    return wp;
  }

  struct LaserPt { double x, y; };

  bool checkWallProximity(
    double & out_min_dist,
    double & out_open_dir_x,
    double & out_open_dir_y,
    const Waypoint & target_wp)
  {
    if (!enable_open_space_escape_) return false;

    sensor_msgs::msg::LaserScan::SharedPtr scan;
    {
      std::lock_guard<std::mutex> lock(scan_mutex_);
      scan = latest_scan_;
    }
    if (!scan || scan->ranges.empty()) {
      return false;
    }

    double laser_tx = 0.0, laser_ty = 0.0, laser_yaw = 3.1415926535;
    try {
      auto tf_laser = tf_buffer_->lookupTransform(base_frame_, scan->header.frame_id, tf2::TimePointZero);
      laser_tx = tf_laser.transform.translation.x;
      laser_ty = tf_laser.transform.translation.y;
      const auto & rq = tf_laser.transform.rotation;
      laser_yaw = std::atan2(2.0 * (rq.w * rq.z + rq.x * rq.y), 1.0 - 2.0 * (rq.y * rq.y + rq.z * rq.z));
    } catch (...) {
      laser_yaw = 3.1415926535;
    }

    std::vector<LaserPt> pts;
    pts.reserve(scan->ranges.size());
    out_min_dist = 999.0;

    const int n = static_cast<int>(scan->ranges.size());
    for (int i = 0; i < n; ++i) {
      const float r = scan->ranges[i];
      if (!std::isfinite(r) || r < scan->range_min || r > scan->range_max) continue;
      const double angle_laser = scan->angle_min + i * scan->angle_increment;
      const double angle_base = normalize_angle(angle_laser + laser_yaw);
      const double px = laser_tx + r * std::cos(angle_base);
      const double py = laser_ty + r * std::sin(angle_base);
      const double d = std::hypot(px, py);
      if (d < 0.22) continue; // Abaikan pantulan rangka/sasis robot

      if (d < out_min_dist) {
        out_min_dist = d;
      }
      pts.push_back({px, py});
    }

    // Jika jarak ke rintangan/dinding terdekat >= threshold, robot aman (tidak dekat dinding)
    if (out_min_dist >= wall_proximity_threshold_) {
      return false;
    }

    // Hitung orientasi relatif ke target waypoint (jika TF map -> base_link tersedia)
    double goal_rel_angle = 0.0;
    try {
      auto trans = tf_buffer_->lookupTransform(global_frame_, base_frame_, tf2::TimePointZero);
      const double rx = trans.transform.translation.x;
      const double ry = trans.transform.translation.y;
      const auto & rq = trans.transform.rotation;
      const double ryaw = std::atan2(2.0 * (rq.w * rq.z + rq.x * rq.y), 1.0 - 2.0 * (rq.y * rq.y + rq.z * rq.z));
      const double goal_map_yaw = std::atan2(target_wp.y - ry, target_wp.x - rx);
      goal_rel_angle = normalize_angle(goal_map_yaw - ryaw);
    } catch (...) {
      goal_rel_angle = 0.0;
    }

    // Evaluasi 36 kandidat arah sudut (-180 sampai +180 derajat)
    const int num_candidates = 36;
    const double step = 2.0 * M_PI / num_candidates;
    const double corridor_half_width = 0.26; // Radius bodi robot (0.24m) + margin 2cm
    const double max_eval_dist = 3.5;

    double best_score = -1e9;
    double best_phi = 0.0;

    for (int c = 0; c < num_candidates; ++c) {
      const double phi = normalize_angle(-M_PI + c * step);
      const double cos_p = std::cos(phi);
      const double sin_p = std::sin(phi);

      double min_fwd = max_eval_dist;
      for (const auto & p : pts) {
        const double fwd = p.x * cos_p + p.y * sin_p;
        const double lat = -p.x * sin_p + p.y * cos_p;

        if (fwd > 0.05 && fwd < max_eval_dist && std::abs(lat) < corridor_half_width) {
          if (fwd < min_fwd) {
            min_fwd = fwd;
          }
        }
      }

      // Prioritas 1: Area paling lapang (min_fwd besar)
      // Prioritas 2: Sedikit bias ke arah target tujuan (+0.35 * cos(phi - goal_rel_angle))
      const double score = min_fwd + 0.35 * std::cos(phi - goal_rel_angle);
      if (score > best_score) {
        best_score = score;
        best_phi = phi;
      }
    }

    out_open_dir_x = std::cos(best_phi);
    out_open_dir_y = std::sin(best_phi);
    return true;
  }

  void moveTowardOpenSpace(double dir_x, double dir_y, double initial_clearance)
  {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "[WALL AVOIDANCE] Terlalu dekat dinding (clearance: %.2fm). Menjauh ke area lapang...", initial_clearance);
    publishStatus(buf);

    const auto t_start = now();
    const double max_duration = escape_distance_ / escape_speed_; // misal 0.35m / 0.12m/s ~= 2.9s
    const double target_clearance = 0.60; // Hentikan jika clearance sudah >= 60cm

    geometry_msgs::msg::Twist twist;
    rclcpp::WallRate rate(20.0);

    while (rclcpp::ok() && (now() - t_start).seconds() < max_duration) {
      double fwd_obs = 999.0;
      double cur_min = 999.0;

      {
        std::lock_guard<std::mutex> lock(scan_mutex_);
        if (latest_scan_ && !latest_scan_->ranges.empty()) {
          const auto & scan = latest_scan_;
          const int n = static_cast<int>(scan->ranges.size());
          for (int i = 0; i < n; ++i) {
            const float r = scan->ranges[i];
            if (!std::isfinite(r) || r < scan->range_min || r > scan->range_max) continue;
            const double angle_base = normalize_angle(scan->angle_min + i * scan->angle_increment + 3.1415926535);
            const double px = r * std::cos(angle_base);
            const double py = r * std::sin(angle_base);
            const double d = std::hypot(px, py);
            if (d < 0.22) continue;
            if (d < cur_min) cur_min = d;

            const double fwd = px * dir_x + py * dir_y;
            const double lat = -px * dir_y + py * dir_x;
            if (fwd > 0.05 && std::abs(lat) < 0.26) {
              if (fwd < fwd_obs) fwd_obs = fwd;
            }
          }
        }
      }

      // Safety check: jika ada halangan di depan arah gerak < 0.28m, segera stop
      if (fwd_obs < 0.28) {
        RCLCPP_WARN(get_logger(), "[WALL AVOIDANCE] Safety stop: Rintangan dekat (%.2fm) di jalur escape", fwd_obs);
        break;
      }

      // Jika sudah bergerak minimal 1.0 detik dan clearance sudah mencapai area aman (>= 0.60m), selesai
      if (cur_min >= target_clearance && (now() - t_start).seconds() >= 1.0) {
        RCLCPP_INFO(get_logger(), "[WALL AVOIDANCE] Ruang lapang tercapai (clearance: %.2fm)", cur_min);
        break;
      }

      twist.linear.x = dir_x * escape_speed_;
      twist.linear.y = dir_y * escape_speed_;
      twist.angular.z = 0.0;
      cmd_vel_pub_->publish(twist);
      rate.sleep();
    }

    // Stop robot
    twist.linear.x = 0.0;
    twist.linear.y = 0.0;
    twist.angular.z = 0.0;
    cmd_vel_pub_->publish(twist);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    publishStatus("[WALL AVOIDANCE] Posisi sudah lapang. Melanjutkan rute Nav2...");
  }

  void handleGoTo(
    const std::string & target_name,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    // Gunakan atomic exchange agar jika sedang sibuk langsung ditolak tanpa blocking
    if (is_navigating_.exchange(true)) {
      res->success = false;
      res->message = "Robot is currently busy with another navigation task";
      publishStatus(res->message);
      return;
    }

    Waypoint wp = getWaypoint(target_name);

    // 1. Cek apakah robot terlalu dekat dinding/rintangan
    double min_clearance = 999.0;
    double open_dir_x = 0.0, open_dir_y = 0.0;
    if (checkWallProximity(min_clearance, open_dir_x, open_dir_y, wp)) {
      moveTowardOpenSpace(open_dir_x, open_dir_y, min_clearance);
    }

    publishStatus(
      "Command received: Navigating to " + target_name +
      " (x=" + std::to_string(wp.x) + ", y=" + std::to_string(wp.y) + ")...");

    bool ok = navigateTo(wp);
    is_navigating_ = false;

    res->success = ok;
    res->message = ok ? "Reached " + target_name : "Failed to reach " + target_name;
    publishStatus(res->message);
  }

  bool navigateTo(const Waypoint & wp)
  {
    if (!nav_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_logger(), "Nav2 action server 'navigate_to_pose' unavailable");
      return false;
    }

    NavigateToPose::Goal goal;
    goal.pose.header.frame_id = global_frame_;
    goal.pose.header.stamp = now();
    goal.pose.pose.position.x = wp.x;
    goal.pose.pose.position.y = wp.y;
    goal.pose.pose.position.z = 0.0;
    goal.pose.pose.orientation = yawToQuaternion(wp.yaw);

    auto send_goal_future = nav_client_->async_send_goal(goal);
    if (send_goal_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "Failed to send navigation goal to %s (timeout)", wp.name.c_str());
      return false;
    }

    auto goal_handle = send_goal_future.get();
    if (!goal_handle) {
      RCLCPP_ERROR(get_logger(), "Navigation goal rejected by Nav2 for %s", wp.name.c_str());
      return false;
    }

    RCLCPP_INFO(get_logger(), "Goal accepted by Nav2 for '%s', waiting for robot to arrive...", wp.name.c_str());

    auto result_future = nav_client_->async_get_result(goal_handle);
    const auto start_time = now();

    while (rclcpp::ok()) {
      if (result_future.wait_for(std::chrono::milliseconds(200)) == std::future_status::ready) {
        auto wrapped_result = result_future.get();
        if (wrapped_result.code == rclcpp_action::ResultCode::SUCCEEDED) {
          RCLCPP_INFO(get_logger(), "Successfully arrived at '%s'", wp.name.c_str());
          return true;
        } else {
          RCLCPP_WARN(
            get_logger(), "Navigation to '%s' ended with code %d",
            wp.name.c_str(), static_cast<int>(wrapped_result.code));
          return false;
        }
      }

      if ((now() - start_time).seconds() > nav_timeout_sec_) {
        RCLCPP_ERROR(
          get_logger(), "Navigation to '%s' timed out after %.1f seconds. Canceling goal...",
          wp.name.c_str(), nav_timeout_sec_);
        nav_client_->async_cancel_goal(goal_handle);
        return false;
      }
    }
    return false;
  }

private:
  std::string global_frame_;
  std::string base_frame_{"base_link"};
  double nav_timeout_sec_{110.0};
  double wall_proximity_threshold_{0.48};
  double escape_distance_{0.35};
  double escape_speed_{0.12};
  bool enable_open_space_escape_{true};
  std::atomic<bool> is_navigating_{false};

  rclcpp::CallbackGroup::SharedPtr srv_cb_group_;
  rclcpp::CallbackGroup::SharedPtr action_cb_group_;
  rclcpp::CallbackGroup::SharedPtr sensor_cb_group_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  std::mutex scan_mutex_;
  sensor_msgs::msg::LaserScan::SharedPtr latest_scan_;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp_action::Client<NavigateToPose>::SharedPtr nav_client_;

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_go_door_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_go_beverage_area_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_go_living_room_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_go_living_room_center_;
};

}  // namespace receptionist

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<receptionist::ReceptionistNode>();

  // Gunakan MultiThreadedExecutor dengan 4 thread untuk service, sensor, dan action callbacks
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
