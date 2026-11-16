#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"

namespace receptionist
{

struct Waypoint
{
  std::string name;
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
};

struct EmptyChairTarget
{
  bool valid{false};
  double angle_err{0.0}; // Radian: >0 to left, <0 to right
  double dist{0.0};      // Meter
  std::string side;      // "left", "center", "right"
  std::string raw_info;
};

geometry_msgs::msg::Quaternion yawToQuaternion(double yaw_deg_or_rad)
{
  // Jika nilai lebih besar dari 2*pi, asumsikan input berupa derajat (misal 90.0, 180.0)
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
    declare_parameter<std::string>("announce_topic", "/receptionist/status");
    declare_parameter<double>("nav_timeout_sec", 110.0);

    // Door waypoint
    declare_parameter<double>("door.x", 0.5);
    declare_parameter<double>("door.y", 0.0);
    declare_parameter<double>("door.yaw", 180.0);

    // Beverage area waypoint
    declare_parameter<double>("beverage_area.x", 1.9);
    declare_parameter<double>("beverage_area.y", 1.9);
    declare_parameter<double>("beverage_area.yaw", 270.0);

    // Living room waypoint
    declare_parameter<double>("living_room.x", 1.5);
    declare_parameter<double>("living_room.y", 0.0);
    declare_parameter<double>("living_room.yaw", 270.0);

    // Chair search & motion parameters
    declare_parameter<std::string>("cmd_vel_topic", "cmd_vel");
    declare_parameter<std::string>("chair_topic", "/vision/chair_detections");
    declare_parameter<std::string>("empty_seat_topic", "/vision/empty_seat");
    declare_parameter<double>("search_chair_angular_speed", 0.35);
    declare_parameter<double>("search_chair_timeout_sec", 40.0);
    declare_parameter<int>("search_chair_confirm_frames", 2);
    declare_parameter<double>("align_chair_p_gain", 1.2);
    declare_parameter<double>("align_chair_min_speed", 0.14);
    declare_parameter<double>("align_chair_max_speed", 0.28);
    declare_parameter<double>("align_chair_tolerance_rad", 0.07);
    declare_parameter<int>("camera_width", 672);

    global_frame_ = get_parameter("global_frame").as_string();
    nav_timeout_sec_ = get_parameter("nav_timeout_sec").as_double();
    std::string announce_topic = get_parameter("announce_topic").as_string();

    cmd_vel_topic_ = get_parameter("cmd_vel_topic").as_string();
    chair_topic_ = get_parameter("chair_topic").as_string();
    empty_seat_topic_ = get_parameter("empty_seat_topic").as_string();
    search_chair_angular_speed_ = get_parameter("search_chair_angular_speed").as_double();
    search_chair_timeout_sec_ = get_parameter("search_chair_timeout_sec").as_double();
    search_chair_confirm_frames_ = get_parameter("search_chair_confirm_frames").as_int();
    align_chair_p_gain_ = get_parameter("align_chair_p_gain").as_double();
    align_chair_min_speed_ = get_parameter("align_chair_min_speed").as_double();
    align_chair_max_speed_ = get_parameter("align_chair_max_speed").as_double();
    align_chair_tolerance_rad_ = get_parameter("align_chair_tolerance_rad").as_double();
    camera_width_ = get_parameter("camera_width").as_int();

    // Callback groups
    srv_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    action_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    sub_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    status_pub_ = create_publisher<std_msgs::msg::String>(announce_topic, 10);
    cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);

    // Subscriptions for chair detection
    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.callback_group = sub_cb_group_;

    chair_sub_ = create_subscription<std_msgs::msg::String>(
      chair_topic_, rclcpp::SensorDataQoS(),
      [this](const std_msgs::msg::String::SharedPtr msg) {
        onChairDetections(msg);
      },
      sub_opts);

    empty_seat_sub_ = create_subscription<std_msgs::msg::String>(
      empty_seat_topic_, 10,
      [this](const std_msgs::msg::String::SharedPtr msg) {
        onEmptySeat(msg);
      },
      sub_opts);

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

    // Service /search_chair (dan alias /receptionist/search_chair)
    srv_search_chair_ = create_service<std_srvs::srv::Trigger>(
      "/search_chair",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleSearchChair(req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    srv_search_chair_alias_ = create_service<std_srvs::srv::Trigger>(
      "/receptionist/search_chair",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleSearchChair(req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    RCLCPP_INFO(get_logger(), "Receptionist Navigation Node ready. Services available:");
    RCLCPP_INFO(get_logger(), " - /receptionist/go_to_door");
    RCLCPP_INFO(get_logger(), " - /receptionist/go_to_beverage_area");
    RCLCPP_INFO(get_logger(), " - /receptionist/go_to_living_room");
    RCLCPP_INFO(get_logger(), " - /search_chair (alias: /receptionist/search_chair)");
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

  void stopRobot()
  {
    geometry_msgs::msg::Twist twist;
    twist.linear.x = 0.0;
    twist.linear.y = 0.0;
    twist.linear.z = 0.0;
    twist.angular.x = 0.0;
    twist.angular.y = 0.0;
    twist.angular.z = 0.0;
    for (int i = 0; i < 5; ++i) {
      cmd_vel_pub_->publish(twist);
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }

  void rotateRight(double speed)
  {
    geometry_msgs::msg::Twist twist;
    twist.linear.x = 0.0;
    twist.linear.y = 0.0;
    twist.linear.z = 0.0;
    twist.angular.x = 0.0;
    twist.angular.y = 0.0;
    // Clockwise (rotasi ke kanan): nilai angular.z negatif sesuai konvensi ROS REP 103
    twist.angular.z = -std::abs(speed);
    cmd_vel_pub_->publish(twist);
  }

  void onChairDetections(const std_msgs::msg::String::SharedPtr msg)
  {
    if (!msg) return;
    std::lock_guard<std::mutex> lock(chair_mutex_);

    // Format dari athome_vision: "chair:x,y,w,h|X,Y,Z|empty/occupied|left/center/right;chair:..."
    // Filter ketat: HANYA kursi berstatus "empty" (kursi kosong) yang dianggap valid
    bool has_empty_chair = false;
    std::string empty_chairs_info;

    if (!msg->data.empty()) {
      std::stringstream ss(msg->data);
      std::string item;
      while (std::getline(ss, item, ';')) {
        // Cek bahwa token adalah kursi dan berstatus "empty" (bukan "occupied")
        if (item.find("chair") != std::string::npos &&
            item.find("empty") != std::string::npos &&
            item.find("occupied") == std::string::npos) {
          has_empty_chair = true;
          if (!empty_chairs_info.empty()) {
            empty_chairs_info += ";";
          }
          empty_chairs_info += item;
        }
      }
    }

    if (has_empty_chair) {
      last_chair_time_ = now();
      latest_chair_info_ = empty_chairs_info;
      consecutive_chair_frames_++;
      if (consecutive_chair_frames_ >= search_chair_confirm_frames_) {
        chair_detected_ = true;
      }
    } else {
      consecutive_chair_frames_ = 0;
      chair_detected_ = false;
    }
  }

  void onEmptySeat(const std_msgs::msg::String::SharedPtr msg)
  {
    if (!msg) return;
    std::lock_guard<std::mutex> lock(chair_mutex_);
    // /vision/empty_seat mempublikasikan sisi kursi kosong ("left", "center", "right") atau "none"
    if (!msg->data.empty() && msg->data != "none") {
      last_chair_time_ = now();
      if (latest_chair_info_.empty() || !chair_detected_) {
        latest_chair_info_ = "empty_seat:" + msg->data;
      }
      consecutive_chair_frames_++;
      if (consecutive_chair_frames_ >= search_chair_confirm_frames_) {
        chair_detected_ = true;
      }
    } else if (msg->data == "none") {
      if (latest_chair_info_.rfind("empty_seat:", 0) == 0) {
        consecutive_chair_frames_ = 0;
        chair_detected_ = false;
      }
    }
  }

  EmptyChairTarget parseEmptyChair(const std::string & data) const
  {
    EmptyChairTarget target;
    if (data.empty()) return target;

    // Format 1: Dari /vision/chair_detections ("chair:x,y,w,h|X,Y,Z|empty|side;...")
    std::stringstream ss(data);
    std::string item;
    double best_dist = 999.0;

    while (std::getline(ss, item, ';')) {
      if (item.find("chair") == std::string::npos ||
          item.find("empty") == std::string::npos ||
          item.find("occupied") != std::string::npos) {
        continue;
      }

      std::vector<std::string> parts;
      std::stringstream part_ss(item);
      std::string p;
      while (std::getline(part_ss, p, '|')) {
        parts.push_back(p);
      }
      if (parts.size() < 4) continue;

      double pos_x = 0.0, pos_y = 0.0, pos_z = 0.0;
      bool has_3d = false;
      if (parts[1].find("nan") == std::string::npos) {
        if (std::sscanf(parts[1].c_str(), "%lf,%lf,%lf", &pos_x, &pos_y, &pos_z) == 3) {
          if (pos_x > 0.1) {
            has_3d = true;
          }
        }
      }

      int rx = 0, ry = 0, rw = 0, rh = 0;
      if (std::sscanf(parts[0].c_str(), "chair:%d,%d,%d,%d", &rx, &ry, &rw, &rh) != 4) {
        continue;
      }

      double angle = 0.0;
      double dist = 0.0;
      if (has_3d) {
        dist = std::hypot(pos_x, pos_y);
        // Konvensi RIGHT_HANDED_Z_UP_X_FWD: X depan, Y kiri (+kiri, -kanan)
        angle = std::atan2(pos_y, pos_x);
      } else {
        // Fallback pixel bbox center jika depth bernilai nan
        int cx = rx + rw / 2;
        double half_w = camera_width_ > 0 ? (camera_width_ / 2.0) : 336.0;
        // cx > half_w (kursi di kanan gambar) -> angle negatif (perlu rotasi ke kanan)
        // cx < half_w (kursi di kiri gambar) -> angle positif (perlu rotasi ke kiri)
        angle = -((cx - half_w) / half_w) * 0.75;
        dist = 1.5;
      }

      if (dist < best_dist || !target.valid) {
        best_dist = dist;
        target.valid = true;
        target.angle_err = angle;
        target.dist = dist;
        target.side = parts[3];
        target.raw_info = item;
      }
    }

    // Format 2: Fallback jika berasal dari /vision/empty_seat (misal "empty_seat:left", "right", atau "center")
    if (!target.valid) {
      std::string seat_str = data;
      if (seat_str.rfind("empty_seat:", 0) == 0) {
        seat_str = seat_str.substr(11);
      }
      if (seat_str == "left") {
        target.valid = true;
        target.angle_err = 0.35; // Kursi di kiri -> putar kiri (+0.35 rad)
        target.dist = 1.5;
        target.side = "left";
        target.raw_info = "empty_seat:left";
      } else if (seat_str == "right") {
        target.valid = true;
        target.angle_err = -0.35; // Kursi di kanan -> putar kanan (-0.35 rad)
        target.dist = 1.5;
        target.side = "right";
        target.raw_info = "empty_seat:right";
      } else if (seat_str == "center") {
        target.valid = true;
        target.angle_err = 0.0; // Sudah tepat di tengah
        target.dist = 1.5;
        target.side = "center";
        target.raw_info = "empty_seat:center";
      }
    }

    return target;
  }

  void handleSearchChair(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    if (is_navigating_.load()) {
      res->success = false;
      res->message = "Robot is currently busy with another navigation task";
      publishStatus(res->message);
      return;
    }

    if (is_searching_chair_.exchange(true)) {
      res->success = false;
      res->message = "Robot is already searching for a chair";
      publishStatus(res->message);
      return;
    }

    publishStatus("Service /search_chair called: Searching and aligning to empty chair...");

    // Helper untuk mengambil data kursi kosong terkini yang masih segar (< 0.8s)
    auto get_current_empty_chair = [this]() -> EmptyChairTarget {
      std::lock_guard<std::mutex> lock(chair_mutex_);
      if (chair_detected_ && (now() - last_chair_time_).seconds() < 0.8) {
        return parseEmptyChair(latest_chair_info_);
      }
      return EmptyChairTarget();
    };

    auto start_time = now();
    bool chair_found = false;
    bool aligned = false;
    EmptyChairTarget current_target;
    rclcpp::Rate loop_rate(20.0); // 20 Hz kontrol loop

    // Cek dulu apakah kursi kosong sudah terlihat saat posisi robot masih diam
    for (int i = 0; i < 6; ++i) {
      EmptyChairTarget t = get_current_empty_chair();
      if (t.valid) {
        chair_found = true;
        current_target = t;
        break;
      }
      loop_rate.sleep();
    }

    // FASE 1: SEARCH - Jika belum terdeteksi, berputar ke kanan sampai mendeteksi kursi kosong
    if (!chair_found) {
      publishStatus("No empty chair in view. Rotating right to search for empty chair...");
      RCLCPP_INFO(
        get_logger(),
        "\033[1;33m[SEARCH CHAIR]\033[0m Rotating right to find empty chair (speed=%.2f rad/s, timeout=%.1fs)...",
        search_chair_angular_speed_, search_chair_timeout_sec_);

      while (rclcpp::ok() && is_searching_chair_.load()) {
        const double elapsed = (now() - start_time).seconds();
        if (elapsed > search_chair_timeout_sec_) {
          RCLCPP_WARN(get_logger(), "[SEARCH CHAIR] Timeout reached (%.1f s)", search_chair_timeout_sec_);
          break;
        }

        EmptyChairTarget t = get_current_empty_chair();
        if (t.valid) {
          chair_found = true;
          current_target = t;
          RCLCPP_INFO(
            get_logger(),
            "\033[1;32m[SEARCH CHAIR]\033[0m Empty chair detected! Angle offset=%.2f rad (side=%s). Now aligning heading...",
            t.angle_err, t.side.c_str());
          publishStatus("Empty chair detected! Aligning robot heading to face chair...");
          break;
        }

        rotateRight(search_chair_angular_speed_);
        loop_rate.sleep();
      }
    }

    // FASE 2: ALIGN - Menyelaraskan arah hadap robot tepat ke arah kursi kosong
    if (chair_found) {
      auto align_start_time = now();
      const double align_timeout_sec = 10.0;
      int centered_frames = 0;

      while (rclcpp::ok() && is_searching_chair_.load()) {
        const double align_elapsed = (now() - align_start_time).seconds();
        if (align_elapsed > align_timeout_sec) {
          RCLCPP_WARN(get_logger(), "[SEARCH CHAIR] Alignment timeout (%.1fs reached), stopping.", align_timeout_sec);
          break;
        }

        EmptyChairTarget t = get_current_empty_chair();
        if (t.valid) {
          current_target = t;
        }

        const double err = current_target.angle_err;

        // Cek apakah arah hadap robot sudah tepat menghadap ke kursi (|err| <= tolerance)
        if (std::abs(err) <= align_chair_tolerance_rad_) {
          centered_frames++;
          if (centered_frames >= 3) { // Terpusat stabil selama ~150 ms
            aligned = true;
            break;
          }
        } else {
          centered_frames = 0;
        }

        // P-controller untuk memutar hadap robot ke arah kursi
        double w = align_chair_p_gain_ * err;
        double spd = std::abs(w);
        if (spd < align_chair_min_speed_) {
          spd = align_chair_min_speed_;
        } else if (spd > align_chair_max_speed_) {
          spd = align_chair_max_speed_;
        }

        // err > 0 -> kursi di sebelah kiri -> putar kiri (+)
        // err < 0 -> kursi di sebelah kanan -> putar kanan (-)
        double angular_z = (err >= 0.0) ? spd : -spd;

        geometry_msgs::msg::Twist twist;
        twist.linear.x = 0.0;
        twist.linear.y = 0.0;
        twist.angular.z = angular_z;
        cmd_vel_pub_->publish(twist);

        loop_rate.sleep();
      }
    }

    // Berhenti penuh setelah arah hadap robot tepat berada pada arah kursi
    stopRobot();
    is_searching_chair_ = false;

    if (chair_found && aligned) {
      res->success = true;
      res->message = "Empty chair found and robot aligned facing chair: " + current_target.raw_info;
      publishStatus(res->message);
      RCLCPP_INFO(
        get_logger(),
        "\033[1;32m[SEARCH CHAIR SUCCESS]\033[0m %s (dist=%.2fm, angle_err=%.2frad)",
        current_target.raw_info.c_str(), current_target.dist, current_target.angle_err);
    } else if (chair_found && !aligned) {
      res->success = true;
      res->message = "Empty chair found (alignment approximate): " + current_target.raw_info;
      publishStatus(res->message);
      RCLCPP_INFO(get_logger(), "\033[1;33m[SEARCH CHAIR PARTIAL]\033[0m %s", res->message.c_str());
    } else {
      res->success = false;
      res->message = "Failed to detect empty chair within timeout (" +
        std::to_string(static_cast<int>(search_chair_timeout_sec_)) + "s)";
      publishStatus(res->message);
      RCLCPP_WARN(get_logger(), "\033[1;31m[SEARCH CHAIR TIMEOUT]\033[0m %s", res->message.c_str());
    }
  }

  void handleGoTo(
    const std::string & target_name,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    if (is_searching_chair_.load()) {
      res->success = false;
      res->message = "Robot is currently busy searching for a chair";
      publishStatus(res->message);
      return;
    }

    if (is_navigating_.exchange(true)) {
      res->success = false;
      res->message = "Robot is currently busy with another navigation task";
      publishStatus(res->message);
      return;
    }

    Waypoint wp = getWaypoint(target_name);

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
    goal.behavior_tree = "";
    goal.pose.header.frame_id = global_frame_;
    goal.pose.header.stamp = now();
    goal.pose.pose.position.x = wp.x;
    goal.pose.pose.position.y = wp.y;
    goal.pose.pose.position.z = 0.0;
    goal.pose.pose.orientation = yawToQuaternion(wp.yaw);

    auto promise = std::make_shared<std::promise<rclcpp_action::ResultCode>>();
    auto future = promise->get_future();

    auto send_opts = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    send_opts.goal_response_callback = [this, wp](const GoalHandleNav::SharedPtr & handle) {
      if (!handle) {
        RCLCPP_WARN(get_logger(), "[NAV] Nav2 rejected goal for '%s'.", wp.name.c_str());
      } else {
        current_goal_handle_ = handle;
        RCLCPP_INFO(get_logger(), "[NAV] Nav2 accepted goal for '%s'. Navigating with recovery & replanning BT...", wp.name.c_str());
      }
    };
    send_opts.result_callback = [promise](const GoalHandleNav::WrappedResult & res) {
      promise->set_value(res.code);
    };

    auto send_future = nav_client_->async_send_goal(goal, send_opts);
    if (send_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready || !send_future.get()) {
      RCLCPP_ERROR(get_logger(), "Failed to send navigation goal to %s", wp.name.c_str());
      return false;
    }

    if (future.wait_for(std::chrono::seconds(static_cast<int>(nav_timeout_sec_))) == std::future_status::ready) {
      const auto code = future.get();
      if (code == rclcpp_action::ResultCode::SUCCEEDED) {
        RCLCPP_INFO(get_logger(), "Successfully arrived at '%s'", wp.name.c_str());
        return true;
      } else {
        RCLCPP_WARN(
          get_logger(), "Navigation to '%s' failed or canceled (code %d)",
          wp.name.c_str(), static_cast<int>(code));
        return false;
      }
    } else {
      RCLCPP_ERROR(
        get_logger(), "Navigation to '%s' timed out after %.1f seconds. Canceling goal...",
        wp.name.c_str(), nav_timeout_sec_);
      if (current_goal_handle_) {
        nav_client_->async_cancel_goal(current_goal_handle_);
      }
      return false;
    }
  }

private:
  std::string global_frame_;
  double nav_timeout_sec_{110.0};
  std::atomic<bool> is_navigating_{false};
  std::atomic<bool> is_searching_chair_{false};
  GoalHandleNav::SharedPtr current_goal_handle_;

  // Chair search parameters & tracking
  std::string cmd_vel_topic_{"cmd_vel"};
  std::string chair_topic_{"/vision/chair_detections"};
  std::string empty_seat_topic_{"/vision/empty_seat"};
  double search_chair_angular_speed_{0.35};
  double search_chair_timeout_sec_{40.0};
  int search_chair_confirm_frames_{2};
  double align_chair_p_gain_{1.2};
  double align_chair_min_speed_{0.14};
  double align_chair_max_speed_{0.28};
  double align_chair_tolerance_rad_{0.07};
  int camera_width_{672};

  std::mutex chair_mutex_;
  bool chair_detected_{false};
  std::string latest_chair_info_;
  rclcpp::Time last_chair_time_{0, 0, RCL_ROS_TIME};
  int consecutive_chair_frames_{0};

  rclcpp::CallbackGroup::SharedPtr srv_cb_group_;
  rclcpp::CallbackGroup::SharedPtr action_cb_group_;
  rclcpp::CallbackGroup::SharedPtr sub_cb_group_;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp_action::Client<NavigateToPose>::SharedPtr nav_client_;

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr chair_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr empty_seat_sub_;

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_go_door_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_go_beverage_area_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_go_living_room_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_search_chair_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_search_chair_alias_;
};

}  // namespace receptionist

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<receptionist::ReceptionistNode>();

  // MultiThreadedExecutor dengan 4 thread untuk memproses service, action, dan vision subscription secara paralel
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
