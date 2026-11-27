#include <algorithm>
#include <atomic>
#include <cctype>
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
#include "nav_msgs/msg/odometry.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2/exceptions.h"

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
  bool has_bbox{false};
  double cx{0.0};        // Center X in pixels
};

struct PersonTarget
{
  bool valid{false};
  double angle_err{0.0}; // Radian: >0 to left, <0 to right
  double dist{0.0};      // Meter
  std::string label;
};

inline std::string toLower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

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

    // Face person parameters
    declare_parameter<std::string>("host_name", "Alex");
    declare_parameter<std::string>("person_label_topic", "/vision/person_label");
    declare_parameter<std::string>("guest_1_name_topic", "/vision/guest_1_name");
    declare_parameter<std::string>("guest_2_name_topic", "/vision/guest_2_name");
    declare_parameter<double>("host_seated_max_dist", 3.0);
    declare_parameter<double>("guest_door_distance", 1.2);
    declare_parameter<double>("guest_door_distance_min", 0.9);
    declare_parameter<double>("guest_door_step_forward_speed", 0.25);
    declare_parameter<double>("guest_door_step_forward_min_speed", 0.10);
    declare_parameter<double>("guest_door_step_forward_p_gain", 0.40);
    declare_parameter<double>("guest_door_step_back_speed", 0.18);
    declare_parameter<double>("guest_door_step_back_min_speed", 0.08);
    declare_parameter<double>("guest_door_step_back_p_gain", 0.35);
    declare_parameter<double>("guest_door_adjust_angular_speed", 0.12);
    declare_parameter<double>("guest_door_timeout_sec", 15.0);
    declare_parameter<double>("guest_approach_distance", 0.7);
    declare_parameter<double>("guest_approach_speed", 0.16);
    declare_parameter<double>("guest_approach_timeout_sec", 15.0);
    declare_parameter<double>("guest_door_align_tolerance", 0.07);
    declare_parameter<double>("face_person_angular_speed", 0.30);
    declare_parameter<double>("face_person_timeout_sec", 25.0);
    declare_parameter<double>("face_person_p_gain", 1.2);
    declare_parameter<double>("face_person_min_speed", 0.14);
    declare_parameter<double>("face_person_max_speed", 0.30);
    declare_parameter<double>("face_person_tolerance_rad", 0.07);

    // Relative 90-degree turn parameters (/receptionist/right)
    declare_parameter<double>("turn_90_angular_speed", 0.30);
    declare_parameter<double>("turn_90_min_speed", 0.14);
    declare_parameter<double>("turn_90_max_speed", 0.35);
    declare_parameter<double>("turn_90_tolerance_rad", 0.04);
    declare_parameter<double>("turn_90_timeout_sec", 15.0);
    declare_parameter<double>("turn_90_p_gain", 1.5);

    // Map-frame absolute yaw turn parameters (/receptionist/left)
    declare_parameter<double>("turn_left_map_yaw_deg", 0.0);
    declare_parameter<double>("turn_left_min_speed", 0.14);
    declare_parameter<double>("turn_left_max_speed", 0.35);
    declare_parameter<double>("turn_left_tolerance_rad", 0.04);
    declare_parameter<double>("turn_left_timeout_sec", 15.0);
    declare_parameter<double>("turn_left_p_gain", 1.5);

    // Chair search & motion parameters
    declare_parameter<std::string>("cmd_vel_topic", "cmd_vel");
    declare_parameter<std::string>("chair_topic", "/vision/chair_detections");
    declare_parameter<std::string>("empty_seat_topic", "/vision/empty_seat");
    declare_parameter<std::string>("odom_topic", "/odom");
    declare_parameter<double>("search_chair_ref_yaw", -1.57);
    declare_parameter<double>("search_chair_angular_speed", 0.30);
    declare_parameter<double>("search_chair_timeout_sec", 40.0);
    declare_parameter<int>("search_chair_confirm_frames", 2);
    declare_parameter<double>("search_chair_sweep_angle_deg", 30.0);
    declare_parameter<double>("chair_max_dist", 3.0);
    declare_parameter<double>("align_chair_p_gain", 1.5);
    declare_parameter<double>("align_chair_min_speed", 0.12);
    declare_parameter<double>("align_chair_max_speed", 0.28);
    declare_parameter<double>("align_chair_tolerance_rad", 0.035);
    camera_width_ = 672;
    declare_parameter<int>("camera_width", 672);

    global_frame_ = get_parameter("global_frame").as_string();
    nav_timeout_sec_ = get_parameter("nav_timeout_sec").as_double();
    std::string announce_topic = get_parameter("announce_topic").as_string();

    host_name_ = get_parameter("host_name").as_string();
    person_label_topic_ = get_parameter("person_label_topic").as_string();
    guest_1_name_topic_ = get_parameter("guest_1_name_topic").as_string();
    guest_2_name_topic_ = get_parameter("guest_2_name_topic").as_string();
    host_seated_max_dist_ = get_parameter("host_seated_max_dist").as_double();
    guest_door_distance_ = get_parameter("guest_door_distance").as_double();
    guest_door_distance_min_ = get_parameter("guest_door_distance_min").as_double();
    guest_door_step_forward_speed_ = get_parameter("guest_door_step_forward_speed").as_double();
    guest_door_step_forward_min_speed_ = get_parameter("guest_door_step_forward_min_speed").as_double();
    guest_door_step_forward_p_gain_ = get_parameter("guest_door_step_forward_p_gain").as_double();
    guest_door_step_back_speed_ = get_parameter("guest_door_step_back_speed").as_double();
    guest_door_step_back_min_speed_ = get_parameter("guest_door_step_back_min_speed").as_double();
    guest_door_step_back_p_gain_ = get_parameter("guest_door_step_back_p_gain").as_double();
    guest_door_adjust_angular_speed_ = get_parameter("guest_door_adjust_angular_speed").as_double();
    guest_door_timeout_sec_ = get_parameter("guest_door_timeout_sec").as_double();
    guest_approach_distance_ = get_parameter("guest_approach_distance").as_double();
    guest_approach_speed_ = get_parameter("guest_approach_speed").as_double();
    guest_approach_timeout_sec_ = get_parameter("guest_approach_timeout_sec").as_double();
    guest_door_align_tolerance_ = get_parameter("guest_door_align_tolerance").as_double();
    face_person_angular_speed_ = get_parameter("face_person_angular_speed").as_double();
    face_person_timeout_sec_ = get_parameter("face_person_timeout_sec").as_double();
    face_person_p_gain_ = get_parameter("face_person_p_gain").as_double();
    face_person_min_speed_ = get_parameter("face_person_min_speed").as_double();
    face_person_max_speed_ = get_parameter("face_person_max_speed").as_double();
    face_person_tolerance_rad_ = get_parameter("face_person_tolerance_rad").as_double();

    turn_90_angular_speed_ = get_parameter("turn_90_angular_speed").as_double();
    turn_90_min_speed_ = get_parameter("turn_90_min_speed").as_double();
    turn_90_max_speed_ = get_parameter("turn_90_max_speed").as_double();
    turn_90_tolerance_rad_ = get_parameter("turn_90_tolerance_rad").as_double();
    turn_90_timeout_sec_ = get_parameter("turn_90_timeout_sec").as_double();
    turn_90_p_gain_ = get_parameter("turn_90_p_gain").as_double();

    turn_left_map_yaw_deg_ = get_parameter("turn_left_map_yaw_deg").as_double();
    turn_left_min_speed_ = get_parameter("turn_left_min_speed").as_double();
    turn_left_max_speed_ = get_parameter("turn_left_max_speed").as_double();
    turn_left_tolerance_rad_ = get_parameter("turn_left_tolerance_rad").as_double();
    turn_left_timeout_sec_ = get_parameter("turn_left_timeout_sec").as_double();
    turn_left_p_gain_ = get_parameter("turn_left_p_gain").as_double();

    cmd_vel_topic_ = get_parameter("cmd_vel_topic").as_string();
    chair_topic_ = get_parameter("chair_topic").as_string();
    empty_seat_topic_ = get_parameter("empty_seat_topic").as_string();
    odom_topic_ = get_parameter("odom_topic").as_string();
    search_chair_ref_yaw_ = get_parameter("search_chair_ref_yaw").as_double();
    search_chair_angular_speed_ = get_parameter("search_chair_angular_speed").as_double();
    search_chair_timeout_sec_ = get_parameter("search_chair_timeout_sec").as_double();
    search_chair_confirm_frames_ = get_parameter("search_chair_confirm_frames").as_int();
    search_chair_sweep_angle_deg_ = get_parameter("search_chair_sweep_angle_deg").as_double();
    chair_max_dist_ = get_parameter("chair_max_dist").as_double();
    align_chair_p_gain_ = get_parameter("align_chair_p_gain").as_double();
    align_chair_min_speed_ = get_parameter("align_chair_min_speed").as_double();
    align_chair_max_speed_ = get_parameter("align_chair_max_speed").as_double();
    align_chair_tolerance_rad_ = get_parameter("align_chair_tolerance_rad").as_double();
    camera_width_ = get_parameter("camera_width").as_int();

    // TF2 Buffer & Listener untuk mendapatkan posisi/yaw robot di frame map
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

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

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, rclcpp::SensorDataQoS(),
      [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
        onOdom(msg);
      },
      sub_opts);

    // Subscriptions for person tracking (Host, Guest 1 & Guest 2)
    person_sub_ = create_subscription<std_msgs::msg::String>(
      person_label_topic_, rclcpp::SensorDataQoS(),
      [this](const std_msgs::msg::String::SharedPtr msg) {
        if (!msg) return;
        std::lock_guard<std::mutex> lock(person_mutex_);
        latest_persons_str_ = msg->data;
        person_detected_ = !msg->data.empty() && (msg->data != "NO_PERSON");
        last_person_time_ = now();
      },
      sub_opts);

    guest_1_name_sub_ = create_subscription<std_msgs::msg::String>(
      guest_1_name_topic_, 10,
      [this](const std_msgs::msg::String::SharedPtr msg) {
        if (!msg || msg->data.empty()) return;
        std::lock_guard<std::mutex> lock(guest_mutex_);
        guest_1_name_ = msg->data;
      },
      sub_opts);

    // Fallback subscriber dari receptionist_manager jika topic berbeda
    if (guest_1_name_topic_ != "/receptionist/guest_1_name") {
      guest_1_name_fallback_sub_ = create_subscription<std_msgs::msg::String>(
        "/receptionist/guest_1_name", 10,
        [this](const std_msgs::msg::String::SharedPtr msg) {
          if (!msg || msg->data.empty()) return;
          std::lock_guard<std::mutex> lock(guest_mutex_);
          guest_1_name_ = msg->data;
        },
        sub_opts);
    }

    guest_2_name_sub_ = create_subscription<std_msgs::msg::String>(
      guest_2_name_topic_, 10,
      [this](const std_msgs::msg::String::SharedPtr msg) {
        if (!msg || msg->data.empty()) return;
        std::lock_guard<std::mutex> lock(guest_mutex_);
        guest_2_name_ = msg->data;
      },
      sub_opts);

    if (guest_2_name_topic_ != "/receptionist/guest_2_name") {
      guest_2_name_fallback_sub_ = create_subscription<std_msgs::msg::String>(
        "/receptionist/guest_2_name", 10,
        [this](const std_msgs::msg::String::SharedPtr msg) {
          if (!msg || msg->data.empty()) return;
          std::lock_guard<std::mutex> lock(guest_mutex_);
          guest_2_name_ = msg->data;
        },
        sub_opts);
    }

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

    // Services for facing persons: /receptionist/host, /receptionist/guest1, /receptionist/guest2
    srv_face_host_ = create_service<std_srvs::srv::Trigger>(
      "/receptionist/host",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleFacePerson("host", req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    srv_face_guest_1_ = create_service<std_srvs::srv::Trigger>(
      "/receptionist/guest1",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleFacePerson("guest_1", req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    srv_face_guest_1_alias_ = create_service<std_srvs::srv::Trigger>(
      "/receptionist/guest_1",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleFacePerson("guest_1", req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    srv_face_guest_2_ = create_service<std_srvs::srv::Trigger>(
      "/receptionist/guest2",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleFacePerson("guest_2", req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    srv_face_guest_2_alias_ = create_service<std_srvs::srv::Trigger>(
      "/receptionist/guest_2",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleFacePerson("guest_2", req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    // Services for adjusting heading and distance to guest at door
    srv_adjust_guest_ = create_service<std_srvs::srv::Trigger>(
      "/receptionist/adjust_guest",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleAdjustGuest(req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    srv_adjust_guest_alias_ = create_service<std_srvs::srv::Trigger>(
      "/receptionist/align_guest",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleAdjustGuest(req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    srv_approach_guest_ = create_service<std_srvs::srv::Trigger>(
      "/receptionist/approach_guest",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleApproachGuest(req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    // Service /receptionist/left menghadap yaw absolut terhadap peta (default 0 deg),
    // sedangkan /receptionist/right tetap rotasi relatif 90 derajat.
    srv_turn_left_ = create_service<std_srvs::srv::Trigger>(
      "/receptionist/left",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleTurnToMapYaw(req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    srv_turn_right_ = create_service<std_srvs::srv::Trigger>(
      "/receptionist/right",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleTurnRelative(-M_PI / 2.0, "right", req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    srv_clear_locks_ = create_service<std_srvs::srv::Trigger>(
      "/receptionist/clear_living_room_locks",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleClearLivingRoomLocks(req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    srv_reset_living_room_yaw_ = create_service<std_srvs::srv::Trigger>(
      "/receptionist/reset_living_room_yaw",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleResetLivingRoomYaw(req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    RCLCPP_INFO(get_logger(), "\033[1;36m[RECEPTIONIST]\033[0m\n  Nav: Node ready");
    publishStatus("Receptionist navigation node ready");
  }

  void publishStatus(const std::string & msg_str)
  {
    std_msgs::msg::String msg;
    msg.data = msg_str;
    status_pub_->publish(msg);
    RCLCPP_INFO(get_logger(), "\033[1;36m[RECEPTIONIST]\033[0m\n  Status: %s", msg_str.c_str());
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


  void onOdom(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    if (!msg) return;
    std::lock_guard<std::mutex> lock(odom_mutex_);
    const auto & q = msg->pose.pose.orientation;
    const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    current_yaw_ = std::atan2(siny_cosp, cosy_cosp);
    has_odom_ = true;
    last_odom_time_ = now();
  }

  bool getRobotYaw(double & yaw_out)
  {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    if (has_odom_ && (now() - last_odom_time_).seconds() < 1.0) {
      yaw_out = current_yaw_;
      return true;
    }
    return false;
  }

  bool getMapYaw(double & yaw_out)
  {
    if (!tf_buffer_) {
      return false;
    }
    try {
      const auto tf_stamped = tf_buffer_->lookupTransform(
        global_frame_, "base_link", tf2::TimePointZero);
      const auto & q = tf_stamped.transform.rotation;
      const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
      const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
      yaw_out = std::atan2(siny_cosp, cosy_cosp);
      return true;
    } catch (const tf2::TransformException &) {
      return false;
    }
  }


  static bool isSeatingLabel(const std::string & item)
  {
    return item.find("chair") != std::string::npos ||
           item.find("sofa") != std::string::npos ||
           item.find("couch") != std::string::npos ||
           item.find("bench") != std::string::npos;
  }

  void onChairDetections(const std_msgs::msg::String::SharedPtr msg)
  {
    if (!msg) return;
    std::lock_guard<std::mutex> lock(chair_mutex_);

    // Format dari athome_vision: "chair:x,y,w,h|X,Y,Z|empty/occupied|left/center/right;chair:..."
    bool has_empty_chair = false;
    std::string empty_chairs_info;

    bool has_occupied_chair = false;
    std::string occupied_chairs_info;

    bool has_any_chair = false;
    std::string all_chairs_info;

    if (!msg->data.empty()) {
      std::stringstream ss(msg->data);
      std::string item;
      while (std::getline(ss, item, ';')) {
        const bool is_seating = isSeatingLabel(item);
        if (!is_seating) {
          continue;
        }

        bool dist_ok = true;
        const auto bar1 = item.find('|');
        if (bar1 != std::string::npos) {
          const auto bar2 = item.find('|', bar1 + 1);
          std::string coord_str = (bar2 != std::string::npos)
            ? item.substr(bar1 + 1, bar2 - bar1 - 1)
            : item.substr(bar1 + 1);
          double px = 0.0, py = 0.0, pz = 0.0;
          if (coord_str.find("nan") == std::string::npos &&
              std::sscanf(coord_str.c_str(), "%lf,%lf,%lf", &px, &py, &pz) == 3) {
            if (px > 0.1) {
              double d = std::hypot(px, py);
              if (d >= chair_max_dist_) {
                dist_ok = false;
              }
            }
          }
        }

        if (dist_ok) {
          has_any_chair = true;
          if (!all_chairs_info.empty()) {
            all_chairs_info += ";";
          }
          all_chairs_info += item;
        }

        // Cek kursi/sofa kosong (empty) dengan filter jarak < chair_max_dist_ (3.0 meter)
        if (dist_ok &&
            item.find("empty") != std::string::npos &&
            item.find("occupied") == std::string::npos) {
          has_empty_chair = true;
          if (!empty_chairs_info.empty()) {
            empty_chairs_info += ";";
          }
          empty_chairs_info += item;
        }
        // Cek orang yang sedang duduk di kursi/sofa (occupied)
        if (item.find("occupied") != std::string::npos) {
          has_occupied_chair = true;
          if (!occupied_chairs_info.empty()) {
            occupied_chairs_info += ";";
          }
          occupied_chairs_info += item;
        }
      }
    }

    if (has_any_chair) {
      last_all_chairs_time_ = now();
      latest_all_chairs_info_ = all_chairs_info;
    }

    if (has_empty_chair) {
      last_chair_time_ = now();
      latest_chair_info_ = empty_chairs_info;
      consecutive_chair_frames_++;
      chair_detected_ = true;
    } else {
      consecutive_chair_frames_ = 0;
      if ((now() - last_chair_time_).seconds() > 2.0) {
        chair_detected_ = false;
      }
    }

    if (has_occupied_chair) {
      latest_occupied_chair_info_ = occupied_chairs_info;
      last_occupied_chair_time_ = now();
      occupied_chair_detected_ = true;
    } else {
      if ((now() - last_occupied_chair_time_).seconds() > 2.0) {
        occupied_chair_detected_ = false;
      }
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
      chair_detected_ = true;
    } else if (msg->data == "none") {
      if (latest_chair_info_.rfind("empty_seat:", 0) == 0) {
        consecutive_chair_frames_ = 0;
        if ((now() - last_chair_time_).seconds() > 2.0) {
          chair_detected_ = false;
        }
      }
    }
  }

  EmptyChairTarget parseEmptyChair(const std::string & data, bool allow_occupied = false) const
  {
    EmptyChairTarget target;
    if (data.empty()) return target;

    // Format 1: Dari /vision/chair_detections ("chair:x,y,w,h|X,Y,Z|empty|side;...")
    std::stringstream ss(data);
    std::string item;
    double best_dist = 1e9;
    double best_bbox_area = 0.0;
    bool target_has_3d = false;
    bool target_is_empty = false;

    while (std::getline(ss, item, ';')) {
      if (!isSeatingLabel(item)) {
        continue;
      }
      const bool item_is_empty =
        (item.find("empty") != std::string::npos && item.find("occupied") == std::string::npos);
      if (!allow_occupied && !item_is_empty) {
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
      const auto colon_pos = parts[0].find(':');
      if (colon_pos == std::string::npos ||
          std::sscanf(parts[0].c_str() + colon_pos + 1, "%d,%d,%d,%d", &rx, &ry, &rw, &rh) != 4) {
        continue;
      }

      double angle = 0.0;
      double dist = 0.0;
      if (has_3d) {
        // Kamera vision membalik sumbu Y (-point.y), kita negasikan kembali (-pos_y)
        // persis seperti di restaurant_node (p.y = -py) agar sesuai standar ROS (+Y Kiri, -Y Kanan).
        const double corrected_y = -pos_y;
        dist = std::hypot(pos_x, corrected_y);
        // Filter: hanya anggap kursi jika jarak < chair_max_dist_ (3.0 meter)
        if (dist >= chair_max_dist_) {
          continue;
        }
        angle = std::atan2(corrected_y, pos_x);
      } else {
        // Fallback pixel bbox center jika depth bernilai nan
        int cx = rx + rw / 2;
        double half_w = camera_width_ > 0 ? (camera_width_ / 2.0) : 336.0;
        angle = -((cx - half_w) / half_w) * 0.75;
        dist = 1.5;
      }

      const double bbox_area = static_cast<double>(rw * rh);
      bool is_better = false;

      if (!target.valid) {
        is_better = true;
      } else if (has_3d && !target_has_3d) {
        // Selalu prioritaskan kursi dengan koordinat 3D valid
        is_better = true;
      } else if (allow_occupied && has_3d == target_has_3d && item_is_empty != target_is_empty) {
        // Jika sama-sama memiliki/tidak memiliki 3D, prioritaskan kursi yang berstatus empty
        is_better = item_is_empty;
      } else if (has_3d && target_has_3d) {
        // Jika jarak kedua kursi hampir sama (< 0.25m), pilih yang sudutnya paling dekat dengan tengah kamera;
        // jika selisih jarak cukup jauh, pilih kursi yang paling dekat (dist terkecil).
        if (std::abs(dist - best_dist) < 0.25) {
          if (std::abs(angle) < std::abs(target.angle_err)) {
            is_better = true;
          }
        } else if (dist < best_dist) {
          is_better = true;
        }
      } else if (!has_3d && !target_has_3d) {
        // Jika keduanya tidak punya 3D, pilih bounding box terbesar (indikasi lebih dekat ke kamera)
        if (bbox_area > best_bbox_area) {
          is_better = true;
        }
      }

      if (is_better) {
        target.valid = true;
        target.angle_err = angle;
        target.dist = dist;
        target.side = parts[3];
        target.raw_info = item;
        target.has_bbox = (rw > 0 && rh > 0);
        target.cx = rx + rw * 0.5;
        target_has_3d = has_3d;
        target_is_empty = item_is_empty;
        best_dist = dist;
        best_bbox_area = bbox_area;
      }
    }

    // Format 2: Fallback jika berasal dari /vision/empty_seat (misal "empty_seat:left|x,y,z|depth" atau "empty_seat:left")
    if (!target.valid) {
      std::string seat_str = data;
      if (seat_str.rfind("empty_seat:", 0) == 0) {
        seat_str = seat_str.substr(11);
      }

      std::string side_str = seat_str;
      double px = 0.0, py = 0.0, pz = 0.0, pdepth = 0.0;
      bool has_coords = false;
      const auto pipe1 = seat_str.find('|');
      if (pipe1 != std::string::npos) {
        side_str = seat_str.substr(0, pipe1);
        const auto pipe2 = seat_str.find('|', pipe1 + 1);
        std::string xyz_str = (pipe2 != std::string::npos)
          ? seat_str.substr(pipe1 + 1, pipe2 - pipe1 - 1)
          : seat_str.substr(pipe1 + 1);
        if (xyz_str.find("nan") == std::string::npos &&
            std::sscanf(xyz_str.c_str(), "%lf,%lf,%lf", &px, &py, &pz) == 3) {
          has_coords = true;
          if (pipe2 != std::string::npos) {
            std::sscanf(seat_str.substr(pipe2 + 1).c_str(), "%lf", &pdepth);
          } else {
            pdepth = std::hypot(px, py);
          }
        }
      }

      if (has_coords && px > 0.1) {
        // Kamera vision membalik sumbu Y (-point.y), kita negasikan kembali (-py)
        // persis seperti di restaurant_node (p.y = -py) agar sesuai standar ROS (+Y Kiri, -Y Kanan).
        const double corrected_y = -py;
        double d = (pdepth > 0.0) ? pdepth : std::hypot(px, corrected_y);
        if (d < chair_max_dist_) {
          target.valid = true;
          target.dist = d;
          target.angle_err = std::atan2(corrected_y, px);
          target.side = (corrected_y >= 0.20) ? "left" : (corrected_y <= -0.20 ? "right" : "center");
          target.raw_info = "empty_seat:" + target.side;
          target_has_3d = true;
        }
      } else {
        // Fallback jika hanya ada format label string:
        // Karena vision mem-publish side terbalik akibat efek mirror, balikkan kembali:
        std::string corrected_side = side_str;
        if (side_str == "left") corrected_side = "right";
        else if (side_str == "right") corrected_side = "left";

        if (corrected_side == "left") {
          target.valid = true;
          target.angle_err = 0.35; // Kursi di kiri -> putar kiri (+0.35 rad)
          target.dist = 1.5;
          target.side = "left";
          target.raw_info = "empty_seat:left";
        } else if (corrected_side == "right") {
          target.valid = true;
          target.angle_err = -0.35; // Kursi di kanan -> putar kanan (-0.35 rad)
          target.dist = 1.5;
          target.side = "right";
          target.raw_info = "empty_seat:right";
        } else if (corrected_side == "center") {
          target.valid = true;
          target.angle_err = 0.0; // Sudah tepat di tengah
          target.dist = 1.5;
          target.side = "center";
          target.raw_info = "empty_seat:center";
        }
      }
    }

    return target;
  }

  EmptyChairTarget parseOccupiedChair(const std::string & data) const
  {
    EmptyChairTarget target;
    if (data.empty()) return target;

    std::stringstream ss(data);
    std::string item;
    double best_dist = 999.0;

    while (std::getline(ss, item, ';')) {
      const bool is_seating = (item.find("chair") != std::string::npos ||
                               item.find("sofa") != std::string::npos ||
                               item.find("couch") != std::string::npos);
      if (!is_seating ||
          item.find("occupied") == std::string::npos) {
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
      const auto colon_pos = parts[0].find(':');
      if (colon_pos == std::string::npos ||
          std::sscanf(parts[0].c_str() + colon_pos + 1, "%d,%d,%d,%d", &rx, &ry, &rw, &rh) != 4) {
        continue;
      }

      double angle = 0.0;
      double dist = 0.0;
      std::string side = parts.size() > 3 ? parts[3] : "center";
      if (has_3d) {
        // Kamera vision membalik sumbu Y (-point.y), kita negasikan kembali (-pos_y)
        // persis seperti di restaurant_node (p.y = -py) agar sesuai standar ROS (+Y Kiri, -Y Kanan).
        const double corrected_y = -pos_y;
        dist = std::hypot(pos_x, corrected_y);
        angle = std::atan2(corrected_y, pos_x);
        side = (corrected_y >= 0.20) ? "left" : (corrected_y <= -0.20 ? "right" : "center");
      } else {
        int cx = rx + rw / 2;
        double half_w = camera_width_ > 0 ? (camera_width_ / 2.0) : 336.0;
        angle = -((cx - half_w) / half_w) * 0.75;
        dist = 1.5;
        if (side == "left") side = "right";
        else if (side == "right") side = "left";
      }

      if (dist < best_dist || !target.valid) {
        best_dist = dist;
        target.valid = true;
        target.angle_err = angle;
        target.dist = dist;
        target.side = side;
        target.raw_info = item;
      }
    }

    return target;
  }

  PersonTarget parseTargetPerson(const std::string & data, const std::string & target_role)
  {
    PersonTarget target;
    if (data.empty() || data == "NO_PERSON") return target;

    auto toLower = [](std::string s) {
      std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
      return s;
    };

    const std::string host_name_lower = toLower(host_name_);
    std::string guest_1_name_lower;
    std::string guest_2_name_lower;
    {
      std::lock_guard<std::mutex> lk(guest_mutex_);
      guest_1_name_lower = toLower(guest_1_name_);
      guest_2_name_lower = toLower(guest_2_name_);
    }

    std::stringstream ss(data);
    std::string item;
    double best_priority = -1e9;

    while (std::getline(ss, item, ';')) {
      if (item.empty()) continue;

      const auto colon = item.find(':');
      const auto bar1 = item.find('|');
      if (colon == std::string::npos || bar1 == std::string::npos || bar1 <= colon) continue;

      std::string label = item.substr(0, colon);
      std::string label_lower = toLower(label);

      // Parse BBox
      int rx = 0, ry = 0, rw = 0, rh = 0;
      std::string bbox_str = item.substr(colon + 1, bar1 - colon - 1);
      std::sscanf(bbox_str.c_str(), "%d,%d,%d,%d", &rx, &ry, &rw, &rh);

      // Parse 3D coords
      const auto bar2 = item.find('|', bar1 + 1);
      std::string coord_str = (bar2 != std::string::npos)
        ? item.substr(bar1 + 1, bar2 - bar1 - 1)
        : item.substr(bar1 + 1);

      double pos_x = 0.0, pos_y = 0.0, pos_z = 0.0;
      bool has_3d = false;
      if (coord_str.find("nan") == std::string::npos) {
        if (std::sscanf(coord_str.c_str(), "%lf,%lf,%lf", &pos_x, &pos_y, &pos_z) == 3) {
          if (pos_x > 0.05 || std::abs(pos_y) > 0.05) {
            has_3d = true;
          }
        }
      }

      double angle = 0.0;
      double dist = 1.5;
      if (has_3d) {
        // Kamera vision membalik sumbu Y (-point.y), kita negasikan kembali (-pos_y)
        // persis seperti di restaurant_node (p.y = -py) agar sesuai standar ROS (+Y Kiri, -Y Kanan).
        const double corrected_y = -pos_y;
        dist = std::hypot(pos_x, corrected_y);
        angle = std::atan2(corrected_y, pos_x);
      } else {
        int cx = rx + rw / 2;
        double half_w = camera_width_ > 0 ? (camera_width_ / 2.0) : 336.0;
        angle = -((cx - half_w) / half_w) * 0.75;
      }

      const bool is_sitting = (label_lower.find("seated") != std::string::npos ||
                               label_lower.find("sitting") != std::string::npos ||
                               label_lower.find("duduk") != std::string::npos ||
                               item.find("sitting=1") != std::string::npos);
      const bool is_standing_3d = (item.find("standing=1") != std::string::npos);

      // Cek apakah full-body terlihat (minimal hip 18/22 dan knee 19/23 valid di kpt, serta bbox tinggi)
      bool has_lower_body_kpt = false;
      const auto kpt_pos = item.find("|kpt=");
      if (kpt_pos != std::string::npos) {
        std::string kpt_str = item.substr(kpt_pos + 5);
        std::vector<double> coords;
        coords.reserve(68);
        std::stringstream kss(kpt_str);
        std::string tok;
        while (std::getline(kss, tok, ',')) {
          try { coords.push_back(std::stod(tok)); } catch (...) { coords.push_back(-1.0); }
        }
        auto kpt_valid = [&](size_t idx) -> bool {
          return (idx * 2 + 1) < coords.size() && coords[idx * 2] >= 0.0 && coords[idx * 2 + 1] >= 0.0;
        };
        const bool has_hip = kpt_valid(18) || kpt_valid(22);
        const bool has_knee = kpt_valid(19) || kpt_valid(23);
        has_lower_body_kpt = (has_hip && has_knee);
      }
      const bool is_tall_bbox = (rw > 0 && rh >= static_cast<int>(rw * 1.70));
      const bool is_fullbody_standing = !is_sitting && (is_standing_3d || (has_lower_body_kpt && is_tall_bbox));

      // Evaluasi apakah orang ini sesuai dengan target_role
      bool is_match = false;
      double priority = 0.0;

      if (target_role == "door_guest" || target_role == "door_guest_approach") {
        // Tamu di pintu harus berdiri di depan robot (bukan duduk, bukan Host)
        if (is_sitting) {
          continue;
        }
        if (!host_name_lower.empty() && label_lower.find(host_name_lower) != std::string::npos) {
          continue;
        }
        if (std::abs(angle) > 0.50) {
          continue;
        }
        // Syarat wajib full-body untuk "door_guest" dinonaktifkan agar tamu yang sebagian
        // tubuh bawahnya terpotong tetap bisa dikunci oleh adjust_guest.
        (void)is_fullbody_standing;
        is_match = true;
        // Prioritaskan tamu yang paling lurus di depan kamera, lalu jarak terdekat
        priority = 100.0 - (std::abs(angle) * 25.0) - (dist * 5.0);
      } else if (target_role == "host" || target_role == "guest") {
        // GATING POSTUR & JARAK: Tamu yang sedang dipandu berdiri dekat robot (< 1.4m)
        // BUKAN Host di ruang tamu!
        if (!is_sitting && dist < 1.4) {
          continue;
        }

        if (!host_name_lower.empty() && label_lower.find(host_name_lower) != std::string::npos) {
          is_match = true;
          priority = 100.0;
        } else if (label_lower.find("host") != std::string::npos) {
          is_match = true;
          priority = 95.0;
        } else if (is_sitting && dist < host_seated_max_dist_) {
          // Orang duduk di kursi yang bukan Guest 1 atau Guest 2
          const bool is_g1 = !guest_1_name_lower.empty() && label_lower.find(guest_1_name_lower) != std::string::npos;
          const bool is_g2 = !guest_2_name_lower.empty() && label_lower.find(guest_2_name_lower) != std::string::npos;
          if (!is_g1 && !is_g2) {
            is_match = true;
            priority = 90.0;
          }
        } else if ((label_lower == "guest" || label_lower == "person" || label_lower == "unknown") &&
                   (guest_1_name_lower.empty() || label_lower.find(guest_1_name_lower) == std::string::npos) &&
                   (guest_2_name_lower.empty() || label_lower.find(guest_2_name_lower) == std::string::npos) &&
                   dist < 5.0) {
          // Fallback jika belum dikenali wajah: hanya terima jika duduk atau berada di area duduk (>= 1.5m)
          if (is_sitting || dist >= 1.5) {
            is_match = true;
            priority = is_sitting ? 85.0 : (70.0 - dist);
          }
        }
      } else if (target_role == "guest_1") {
        // GATING POSTUR & JARAK: Di Cycle 2 Guest 1 sudah duduk di ruang tamu.
        // Tamu yang berdiri dekat robot (< 1.4m) adalah Guest 2, BUKAN Guest 1!
        if (!is_sitting && dist < 1.4) {
          continue;
        }

        if (!guest_1_name_lower.empty() && label_lower.find(guest_1_name_lower) != std::string::npos) {
          is_match = true;
          priority = 100.0;
        } else if (label_lower.find("guest_1") != std::string::npos || label_lower.find("guest 1") != std::string::npos) {
          is_match = true;
          priority = 90.0;
        } else if (is_sitting && dist < host_seated_max_dist_) {
          const bool is_host = (!host_name_lower.empty() && label_lower.find(host_name_lower) != std::string::npos) ||
                               label_lower.find("host") != std::string::npos;
          const bool is_g2 = !guest_2_name_lower.empty() && label_lower.find(guest_2_name_lower) != std::string::npos;
          if (!is_host && !is_g2) {
            is_match = true;
            priority = 80.0;
          }
        } else if ((guest_2_name_lower.empty() || label_lower.find(guest_2_name_lower) == std::string::npos) &&
                   (host_name_lower.empty() || label_lower.find(host_name_lower) == std::string::npos) &&
                   label_lower.find("host") == std::string::npos) {
          if (is_sitting || dist >= 1.5) {
            is_match = true;
            priority = is_sitting ? 75.0 : (60.0 - dist);
          }
        }
      } else if (target_role == "guest_2") {
        // GATING POSTUR & JARAK: Host dan Guest 1 sedang duduk di kursi (> 1.4m dan duduk).
        // Jangan pilih orang duduk jauh sebagai Guest 2!
        if (is_sitting && dist > 1.4) {
          continue;
        }

        if (!guest_2_name_lower.empty() && label_lower.find(guest_2_name_lower) != std::string::npos) {
          is_match = true;
          priority = 100.0;
        } else if (label_lower.find("guest_2") != std::string::npos || label_lower.find("guest 2") != std::string::npos) {
          is_match = true;
          priority = 90.0;
        } else if (!is_sitting && dist < 1.8) {
          const bool is_host = (!host_name_lower.empty() && label_lower.find(host_name_lower) != std::string::npos) ||
                               label_lower.find("host") != std::string::npos;
          const bool is_g1 = !guest_1_name_lower.empty() && label_lower.find(guest_1_name_lower) != std::string::npos;
          if (!is_host && !is_g1) {
            is_match = true;
            priority = 85.0 - (dist * 5.0);
          }
        } else if ((guest_1_name_lower.empty() || label_lower.find(guest_1_name_lower) == std::string::npos) &&
                   (host_name_lower.empty() || label_lower.find(host_name_lower) == std::string::npos) &&
                   label_lower.find("host") == std::string::npos) {
          if (!is_sitting && dist < 2.2) {
            is_match = true;
            priority = 50.0 - (dist * 5.0);
          }
        }
      }

      if (is_match && priority > best_priority) {
        best_priority = priority;
        target.valid = true;
        target.angle_err = angle;
        target.dist = dist;
        target.label = label;
      }
    }

    return target;
  }

  bool rotateToAbsYaw(
    double target_abs_yaw,
    std::atomic<bool> & active_flag,
    double tol_rad,
    double p_gain,
    double min_spd,
    double max_spd,
    double timeout_sec,
    const std::string & tag)
  {
    auto start_time = now();
    rclcpp::Rate loop_rate(25.0);
    int centered_frames = 0;
    bool aligned = false;

    while (rclcpp::ok() && active_flag.load()) {
      if ((now() - start_time).seconds() > timeout_sec) {
        RCLCPP_WARN(get_logger(), "[%s] Rotate to locked yaw %.2f rad timed out", tag.c_str(), target_abs_yaw);
        break;
      }
      double cur_yaw = 0.0;
      if (!getRobotYaw(cur_yaw)) {
        loop_rate.sleep();
        continue;
      }
      const double err = normalize_angle(target_abs_yaw - cur_yaw);
      if (std::abs(err) <= tol_rad) {
        centered_frames++;
        if (centered_frames >= 3) {
          aligned = true;
          break;
        }
      } else {
        centered_frames = 0;
      }

      double w = p_gain * err;
      double spd = std::clamp(std::abs(w), min_spd, max_spd);
      geometry_msgs::msg::Twist twist;
      twist.angular.z = (err >= 0.0) ? spd : -spd;
      cmd_vel_pub_->publish(twist);
      loop_rate.sleep();
    }

    stopRobot();
    return aligned;
  }

  void clearLivingRoomLocks()
  {
    std::lock_guard<std::mutex> lock(living_room_lock_mutex_);
    has_locked_host_yaw_ = false;
    has_locked_guest1_yaw_ = false;
    has_locked_guest2_yaw_ = false;
    has_locked_empty_chair_yaw_ = false;
    locked_host_yaw_ = 0.0;
    locked_guest1_yaw_ = 0.0;
    locked_guest2_yaw_ = 0.0;
    locked_empty_chair_yaw_ = 0.0;
  }

  bool isYawCollidingWithLockedPeople(double abs_yaw, double min_sep_rad = 0.25)
  {
    std::lock_guard<std::mutex> lock(living_room_lock_mutex_);
    if (has_locked_host_yaw_ && std::abs(normalize_angle(abs_yaw - locked_host_yaw_)) < min_sep_rad) {
      return true;
    }
    if (has_locked_guest1_yaw_ && std::abs(normalize_angle(abs_yaw - locked_guest1_yaw_)) < min_sep_rad) {
      return true;
    }
    return false;
  }

  void handleClearLivingRoomLocks(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    clearLivingRoomLocks();
    res->success = true;
    res->message = "Cleared temporary living room position locks";
    RCLCPP_INFO(get_logger(), "\033[1;36m[LIVING ROOM]\033[0m %s", res->message.c_str());
  }

  void handleResetLivingRoomYaw(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    if (is_turning_.load()) {
      res->success = false;
      res->message = "Robot is currently busy turning";
      publishStatus(res->message);
      return;
    }
    if (is_navigating_.load()) {
      res->success = false;
      res->message = "Robot is currently busy with navigation task";
      publishStatus(res->message);
      return;
    }
    if (is_searching_chair_.load()) {
      res->success = false;
      res->message = "Robot is currently busy searching for a chair";
      publishStatus(res->message);
      return;
    }
    if (is_facing_person_.exchange(true)) {
      res->success = false;
      res->message = "Robot is already busy rotating/facing";
      publishStatus(res->message);
      return;
    }

    const double target_yaw = has_living_room_arrival_yaw_
      ? living_room_arrival_yaw_
      : normalize_angle(getWaypoint("living_room").yaw * M_PI / 180.0);

    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[LIVING ROOM]\033[0m Resetting yaw in-place to %.2f rad (%.1f deg, source=%s)...",
      target_yaw, target_yaw * 180.0 / M_PI,
      has_living_room_arrival_yaw_ ? "arrival_yaw" : "waypoint_yaw");

    const bool aligned = rotateToAbsYaw(
      target_yaw, is_facing_person_,
      face_person_tolerance_rad_, face_person_p_gain_,
      face_person_min_speed_, face_person_max_speed_, 6.0, "RESET_LIVING_ROOM_YAW");

    is_facing_person_ = false;
    res->success = aligned;
    res->message = aligned
      ? "Robot rotated back to living_room yaw"
      : "Reset living_room yaw timed out";
    publishStatus(res->message);
  }

  void handleSearchChair(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    if (is_turning_.load()) {
      res->success = false;
      res->message = "Robot is currently busy turning";
      publishStatus(res->message);
      return;
    }

    if (is_navigating_.load()) {
      res->success = false;
      res->message = "Robot is currently busy with another navigation task";
      publishStatus(res->message);
      return;
    }

    if (is_facing_person_.load()) {
      res->success = false;
      res->message = "Robot is currently busy facing person";
      publishStatus(res->message);
      return;
    }

    if (is_searching_chair_.exchange(true)) {
      res->success = false;
      res->message = "Robot is already searching for a chair";
      publishStatus(res->message);
      return;
    }

    publishStatus("Service /search_chair called: Aligning to detected chair in view...");

    // Helper untuk mengambil data kursi terdeteksi terkini (empty maupun occupied) yang tidak bertabrakan dengan kursi Host/Guest 1
    // Jika use_locked_yaw = true, kunci pilihan pada kursi yang sudutnya paling dekat dengan locked_yaw
    auto get_current_empty_chair = [this](bool use_locked_yaw = false, double locked_yaw = 0.0) -> EmptyChairTarget {
      std::string raw;
      {
        std::lock_guard<std::mutex> lock(chair_mutex_);
        if (!latest_all_chairs_info_.empty() && (now() - last_all_chairs_time_).seconds() < 2.5) {
          raw = latest_all_chairs_info_;
        } else if (!latest_chair_info_.empty() && (now() - last_chair_time_).seconds() < 2.5) {
          raw = latest_chair_info_;
        }
      }
      if (raw.empty()) return EmptyChairTarget();

      double cur_yaw = 0.0;
      const bool have_yaw = getRobotYaw(cur_yaw);
      if (raw.rfind("empty_seat:", 0) == 0 || !have_yaw) {
        EmptyChairTarget ec = parseEmptyChair(raw, true);
        if (ec.valid && have_yaw && isYawCollidingWithLockedPeople(normalize_angle(cur_yaw + ec.angle_err))) {
          ec.valid = false;
        }
        return ec;
      }

      // Filter setiap kursi terdeteksi di raw terhadap posisi Host/Guest 1 yang terkunci
      std::string filtered;
      EmptyChairTarget best_locked;
      double best_yaw_diff = 0.35; // toleransi ~20 deg untuk mempertahankan kursi yang sama
      std::stringstream ss(raw);
      std::string item;
      while (std::getline(ss, item, ';')) {
        if (item.empty()) continue;
        EmptyChairTarget one = parseEmptyChair(item, true);
        if (!one.valid) continue;
        const double abs_yaw = normalize_angle(cur_yaw + one.angle_err);
        if (isYawCollidingWithLockedPeople(abs_yaw)) continue;
        if (use_locked_yaw) {
          const double y_diff = std::abs(normalize_angle(abs_yaw - locked_yaw));
          if (y_diff <= best_yaw_diff) {
            best_yaw_diff = y_diff;
            best_locked = one;
          }
        }
        if (!filtered.empty()) filtered += ";";
        filtered += item;
      }
      if (use_locked_yaw && best_locked.valid) {
        return best_locked;
      }
      return parseEmptyChair(filtered, true);
    };

    auto start_time = now();
    bool chair_found = false;
    bool aligned = false;
    EmptyChairTarget current_target;
    rclcpp::Rate loop_rate(20.0); // 20 Hz kontrol loop

    // Cek target kursi kosong dalam pandangan kamera saat ini (tunggu hingga ~1.25s jika frame sedang sinkron)
    for (int i = 0; i < 25; ++i) {
      EmptyChairTarget t = get_current_empty_chair();
      if (t.valid && t.dist < chair_max_dist_) {
        chair_found = true;
        current_target = t;
        break;
      }
      loop_rate.sleep();
    }

    // Jika pandangan tengah belum menemukan kursi kosong tetapi scan living room sudah mengunci sudutnya, putar ke sana
    bool locked_empty_used = false;
    double locked_empty_yaw = 0.0;
    if (!chair_found) {
      std::lock_guard<std::mutex> lock(living_room_lock_mutex_);
      if (has_locked_empty_chair_yaw_) {
        locked_empty_used = true;
        locked_empty_yaw = locked_empty_chair_yaw_;
      }
    }

    if (locked_empty_used) {
      RCLCPP_INFO(
        get_logger(),
        "\033[1;32m[CHAIR]\033[0m Rotating to locked empty chair yaw: %.2f rad",
        locked_empty_yaw);
      aligned = rotateToAbsYaw(locked_empty_yaw, is_searching_chair_,
        align_chair_tolerance_rad_, align_chair_p_gain_,
        align_chair_min_speed_, align_chair_max_speed_, 6.0, "CHAIR");
      stopRobot();

      // Beri jeda sejenak untuk kamera stabil dan deteksi kursi langsung di hadapan
      for (int i = 0; i < 20 && rclcpp::ok() && is_searching_chair_.load(); ++i) {
        EmptyChairTarget t = get_current_empty_chair();
        if (t.valid && t.dist < chair_max_dist_) {
          current_target = t;
          chair_found = true;
          break;
        }
        loop_rate.sleep();
      }

      if (!chair_found) {
        current_target.valid = true;
        current_target.dist = 1.8;
        current_target.side = "locked";
        current_target.raw_info = "locked_empty_chair";
        chair_found = true;
      }
    }

    // Jika kursi belum terdeteksi (baik di FOV awal maupun via locked yaw), lakukan sweep scanning
    // dan langsung berhenti seketika begitu kursi kosong terlihat di kamera!
    if (!chair_found) {
      const double sweep_rad = search_chair_sweep_angle_deg_ * M_PI / 180.0;
      double cur_yaw = 0.0;
      if (getRobotYaw(cur_yaw)) {
        for (double offset : {sweep_rad, -sweep_rad}) {
          if (!rclcpp::ok() || !is_searching_chair_.load() || chair_found) break;
          const double target_yaw = normalize_angle(cur_yaw + offset);
          auto sweep_start = now();
          while (rclcpp::ok() && is_searching_chair_.load()) {
            EmptyChairTarget t = get_current_empty_chair();
            if (t.valid && t.dist < chair_max_dist_) {
              chair_found = true;
              current_target = t;
              stopRobot();
              break;
            }
            if ((now() - sweep_start).seconds() > 3.5) break;
            double now_yaw = 0.0;
            if (!getRobotYaw(now_yaw)) break;
            const double y_err = normalize_angle(target_yaw - now_yaw);
            if (std::abs(y_err) <= align_chair_tolerance_rad_) break;
            double spd = std::clamp(std::abs(align_chair_p_gain_ * y_err),
              align_chair_min_speed_, align_chair_max_speed_);
            geometry_msgs::msg::Twist tw;
            tw.angular.z = (y_err >= 0.0) ? spd : -spd;
            cmd_vel_pub_->publish(tw);
            loop_rate.sleep();
          }
          stopRobot();
          if (chair_found) break;
          for (int i = 0; i < 10 && rclcpp::ok() && is_searching_chair_.load(); ++i) {
            EmptyChairTarget t = get_current_empty_chair();
            if (t.valid && t.dist < chair_max_dist_) {
              chair_found = true;
              current_target = t;
              break;
            }
            loop_rate.sleep();
          }
          if (chair_found) break;
        }
      }
    }

    // ALIGN: Sempurnakan hadap robot agar tepat presisi di titik tengah kursi via visual feedback
    if (chair_found && current_target.valid && current_target.side != "locked") {
      const double frame_w = camera_width_ > 0 ? static_cast<double>(camera_width_) : 1280.0;
      const double target_cx = frame_w * 0.5;
      const double align_tol_px = (frame_w >= 1000.0) ? 10.0 : 5.0;

      RCLCPP_INFO(
        get_logger(),
        "\033[1;32m[CHAIR]\033[0m Target: %s (%.2fm, %s, cx=%.1f, target_cx=%.1f, tol=%.1f px), visual aligning...",
        current_target.raw_info.c_str(), current_target.dist, current_target.side.c_str(),
        current_target.cx, target_cx, align_tol_px);
      publishStatus("Chair in view! Aligning robot heading to face chair...");

      const bool already_aligned = current_target.has_bbox
        ? (std::abs(current_target.cx - target_cx) <= align_tol_px)
        : (std::abs(current_target.angle_err) <= align_chair_tolerance_rad_);

      if (already_aligned) {
        aligned = true;
        RCLCPP_INFO(
          get_logger(),
          "\033[1;32m[CHAIR]\033[0m Already aligned: cx=%.1f (target=%.1f), err=%.2frad",
          current_target.cx, target_cx, current_target.angle_err);
      } else {
        auto align_start_time = now();
        const double align_timeout_sec = 6.0;
        int centered_frames = 0;
        double locked_chair_yaw = 0.0;
        bool has_locked_chair_yaw = getRobotYaw(locked_chair_yaw);
        if (has_locked_chair_yaw) {
          locked_chair_yaw = normalize_angle(locked_chair_yaw + current_target.angle_err);
        }

        while (rclcpp::ok() && is_searching_chair_.load()) {
          const double align_elapsed = (now() - align_start_time).seconds();
          if (align_elapsed > align_timeout_sec) {
            RCLCPP_WARN(get_logger(), "\033[1;33m[CHAIR]\033[0m Alignment timeout (%.1fs)", align_timeout_sec);
            break;
          }

          EmptyChairTarget t = get_current_empty_chair(has_locked_chair_yaw, locked_chair_yaw);
          if (!t.valid) {
            centered_frames = 0;
            geometry_msgs::msg::Twist stop_twist;
            cmd_vel_pub_->publish(stop_twist);
            loop_rate.sleep();
            continue;
          }
          current_target = t;
          double cur_yaw = 0.0;
          if (getRobotYaw(cur_yaw)) {
            locked_chair_yaw = normalize_angle(cur_yaw + current_target.angle_err);
            has_locked_chair_yaw = true;
          }

          bool is_aligned = false;
          double angular_z = 0.0;

          if (current_target.has_bbox) {
            const double err_px_x = current_target.cx - target_cx;
            if (std::abs(err_px_x) <= align_tol_px) {
              is_aligned = true;
            } else {
              // err_px_x < 0: chair on left -> rotate left (+z); err_px_x > 0: chair on right -> rotate right (-z)
              double w = -0.0012 * err_px_x;
              double spd = std::clamp(std::abs(w), align_chair_min_speed_, align_chair_max_speed_);
              angular_z = (err_px_x < 0.0) ? spd : -spd;
            }
          } else {
            const double err = current_target.angle_err;
            if (std::abs(err) <= align_chair_tolerance_rad_) {
              is_aligned = true;
            } else {
              double w = align_chair_p_gain_ * err;
              double spd = std::clamp(std::abs(w), align_chair_min_speed_, align_chair_max_speed_);
              angular_z = (err >= 0.0) ? spd : -spd;
            }
          }

          if (is_aligned) {
            centered_frames++;
            geometry_msgs::msg::Twist stop_twist;
            cmd_vel_pub_->publish(stop_twist);
            if (centered_frames >= 3) { // Terpusat stabil selama ~150 ms
              aligned = true;
              break;
            }
            loop_rate.sleep();
            continue;
          } else {
            centered_frames = 0;
          }

          geometry_msgs::msg::Twist twist;
          twist.linear.x = 0.0;
          twist.linear.y = 0.0;
          twist.angular.z = angular_z;
          cmd_vel_pub_->publish(twist);

          loop_rate.sleep();
        }
      }
    }

    // Berhenti penuh setelah arah hadap robot tepat berada pada arah kursi
    stopRobot();
    is_searching_chair_ = false;

    if (chair_found) {
      bool is_guest1_session = false;
      bool is_guest2_session = false;
      {
        std::lock_guard<std::mutex> lk(guest_mutex_);
        is_guest1_session = !guest_1_name_.empty();
        is_guest2_session = !guest_2_name_.empty();
      }
      if (is_guest2_session || is_guest1_session) {
        double cur_yaw = 0.0;
        const bool have_yaw = getRobotYaw(cur_yaw);
        double locked_yaw = 0.0;
        bool locked_ok = false;
        {
          std::lock_guard<std::mutex> lock(living_room_lock_mutex_);
          if (have_yaw) {
            locked_yaw = cur_yaw;
            locked_ok = true;
          } else if (has_locked_empty_chair_yaw_) {
            locked_yaw = locked_empty_chair_yaw_;
            locked_ok = true;
          }
          if (locked_ok) {
            if (is_guest2_session) {
              locked_guest2_yaw_ = locked_yaw;
              has_locked_guest2_yaw_ = true;
            } else {
              locked_guest1_yaw_ = locked_yaw;
              has_locked_guest1_yaw_ = true;
            }
          }
        }
        if (locked_ok) {
          RCLCPP_INFO(
            get_logger(),
            "\033[1;36m[LIVING ROOM LOCK]\033[0m Locked %s seated yaw: %.2f rad (%.1f deg)",
            is_guest2_session ? "Guest 2" : "Guest 1",
            locked_yaw, locked_yaw * 180.0 / M_PI);
        }
      }
      res->success = true;
      res->message = "Chair found and robot aligned facing chair: " + current_target.raw_info;
      publishStatus(res->message);
      RCLCPP_INFO(
        get_logger(),
        "\033[1;32m[CHAIR]\033[0m Found: %s (%.2fm, %s)",
        current_target.raw_info.c_str(), current_target.dist,
        aligned ? "aligned" : "approx");
    } else {
      res->success = false;
      res->message = "No chair detected in current view";
      publishStatus(res->message);
      RCLCPP_WARN(get_logger(), "\033[1;33m[CHAIR]\033[0m No empty chair detected");
    }
  }

  void handleFacePerson(
    const std::string & target_role,
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    if (is_turning_.load()) {
      res->success = false;
      res->message = "Robot is currently busy turning";
      publishStatus(res->message);
      return;
    }
    if (is_navigating_.load()) {
      res->success = false;
      res->message = "Robot is currently busy with navigation task";
      publishStatus(res->message);
      return;
    }
    if (is_searching_chair_.load()) {
      res->success = false;
      res->message = "Robot is currently busy searching for a chair";
      publishStatus(res->message);
      return;
    }
    if (is_facing_person_.exchange(true)) {
      res->success = false;
      res->message = "Robot is already busy facing person";
      publishStatus(res->message);
      return;
    }

    std::string current_guest_1_name;
    std::string current_guest_2_name;
    {
      std::lock_guard<std::mutex> lk(guest_mutex_);
      current_guest_1_name = guest_1_name_;
      current_guest_2_name = guest_2_name_;
    }

    std::string display_name;
    std::string srv_name;
    if (target_role == "host") {
      display_name = "Host (" + host_name_ + " / Seated)";
      srv_name = "/receptionist/host";
    } else if (target_role == "guest_1") {
      display_name = current_guest_1_name.empty() ? "Guest 1" : ("Guest 1 (" + current_guest_1_name + ")");
      srv_name = "/receptionist/guest1";
    } else if (target_role == "guest_2") {
      display_name = current_guest_2_name.empty() ? "Guest 2" : ("Guest 2 (" + current_guest_2_name + ")");
      srv_name = "/receptionist/guest2";
    } else {
      display_name = target_role;
      srv_name = "/receptionist/" + target_role;
    }
    publishStatus("Service " + srv_name + " called: Aligning to " + display_name + " in view...");

    // Gunakan sudut yaw terkunci dari hasil scan awal di living room jika tersedia
    bool use_locked_yaw = false;
    double target_locked_yaw = 0.0;
    {
      std::lock_guard<std::mutex> lock(living_room_lock_mutex_);
      if (target_role == "host" && has_locked_host_yaw_) {
        use_locked_yaw = true;
        target_locked_yaw = locked_host_yaw_;
      } else if (target_role == "guest_1" && has_locked_guest1_yaw_) {
        use_locked_yaw = true;
        target_locked_yaw = locked_guest1_yaw_;
      } else if (target_role == "guest_2" && has_locked_guest2_yaw_) {
        use_locked_yaw = true;
        target_locked_yaw = locked_guest2_yaw_;
      }
    }

    bool aligned_locked = false;
    if (use_locked_yaw) {
      RCLCPP_INFO(
        get_logger(),
        "\033[1;32m[FACE PERSON]\033[0m Rotating to locked %s yaw: %.2f rad (%.1f deg)",
        display_name.c_str(), target_locked_yaw, target_locked_yaw * 180.0 / M_PI);
      aligned_locked = rotateToAbsYaw(
        target_locked_yaw, is_facing_person_,
        face_person_tolerance_rad_, face_person_p_gain_,
        face_person_min_speed_, face_person_max_speed_, 6.0, "FACE PERSON");

      // Berhenti sejenak untuk menstabilkan kamera dan mengecek fine-centering visual
      stopRobot();
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    auto get_current_person_target = [this, &target_role, use_locked_yaw, target_locked_yaw]() -> PersonTarget {
      PersonTarget target;
      double cur_yaw = 0.0;
      const bool have_yaw = getRobotYaw(cur_yaw);

      // PRIORITAS UTAMA: Cek deteksi orang dari vision person_label
      // (mengenali Host bernama maupun "guest" tanpa nama, dan mengecualikan Guest 1/2)
      {
        std::lock_guard<std::mutex> lock(person_mutex_);
        if (person_detected_ && (now() - last_person_time_).seconds() < 1.0) {
          target = parseTargetPerson(latest_persons_str_, target_role);
          if (target.valid) {
            if (use_locked_yaw && have_yaw) {
              const double cand_yaw = normalize_angle(cur_yaw + target.angle_err);
              if (std::abs(normalize_angle(cand_yaw - target_locked_yaw)) > 0.30) {
                // Jangan melompat ke kursi orang lain saat fine-centering ke locked yaw
                target.valid = false;
              }
            }
            if (target.valid) {
              return target;
            }
          }
        }
      }

      // PRIORITAS KEDUA (Fallback khusus Host bila belum ada locked yaw): Orang yang sedang DUDUK di kursi
      if (target_role == "host" && !use_locked_yaw) {
        std::lock_guard<std::mutex> lock(chair_mutex_);
        if (occupied_chair_detected_ && (now() - last_occupied_chair_time_).seconds() < 2.0) {
          EmptyChairTarget occ = parseOccupiedChair(latest_occupied_chair_info_);
          if (occ.valid && occ.dist < host_seated_max_dist_) {
            target.valid = true;
            target.angle_err = occ.angle_err;
            target.dist = occ.dist;
            target.label = "Seated Guest (" + occ.side + ")";
            return target;
          }
        }
      }
      return target;
    };

    auto start_time = now();
    bool person_found = false;
    bool aligned = false;
    PersonTarget current_target;
    rclcpp::Rate loop_rate(20.0); // 20 Hz kontrol loop

    // Cek target dalam pandangan kamera saat ini (tunggu hingga ~1.25s jika frame sedang sinkron)
    for (int i = 0; i < 25; ++i) {
      PersonTarget t = get_current_person_target();
      if (t.valid) {
        person_found = true;
        current_target = t;
        break;
      }
      loop_rate.sleep();
    }

    // ALIGN: Menyelaraskan arah hadap robot tepat ke target (Host / Guest 1 / Guest 2)
    if (person_found) {
      RCLCPP_INFO(
        get_logger(),
        "\033[1;32m[FACE PERSON]\033[0m %s terdeteksi dalam pandangan! Label: %s (dist=%.2fm, angle=%.2frad). Menyelaraskan arah hadap...",
        display_name.c_str(), current_target.label.c_str(), current_target.dist, current_target.angle_err);
      publishStatus(display_name + " detected in view! Aligning robot heading to face target...");

      if (std::abs(current_target.angle_err) <= face_person_tolerance_rad_) {
        aligned = true;
        RCLCPP_INFO(
          get_logger(),
          "\033[1;32m[FACE PERSON]\033[0m Robot sudah tepat menghadap %s (angle_err=%.2frad <= tol=%.2frad).",
          display_name.c_str(), current_target.angle_err, face_person_tolerance_rad_);
      } else {
        auto align_start_time = now();
        const double align_timeout_sec = 6.0;
        int centered_frames = 0;

        while (rclcpp::ok() && is_facing_person_.load()) {
          const double align_elapsed = (now() - align_start_time).seconds();
          if (align_elapsed > align_timeout_sec) {
            RCLCPP_WARN(get_logger(), "[FACE PERSON] Alignment timeout (%.1fs reached), stopping.", align_timeout_sec);
            break;
          }

          PersonTarget t = get_current_person_target();
          if (!t.valid) {
            centered_frames = 0;
            geometry_msgs::msg::Twist stop_twist;
            cmd_vel_pub_->publish(stop_twist);
            loop_rate.sleep();
            continue;
          }
          current_target = t;

          const double err = current_target.angle_err;

          if (std::abs(err) <= face_person_tolerance_rad_) {
            centered_frames++;
            if (centered_frames >= 2) { // Terpusat stabil ~100 ms
              aligned = true;
              break;
            }
          } else {
            centered_frames = 0;
          }

          double w = face_person_p_gain_ * err;
          double spd = std::abs(w);
          if (spd < face_person_min_speed_) {
            spd = face_person_min_speed_;
          } else if (spd > face_person_max_speed_) {
            spd = face_person_max_speed_;
          }

          double angular_z = (err >= 0.0) ? spd : -spd;

          geometry_msgs::msg::Twist twist;
          twist.linear.x = 0.0;
          twist.linear.y = 0.0;
          twist.angular.z = angular_z;
          cmd_vel_pub_->publish(twist);

          loop_rate.sleep();
        }
      }
    }

    // Hentikan robot
    stopRobot();
    is_facing_person_ = false;

    if (person_found && aligned) {
      res->success = true;
      res->message = "Robot aligned facing " + display_name + " (detected: " + current_target.label + ")";
      publishStatus(res->message);
      RCLCPP_INFO(get_logger(), "\033[1;36m[RECEPTIONIST]\033[0m\n  Face: %s", res->message.c_str());
    } else if (person_found && !aligned) {
      res->success = true;
      res->message = "Robot aligned approximately facing " + display_name + " (detected: " + current_target.label + ")";
      publishStatus(res->message);
      RCLCPP_INFO(get_logger(), "\033[1;36m[RECEPTIONIST]\033[0m\n  Face: %s", res->message.c_str());
    } else if (use_locked_yaw) {
      res->success = true;
      res->message = "Robot aligned to locked " + display_name + (aligned_locked ? " (exact)" : " (approx)");
      publishStatus(res->message);
      RCLCPP_INFO(get_logger(), "\033[1;36m[RECEPTIONIST]\033[0m\n  Face: %s", res->message.c_str());
    } else {
      res->success = false;
      res->message = "No " + display_name + " detected in current view";
      publishStatus(res->message);
      RCLCPP_WARN(get_logger(), "\033[1;33m[FACE PERSON]\033[0m %s", res->message.c_str());
    }
  }

  void handleAdjustGuest(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    if (is_turning_.load()) {
      res->success = false;
      res->message = "Robot is currently busy turning";
      publishStatus(res->message);
      return;
    }
    if (is_navigating_.load()) {
      res->success = false;
      res->message = "Robot is currently busy with navigation task";
      publishStatus(res->message);
      return;
    }
    if (is_searching_chair_.load()) {
      res->success = false;
      res->message = "Robot is currently busy searching for a chair";
      publishStatus(res->message);
      return;
    }
    if (is_facing_person_.exchange(true)) {
      res->success = false;
      res->message = "Robot is already busy adjusting facing or position";
      publishStatus(res->message);
      return;
    }

    publishStatus(
      "Service /receptionist/adjust_guest called: Aligning to guest and adjusting distance to " +
      std::to_string(guest_door_distance_min_) + "m-" + std::to_string(guest_door_distance_) + "m...");

    auto get_door_guest_target = [this]() -> PersonTarget {
      std::lock_guard<std::mutex> lock(person_mutex_);
      if (person_detected_ && (now() - last_person_time_).seconds() < 1.0) {
        return parseTargetPerson(latest_persons_str_, "door_guest");
      }
      return PersonTarget();
    };

    rclcpp::Rate loop_rate(20.0); // 20 Hz
    bool guest_found = false;
    PersonTarget current_target;

    // Cek target dalam 10 frame (500 ms)
    for (int i = 0; i < 10; ++i) {
      PersonTarget t = get_door_guest_target();
      if (t.valid) {
        guest_found = true;
        current_target = t;
        break;
      }
      loop_rate.sleep();
    }

    // FASE 1: Menyelaraskan arah hadap robot tepat ke tamu (Align heading)
    if (guest_found) {
      int centered_frames = 0;
      auto align_start_time = now();
      const double align_timeout = 6.0;

      while (rclcpp::ok() && is_facing_person_.load()) {
        if ((now() - align_start_time).seconds() > align_timeout) {
          break;
        }

        PersonTarget t = get_door_guest_target();
        if (t.valid) {
          current_target = t;
        }

        const double err = current_target.angle_err;
        if (std::abs(err) <= guest_door_align_tolerance_) {
          centered_frames++;
          if (centered_frames >= 3) {
            break;
          }
        } else {
          centered_frames = 0;
        }

        double w = face_person_p_gain_ * err;
        double spd = std::abs(w);
        if (spd < face_person_min_speed_) spd = face_person_min_speed_;
        else if (spd > face_person_max_speed_) spd = face_person_max_speed_;

        geometry_msgs::msg::Twist twist;
        twist.linear.x = 0.0;
        twist.linear.y = 0.0;
        twist.angular.z = (err >= 0.0) ? spd : -spd;
        cmd_vel_pub_->publish(twist);

        loop_rate.sleep();
      }
      stopRobot();
    }

    // FASE 2: Penyesuaian Jarak Adaptif Dua Arah (Smart Distance Adjustment)
    // Target rentang ideal: 0.9m - 1.2m
    auto adjust_dist_start = now();
    int in_ideal_range_frames = 0;
    const double target_max = guest_door_distance_;     // 1.2m
    const double target_min = guest_door_distance_min_; // 0.9m

    while (rclcpp::ok() && is_facing_person_.load()) {
      if ((now() - adjust_dist_start).seconds() > guest_door_timeout_sec_) {
        RCLCPP_WARN(get_logger(), "[ADJUST GUEST] Distance adjustment timeout (%.1fs)", guest_door_timeout_sec_);
        break;
      }

      PersonTarget t = get_door_guest_target();
      if (t.valid) {
        current_target = t;
      }

      const double dist = current_target.dist;

      // Cek apakah sudah berada di rentang ideal (0.9m - 1.2m)
      if (dist >= target_min && dist <= target_max) {
        in_ideal_range_frames++;
        if (in_ideal_range_frames >= 3) {
          break; // Stabil di zona ideal
        }
      } else {
        in_ideal_range_frames = 0;
      }

      geometry_msgs::msg::Twist twist;
      twist.linear.y = 0.0;

      if (dist > target_max) {
        // Tamu berdiri agak jauh (> target_max): Robot maju mendekat secara proporsional
        const double err = dist - target_max;
        const double max_fwd = std::abs(guest_door_step_forward_speed_);
        const double min_fwd = std::min(std::abs(guest_door_step_forward_min_speed_), max_fwd);
        double spd = std::clamp(guest_door_step_forward_p_gain_ * err, min_fwd, max_fwd);
        twist.linear.x = spd;
      } else if (dist < target_min && dist > 0.10) {
        // Tamu berdiri terlalu dekat (< target_min): Robot mundur sedikit secara proporsional
        const double err = target_min - dist;
        const double max_back = std::abs(guest_door_step_back_speed_);
        const double min_back = std::min(std::abs(guest_door_step_back_min_speed_), max_back);
        double spd = std::clamp(guest_door_step_back_p_gain_ * err, min_back, max_back);
        twist.linear.x = -spd;
      } else {
        twist.linear.x = 0.0;
      }

      // Koreksi sudut halus sambil bergerak jika sudut agak melenceng
      if (std::abs(current_target.angle_err) > 0.08) {
        const double ang_spd = std::abs(guest_door_adjust_angular_speed_);
        twist.angular.z = (current_target.angle_err >= 0.0) ? ang_spd : -ang_spd;
      } else {
        twist.angular.z = 0.0;
      }

      cmd_vel_pub_->publish(twist);
      loop_rate.sleep();
    }

    stopRobot();
    is_facing_person_ = false;

    res->success = true;
    res->message = "Guest alignment and adaptive distance adjustment completed (dist=" +
                   std::to_string(current_target.dist) + "m, ideal=" +
                   std::to_string(target_min) + "m-" + std::to_string(target_max) + "m)";
    publishStatus(res->message);
    RCLCPP_INFO(get_logger(), "\033[1;36m[RECEPTIONIST]\033[0m\n  Align: %s", res->message.c_str());
  }

  void handleApproachGuest(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    if (is_turning_.load()) {
      res->success = false;
      res->message = "Robot is currently busy turning";
      publishStatus(res->message);
      return;
    }
    if (is_navigating_.load()) {
      res->success = false;
      res->message = "Robot is currently busy with navigation task";
      publishStatus(res->message);
      return;
    }
    if (is_searching_chair_.load()) {
      res->success = false;
      res->message = "Robot is currently busy searching for a chair";
      publishStatus(res->message);
      return;
    }
    if (is_facing_person_.exchange(true)) {
      res->success = false;
      res->message = "Robot is already busy adjusting facing or position";
      publishStatus(res->message);
      return;
    }

    publishStatus(
      "Service /receptionist/approach_guest called: Aligning and approaching guest to " +
      std::to_string(guest_approach_distance_) + "m...");

    auto get_door_guest_target = [this]() -> PersonTarget {
      std::lock_guard<std::mutex> lock(person_mutex_);
      if (person_detected_ && (now() - last_person_time_).seconds() < 1.0) {
        return parseTargetPerson(latest_persons_str_, "door_guest_approach");
      }
      return PersonTarget();
    };

    rclcpp::Rate loop_rate(20.0);
    bool guest_found = false;
    PersonTarget current_target;

    for (int i = 0; i < 20; ++i) {
      PersonTarget t = get_door_guest_target();
      if (t.valid) {
        guest_found = true;
        current_target = t;
        break;
      }
      loop_rate.sleep();
    }

    if (!guest_found) {
      stopRobot();
      is_facing_person_ = false;
      res->success = false;
      res->message = "No guest detected in front to approach";
      publishStatus(res->message);
      RCLCPP_WARN(get_logger(), "\033[1;33m[APPROACH GUEST]\033[0m %s", res->message.c_str());
      return;
    }

    // FASE 1: Align heading toward guest before moving forward
    {
      int centered_frames = 0;
      auto align_start_time = now();
      const double align_timeout = 5.0;

      while (rclcpp::ok() && is_facing_person_.load()) {
        if ((now() - align_start_time).seconds() > align_timeout) {
          break;
        }
        PersonTarget t = get_door_guest_target();
        if (t.valid) {
          current_target = t;
        }
        const double err = current_target.angle_err;
        if (std::abs(err) <= guest_door_align_tolerance_) {
          centered_frames++;
          if (centered_frames >= 3) {
            break;
          }
        } else {
          centered_frames = 0;
        }

        double w = face_person_p_gain_ * err;
        double spd = std::clamp(std::abs(w), face_person_min_speed_, face_person_max_speed_);

        geometry_msgs::msg::Twist twist;
        twist.angular.z = (err >= 0.0) ? spd : -spd;
        cmd_vel_pub_->publish(twist);
        loop_rate.sleep();
      }
      stopRobot();
    }

    // FASE 2: Maju perlahan hingga jarak mencapai <= guest_approach_distance_ (0.7m)
    auto approach_start = now();
    auto last_valid_time = now();
    int reached_dist_frames = 0;
    bool target_lost = false;

    while (rclcpp::ok() && is_facing_person_.load()) {
      if ((now() - approach_start).seconds() > guest_approach_timeout_sec_) {
        RCLCPP_WARN(get_logger(), "[APPROACH GUEST] Timeout reached (%.1fs)", guest_approach_timeout_sec_);
        break;
      }

      PersonTarget t = get_door_guest_target();
      if (t.valid) {
        current_target = t;
        last_valid_time = now();
      } else if ((now() - last_valid_time).seconds() > 1.0) {
        RCLCPP_WARN(get_logger(), "[APPROACH GUEST] Guest lost during approach; stopping safely.");
        target_lost = true;
        break;
      }

      if (current_target.dist <= guest_approach_distance_) {
        reached_dist_frames++;
        if (reached_dist_frames >= 2) {
          break;
        }
      } else {
        reached_dist_frames = 0;
      }

      geometry_msgs::msg::Twist twist;
      const double max_vx = std::abs(guest_approach_speed_);
      const double min_vx = std::min(0.10, max_vx);
      const double slow_zone = 0.50;
      const double dist_remaining = std::max(0.0, current_target.dist - guest_approach_distance_);
      const double ratio = std::clamp(dist_remaining / slow_zone, 0.0, 1.0);
      twist.linear.x = min_vx + (max_vx - min_vx) * ratio;
      if (std::abs(current_target.angle_err) > 0.10) {
        twist.angular.z = (current_target.angle_err >= 0.0) ? 0.12 : -0.12;
      } else {
        twist.angular.z = 0.0;
      }
      cmd_vel_pub_->publish(twist);
      loop_rate.sleep();
    }

    stopRobot();
    is_facing_person_ = false;

    res->success = !target_lost;
    res->message = target_lost
      ? "Stopped approach early: guest lost from view (dist=" + std::to_string(current_target.dist) + "m)"
      : "Guest approach completed (dist=" + std::to_string(current_target.dist) +
        "m, target=" + std::to_string(guest_approach_distance_) + "m)";
    publishStatus(res->message);
    RCLCPP_INFO(get_logger(), "\033[1;36m[RECEPTIONIST]\033[0m\n  Approach: %s", res->message.c_str());
  }

  static double normalize_angle(double angle)
  {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
  }

  void handleTurnToMapYaw(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    if (is_navigating_.load()) {
      res->success = false;
      res->message = "Robot is currently busy with navigation task";
      publishStatus(res->message);
      return;
    }

    if (is_facing_person_.load()) {
      res->success = false;
      res->message = "Robot is currently busy facing person";
      publishStatus(res->message);
      return;
    }

    if (is_searching_chair_.load()) {
      res->success = false;
      res->message = "Robot is currently busy searching for a chair";
      publishStatus(res->message);
      return;
    }

    if (is_turning_.exchange(true)) {
      res->success = false;
      res->message = "Robot is already executing a turn";
      publishStatus(res->message);
      return;
    }

    const double target_yaw = normalize_angle(turn_left_map_yaw_deg_ * M_PI / 180.0);
    publishStatus(
      "Service /receptionist/left called: Rotating to map yaw " +
      std::to_string(turn_left_map_yaw_deg_) + " deg...");

    // Tunggu pembacaan yaw terhadap peta (TF map -> base_link) maksimal ~1 detik
    double cur_yaw = 0.0;
    bool got_yaw = false;
    for (int i = 0; i < 20; ++i) {
      if (getMapYaw(cur_yaw)) {
        got_yaw = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    if (!got_yaw) {
      is_turning_ = false;
      res->success = false;
      res->message = "Failed to turn: Map TF yaw (" + global_frame_ + " -> base_link) not available";
      publishStatus(res->message);
      RCLCPP_ERROR(
        get_logger(),
        "\033[1;31m[TURN MAP]\033[0m Map TF yaw (%s -> base_link) not available",
        global_frame_.c_str());
      return;
    }

    RCLCPP_INFO(
      get_logger(),
      "\033[1;32m[TURN MAP]\033[0m Rotating to map yaw %.1f deg (%.2f rad), current map yaw: %.2f rad",
      turn_left_map_yaw_deg_, target_yaw, cur_yaw);

    auto start_time = now();
    rclcpp::Rate loop_rate(30.0);
    int centered_count = 0;
    bool success = false;
    const double max_spd = std::abs(turn_left_max_speed_);
    const double min_spd = std::min(std::abs(turn_left_min_speed_), max_spd);

    while (rclcpp::ok() && is_turning_.load()) {
      if ((now() - start_time).seconds() > turn_left_timeout_sec_) {
        RCLCPP_WARN(
          get_logger(),
          "\033[1;33m[TURN MAP]\033[0m Timeout reached (%.1fs)",
          turn_left_timeout_sec_);
        break;
      }

      if (!getMapYaw(cur_yaw)) {
        loop_rate.sleep();
        continue;
      }

      const double err = normalize_angle(target_yaw - cur_yaw);
      if (std::abs(err) <= turn_left_tolerance_rad_) {
        centered_count++;
        if (centered_count >= 3) {
          success = true;
          break;
        }
      } else {
        centered_count = 0;
      }

      double w = turn_left_p_gain_ * err;
      double spd = std::clamp(std::abs(w), min_spd, max_spd);

      geometry_msgs::msg::Twist twist;
      twist.linear.x = 0.0;
      twist.linear.y = 0.0;
      twist.angular.z = (err >= 0.0) ? spd : -spd;
      cmd_vel_pub_->publish(twist);

      loop_rate.sleep();
    }

    stopRobot();
    is_turning_ = false;

    if (success) {
      res->success = true;
      res->message = "Rotated to map yaw " + std::to_string(turn_left_map_yaw_deg_) + " deg";
      publishStatus(res->message);
      RCLCPP_INFO(get_logger(), "\033[1;36m[RECEPTIONIST]\033[0m\n  Turn Map: %s", res->message.c_str());
    } else {
      res->success = false;
      res->message = "Turn to map yaw timed out or interrupted";
      publishStatus(res->message);
      RCLCPP_WARN(get_logger(), "\033[1;36m[RECEPTIONIST]\033[0m\n  Turn Map: %s", res->message.c_str());
    }
  }

  void handleTurnRelative(
    double angle_rad,
    const std::string & direction_name,
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    if (is_navigating_.load()) {
      res->success = false;
      res->message = "Robot is currently busy with navigation task";
      publishStatus(res->message);
      return;
    }

    if (is_facing_person_.load()) {
      res->success = false;
      res->message = "Robot is currently busy facing person";
      publishStatus(res->message);
      return;
    }

    if (is_searching_chair_.load()) {
      res->success = false;
      res->message = "Robot is currently busy searching for a chair";
      publishStatus(res->message);
      return;
    }

    if (is_turning_.exchange(true)) {
      res->success = false;
      res->message = "Robot is already executing a turn";
      publishStatus(res->message);
      return;
    }

    publishStatus("Service /receptionist/" + direction_name + " called: Turning 90 degrees " + direction_name + "...");

    // Tunggu pembacaan yaw awal dari odometry (maksimal ~1 detik)
    double start_yaw = 0.0;
    bool got_yaw = false;
    for (int i = 0; i < 20; ++i) {
      if (getRobotYaw(start_yaw)) {
        got_yaw = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    if (!got_yaw) {
      is_turning_ = false;
      res->success = false;
      res->message = "Failed to turn: Odometry yaw not available";
      publishStatus(res->message);
      RCLCPP_ERROR(get_logger(), "\033[1;31m[TURN]\033[0m Odometry yaw not available");
      return;
    }

    RCLCPP_INFO(
      get_logger(),
      "\033[1;32m[TURN]\033[0m Starting 90-degree turn %s (target delta: %.2f rad, start yaw: %.2f rad)",
      direction_name.c_str(), angle_rad, start_yaw);

    const double target_turn = angle_rad;
    double last_yaw = start_yaw;
    double accumulated_turn = 0.0;
    auto start_time = now();
    rclcpp::Rate loop_rate(30.0);
    int centered_count = 0;
    bool success = false;

    while (rclcpp::ok() && is_turning_.load()) {
      const double elapsed = (now() - start_time).seconds();
      if (elapsed > turn_90_timeout_sec_) {
        RCLCPP_WARN(get_logger(), "\033[1;33m[TURN]\033[0m Timeout reached (%.1fs)", turn_90_timeout_sec_);
        break;
      }

      double cur_yaw = 0.0;
      if (getRobotYaw(cur_yaw)) {
        double dyaw = normalize_angle(cur_yaw - last_yaw);
        accumulated_turn += dyaw;
        last_yaw = cur_yaw;
      }

      const double remaining = target_turn - accumulated_turn;

      if (std::abs(remaining) <= turn_90_tolerance_rad_) {
        centered_count++;
        if (centered_count >= 3) {
          success = true;
          break;
        }
      } else {
        centered_count = 0;
      }

      // Hitung kecepatan sudut proporsional
      double w = turn_90_p_gain_ * remaining;
      double spd = std::abs(w);
      if (spd < turn_90_min_speed_) spd = turn_90_min_speed_;
      else if (spd > turn_90_max_speed_) spd = turn_90_max_speed_;

      geometry_msgs::msg::Twist twist;
      twist.linear.x = 0.0;
      twist.linear.y = 0.0;
      twist.angular.z = (remaining >= 0.0) ? spd : -spd;
      cmd_vel_pub_->publish(twist);

      loop_rate.sleep();
    }

    stopRobot();
    is_turning_ = false;

    const double turned_deg = accumulated_turn * 180.0 / M_PI;
    if (success) {
      res->success = true;
      res->message = "Successfully turned 90 degrees " + direction_name +
                     " (turned " + std::to_string(turned_deg) + " deg)";
      publishStatus(res->message);
      RCLCPP_INFO(
        get_logger(),
        "\033[1;32m[TURN]\033[0m Complete: turned %.1f deg %s",
        turned_deg, direction_name.c_str());
    } else {
      res->success = false;
      res->message = "Turn 90 degrees " + direction_name + " timed out or incomplete" +
                     " (turned " + std::to_string(turned_deg) + " deg)";
      publishStatus(res->message);
      RCLCPP_WARN(
        get_logger(),
        "\033[1;33m[TURN]\033[0m Incomplete: turned %.1f deg %s",
        turned_deg, direction_name.c_str());
    }
  }

  void handleGoTo(
    const std::string & target_name,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    if (is_turning_.load()) {
      res->success = false;
      res->message = "Robot is currently busy turning";
      publishStatus(res->message);
      return;
    }

    if (is_searching_chair_.load()) {
      res->success = false;
      res->message = "Robot is currently busy searching for a chair";
      publishStatus(res->message);
      return;
    }

    if (is_facing_person_.load()) {
      res->success = false;
      res->message = "Robot is currently busy facing person";
      publishStatus(res->message);
      return;
    }

    if (is_navigating_.exchange(true)) {
      res->success = false;
      res->message = "Robot is currently busy with another navigation task";
      publishStatus(res->message);
      return;
    }

    if (target_name == "door") {
      clearLivingRoomLocks();
    }

    Waypoint wp = getWaypoint(target_name);

    publishStatus(
      "Command received: Navigating to " + target_name +
      " (x=" + std::to_string(wp.x) + ", y=" + std::to_string(wp.y) + ")...");

    bool ok = navigateTo(wp);
    is_navigating_ = false;

    if (target_name == "living_room" && ok) {
      double cur_yaw = 0.0;
      if (getRobotYaw(cur_yaw)) {
        living_room_arrival_yaw_ = cur_yaw;
        has_living_room_arrival_yaw_ = true;
      } else {
        living_room_arrival_yaw_ = wp.yaw * M_PI / 180.0;
        has_living_room_arrival_yaw_ = true;
      }
      RCLCPP_INFO(
        get_logger(),
        "\033[1;36m[NAV]\033[0m Reached living room. Recorded arrival heading: %.2f rad (%.1f deg)",
        living_room_arrival_yaw_, living_room_arrival_yaw_ * 180.0 / M_PI);
    }

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
        RCLCPP_WARN(get_logger(), "\033[1;31m[NAV]\033[0m Goal rejected: %s", wp.name.c_str());
      } else {
        current_goal_handle_ = handle;
        RCLCPP_INFO(get_logger(), "\033[1;36m[RECEPTIONIST]\033[0m\n  Nav: Navigating to '%s'", wp.name.c_str());
      }
    };
    send_opts.result_callback = [promise](const GoalHandleNav::WrappedResult & res) {
      promise->set_value(res.code);
    };

    auto send_future = nav_client_->async_send_goal(goal, send_opts);
    if (send_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready || !send_future.get()) {
      RCLCPP_ERROR(get_logger(), "\033[1;31m[NAV]\033[0m Failed sending goal to %s", wp.name.c_str());
      return false;
    }

    if (future.wait_for(std::chrono::seconds(static_cast<int>(nav_timeout_sec_))) == std::future_status::ready) {
      const auto code = future.get();
      if (code == rclcpp_action::ResultCode::SUCCEEDED) {
        RCLCPP_INFO(get_logger(), "\033[1;36m[RECEPTIONIST]\033[0m\n  Nav: Arrived at '%s'", wp.name.c_str());
        return true;
      } else {
        RCLCPP_WARN(
          get_logger(), "\033[1;33m[NAV]\033[0m Failed: '%s' (code %d)",
          wp.name.c_str(), static_cast<int>(code));
        return false;
      }
    } else {
      RCLCPP_ERROR(
        get_logger(), "\033[1;31m[NAV]\033[0m Timeout: '%s' (%.1fs)",
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
  double search_chair_ref_yaw_{-1.57};
  double search_chair_angular_speed_{0.30};
  double search_chair_timeout_sec_{40.0};
  int search_chair_confirm_frames_{2};
  double search_chair_sweep_angle_deg_{30.0};
  double chair_max_dist_{3.0};
  double align_chair_p_gain_{1.2};
  double align_chair_min_speed_{0.14};
  double align_chair_max_speed_{0.28};
  double align_chair_tolerance_rad_{0.07};
  int camera_width_{672};

  // TF2
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // Odometry tracking
  std::string odom_topic_{"/odom"};
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  std::mutex odom_mutex_;
  bool has_odom_{false};
  double current_yaw_{0.0};
  rclcpp::Time last_odom_time_{0, 0, RCL_ROS_TIME};

  // Living room arrival reference yaw & temporary locked target yaws
  bool has_living_room_arrival_yaw_{false};
  double living_room_arrival_yaw_{0.0};
  std::mutex living_room_lock_mutex_;
  bool has_locked_host_yaw_{false};
  double locked_host_yaw_{0.0};
  bool has_locked_guest1_yaw_{false};
  double locked_guest1_yaw_{0.0};
  bool has_locked_guest2_yaw_{false};
  double locked_guest2_yaw_{0.0};
  bool has_locked_empty_chair_yaw_{false};
  double locked_empty_chair_yaw_{0.0};

  std::mutex chair_mutex_;
  bool chair_detected_{false};
  std::string latest_chair_info_;
  rclcpp::Time last_chair_time_{0, 0, RCL_ROS_TIME};
  std::string latest_all_chairs_info_;
  rclcpp::Time last_all_chairs_time_{0, 0, RCL_ROS_TIME};
  int consecutive_chair_frames_{0};
  bool occupied_chair_detected_{false};
  std::string latest_occupied_chair_info_;
  rclcpp::Time last_occupied_chair_time_{0, 0, RCL_ROS_TIME};

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

  // Face person tracking & services
  std::atomic<bool> is_facing_person_{false};
  std::string host_name_{"Alex"};
  std::string person_label_topic_{"/vision/person_label"};
  std::string guest_1_name_topic_{"/vision/guest_1_name"};
  std::string guest_2_name_topic_{"/vision/guest_2_name"};
  double host_seated_max_dist_{3.0};
  double guest_door_distance_{1.2};
  double guest_door_distance_min_{0.9};
  double guest_door_step_forward_speed_{0.25};
  double guest_door_step_forward_min_speed_{0.10};
  double guest_door_step_forward_p_gain_{0.40};
  double guest_door_step_back_speed_{0.18};
  double guest_door_step_back_min_speed_{0.08};
  double guest_door_step_back_p_gain_{0.35};
  double guest_door_adjust_angular_speed_{0.12};
  double guest_door_timeout_sec_{15.0};
  double guest_approach_distance_{0.7};
  double guest_approach_speed_{0.16};
  double guest_approach_timeout_sec_{15.0};
  double guest_door_align_tolerance_{0.07};
  double face_person_angular_speed_{0.30};
  double face_person_timeout_sec_{25.0};
  double face_person_p_gain_{1.2};
  double face_person_min_speed_{0.14};
  double face_person_max_speed_{0.30};
  double face_person_tolerance_rad_{0.07};

  std::mutex person_mutex_;
  bool person_detected_{false};
  std::string latest_persons_str_;
  rclcpp::Time last_person_time_{0, 0, RCL_ROS_TIME};

  std::mutex guest_mutex_;
  std::string guest_1_name_;
  std::string guest_2_name_;

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr person_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr guest_1_name_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr guest_1_name_fallback_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr guest_2_name_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr guest_2_name_fallback_sub_;

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_face_host_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_face_guest_1_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_face_guest_1_alias_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_face_guest_2_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_face_guest_2_alias_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_adjust_guest_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_adjust_guest_alias_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_approach_guest_;

  // 90-degree relative turn (/receptionist/left dan /receptionist/right)
  std::atomic<bool> is_turning_{false};
  double turn_90_angular_speed_{0.30};
  double turn_90_min_speed_{0.14};
  double turn_90_max_speed_{0.35};
  double turn_90_tolerance_rad_{0.04};
  double turn_90_timeout_sec_{15.0};
  double turn_90_p_gain_{1.5};

  double turn_left_map_yaw_deg_{0.0};
  double turn_left_min_speed_{0.14};
  double turn_left_max_speed_{0.35};
  double turn_left_tolerance_rad_{0.04};
  double turn_left_timeout_sec_{15.0};
  double turn_left_p_gain_{1.5};

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_turn_left_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_turn_right_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_clear_locks_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_reset_living_room_yaw_;
};

}  // namespace receptionist

int main(int argc, char ** argv)
{
  setenv("RCUTILS_CONSOLE_OUTPUT_FORMAT", "[{severity}] {message}", 1);
  rclcpp::init(argc, argv);
  auto node = std::make_shared<receptionist::ReceptionistNode>();

  // MultiThreadedExecutor dengan 4 thread untuk memproses service, action, dan vision subscription secara paralel
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
