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
    declare_parameter<double>("guest_door_distance", 1.0);
    declare_parameter<double>("guest_door_step_back_speed", 0.16);
    declare_parameter<double>("guest_door_timeout_sec", 15.0);
    declare_parameter<double>("guest_door_align_tolerance", 0.07);
    declare_parameter<double>("face_person_angular_speed", 0.30);
    declare_parameter<double>("face_person_timeout_sec", 25.0);
    declare_parameter<double>("face_person_p_gain", 1.2);
    declare_parameter<double>("face_person_min_speed", 0.14);
    declare_parameter<double>("face_person_max_speed", 0.30);
    declare_parameter<double>("face_person_tolerance_rad", 0.07);

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
    declare_parameter<double>("align_chair_p_gain", 1.2);
    declare_parameter<double>("align_chair_min_speed", 0.14);
    declare_parameter<double>("align_chair_max_speed", 0.28);
    declare_parameter<double>("align_chair_tolerance_rad", 0.07);
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
    guest_door_step_back_speed_ = get_parameter("guest_door_step_back_speed").as_double();
    guest_door_timeout_sec_ = get_parameter("guest_door_timeout_sec").as_double();
    guest_door_align_tolerance_ = get_parameter("guest_door_align_tolerance").as_double();
    face_person_angular_speed_ = get_parameter("face_person_angular_speed").as_double();
    face_person_timeout_sec_ = get_parameter("face_person_timeout_sec").as_double();
    face_person_p_gain_ = get_parameter("face_person_p_gain").as_double();
    face_person_min_speed_ = get_parameter("face_person_min_speed").as_double();
    face_person_max_speed_ = get_parameter("face_person_max_speed").as_double();
    face_person_tolerance_rad_ = get_parameter("face_person_tolerance_rad").as_double();

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

    guest_2_name_sub_ = create_subscription<std_msgs::msg::String>(
      guest_2_name_topic_, 10,
      [this](const std_msgs::msg::String::SharedPtr msg) {
        if (!msg || msg->data.empty()) return;
        std::lock_guard<std::mutex> lock(guest_mutex_);
        guest_2_name_ = msg->data;
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

  void rotateLeft(double speed)
  {
    geometry_msgs::msg::Twist twist;
    twist.linear.x = 0.0;
    twist.linear.y = 0.0;
    twist.linear.z = 0.0;
    twist.angular.x = 0.0;
    twist.angular.y = 0.0;
    // Counter-clockwise (rotasi ke kiri): nilai angular.z positif sesuai konvensi ROS REP 103
    twist.angular.z = std::abs(speed);
    cmd_vel_pub_->publish(twist);
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

  bool getRobotMapYaw(double & yaw_out)
  {
    if (tf_buffer_) {
      try {
        auto trans = tf_buffer_->lookupTransform(global_frame_, "base_footprint", tf2::TimePointZero);
        const auto & q = trans.transform.rotation;
        const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
        const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        yaw_out = std::atan2(siny_cosp, cosy_cosp);
        return true;
      } catch (const tf2::TransformException &) {
        // Fallback ke odometry jika TF map belum tersedia
      }
    }
    return getRobotYaw(yaw_out);
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

    if (!msg->data.empty()) {
      std::stringstream ss(msg->data);
      std::string item;
      while (std::getline(ss, item, ';')) {
        const bool is_seating = (item.find("chair") != std::string::npos ||
                                 item.find("sofa") != std::string::npos ||
                                 item.find("couch") != std::string::npos);
        // Cek kursi/sofa kosong (empty) dengan filter jarak < chair_max_dist_ (3.0 meter)
        if (is_seating &&
            item.find("empty") != std::string::npos &&
            item.find("occupied") == std::string::npos) {
          
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
            has_empty_chair = true;
            if (!empty_chairs_info.empty()) {
              empty_chairs_info += ";";
            }
            empty_chairs_info += item;
          }
        }
        // Cek orang yang sedang duduk di kursi/sofa (occupied)
        if (is_seating &&
            item.find("occupied") != std::string::npos) {
          has_occupied_chair = true;
          if (!occupied_chairs_info.empty()) {
            occupied_chairs_info += ";";
          }
          occupied_chairs_info += item;
        }
      }
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
      occupied_chair_detected_ = false;
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

  EmptyChairTarget parseEmptyChair(const std::string & data) const
  {
    EmptyChairTarget target;
    if (data.empty()) return target;

    // Format 1: Dari /vision/chair_detections ("chair:x,y,w,h|X,Y,Z|empty|side;...")
    std::stringstream ss(data);
    std::string item;
    double best_dist = 1e9;
    double best_bbox_area = 0.0;
    bool target_has_3d = false;

    while (std::getline(ss, item, ';')) {
      const bool is_seating = (item.find("chair") != std::string::npos ||
                               item.find("sofa") != std::string::npos ||
                               item.find("couch") != std::string::npos);
      if (!is_seating ||
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
        // Selalu prioritaskan kursi kosong dengan koordinat 3D valid
        is_better = true;
      } else if (has_3d && target_has_3d) {
        // Jika keduanya punya 3D, pilih kursi yang jaraknya paling dekat (dist terkecil)
        if (dist < best_dist) {
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
        target_has_3d = has_3d;
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

      // Evaluasi apakah orang ini sesuai dengan target_role
      bool is_match = false;
      double priority = 0.0;

      if (target_role == "door_guest") {
        is_match = true;
        // Tamu di pintu: prioritaskan orang terdekat dan paling lurus di depan kamera
        priority = 100.0 - (dist * 10.0) - (std::abs(angle) * 5.0);
      } else if (target_role == "host") {
        if (!host_name_lower.empty() && label_lower.find(host_name_lower) != std::string::npos) {
          is_match = true;
          priority = 100.0;
        } else if (label_lower.find("host") != std::string::npos) {
          is_match = true;
          priority = 85.0;
        } else if ((label_lower.find("seated") != std::string::npos ||
                    label_lower.find("sitting") != std::string::npos ||
                    label_lower.find("duduk") != std::string::npos) &&
                   dist < host_seated_max_dist_) {
          // Logic OR: orang yang sedang duduk (seated) jika jarak < 3.0m
          is_match = true;
          priority = 90.0;
        }
      } else if (target_role == "guest_1") {
        if (!guest_1_name_lower.empty() && label_lower.find(guest_1_name_lower) != std::string::npos) {
          is_match = true;
          priority = 100.0;
        } else if (label_lower.find("guest_1") != std::string::npos || label_lower.find("guest 1") != std::string::npos) {
          is_match = true;
          priority = 70.0;
        } else if ((guest_2_name_lower.empty() || label_lower.find(guest_2_name_lower) == std::string::npos) &&
                   (host_name_lower.empty() || label_lower.find(host_name_lower) == std::string::npos) &&
                   label_lower.find("host") == std::string::npos) {
          // Fallback kandidat: orang yang bukan host dan bukan guest 2
          is_match = true;
          priority = 20.0;
        }
      } else if (target_role == "guest_2") {
        if (!guest_2_name_lower.empty() && label_lower.find(guest_2_name_lower) != std::string::npos) {
          is_match = true;
          priority = 100.0;
        } else if (label_lower.find("guest_2") != std::string::npos || label_lower.find("guest 2") != std::string::npos) {
          is_match = true;
          priority = 70.0;
        } else if ((guest_1_name_lower.empty() || label_lower.find(guest_1_name_lower) == std::string::npos) &&
                   (host_name_lower.empty() || label_lower.find(host_name_lower) == std::string::npos) &&
                   label_lower.find("host") == std::string::npos) {
          // Fallback kandidat: orang yang bukan host dan bukan guest 1
          is_match = true;
          priority = 20.0;
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

    publishStatus("Service /search_chair called: Aligning to empty chair in view...");

    // Helper untuk mengambil data kursi kosong terkini yang masih segar (< 2.5s)
    auto get_current_empty_chair = [this]() -> EmptyChairTarget {
      std::lock_guard<std::mutex> lock(chair_mutex_);
      if (!latest_chair_info_.empty() && (now() - last_chair_time_).seconds() < 2.5) {
        return parseEmptyChair(latest_chair_info_);
      }
      return EmptyChairTarget();
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

    // ALIGN: Langsung memutar hadap robot tepat ke arah kursi kosong terdekat
    if (chair_found) {
      RCLCPP_INFO(
        get_logger(),
        "\033[1;32m[CHAIR]\033[0m Target: %s (%.2fm, %s), aligning...",
        current_target.raw_info.c_str(), current_target.dist, current_target.side.c_str());
      publishStatus("Empty chair in view! Aligning robot heading to face chair...");

      if (std::abs(current_target.angle_err) <= align_chair_tolerance_rad_) {
        aligned = true;
        RCLCPP_INFO(
          get_logger(),
          "\033[1;32m[CHAIR]\033[0m Aligned: err=%.2frad",
          current_target.angle_err);
      } else {
        auto align_start_time = now();
        const double align_timeout_sec = 6.0;
        int centered_frames = 0;

        while (rclcpp::ok() && is_searching_chair_.load()) {
          const double align_elapsed = (now() - align_start_time).seconds();
          if (align_elapsed > align_timeout_sec) {
            RCLCPP_WARN(get_logger(), "\033[1;33m[CHAIR]\033[0m Alignment timeout (%.1fs)", align_timeout_sec);
            break;
          }

          EmptyChairTarget t = get_current_empty_chair();
          if (t.valid) {
            current_target = t;
          }

          const double err = current_target.angle_err;

          if (std::abs(err) <= align_chair_tolerance_rad_) {
            centered_frames++;
            if (centered_frames >= 2) { // Terpusat stabil selama ~100 ms
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

    // Berhenti penuh setelah arah hadap robot tepat berada pada arah kursi
    stopRobot();
    is_searching_chair_ = false;

    if (chair_found) {
      res->success = true;
      res->message = "Empty chair found and robot aligned facing chair: " + current_target.raw_info;
      publishStatus(res->message);
      RCLCPP_INFO(
        get_logger(),
        "\033[1;32m[CHAIR]\033[0m Found: %s (%.2fm, %s)",
        current_target.raw_info.c_str(), current_target.dist,
        aligned ? "aligned" : "approx");
    } else {
      res->success = false;
      res->message = "No empty chair detected in current view";
      publishStatus(res->message);
      RCLCPP_WARN(get_logger(), "\033[1;33m[CHAIR]\033[0m No empty chair detected");
    }
  }

  void handleFacePerson(
    const std::string & target_role,
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
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

    auto get_current_person_target = [this, &target_role]() -> PersonTarget {
      PersonTarget target;

      // Khusus saat mencari Host / orang di living room:
      // PRIORITAS UTAMA: Orang yang sedang DUDUK di kursi (occupied_chair < 3.0m)
      if (target_role == "host") {
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

      // PRIORITAS KEDUA: Cek deteksi orang dari vision person_label
      {
        std::lock_guard<std::mutex> lock(person_mutex_);
        if (person_detected_ && (now() - last_person_time_).seconds() < 1.0) {
          target = parseTargetPerson(latest_persons_str_, target_role);
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
          if (t.valid) {
            current_target = t;
          }

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
      std::to_string(guest_door_distance_) + "m...");

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

    // FASE 2: Mundur hingga jaraknya dengan tamu mencapai target (misal 1.5 meter)
    auto step_back_start = now();
    int reached_dist_frames = 0;
    while (rclcpp::ok() && is_facing_person_.load()) {
      if ((now() - step_back_start).seconds() > guest_door_timeout_sec_) {
        RCLCPP_WARN(get_logger(), "[ADJUST GUEST] Step back timeout reached (%.1fs)", guest_door_timeout_sec_);
        break;
      }

      PersonTarget t = get_door_guest_target();
      if (t.valid) {
        current_target = t;
      }

      // Periksa apakah jarak saat ini sudah mencapai atau melebihi guest_door_distance_
      if (current_target.dist >= guest_door_distance_) {
        reached_dist_frames++;
        if (reached_dist_frames >= 2) {
          break; // Jarak sudah terpenuhi (>= 1.5m)
        }
      } else {
        reached_dist_frames = 0;
      }

      // Mundur perlahan
      geometry_msgs::msg::Twist twist;
      twist.linear.x = -std::abs(guest_door_step_back_speed_);
      twist.linear.y = 0.0;
      // Koreksi sudut halus sambil mundur jika sudut agak miring
      if (std::abs(current_target.angle_err) > 0.12) {
        twist.angular.z = (current_target.angle_err >= 0.0) ? 0.12 : -0.12;
      } else {
        twist.angular.z = 0.0;
      }
      cmd_vel_pub_->publish(twist);

      loop_rate.sleep();
    }

    stopRobot();
    is_facing_person_ = false;

    res->success = true;
    res->message = "Guest alignment and distance adjustment completed (dist=" +
                   std::to_string(current_target.dist) + "m, target=" +
                   std::to_string(guest_door_distance_) + "m)";
    publishStatus(res->message);
    RCLCPP_INFO(get_logger(), "\033[1;36m[RECEPTIONIST]\033[0m\n  Align: %s", res->message.c_str());
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

  // Living room arrival reference yaw
  bool has_living_room_arrival_yaw_{false};
  double living_room_arrival_yaw_{0.0};

  std::mutex chair_mutex_;
  bool chair_detected_{false};
  std::string latest_chair_info_;
  rclcpp::Time last_chair_time_{0, 0, RCL_ROS_TIME};
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
  double guest_door_distance_{1.0};
  double guest_door_step_back_speed_{0.16};
  double guest_door_timeout_sec_{15.0};
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
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr guest_2_name_sub_;

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_face_host_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_face_guest_1_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_face_guest_1_alias_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_face_guest_2_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_face_guest_2_alias_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_adjust_guest_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_adjust_guest_alias_;
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
