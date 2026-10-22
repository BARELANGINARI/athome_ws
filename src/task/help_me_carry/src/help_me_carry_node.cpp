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
#include "person_kalman_filter.hpp"

namespace
{
constexpr const char * kNoPerson = "NO_PERSON";

struct Person
{
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
    const auto share = ament_index_cpp::get_package_share_directory("athome_navigation");
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
    person_topic_ = declare_parameter<std::string>("person_topic", "/zed/person_label");
    gesture_topic_ = declare_parameter<std::string>("gesture_topic", "/zed/gesture");
    bag_topic_ = declare_parameter<std::string>("bag_topic", "/zed/bag_detections");
    yaw_topic_ = declare_parameter<std::string>("yaw_topic", "/zed/yaw_deg");
    goal_update_topic_ = declare_parameter<std::string>("goal_update_topic", "/goal_update");
    action_name_ = declare_parameter<std::string>("action_name", "navigate_to_pose");
    frame_map_ = declare_parameter<std::string>("frame_map", "map");
    frame_base_ = declare_parameter<std::string>("frame_base", "base_link");
    target_index_ = declare_parameter<int>("target_index", 0);
    stop_distance_ = declare_parameter<double>("follow_stop_distance", 0.5);  // Jarak berhenti (0.5 m)
    start_distance_ = declare_parameter<double>("follow_start_distance", 0.7); // Jarak mulai maju (0.7 m)
    max_range_ = declare_parameter<double>("max_follow_range", 3.0);
    max_lost_time_ = declare_parameter<double>("max_lost_time", 3.0);
    bt_xml_ = declare_parameter<std::string>("follow_bt_xml", "");

    if (bt_xml_.empty()) {
      bt_xml_ = find_bt_xml();
    }

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_, this, false);
    action_client_ = rclcpp_action::create_client<NavigateToPose>(this, action_name_);

    goal_update_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      goal_update_topic_, rclcpp::QoS(10).reliable());
    cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>(
      "/cmd_vel", rclcpp::QoS(10));

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
      });

    bag_sub_ = create_subscription<std_msgs::msg::String>(
      bag_topic_, rclcpp::SensorDataQoS(),
      [this](const std_msgs::msg::String::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(bag_mutex_);
        latest_bags_ = parse_bags(msg->data);
      });

    yaw_sub_ = create_subscription<std_msgs::msg::Float64>(
      yaw_topic_, rclcpp::SensorDataQoS(),
      [this](const std_msgs::msg::Float64::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(bag_mutex_);
        current_yaw_deg_ = msg->data;
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

    // Dapatkan posisi robot di peta (map frame)
    geometry_msgs::msg::TransformStamped trans;
    try {
      trans = tf_buffer_->lookupTransform(frame_map_, frame_base_, tf2::TimePointZero);
    } catch (const tf2::TransformException & exc) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "TF %s <- %s belum siap! Pastikan lokalisasi/SLAM aktif: %s",
        frame_map_.c_str(), frame_base_.c_str(), exc.what());
      return;
    }

    const double rx = trans.transform.translation.x;
    const double ry = trans.transform.translation.y;
    const auto & rq = trans.transform.rotation;
    const double siny_cosp = 2.0 * (rq.w * rq.z + rq.x * rq.y);
    const double cosy_cosp = 1.0 - 2.0 * (rq.y * rq.y + rq.z * rq.z);
    const double ryaw = std::atan2(siny_cosp, cosy_cosp);

    // Hitung dt untuk prediksi Kalman Filter
    double dt = 0.033;
    if (has_last_filter_time_) {
      dt = (now() - last_filter_time_).seconds();
      if (dt <= 0.0) dt = 0.033;
      if (dt > 1.0) dt = 1.0;
    }
    last_filter_time_ = now();
    has_last_filter_time_ = true;

    // Prediksi posisi target saat ini berdasarkan kecepatan gerak sebelumnya
    if (kf_.is_initialized()) {
      kf_.predict(dt);
    }

    const auto people = parse_persons(msg->data);
    struct PersonCandidate {
      bool is_operator{false};
      double mx{0.0};
      double my{0.0};
      double dist{0.0};
      double direct_bearing{0.0};
      double dist_to_kf{0.0};
    };

    std::vector<PersonCandidate> candidates;
    for (const auto & person : people) {
      if (!std::isfinite(person.x) || !std::isfinite(person.y)) continue;
      const double dist = std::hypot(person.x, person.y);
      if (dist > max_range_) continue;

      // Bearing ke person dari kamera ZED (sudut relatif robot).
      // Mengikuti konvensi referensi: thetaBot = atan2(-robotPos.y, robotPos.x)
      // karena y sudah di-negate di get3dCoordinatesWithOffset (y = -point.y + 3)
      // maka bearing = atan2(-person.y, person.x) agar arah putar robot benar.
      const double bearing = std::atan2(-person.y, person.x);
      double abs_yaw = ryaw + bearing;
      abs_yaw = std::atan2(std::sin(abs_yaw), std::cos(abs_yaw));

      const double mx = rx + dist * std::cos(abs_yaw);
      const double my = ry + dist * std::sin(abs_yaw);

      PersonCandidate cand;
      cand.is_operator = person.is_operator;
      cand.mx = mx;
      cand.my = my;
      cand.dist = dist;
      cand.direct_bearing = bearing;
      cand.dist_to_kf = kf_.is_initialized() ? kf_.distance_to(mx, my) : dist;
      candidates.push_back(cand);
    }

    bool person_matched = false;
    double direct_bearing = 0.0;
    double direct_dist = 0.0;

    if (!candidates.empty()) {
      if (!kf_.is_initialized()) {
        // Inisialisasi awal target: utamakan kandidat berlabel operator, atau index sesuai target_index_
        auto op_it = std::find_if(candidates.begin(), candidates.end(), [](const PersonCandidate & c) {
          return c.is_operator;
        });
        std::size_t idx = 0;
        if (op_it != candidates.end()) {
          idx = std::distance(candidates.begin(), op_it);
        } else {
          idx = static_cast<std::size_t>(
            std::min<int>(std::max(target_index_, 0), static_cast<int>(candidates.size()) - 1));
        }
        kf_.init(candidates[idx].mx, candidates[idx].my);
        last_seen_sec_ = now().seconds();
        person_matched = true;
        direct_bearing = candidates[idx].direct_bearing;
        direct_dist = candidates[idx].dist;
        RCLCPP_INFO(get_logger(),
          "\033[92m[FOLLOW] Target dikunci: dist=%.2f m | bearing=%+.1f deg (pos map: %.2f, %.2f)\033[0m",
          direct_dist, direct_bearing * 180.0 / M_PI, kf_.x(), kf_.y());
      } else {
        // Cari orang yang posisinya paling cocok dengan prediksi Kalman Filter (Gating)
        const double gate_radius = kf_.get_gate_radius(0.9, 2.0);
        auto best_it = std::min_element(
          candidates.begin(), candidates.end(),
          [](const PersonCandidate & a, const PersonCandidate & b) {
            return a.dist_to_kf < b.dist_to_kf;
          });

        if (best_it != candidates.end() && best_it->dist_to_kf <= gate_radius) {
          // Cocok! Ini target asli kita (bahkan jika ada orang lain yang lewat memotong)
          kf_.update(best_it->mx, best_it->my);
          last_seen_sec_ = now().seconds();
          person_matched = true;
          direct_bearing = best_it->direct_bearing;
          direct_dist = best_it->dist;
        } else {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 1500,
            "\033[93m[FOLLOW] Target di luar gating / terhalang (gate=%.2f m, dist=%.2f m)\033[0m",
            gate_radius, best_it != candidates.end() ? best_it->dist_to_kf : -1.0);
        }
      }
    }

    if (!person_matched) {
      // Operator tidak terlihat di frame ini: hentikan robot seketika
      geometry_msgs::msg::Twist stop_twist;
      cmd_vel_pub_->publish(stop_twist);
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1500,
        "\033[93m[FOLLOW] Menunggu: Operator tidak terdeteksi dalam frame kamera ZED\033[0m");
      return;
    }

    // =====================================================================
    // MURNI BASED ON DEPTH KAMERA ZED (DIRECT VELOCITY CONTROLLER)
    // =====================================================================
    // direct_dist    : jarak langsung (hypot) ke operator dari depth ZED (meter)
    // direct_bearing : sudut arah hadap operator relatif terhadap kamera robot (rad)
    // stop_distance_ : 0.3 m (sesuai depth_minimum_distance 30 cm)

    geometry_msgs::msg::Twist twist;
    std::string status_str = "STOP";

    // 1. KONTROL ANGULAR (Rotasi hadap orang)
    if (std::abs(direct_bearing) > 0.08) {
      double w = std::clamp(1.5 * direct_bearing, -0.6, 0.6);
      if (std::abs(w) < 0.12) {
        w = (direct_bearing > 0.0 ? 0.12 : -0.12);
      }
      twist.angular.z = w;
      status_str = "ALIGN YAW";
    } else {
      twist.angular.z = 0.0;
    }

    // 2. KONTROL LINEAR (Maju di 0.7m, Berhenti di 0.5m)
    if (!is_driving_forward_) {
      // Robot sedang diam: hanya mulai maju jika jarak orang melebihi start_distance (0.7m)
      if (direct_dist > start_distance_) {
        is_driving_forward_ = true;
      }
    } else {
      // Robot sedang maju: terus maju sampai jarak orang sudah <= stop_distance (0.5m)
      if (direct_dist <= stop_distance_) {
        is_driving_forward_ = false;
      }
    }

    if (is_driving_forward_) {
      // Kecepatan maju proporsional terhadap selisih jarak
      double v_x = 0.5 * (direct_dist - stop_distance_);
      // Batasi kecepatan maju: min 0.10 m/s, max 0.35 m/s
      v_x = std::clamp(v_x, 0.10, 0.35);

      // Jika sudut kemiringan masih terlalu miring (> 35° / 0.6 rad), pelankan maju agar hadap dulu
      if (std::abs(direct_bearing) > 0.6) {
        v_x *= 0.3;
      }

      twist.linear.x = v_x;
      status_str = "MAJU MENGEJAR (ZED)";
    } else {
      twist.linear.x = 0.0;
      if (status_str != "ALIGN YAW") {
        status_str = "STOP (Jarak <= 0.5m)";
      }
    }

    // Publish langsung ke motor base robot (/cmd_vel)
    cmd_vel_pub_->publish(twist);

    // Print status debug teratur di terminal (2 Hz)
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 500,
      "\033[94m[DEBUG] Operator: dist=%.2f m | bearing=%+.1f deg | Status: %s | vx=%.2f m/s | wz=%.2f rad/s\033[0m",
      direct_dist, direct_bearing * 180.0 / M_PI,
      status_str.c_str(), twist.linear.x, twist.angular.z);
  }

  void send_nav2_goal(const geometry_msgs::msg::PoseStamped & target_pose)
  {
    if (!is_following_ || is_nav_active_) {
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
        RCLCPP_INFO(get_logger(), "\033[92m[FOLLOW] Goal following diterima oleh Nav2.\033[0m");
      }
    };
    send_opts.result_callback = [this](const GoalHandle::WrappedResult & res) {
      is_nav_active_ = false;
      goal_handle_.reset();
      if (res.code == rclcpp_action::ResultCode::SUCCEEDED) {
        RCLCPP_INFO(get_logger(), "\033[92m[FOLLOW] Sampai di dekat operator. Menunggu operator melangkah lagi...\033[0m");
      } else if (res.code == rclcpp_action::ResultCode::CANCELED) {
        RCLCPP_INFO(get_logger(), "\033[93m[FOLLOW] Goal following Nav2 dibatalkan.\033[0m");
      } else if (res.code == rclcpp_action::ResultCode::ABORTED) {
        last_nav_fail_time_ = now();
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "\033[91m[FOLLOW] Nav2 gagal/di-abort (rintangan/unreachable). Jeda sejenak sebelum mencoba lagi...\033[0m");
      }
    };

    is_nav_active_ = true;
    action_client_->async_send_goal(goal, send_opts);
  }

  void handle_start_follow(std::shared_ptr<Trigger::Response> resp)
  {
    if (is_following_) {
      resp->success = true;
      resp->message = "Following sudah aktif.";
      return;
    }

    kf_.reset();
    has_last_filter_time_ = false;
    is_following_ = true;
    is_nav_active_ = false;
    is_driving_forward_ = false;
    last_seen_sec_ = now().seconds();

    resp->success = true;
    resp->message = "Person following aktif (Murni Depth ZED). Menunggu operator...";
    RCLCPP_INFO(get_logger(), "%s", resp->message.c_str());
    IC(resp->success, resp->message);
  }

  void handle_stop_follow(std::shared_ptr<Trigger::Response> resp)
  {
    is_following_ = false;
    is_nav_active_ = false;
    is_driving_forward_ = false;
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

    resp->success = true;
    resp->message = "Person following dihentikan.";
    RCLCPP_INFO(get_logger(), "%s", resp->message.c_str());
    IC(resp->success, resp->message);
  }

  void check_safety_watchdog()
  {
    if (!is_following_ || max_lost_time_ <= 0.0) {
      return;
    }

    const double elapsed = now().seconds() - last_seen_sec_;
    if (elapsed > max_lost_time_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 4000,
        "Watchdog: Operator tidak terlihat selama %.1f detik (maks: %.1f s)! Menghentikan robot.",
        elapsed, max_lost_time_);
      if (is_nav_active_ && goal_handle_) {
        action_client_->async_cancel_goal(goal_handle_);
        goal_handle_.reset();
        is_nav_active_ = false;
      }
      // Reset filter agar saat operator terlihat lagi langsung re-lock tanpa tertolak gating
      kf_.reset();
      has_last_filter_time_ = false;

      geometry_msgs::msg::Twist stop_twist;
      cmd_vel_pub_->publish(stop_twist);
    }
  }

  void handle_align_bag(
    const std::shared_ptr<Trigger::Request> /*req*/,
    std::shared_ptr<Trigger::Response> resp)
  {
    // 1. Hentikan person following jika aktif
    if (is_following_) {
      auto dummy = std::make_shared<Trigger::Response>();
      handle_stop_follow(dummy);
    }

    // 2. Ambil data gesture dan tas terakhir dari ZED
    std::string gest;
    std::vector<Bag> bags;
    {
      std::lock_guard<std::mutex> lock(bag_mutex_);
      gest = latest_gesture_;
      bags = latest_bags_;
    }

    const bool is_right = (gest.find("RIGHT") != std::string::npos || gest.find("KANAN") != std::string::npos);
    const bool is_left  = (gest.find("LEFT") != std::string::npos || gest.find("KIRI") != std::string::npos);

    if (!is_right && !is_left) {
      resp->success = false;
      resp->message = "Gesture tidak valid (harus kiri atau kanan): " + gest;
      return;
    }

    // 3. Saring tas ZED di sisi yang dipilih (y < 0: kanan, y > 0: kiri)
    Bag target_bag{0.0, 0.0};
    double min_d = 1e9;
    for (const auto & b : bags) {
      if ((is_right && b.y < 0.0) || (is_left && b.y > 0.0)) {
        const double d = std::hypot(b.x, b.y);
        if (d < min_d) { min_d = d; target_bag = b; }
      }
    }

    if (min_d > 1e8) {
      resp->success = false;
      resp->message = std::string("Tas di sisi ") + (is_right ? "kanan" : "kiri") + " tidak ditemukan.";
      return;
    }

    // 4. Hitung sudut putar menggunakan data yaw dari vision (/zed/yaw_deg)
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

    // 5. Berhenti seketika
    twist.angular.z = 0.0;
    cmd_vel_pub_->publish(twist);

    resp->success = true;
    resp->message = std::string("Robot telah lurus sejajar tas ") + (is_right ? "kanan" : "kiri") + " (yaw vision) dan berhenti.";
    RCLCPP_INFO(get_logger(), "%s", resp->message.c_str());
    IC(resp->success, resp->message);
  }

  void handle_homming(
    const std::shared_ptr<Trigger::Request> /*req*/,
    std::shared_ptr<Trigger::Response> resp)
  {
    RCLCPP_INFO(get_logger(), "[Homing] Menerima panggilan service /homing...");

    // 1. Hentikan person following jika sedang aktif
    if (is_following_) {
      auto dummy = std::make_shared<Trigger::Response>();
      handle_stop_follow(dummy);
    }

    // 2. Pastikan action server Nav2 aktif
    if (!action_client_->wait_for_action_server(std::chrono::seconds(3))) {
      resp->success = false;
      resp->message = "Action server 'navigate_to_pose' Nav2 tidak aktif.";
      RCLCPP_ERROR(get_logger(), "%s", resp->message.c_str());
      return;
    }

    // 3. Goal titik origin (x=0, y=0, w=1) pada frame map dengan BT replanning & recovery standar
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
        RCLCPP_WARN(get_logger(), "[Homing] Nav2 menolak goal origin.");
      } else {
        homing_goal_handle_ = handle;
        RCLCPP_INFO(get_logger(), "[Homing] Nav2 menerima goal origin. Bergerak ke titik awal...");
      }
    };
    send_opts.result_callback = [promise](const GoalHandle::WrappedResult & res) {
      promise->set_value(res.code);
    };

    auto send_future = action_client_->async_send_goal(goal, send_opts);
    if (send_future.wait_for(std::chrono::seconds(4)) != std::future_status::ready || !send_future.get()) {
      resp->success = false;
      resp->message = "Gagal mengirim goal origin ke Nav2.";
      return;
    }

    // 4. Tunggu robot tiba di origin
    if (future.wait_for(std::chrono::seconds(120)) == std::future_status::ready) {
      const auto code = future.get();
      if (code == rclcpp_action::ResultCode::SUCCEEDED) {
        resp->success = true;
        resp->message = "Robot berhasil tiba kembali di posisi awal start (origin).";
        RCLCPP_INFO(get_logger(), "%s", resp->message.c_str());
      } else {
        resp->success = false;
        resp->message = "Navigasi kembali ke titik awal dibatalkan atau gagal.";
        RCLCPP_WARN(get_logger(), "%s", resp->message.c_str());
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
  std::string goal_update_topic_;
  std::string action_name_;
  std::string frame_map_;
  std::string frame_base_;
  int target_index_{0};
  double stop_distance_{0.5};
  double start_distance_{0.7};
  double max_range_{3.0};
  double max_lost_time_{3.0};
  std::string bt_xml_;

  std::atomic<bool> is_following_{false};
  std::atomic<bool> is_nav_active_{false};
  bool is_driving_forward_{false};
  double last_seen_sec_{0.0};
  rclcpp::Time last_goal_pub_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_nav_fail_time_{0, 0, RCL_ROS_TIME};

  std::mutex bag_mutex_;
  std::string latest_gesture_{"NONE"};
  std::vector<Bag> latest_bags_;
  double current_yaw_deg_{0.0};

  help_me_carry::PersonKalmanFilter kf_{1.2, 0.15};
  rclcpp::Time last_filter_time_;
  bool has_last_filter_time_{false};

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp_action::Client<NavigateToPose>::SharedPtr action_client_;
  GoalHandle::SharedPtr goal_handle_;
  GoalHandle::SharedPtr homing_goal_handle_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_update_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr person_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr gesture_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr bag_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr yaw_sub_;
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
