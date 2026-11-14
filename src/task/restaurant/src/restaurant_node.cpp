#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

namespace restaurant
{

inline double normalize_angle(double angle)
{
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

geometry_msgs::msg::Quaternion yawToQuaternion(double yaw_deg_or_rad)
{
  double yaw_rad = (std::abs(yaw_deg_or_rad) > 6.2831853) ? (yaw_deg_or_rad * M_PI / 180.0) : yaw_deg_or_rad;
  geometry_msgs::msg::Quaternion q;
  q.x = 0.0;
  q.y = 0.0;
  q.z = std::sin(yaw_rad * 0.5);
  q.w = std::cos(yaw_rad * 0.5);
  return q;
}

// Struktur representasi deteksi orang beserta BBox dan koordinat 3D
struct DetectedPerson
{
  int bbox_x{0};
  int bbox_y{0};
  int bbox_w{0};
  int bbox_h{0};
  double bbox_cx{0.0};
  double bbox_cy{0.0};
  double x{0.0};
  double y{0.0};
  double z{0.0};
  bool has_3d{false};
  int id{-1};
  std::string label;
};

// Parser untuk topik /vision/person_label ("label:x,y,w,h|X,Y,Z|id=...;...")
inline std::vector<DetectedPerson> parsePersonLabel(const std::string & data)
{
  std::vector<DetectedPerson> persons;
  if (data.empty() || data.find("NO_PERSON") != std::string::npos) {
    return persons;
  }

  std::stringstream ss(data);
  std::string chunk;
  while (std::getline(ss, chunk, ';')) {
    if (chunk.empty()) continue;
    auto sep1 = chunk.find(':');
    auto bar1 = chunk.find('|');
    if (sep1 == std::string::npos || bar1 == std::string::npos || bar1 <= sep1) continue;

    DetectedPerson p;
    p.label = chunk.substr(0, sep1);

    // 1. Ekstraksi Bounding Box (x, y, w, h)
    std::string bbox_str = chunk.substr(sep1 + 1, bar1 - sep1 - 1);
    std::stringstream bss(bbox_str);
    std::string val;
    std::vector<int> bvals;
    while (std::getline(bss, val, ',')) {
      try { bvals.push_back(std::stoi(val)); } catch (...) {}
    }
    if (bvals.size() >= 4) {
      p.bbox_x = bvals[0];
      p.bbox_y = bvals[1];
      p.bbox_w = bvals[2];
      p.bbox_h = bvals[3];
      p.bbox_cx = p.bbox_x + p.bbox_w * 0.5;
      p.bbox_cy = p.bbox_y + p.bbox_h * 0.5;
    }

    // 2. Ekstraksi Koordinat 3D (x, y, z) jika tersedia
    auto bar2 = chunk.find('|', bar1 + 1);
    std::string xyz_str = (bar2 != std::string::npos) ? chunk.substr(bar1 + 1, bar2 - bar1 - 1) : chunk.substr(bar1 + 1);
    if (xyz_str.find("nan") == std::string::npos) {
      double px = 0.0, py = 0.0, pz = 0.0;
      if (std::sscanf(xyz_str.c_str(), "%lf,%lf,%lf", &px, &py, &pz) >= 2) {
        if (std::isfinite(px) && std::isfinite(py)) {
          p.x = px;
          p.y = -py; // Negasikan kembali agar sesuai standar ROS base_link (+Y Kiri, -Y Kanan)
          p.z = pz;
          p.has_3d = true;
        }
      }
    }

    // 3. ID pelacakan orang
    auto id_pos = chunk.find("id=");
    if (id_pos != std::string::npos) {
      try { p.id = std::stoi(chunk.substr(id_pos + 3)); } catch (...) {}
    }

    persons.push_back(p);
  }
  return persons;
}

// Helper parser yang fleksibel untuk berbagai format input string posisi
bool parseCoordinates(const std::string & raw_data, double & out_x, double & out_y)
{
  if (raw_data.empty()) return false;

  std::string data = raw_data;

  // 1. Format pipeline vision (misal: "order1:bbox|x,y,z|conf")
  if (data.find('|') != std::string::npos) {
    std::stringstream ss(data);
    std::string seg;
    std::vector<std::string> parts;
    while (std::getline(ss, seg, '|')) {
      parts.push_back(seg);
    }
    if (parts.size() >= 2) {
      data = parts[1]; // Ambil bagian koordinat x,y,z
    }
  }

  // 2. Format JSON (misal: {"x": 1.5, "y": 2.0})
  std::regex json_regex(R"("x"\s*:\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*"y"\s*:\s*([-+]?[0-9]*\.?[0-9]+))");
  std::smatch json_match;
  if (std::regex_search(data, json_match, json_regex)) {
    try {
      out_x = std::stod(json_match[1].str());
      out_y = std::stod(json_match[2].str());
      return std::isfinite(out_x) && std::isfinite(out_y);
    } catch (...) {}
  }

  // 3. Hapus prefix jika ada label (misal: "order1: 1.5, 2.0" -> "1.5, 2.0")
  auto colon_pos = data.find(':');
  if (colon_pos != std::string::npos) {
    data = data.substr(colon_pos + 1);
  }

  // Ganti koma dan karakter non-angka dengan spasi
  for (char & c : data) {
    if (c == ',' || c == ';' || c == '(' || c == ')' || c == '[' || c == ']') {
      c = ' ';
    }
  }

  std::stringstream ss(data);
  double x_val, y_val;
  if (ss >> x_val >> y_val) {
    if (std::isfinite(x_val) && std::isfinite(y_val)) {
      out_x = x_val;
      out_y = y_val;
      return true;
    }
  }

  return false;
}

class RestaurantNode : public rclcpp::Node
{
public:
  using NavigateToPose = nav2_msgs::action::NavigateToPose;
  using GoalHandleNav = rclcpp_action::ClientGoalHandle<NavigateToPose>;

  RestaurantNode()
  : Node("restaurant_node")
  {
    // Parameter deklarasi
    declare_parameter("origin_x", 0.0);
    declare_parameter("origin_y", 0.0);
    declare_parameter("origin_yaw", 0.0);
    declare_parameter("global_frame", "map");
    declare_parameter("base_frame", "base_link");
    declare_parameter("order_topic", "/vision/order");
    declare_parameter("order_frame", "base_link"); // "base_link" (relatif thd robot) atau "map" (global)
    declare_parameter("standoff_distance", 0.85);   // Jarak henti aman di depan orang (meter) agar tidak menabrak / masuk obstacle
    declare_parameter("auto_navigate", false);      // Langsung navigasi saat data order masuk atau tunggu service
    declare_parameter("nav_timeout_sec", 120.0);
    declare_parameter("search_sweep_angle_deg", 90.0); // Rotasi pencarian ke kiri 90 deg dan ke kanan 90 deg
    declare_parameter("search_angular_speed", 0.35);    // Kecepatan sudut rotasi (rad/s)
    declare_parameter("search_hold_time_sec", 0.5);     // Waktu diam sejenak di ujung kiri/kanan (detik)
    declare_parameter("search_timeout_sec", 35.0);      // Timeout keseluruhan fungsi search (detik)
    declare_parameter("cmd_vel_topic", "/cmd_vel");
    declare_parameter("gesture_topic", "/vision/gesture");
    declare_parameter("align_to_person_enabled", true);
    declare_parameter("person_label_topic", "/vision/person_label");
    declare_parameter("align_timeout_sec", 10.0);
    declare_parameter("align_tolerance_px", 30.0);
    declare_parameter("align_p_gain", 0.6);
    declare_parameter("align_min_speed", 0.12);
    declare_parameter("align_max_speed", 0.35);
    declare_parameter("camera_image_width", 672.0);
    declare_parameter("align_invert_direction", false);

    origin_x_ = get_parameter("origin_x").as_double();
    origin_y_ = get_parameter("origin_y").as_double();
    origin_yaw_ = get_parameter("origin_yaw").as_double();
    global_frame_ = get_parameter("global_frame").as_string();
    base_frame_ = get_parameter("base_frame").as_string();
    order_topic_ = get_parameter("order_topic").as_string();
    order_frame_ = get_parameter("order_frame").as_string();
    standoff_dist_ = get_parameter("standoff_distance").as_double();
    auto_navigate_ = get_parameter("auto_navigate").as_bool();
    nav_timeout_sec_ = get_parameter("nav_timeout_sec").as_double();
    search_sweep_angle_deg_ = get_parameter("search_sweep_angle_deg").as_double();
    search_angular_speed_ = get_parameter("search_angular_speed").as_double();
    search_hold_time_sec_ = get_parameter("search_hold_time_sec").as_double();
    search_timeout_sec_ = get_parameter("search_timeout_sec").as_double();
    cmd_vel_topic_ = get_parameter("cmd_vel_topic").as_string();
    gesture_topic_ = get_parameter("gesture_topic").as_string();
    align_to_person_enabled_ = get_parameter("align_to_person_enabled").as_bool();
    person_label_topic_ = get_parameter("person_label_topic").as_string();
    align_timeout_sec_ = get_parameter("align_timeout_sec").as_double();
    align_tolerance_px_ = get_parameter("align_tolerance_px").as_double();
    align_p_gain_ = get_parameter("align_p_gain").as_double();
    align_min_speed_ = get_parameter("align_min_speed").as_double();
    align_max_speed_ = get_parameter("align_max_speed").as_double();
    camera_image_width_ = get_parameter("camera_image_width").as_double();
    align_invert_direction_ = get_parameter("align_invert_direction").as_bool();

    // TF2 Buffer & Listener untuk konversi koordinat base_link -> map
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Callback groups
    srv_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    action_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    sub_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    // Nav2 Action Client
    nav_client_ = rclcpp_action::create_client<NavigateToPose>(
      this, "navigate_to_pose", action_cb_group_);

    // Publisher cmd_vel untuk kontrol rotasi pencarian di tempat
    cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);

    // Subscription /vision/order
    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.callback_group = sub_cb_group_;
    order_sub_ = create_subscription<std_msgs::msg::String>(
      order_topic_, 10,
      [this](const std_msgs::msg::String::SharedPtr msg) {
        onOrderMessage(msg);
      },
      sub_opts);

    // Subscription /vision/gesture untuk mendeteksi lambaian tangan secara real-time
    gesture_sub_ = create_subscription<std_msgs::msg::String>(
      gesture_topic_, 10,
      [this](const std_msgs::msg::String::SharedPtr msg) {
        onGestureMessage(msg);
      },
      sub_opts);

    // Subscription /vision/person_label untuk membaca bounding box orang di depan robot
    person_label_sub_ = create_subscription<std_msgs::msg::String>(
      person_label_topic_, rclcpp::SensorDataQoS(),
      [this](const std_msgs::msg::String::SharedPtr msg) {
        onPersonLabelMessage(msg);
      },
      sub_opts);

    // Service /homing
    srv_homing_ = create_service<std_srvs::srv::Trigger>(
      "/homing",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleHoming(req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    // Service alias /restaurant/homing
    srv_homing_alias_ = create_service<std_srvs::srv::Trigger>(
      "/restaurant/homing",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleHoming(req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    // Service /go_to_order untuk menggerakkan robot ke posisi orang yang memesan
    srv_go_order_ = create_service<std_srvs::srv::Trigger>(
      "/go_to_order",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleGoToOrder(req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    srv_go_order_alias_ = create_service<std_srvs::srv::Trigger>(
      "/restaurant/go_to_order",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleGoToOrder(req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    // Alias kompatibilitas ke belakang untuk /go_to_order1
    srv_go_order1_compat_ = create_service<std_srvs::srv::Trigger>(
      "/go_to_order1",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleGoToOrder(req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    srv_go_order1_alias_compat_ = create_service<std_srvs::srv::Trigger>(
      "/restaurant/go_to_order1",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleGoToOrder(req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    // Service /search untuk memutar robot kiri 90 deg dan kanan 90 deg mencari pelanggan melambaikan tangan
    srv_search_ = create_service<std_srvs::srv::Trigger>(
      "/search",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleSearch(req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    srv_search_alias_ = create_service<std_srvs::srv::Trigger>(
      "/restaurant/search",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleSearch(req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    // Service /deliver_order untuk kembali ke koordinat pelanggan yang telah disimpan saat /go_to_order
    srv_deliver_order_ = create_service<std_srvs::srv::Trigger>(
      "/deliver_order",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleDeliverOrder(req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    srv_deliver_order_alias_ = create_service<std_srvs::srv::Trigger>(
      "/restaurant/deliver_order",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        handleDeliverOrder(req, res);
      },
      rclcpp::ServicesQoS(),
      srv_cb_group_);

    // Publisher pose pelanggan tersimpan untuk visualisasi RViz
    rclcpp::QoS latched_qos(1);
    latched_qos.transient_local();
    latched_qos.reliable();
    saved_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      "/restaurant/saved_order_pose", latched_qos);

    RCLCPP_INFO(
      get_logger(),
      "\033[1;32m[RestaurantNode]\033[0m Ready! Subscribed to %s | Services: /search, /go_to_order, /deliver_order, /homing",
      order_topic_.c_str());
  }

private:
  // Callback saat menerima data deteksi orang beserta bounding box dari /vision/person_label
  void onPersonLabelMessage(const std_msgs::msg::String::SharedPtr msg)
  {
    if (!msg || msg->data.empty()) return;
    auto parsed = parsePersonLabel(msg->data);
    std::lock_guard<std::mutex> lock(person_mutex_);
    latest_persons_ = std::move(parsed);
    latest_person_time_ = now();
  }

  // Callback saat menerima pesan di /vision/order
  void onOrderMessage(const std_msgs::msg::String::SharedPtr msg)
  {
    double parsed_x = 0.0, parsed_y = 0.0;
    if (!parseCoordinates(msg->data, parsed_x, parsed_y)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 3000,
        "[Order] Failed parsing coordinates from message: '%s'", msg->data.c_str());
      return;
    }

    // Kamera vision membalik sumbu Y (-point.y), kita negasikan kembali (-parsed_y)
    // agar sesuai dengan standar koordinat ROS base_link (+Y = Kiri, -Y = Kanan)
    const double corrected_y = -parsed_y;

    {
      std::lock_guard<std::mutex> lock(order_mutex_);
      latest_order_x_ = parsed_x;
      latest_order_y_ = corrected_y;
      latest_order_time_ = now();
      has_order_ = true;
      if (is_searching_.load()) {
        wave_seen_during_search_ = true;
      }
    }

    RCLCPP_INFO(
      get_logger(),
      "\033[1;35m[Order]\033[0m Received customer position (%s): x=%.2f, y=%.2f (raw vision y=%.2f)",
      order_frame_.c_str(), parsed_x, corrected_y, parsed_y);

    if (auto_navigate_ && !is_navigating_.load() && !is_searching_.load()) {
      std::thread([this]() {
        auto dummy_req = std::make_shared<std_srvs::srv::Trigger::Request>();
        auto dummy_res = std::make_shared<std_srvs::srv::Trigger::Response>();
        handleGoToOrder(dummy_req, dummy_res);
      }).detach();
    }
  }

  // Callback saat menerima deteksi lambaian tangan dari /vision/gesture
  void onGestureMessage(const std_msgs::msg::String::SharedPtr msg)
  {
    if (!msg || msg->data.empty()) return;
    std::string gest = msg->data;
    std::transform(gest.begin(), gest.end(), gest.begin(), ::toupper);
    if (gest.find("WAVE") != std::string::npos ||
        gest.find("RAISE") != std::string::npos)
    {
      if (is_searching_.load()) {
        wave_seen_during_search_ = true;
      }
    }
  }

  // Menyelaraskan arah hadap robot tepat ke tengah bounding box (bbox) orang di depannya
  bool alignHeadingToPersonInFront()
  {
    if (!align_to_person_enabled_) {
      return true;
    }

    RCLCPP_INFO(
      get_logger(),
      "\033[1;33m[Align]\033[0m Reached destination! Detecting person bbox in front and adjusting robot heading...");

    const auto start_time = now();
    const double timeout_sec = align_timeout_sec_;
    const double img_center_x = camera_image_width_ * 0.5;
    const double tol_px = align_tolerance_px_;
    int centered_count = 0;
    rclcpp::Rate rate(20.0);

    bool person_ever_seen = false;

    while (rclcpp::ok()) {
      if ((now() - start_time).seconds() > timeout_sec) {
        RCLCPP_WARN(
          get_logger(),
          "[Align] Alignment timeout (%.1fs) reached. Finishing alignment.", timeout_sec);
        break;
      }

      std::vector<DetectedPerson> current_persons;
      rclcpp::Time ptime;
      {
        std::lock_guard<std::mutex> lock(person_mutex_);
        current_persons = latest_persons_;
        ptime = latest_person_time_;
      }

      // Filter deteksi orang yang masih segar (< 1.5 detik yang lalu)
      bool has_fresh_person = false;
      DetectedPerson target_person;
      double best_score = -1e9;

      if (!current_persons.empty() && (now() - ptime).seconds() < 1.5) {
        for (const auto & p : current_persons) {
          // Abaikan deteksi kecil/noise
          if (p.bbox_w < 25 || p.bbox_h < 45) continue;

          // Skor: prioritaskan orang terdekat (3D range kecil) atau bbox terbesar
          double score = 0.0;
          if (p.has_3d && p.x > 0.1 && p.x < 3.5) {
            score = 1000.0 - std::hypot(p.x, p.y);
          } else {
            score = static_cast<double>(p.bbox_w * p.bbox_h);
          }

          if (score > best_score) {
            best_score = score;
            target_person = p;
            has_fresh_person = true;
          }
        }
      }

      if (!has_fresh_person) {
        geometry_msgs::msg::Twist stop_cmd;
        cmd_vel_pub_->publish(stop_cmd);
        rate.sleep();
        continue;
      }

      person_ever_seen = true;

      // Hitung selisih piksel tengah bbox terhadap tengah gambar kamera
      // error_px > 0: orang di sebelah kiri kamera -> putar robot ke kiri (+)
      // error_px < 0: orang di sebelah kanan kamera -> putar robot ke kanan (-)
      double error_px = img_center_x - target_person.bbox_cx;
      if (align_invert_direction_) {
        error_px = -error_px;
      }

      // Cek apakah posisi tengah bbox orang sudah berada di dalam toleransi tengah kamera
      if (std::abs(error_px) <= tol_px) {
        centered_count++;
        geometry_msgs::msg::Twist stop_cmd;
        cmd_vel_pub_->publish(stop_cmd);

        if (centered_count >= 3) {
          stopRobot();
          RCLCPP_INFO(
            get_logger(),
            "\033[1;32m[Align SUCCESS]\033[0m Robot perfectly aligned with customer bbox (cx=%.1f, err=%.1f px)!",
            target_person.bbox_cx, error_px);
          return true;
        }
      } else {
        centered_count = 0;

        // P-controller untuk kecepatan sudut rotasi
        const double norm_err = error_px / img_center_x; // -1.0 .. +1.0
        double wz = align_p_gain_ * norm_err;

        // Batasi dengan kecepatan min dan max
        wz = std::clamp(wz, -align_max_speed_, align_max_speed_);
        if (std::abs(wz) < align_min_speed_) {
          wz = (error_px > 0.0) ? align_min_speed_ : -align_min_speed_;
        }

        geometry_msgs::msg::Twist cmd;
        cmd.linear.x = 0.0;
        cmd.linear.y = 0.0;
        cmd.angular.z = wz;
        cmd_vel_pub_->publish(cmd);

        RCLCPP_DEBUG_THROTTLE(
          get_logger(), *get_clock(), 500,
          "[Align] bbox cx=%.1f, err=%.1f px -> wz=%.2f rad/s",
          target_person.bbox_cx, error_px, wz);
      }

      rate.sleep();
    }

    stopRobot();

    if (person_ever_seen) {
      RCLCPP_INFO(
        get_logger(),
        "\033[1;32m[Align DONE]\033[0m Alignment finished (heading aligned close to center).");
    } else {
      RCLCPP_WARN(
        get_logger(),
        "[Align] No person bbox detected in front within timeout. Preserving Nav2 arrival heading.");
    }

    return true;
  }

  // Handler service /go_to_order
  void handleGoToOrder(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    if (is_searching_.load()) {
      res->success = false;
      res->message = "Cannot navigate: Search rotation is currently active.";
      RCLCPP_WARN(get_logger(), "[Order] %s", res->message.c_str());
      return;
    }

    if (is_navigating_.exchange(true)) {
      res->success = false;
      res->message = "Robot is currently busy with another navigation task.";
      RCLCPP_WARN(get_logger(), "[Order] %s", res->message.c_str());
      return;
    }

    double ox = 0.0, oy = 0.0;
    {
      std::lock_guard<std::mutex> lock(order_mutex_);
      if (!has_order_) {
        is_navigating_ = false;
        res->success = false;
        res->message = "No order position data received yet on topic " + order_topic_;
        RCLCPP_WARN(get_logger(), "[Order] %s", res->message.c_str());
        return;
      }
      ox = latest_order_x_;
      oy = latest_order_y_;
    }

    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[Order]\033[0m Calculating safe path and navigating to customer at (%.2f, %.2f)...",
      ox, oy);

    bool ok = navigateToPersonWithSafeStandoff(ox, oy);
    if (ok) {
      RCLCPP_INFO(
        get_logger(),
        "\033[1;36m[Order]\033[0m Reached customer position. Adjusting heading to face customer bbox...");
      alignHeadingToPersonInFront();
    }
    is_navigating_ = false;

    res->success = ok;
    res->message = ok ? "Successfully arrived safely and aligned facing customer."
                      : "Failed or timed out navigating to customer.";
  }

  // Mengambil posisi robot saat ini pada frame map (atau odom sebagai fallback)
  bool getRobotPose(double & rx, double & ry, double & ryaw)
  {
    try {
      auto trans = tf_buffer_->lookupTransform(global_frame_, base_frame_, tf2::TimePointZero);
      rx = trans.transform.translation.x;
      ry = trans.transform.translation.y;
      const auto & q = trans.transform.rotation;
      const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
      const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
      ryaw = std::atan2(siny_cosp, cosy_cosp);
      return true;
    } catch (const tf2::TransformException & ex) {
      try {
        auto trans = tf_buffer_->lookupTransform("odom", base_frame_, tf2::TimePointZero);
        rx = trans.transform.translation.x;
        ry = trans.transform.translation.y;
        const auto & q = trans.transform.rotation;
        const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
        const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        ryaw = std::atan2(siny_cosp, cosy_cosp);
        return true;
      } catch (const tf2::TransformException & ex2) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "[TF] Cannot lookup transform (%s or odom -> %s): %s",
          global_frame_.c_str(), base_frame_.c_str(), ex.what());
        return false;
      }
    }
  }

  // Mengirim target pose ke Nav2 Action Server dengan penghindaran rintangan dinamis secara real-time
  bool navigateToMapPose(double target_x, double target_y, double target_yaw, const std::string & label)
  {
    if (!nav_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_logger(), "[%s] Nav2 action server 'navigate_to_pose' is unavailable!", label.c_str());
      return false;
    }

    NavigateToPose::Goal goal;
    goal.pose.header.frame_id = global_frame_;
    goal.pose.header.stamp = now();
    goal.pose.pose.position.x = target_x;
    goal.pose.pose.position.y = target_y;
    goal.pose.pose.position.z = 0.0;
    goal.pose.pose.orientation = yawToQuaternion(target_yaw);

    auto promise = std::make_shared<std::promise<rclcpp_action::ResultCode>>();
    auto future = promise->get_future();

    auto send_opts = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    send_opts.goal_response_callback = [this, label](const GoalHandleNav::SharedPtr & handle) {
      if (!handle) {
        RCLCPP_WARN(get_logger(), "[%s] Nav2 rejected goal!", label.c_str());
      } else {
        current_goal_handle_ = handle;
        RCLCPP_INFO(get_logger(), "[%s] Nav2 accepted goal! Navigating with real-time obstacle avoidance...", label.c_str());
      }
    };
    send_opts.result_callback = [promise](const GoalHandleNav::WrappedResult & res) {
      promise->set_value(res.code);
    };

    auto send_future = nav_client_->async_send_goal(goal, send_opts);
    if (send_future.wait_for(std::chrono::seconds(5)) != std::future_status::ready || !send_future.get()) {
      RCLCPP_ERROR(get_logger(), "[%s] Failed sending goal to Nav2 action server!", label.c_str());
      return false;
    }

    if (future.wait_for(std::chrono::seconds(static_cast<int>(nav_timeout_sec_))) == std::future_status::ready) {
      const auto code = future.get();
      if (code == rclcpp_action::ResultCode::SUCCEEDED) {
        RCLCPP_INFO(get_logger(), "\033[1;32m[%s] Arrived safely at destination position!\033[0m", label.c_str());
        return true;
      } else {
        RCLCPP_WARN(get_logger(), "[%s] Navigation canceled or aborted by Nav2 (code %d).", label.c_str(), static_cast<int>(code));
        return false;
      }
    } else {
      RCLCPP_ERROR(get_logger(), "[%s] Navigation timed out after %.1fs. Canceling goal...", label.c_str(), nav_timeout_sec_);
      if (current_goal_handle_) {
        nav_client_->async_cancel_goal(current_goal_handle_);
      }
      return false;
    }
  }

  // Fungsi utama: Menghitung koordinat global, standoff offset, menyimpan koordinat untuk deliver_order, dan mengirim ke Nav2
  bool navigateToPersonWithSafeStandoff(double px, double py)
  {
    double rx = 0.0, ry = 0.0, ryaw = 0.0;
    if (!getRobotPose(rx, ry, ryaw)) {
      RCLCPP_ERROR(get_logger(), "[Navigation] Failed to get robot current pose from TF!");
      return false;
    }

    // 1. Transformasikan ke koordinat global (map) jika input berasal dari frame base_link
    double target_map_x = px;
    double target_map_y = py;
    if (order_frame_ == "base_link") {
      target_map_x = rx + px * std::cos(ryaw) - py * std::sin(ryaw);
      target_map_y = ry + px * std::sin(ryaw) + py * std::cos(ryaw);
      RCLCPP_INFO(
        get_logger(),
        "[Transform] base_link (%.2f, %.2f) -> map (%.2f, %.2f)",
        px, py, target_map_x, target_map_y);
    }

    // 2. Hitung jarak dan sudut dari robot ke orang
    const double dx = target_map_x - rx;
    const double dy = target_map_y - ry;
    const double dist_to_person = std::hypot(dx, dy);
    const double angle_to_person = std::atan2(dy, dx);

    // 3. Tentukan Titik Berhenti Aman (Standoff Offset)
    // Robot berhenti sejauh standoff_dist_ (misal 0.85m) di depan orang dan menghadap langsung ke orang
    double goal_x = target_map_x;
    double goal_y = target_map_y;
    double goal_yaw = angle_to_person;

    if (dist_to_person > standoff_dist_) {
      goal_x = target_map_x - standoff_dist_ * std::cos(angle_to_person);
      goal_y = target_map_y - standoff_dist_ * std::sin(angle_to_person);
    } else {
      // Jika robot sudah sangat dekat dengan orang, pertahankan posisi robot sekarang dan hadapkan ke orang
      goal_x = rx;
      goal_y = ry;
    }

    RCLCPP_INFO(
      get_logger(),
      "\033[1;32m[Safe Goal]\033[0m Target Person: (%.2f, %.2f) | Goal Pose: (%.2f, %.2f, yaw=%.1f deg)",
      target_map_x, target_map_y, goal_x, goal_y, goal_yaw * 180.0 / M_PI);

    // 4. Simpan koordinat tujuan ini agar bisa digunakan kembali oleh service /deliver_order
    {
      std::lock_guard<std::mutex> lock(saved_order_mutex_);
      saved_order_x_ = goal_x;
      saved_order_y_ = goal_y;
      saved_order_yaw_ = goal_yaw;
      has_saved_order_ = true;
    }

    // Publikasikan juga ke topic /restaurant/saved_order_pose untuk visualisasi di RViz
    if (saved_pose_pub_) {
      geometry_msgs::msg::PoseStamped p;
      p.header.frame_id = global_frame_;
      p.header.stamp = now();
      p.pose.position.x = goal_x;
      p.pose.position.y = goal_y;
      p.pose.position.z = 0.0;
      p.pose.orientation = yawToQuaternion(goal_yaw);
      saved_pose_pub_->publish(p);
    }

    RCLCPP_INFO(
      get_logger(),
      "\033[1;34m[Order Caching]\033[0m Saved customer delivery pose: (x=%.2f, y=%.2f, yaw=%.1f deg) for /deliver_order",
      goal_x, goal_y, goal_yaw * 180.0 / M_PI);

    // 5. Kirim Goal Pose ke Nav2 Action Server
    return navigateToMapPose(goal_x, goal_y, goal_yaw, "Order");
  }

  // Handler service /deliver_order: Berlayar kembali ke koordinat pelanggan yang telah disimpan
  void handleDeliverOrder(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    if (is_searching_.load()) {
      res->success = false;
      res->message = "Cannot deliver: Search rotation is currently active.";
      RCLCPP_WARN(get_logger(), "[Deliver] %s", res->message.c_str());
      return;
    }

    if (is_navigating_.exchange(true)) {
      res->success = false;
      res->message = "Robot is currently busy with another navigation task.";
      RCLCPP_WARN(get_logger(), "[Deliver] %s", res->message.c_str());
      return;
    }

    double dx = 0.0, dy = 0.0, dyaw = 0.0;
    {
      std::lock_guard<std::mutex> lock(saved_order_mutex_);
      if (!has_saved_order_) {
        is_navigating_ = false;
        res->success = false;
        res->message = "No saved customer order position found. Please call /go_to_order first!";
        RCLCPP_WARN(get_logger(), "[Deliver] %s", res->message.c_str());
        return;
      }
      dx = saved_order_x_;
      dy = saved_order_y_;
      dyaw = saved_order_yaw_;
    }

    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[Deliver]\033[0m Navigating robot back to customer table at (x=%.2f, y=%.2f, yaw=%.1f deg)...",
      dx, dy, dyaw * 180.0 / M_PI);

    bool ok = navigateToMapPose(dx, dy, dyaw, "Deliver");
    if (ok) {
      RCLCPP_INFO(
        get_logger(),
        "\033[1;36m[Deliver]\033[0m Reached customer table. Adjusting heading to face customer bbox...");
      alignHeadingToPersonInFront();
    }
    is_navigating_ = false;

    res->success = ok;
    res->message = ok ? "Successfully arrived at customer table and aligned facing customer."
                      : "Failed or timed out navigating to customer table.";
  }

  // Handler Homing
  void handleHoming(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    if (is_searching_.load()) {
      res->success = false;
      res->message = "Cannot homing: Search rotation is currently active.";
      RCLCPP_WARN(get_logger(), "[Homing] %s", res->message.c_str());
      return;
    }

    if (is_navigating_.exchange(true)) {
      res->success = false;
      res->message = "Robot is currently busy navigating.";
      RCLCPP_WARN(get_logger(), "[Homing] %s", res->message.c_str());
      return;
    }

    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[Homing]\033[0m Navigating robot back to origin (x=%.2f, y=%.2f)...",
      origin_x_, origin_y_);

    bool ok = navigateToOrigin();
    is_navigating_ = false;

    res->success = ok;
    res->message = ok ? "Robot successfully arrived at homing position."
                      : "Failed or timed out navigating to homing position.";
  }

  bool navigateToOrigin()
  {
    return navigateToMapPose(origin_x_, origin_y_, origin_yaw_, "Homing");
  }

  // Menghentikan pergerakan robot dengan mempublikasikan nol pada /cmd_vel
  void stopRobot()
  {
    geometry_msgs::msg::Twist stop_twist;
    stop_twist.linear.x = 0.0;
    stop_twist.linear.y = 0.0;
    stop_twist.linear.z = 0.0;
    stop_twist.angular.x = 0.0;
    stop_twist.angular.y = 0.0;
    stop_twist.angular.z = 0.0;
    for (int i = 0; i < 3; ++i) {
      cmd_vel_pub_->publish(stop_twist);
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }

  // Menghitung kecepatan sudut rotasi proporsional dengan clamping
  double computeAngularVel(double cur_yaw, double target_yaw)
  {
    const double err = normalize_angle(target_yaw - cur_yaw);
    double wz = std::clamp(1.5 * err, -search_angular_speed_, search_angular_speed_);
    const double min_speed = 0.12;
    if (std::abs(wz) < min_speed && std::abs(err) > 0.02) {
      wz = (err > 0.0) ? min_speed : -min_speed;
    }
    return wz;
  }

  // Handler service /search: Memutar robot ke kiri 90 deg lalu ke kanan 90 deg
  // Jika saat rotasi melihat orang melambaikan tangan, service langsung berhenti dan success
  void handleSearch(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    if (is_navigating_.load()) {
      res->success = false;
      res->message = "Cannot search: Robot is currently navigating.";
      RCLCPP_WARN(get_logger(), "[Search] %s", res->message.c_str());
      return;
    }

    if (is_searching_.exchange(true)) {
      res->success = false;
      res->message = "Search is already running.";
      RCLCPP_WARN(get_logger(), "[Search] %s", res->message.c_str());
      return;
    }

    RCLCPP_INFO(
      get_logger(),
      "\033[1;33m[Search]\033[0m Starting search rotation (Left: %.1f deg, Right: %.1f deg)...",
      search_sweep_angle_deg_, search_sweep_angle_deg_);

    wave_seen_during_search_ = false;
    const auto search_start_time = now();

    double rx = 0.0, ry = 0.0, initial_yaw = 0.0;
    if (!getRobotPose(rx, ry, initial_yaw)) {
      stopRobot();
      is_searching_ = false;
      res->success = false;
      res->message = "Failed to get robot pose from TF before search.";
      RCLCPP_ERROR(get_logger(), "[Search] %s", res->message.c_str());
      return;
    }

    const double sweep_rad = search_sweep_angle_deg_ * M_PI / 180.0;
    const double target_left = normalize_angle(initial_yaw + sweep_rad);
    const double target_right = normalize_angle(initial_yaw - sweep_rad);
    const double target_center = initial_yaw;

    enum class SearchPhase {
      CHECK_FRONT,
      SWEEP_LEFT,
      HOLD_LEFT,
      RETURN_FROM_LEFT,
      SWEEP_RIGHT,
      HOLD_RIGHT,
      RETURN_FROM_RIGHT,
      DONE
    };

    SearchPhase phase = SearchPhase::CHECK_FRONT;
    rclcpp::Time phase_start_time = now();
    rclcpp::Rate rate(20.0);

    bool wave_found = false;

    while (rclcpp::ok() && is_searching_.load()) {
      // 1. Cek timeout keseluruhan
      if ((now() - search_start_time).seconds() > search_timeout_sec_) {
        RCLCPP_WARN(get_logger(), "[Search] Timeout %.1fs reached.", search_timeout_sec_);
        break;
      }

      // 2. Cek apakah orang melambaikan tangan terdeteksi secara real-time
      if (wave_seen_during_search_.load()) {
        wave_found = true;
        RCLCPP_INFO(
          get_logger(),
          "\033[1;32m[Search]\033[0m Waving customer detected during search rotation! Stopping rotation immediately.");
        break;
      }

      // Ambil heading robot terkini
      double cur_x = 0.0, cur_y = 0.0, cur_yaw = 0.0;
      if (!getRobotPose(cur_x, cur_y, cur_yaw)) {
        rate.sleep();
        continue;
      }

      geometry_msgs::msg::Twist cmd;

      switch (phase) {
        case SearchPhase::CHECK_FRONT: {
          cmd_vel_pub_->publish(cmd); // Diam sejenak cek depan
          if ((now() - phase_start_time).seconds() >= 0.4) {
            phase = SearchPhase::SWEEP_LEFT;
            phase_start_time = now();
            RCLCPP_INFO(get_logger(), "[Search] Sweeping Left to +%.1f deg...", search_sweep_angle_deg_);
          }
          break;
        }

        case SearchPhase::SWEEP_LEFT: {
          const double err = normalize_angle(target_left - cur_yaw);
          if (std::abs(err) <= 0.05 || (now() - phase_start_time).seconds() > 10.0) {
            cmd_vel_pub_->publish(cmd); // Stop sejenak
            phase = SearchPhase::HOLD_LEFT;
            phase_start_time = now();
            RCLCPP_INFO(get_logger(), "[Search] Reached Left position. Holding for visual detection...");
          } else {
            cmd.angular.z = computeAngularVel(cur_yaw, target_left);
            cmd_vel_pub_->publish(cmd);
          }
          break;
        }

        case SearchPhase::HOLD_LEFT: {
          cmd_vel_pub_->publish(cmd); // Diam di posisi kiri
          if ((now() - phase_start_time).seconds() >= search_hold_time_sec_) {
            phase = SearchPhase::RETURN_FROM_LEFT;
            phase_start_time = now();
            RCLCPP_INFO(get_logger(), "[Search] Returning to center from left...");
          }
          break;
        }

        case SearchPhase::RETURN_FROM_LEFT: {
          const double err = normalize_angle(target_center - cur_yaw);
          if (std::abs(err) <= 0.06 || (now() - phase_start_time).seconds() > 10.0) {
            cmd_vel_pub_->publish(cmd); // Stop
            phase = SearchPhase::SWEEP_RIGHT;
            phase_start_time = now();
            RCLCPP_INFO(get_logger(), "[Search] At center. Sweeping Right to -%.1f deg...", search_sweep_angle_deg_);
          } else {
            cmd.angular.z = computeAngularVel(cur_yaw, target_center);
            cmd_vel_pub_->publish(cmd);
          }
          break;
        }

        case SearchPhase::SWEEP_RIGHT: {
          const double err = normalize_angle(target_right - cur_yaw);
          if (std::abs(err) <= 0.05 || (now() - phase_start_time).seconds() > 10.0) {
            cmd_vel_pub_->publish(cmd); // Stop
            phase = SearchPhase::HOLD_RIGHT;
            phase_start_time = now();
            RCLCPP_INFO(get_logger(), "[Search] Reached Right position. Holding for visual detection...");
          } else {
            cmd.angular.z = computeAngularVel(cur_yaw, target_right);
            cmd_vel_pub_->publish(cmd);
          }
          break;
        }

        case SearchPhase::HOLD_RIGHT: {
          cmd_vel_pub_->publish(cmd); // Diam di posisi kanan
          if ((now() - phase_start_time).seconds() >= search_hold_time_sec_) {
            phase = SearchPhase::RETURN_FROM_RIGHT;
            phase_start_time = now();
            RCLCPP_INFO(get_logger(), "[Search] Returning to center from right...");
          }
          break;
        }

        case SearchPhase::RETURN_FROM_RIGHT: {
          const double err = normalize_angle(target_center - cur_yaw);
          if (std::abs(err) <= 0.06 || (now() - phase_start_time).seconds() > 10.0) {
            cmd_vel_pub_->publish(cmd); // Stop
            phase = SearchPhase::DONE;
            RCLCPP_INFO(get_logger(), "[Search] Rotation search sweep finished.");
          } else {
            cmd.angular.z = computeAngularVel(cur_yaw, target_center);
            cmd_vel_pub_->publish(cmd);
          }
          break;
        }

        case SearchPhase::DONE:
          break;
      }

      if (phase == SearchPhase::DONE) {
        break;
      }

      rate.sleep();
    }

    stopRobot();
    is_searching_ = false;

    if (wave_found) {
      double ox = 0.0, oy = 0.0;
      bool has_coords = false;
      {
        std::lock_guard<std::mutex> lock(order_mutex_);
        if (has_order_) {
          ox = latest_order_x_;
          oy = latest_order_y_;
          has_coords = true;
        }
      }

      res->success = true;
      if (has_coords) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "Waving customer found at (x=%.2f, y=%.2f)!", ox, oy);
        res->message = buf;
      } else {
        res->message = "Waving customer detected during search rotation!";
      }
      RCLCPP_INFO(get_logger(), "\033[1;32m[Search SUCCESS]\033[0m %s", res->message.c_str());
    } else {
      res->success = false;
      res->message = "Search rotation completed. No waving customer detected.";
      RCLCPP_INFO(get_logger(), "\033[1;33m[Search FINISHED]\033[0m %s", res->message.c_str());
    }
  }

  // Parameters
  double origin_x_{0.0};
  double origin_y_{0.0};
  double origin_yaw_{0.0};
  std::string global_frame_{"map"};
  std::string base_frame_{"base_link"};
  std::string order_topic_{"/vision/order"};
  std::string order_frame_{"base_link"};
  double standoff_dist_{0.85};
  bool auto_navigate_{false};
  double nav_timeout_sec_{120.0};

  // Search Settings & State
  double search_sweep_angle_deg_{90.0};
  double search_angular_speed_{0.35};
  double search_hold_time_sec_{0.5};
  double search_timeout_sec_{35.0};
  std::string cmd_vel_topic_{"/cmd_vel"};
  std::string gesture_topic_{"/vision/gesture"};

  // State & Data
  std::atomic<bool> is_navigating_{false};
  std::atomic<bool> is_searching_{false};
  std::atomic<bool> wave_seen_during_search_{false};
  GoalHandleNav::SharedPtr current_goal_handle_;
  std::mutex order_mutex_;
  bool has_order_{false};
  double latest_order_x_{0.0};
  double latest_order_y_{0.0};
  rclcpp::Time latest_order_time_;

  // TF2
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // Callback groups
  rclcpp::CallbackGroup::SharedPtr srv_cb_group_;
  rclcpp::CallbackGroup::SharedPtr action_cb_group_;
  rclcpp::CallbackGroup::SharedPtr sub_cb_group_;

  // ROS Interfaces
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr order_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr gesture_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp_action::Client<NavigateToPose>::SharedPtr nav_client_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_homing_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_homing_alias_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_go_order_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_go_order_alias_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_go_order1_compat_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_go_order1_alias_compat_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_search_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_search_alias_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_deliver_order_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_deliver_order_alias_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr saved_pose_pub_;

  // Saved Customer Order Pose (cached during /go_to_order for /deliver_order)
  std::mutex saved_order_mutex_;
  bool has_saved_order_{false};
  double saved_order_x_{0.0};
  double saved_order_y_{0.0};
  double saved_order_yaw_{0.0};

  // Align Heading to Person BBox Settings & State
  bool align_to_person_enabled_{true};
  std::string person_label_topic_{"/vision/person_label"};
  double align_timeout_sec_{10.0};
  double align_tolerance_px_{30.0};
  double align_p_gain_{0.6};
  double align_min_speed_{0.12};
  double align_max_speed_{0.35};
  double camera_image_width_{672.0};
  bool align_invert_direction_{false};

  std::mutex person_mutex_;
  std::vector<DetectedPerson> latest_persons_;
  rclcpp::Time latest_person_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr person_label_sub_;
};

}  // namespace restaurant

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<restaurant::RestaurantNode>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
