// ==============================================================================
// Barelang Inari - RoboCup@Home 2026
// Package  : restaurant
// Node     : restaurant_manager (RestaurantTaskNode)
// Deskripsi: Finite State Machine (FSM) utama untuk tugas Restaurant.
//            Mengatur inisialisasi origin, STT/TTS (athome_hri), deteksi
//            lambaian tangan & wajah (ZED Vision), navigasi otonom (Nav2 /
//            restaurant_node), manipulasi lengan (athome_arm), serta pelayanan
//            2 pelanggan secara berurutan.
// ==============================================================================

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <functional>
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
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

using namespace std::chrono_literals;

namespace restaurant
{

// ==============================================================================
// 1. DEFINISI STATE FINITE STATE MACHINE (FSM)
// ==============================================================================
enum class RestaurantState {
  INIT,                           // 1. Inisialisasi origin_position, STT & TTS engine
  ANNOUNCE_SERVER,                // 2a. Sapaan bar & instruksi melambaikan tangan
  DETECT_WAVE_HAND,               // 2b. Deteksi orang melambaikan tangan -> "i'm coming" -> call zed/wave_hand
  NAV_TO_WAVING_CUSTOMER,         // 3. Navigasi menuju pelanggan yang melambaikan tangan
  SAVE_CUSTOMER_POSE_ASK_NAME,    // 4a. Simpan koordinat pelanggan & tanya nama
  WAIT_CUSTOMER_NAME,             // 4b. Menunggu respon nama dari pelanggan
  CAPTURE_CUSTOMER_FACE,          // 5. "Hello {name}! please dont move..." -> call zed/camera_capture_face
  ASK_CUSTOMER_ORDER,             // 6a. "it's finished, May i take your order please?"
  WAIT_CUSTOMER_ORDER,            // 6b. Menunggu respon pesanan ({order_1} dan {order_2})
  PRE_CONFIRM_ORDER,              // 6c. "i will confirm your order, please answer with yes..."
  WAIT_BEFORE_CONFIRM_SPEECH,     // 6d. Jeda 0.5 detik sebelum menyebutkan detail pesanan
  SPEAK_ORDER_CONFIRMATION,       // 6e. "your order is {order_1} and {order_2} is it correct?"
  WAIT_ORDER_CONFIRMATION,        // 6f. Menunggu jawaban yes / no dari pelanggan
  NAV_TO_ORIGIN_BARMAN,           // 7. Navigasi kembali ke origin (barman place)
  ANNOUNCE_ORDER_TO_BARMAN,       // 8a. "The first/second customer order..." -> "please place {order_1}..."
  DETECT_AND_NAV_TO_ORDER1,       // 8b. Deteksi posisi {order_1} & navigasi ke posisi tersebut
  GRASP_ORDER1,                   // 8c. Panggil srv /arm/ambil_tas_simple -> instruksi taruh {order_2} di tray
  WAIT_TRAY_SILENT_5S,            // 8d. Diam selama 5 detik menunggu {order_2} ditaruh di tray
  ASK_TRAY_CONFIRMATION,          // 8e. "is the order had take in the tray, please answer with yes robot or no robot"
  WAIT_TRAY_CONFIRMATION,         // 8f. Menunggu jawaban yes robot / no robot
  WAIT_TRAY_RETRY_3S,             // 8g. Jika no, tunggu 3 detik lagi lalu ulangi callback pertanyaan
  NAV_DELIVER_TO_CUSTOMER,        // 9a. Navigasi kembali ke koordinat pelanggan
  GREET_DELIVERY_CUSTOMER,        // 9b. "Hello {name}, i have brought the order you requested"
  WAIT_1S_BEFORE_RELEASE_PROMPT,  // 9c. Tunggu 1 detik
  PROMPT_RELEASE_ORDER1,          // 9d. "for {order_1}, please hold out your hand beneath this order..."
  RELEASE_GRIPPER_AND_TORQUE_OFF, // 9e. Panggil service release gripper lalu matikan torsi arm (/arm/free)
  PROMPT_TAKE_ORDER2_TRAY,        // 9f. "for {order_2} please take it from the tray"
  WAIT_5S_CUSTOMER_TAKE_TRAY,     // 9g. Tunggu 5 detik -> "thank you for your order"
  NAV_RETURN_ORIGIN_AFTER_SERVE,  // 9h/10. Kembali ke origin_position, lalu cek loop pelanggan ke-2
  FINISH_TASK,                    // 11. Ucapkan "completed restaurant task"
  COMPLETED,                      // Tugas selesai
  ABORTED                         // Tugas dibatalkan karena error kritis / timeout global
};

static std::string stateToString(RestaurantState state)
{
  switch (state) {
    case RestaurantState::INIT:                           return "INIT";
    case RestaurantState::ANNOUNCE_SERVER:                return "ANNOUNCE_SERVER";
    case RestaurantState::DETECT_WAVE_HAND:               return "DETECT_WAVE_HAND";
    case RestaurantState::NAV_TO_WAVING_CUSTOMER:         return "NAV_TO_WAVING_CUSTOMER";
    case RestaurantState::SAVE_CUSTOMER_POSE_ASK_NAME:    return "SAVE_CUSTOMER_POSE_ASK_NAME";
    case RestaurantState::WAIT_CUSTOMER_NAME:             return "WAIT_CUSTOMER_NAME";
    case RestaurantState::CAPTURE_CUSTOMER_FACE:          return "CAPTURE_CUSTOMER_FACE";
    case RestaurantState::ASK_CUSTOMER_ORDER:             return "ASK_CUSTOMER_ORDER";
    case RestaurantState::WAIT_CUSTOMER_ORDER:            return "WAIT_CUSTOMER_ORDER";
    case RestaurantState::PRE_CONFIRM_ORDER:              return "PRE_CONFIRM_ORDER";
    case RestaurantState::WAIT_BEFORE_CONFIRM_SPEECH:     return "WAIT_BEFORE_CONFIRM_SPEECH";
    case RestaurantState::SPEAK_ORDER_CONFIRMATION:       return "SPEAK_ORDER_CONFIRMATION";
    case RestaurantState::WAIT_ORDER_CONFIRMATION:        return "WAIT_ORDER_CONFIRMATION";
    case RestaurantState::NAV_TO_ORIGIN_BARMAN:           return "NAV_TO_ORIGIN_BARMAN";
    case RestaurantState::ANNOUNCE_ORDER_TO_BARMAN:       return "ANNOUNCE_ORDER_TO_BARMAN";
    case RestaurantState::DETECT_AND_NAV_TO_ORDER1:       return "DETECT_AND_NAV_TO_ORDER1";
    case RestaurantState::GRASP_ORDER1:                   return "GRASP_ORDER1";
    case RestaurantState::WAIT_TRAY_SILENT_5S:            return "WAIT_TRAY_SILENT_5S";
    case RestaurantState::ASK_TRAY_CONFIRMATION:          return "ASK_TRAY_CONFIRMATION";
    case RestaurantState::WAIT_TRAY_CONFIRMATION:         return "WAIT_TRAY_CONFIRMATION";
    case RestaurantState::WAIT_TRAY_RETRY_3S:             return "WAIT_TRAY_RETRY_3S";
    case RestaurantState::NAV_DELIVER_TO_CUSTOMER:        return "NAV_DELIVER_TO_CUSTOMER";
    case RestaurantState::GREET_DELIVERY_CUSTOMER:        return "GREET_DELIVERY_CUSTOMER";
    case RestaurantState::WAIT_1S_BEFORE_RELEASE_PROMPT:  return "WAIT_1S_BEFORE_RELEASE_PROMPT";
    case RestaurantState::PROMPT_RELEASE_ORDER1:          return "PROMPT_RELEASE_ORDER1";
    case RestaurantState::RELEASE_GRIPPER_AND_TORQUE_OFF: return "RELEASE_GRIPPER_AND_TORQUE_OFF";
    case RestaurantState::PROMPT_TAKE_ORDER2_TRAY:        return "PROMPT_TAKE_ORDER2_TRAY";
    case RestaurantState::WAIT_5S_CUSTOMER_TAKE_TRAY:     return "WAIT_5S_CUSTOMER_TAKE_TRAY";
    case RestaurantState::NAV_RETURN_ORIGIN_AFTER_SERVE:  return "NAV_RETURN_ORIGIN_AFTER_SERVE";
    case RestaurantState::FINISH_TASK:                    return "FINISH_TASK";
    case RestaurantState::COMPLETED:                      return "COMPLETED";
    case RestaurantState::ABORTED:                        return "ABORTED";
    default:                                              return "UNKNOWN";
  }
}

// ==============================================================================
// 2. STRUKTUR DATA PELANGGAN & DETEKSI OBJEK VISION
// ==============================================================================
struct CustomerRecord {
  int id{1};
  std::string name;
  std::string order_1;
  std::string order_2;
  double map_x{0.0};
  double map_y{0.0};
  double map_yaw{0.0};
  bool pose_saved{false};
};

struct DetectedItem {
  std::string name;
  int bbox_x{0};
  int bbox_y{0};
  int bbox_w{0};
  int bbox_h{0};
  double x{0.0};
  double y{0.0};
  double z{0.0};
  bool has_3d{false};
};

static geometry_msgs::msg::Quaternion yawToQuaternion(double yaw_rad)
{
  geometry_msgs::msg::Quaternion q;
  q.x = 0.0;
  q.y = 0.0;
  q.z = std::sin(yaw_rad * 0.5);
  q.w = std::cos(yaw_rad * 0.5);
  return q;
}

static bool parseCoordinates(const std::string & raw_data, double & out_x, double & out_y)
{
  if (raw_data.empty()) return false;
  std::string data = raw_data;

  if (data.find('|') != std::string::npos) {
    std::stringstream ss(data);
    std::string seg;
    std::vector<std::string> parts;
    while (std::getline(ss, seg, '|')) {
      parts.push_back(seg);
    }
    if (parts.size() >= 2) {
      data = parts[1];
    }
  }

  std::regex json_regex(R"("x"\s*:\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*"y"\s*:\s*([-+]?[0-9]*\.?[0-9]+))");
  std::smatch json_match;
  if (std::regex_search(data, json_match, json_regex)) {
    try {
      out_x = std::stod(json_match[1].str());
      out_y = std::stod(json_match[2].str());
      return std::isfinite(out_x) && std::isfinite(out_y);
    } catch (...) {}
  }

  auto colon_pos = data.find(':');
  if (colon_pos != std::string::npos) {
    data = data.substr(colon_pos + 1);
  }

  for (char & c : data) {
    if (c == ',' || c == ';' || c == '(' || c == ')' || c == '[' || c == ']') {
      c = ' ';
    }
  }

  std::stringstream ss(data);
  double x_val = 0.0, y_val = 0.0;
  if (ss >> x_val >> y_val) {
    if (std::isfinite(x_val) && std::isfinite(y_val)) {
      out_x = x_val;
      out_y = y_val;
      return true;
    }
  }
  return false;
}

// ==============================================================================
// 3. KELAS UTAMA: RestaurantTaskNode
// ==============================================================================
class RestaurantTaskNode : public rclcpp::Node
{
public:
  using Trigger = std_srvs::srv::Trigger;
  using TriggerClient = rclcpp::Client<Trigger>;
  using TriggerResult = std::function<void(bool, const std::string &)>;
  using NavigateToPose = nav2_msgs::action::NavigateToPose;
  using GoalHandleNav = rclcpp_action::ClientGoalHandle<NavigateToPose>;

  RestaurantTaskNode()
  : Node("restaurant_manager")
  {
    // --------------------------------------------------------------------------
    // Deklarasi Parameter Konfigurasi
    // --------------------------------------------------------------------------
    declare_parameter<double>("origin_x", 0.0);
    declare_parameter<double>("origin_y", 0.0);
    declare_parameter<double>("origin_yaw", 0.0);
    declare_parameter<bool>("capture_tf_origin_on_init", true);
    declare_parameter<std::string>("global_frame", "map");
    declare_parameter<std::string>("base_frame", "base_link");
    declare_parameter<std::string>("order_frame", "base_link");
    declare_parameter<double>("customer_standoff_distance", 0.85);
    declare_parameter<double>("object_standoff_distance", 0.45);
    declare_parameter<double>("nav_timeout_sec", 120.0);
    declare_parameter<double>("order1_detect_timeout_sec", 8.0);
    declare_parameter<int>("max_customers", 2);

    origin_x_ = get_parameter("origin_x").as_double();
    origin_y_ = get_parameter("origin_y").as_double();
    origin_yaw_ = get_parameter("origin_yaw").as_double();
    capture_tf_origin_on_init_ = get_parameter("capture_tf_origin_on_init").as_bool();
    global_frame_ = get_parameter("global_frame").as_string();
    base_frame_ = get_parameter("base_frame").as_string();
    order_frame_ = get_parameter("order_frame").as_string();
    customer_standoff_dist_ = get_parameter("customer_standoff_distance").as_double();
    object_standoff_dist_ = get_parameter("object_standoff_distance").as_double();
    nav_timeout_sec_ = get_parameter("nav_timeout_sec").as_double();
    order1_detect_timeout_sec_ = get_parameter("order1_detect_timeout_sec").as_double();
    max_customers_ = std::max(1, static_cast<int>(get_parameter("max_customers").as_int()));

    // TF2 Buffer & Listener
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.callback_group = callback_group_;

    // --------------------------------------------------------------------------
    // Publisher & Subscriber HRI (TTS & STT)
    // --------------------------------------------------------------------------
    tts_pub_ = create_publisher<std_msgs::msg::String>("/hri/tts_say", 10);
    status_pub_ = create_publisher<std_msgs::msg::String>("/restaurant/status", 10);
    person_name_pub_ = create_publisher<std_msgs::msg::String>("/vision/set_person_name", 10);

    tts_done_sub_ = create_subscription<std_msgs::msg::String>(
      "/hri/tts_done", 10,
      std::bind(&RestaurantTaskNode::onTtsDone, this, std::placeholders::_1),
      sub_opts);

    stt_sub_ = create_subscription<std_msgs::msg::String>(
      "/hri/stt_transcript", 10,
      std::bind(&RestaurantTaskNode::onTranscript, this, std::placeholders::_1),
      sub_opts);

    emergency_stt_sub_ = create_subscription<std_msgs::msg::String>(
      "/stt/keyword_interrupt", 10,
      std::bind(&RestaurantTaskNode::onTranscript, this, std::placeholders::_1),
      sub_opts);

    // --------------------------------------------------------------------------
    // Subscriber Vision (Camera)
    // --------------------------------------------------------------------------
    gesture_sub_ = create_subscription<std_msgs::msg::String>(
      "/vision/gesture", 10,
      std::bind(&RestaurantTaskNode::onGesture, this, std::placeholders::_1),
      sub_opts);

    person_sub_ = create_subscription<std_msgs::msg::String>(
      "/vision/person_label", rclcpp::SensorDataQoS(),
      std::bind(&RestaurantTaskNode::onPersonLabel, this, std::placeholders::_1),
      sub_opts);

    order_pos_sub_ = create_subscription<std_msgs::msg::String>(
      "/vision/order", 10,
      std::bind(&RestaurantTaskNode::onOrderTopic, this, std::placeholders::_1),
      sub_opts);

    drink_sub_ = create_subscription<std_msgs::msg::String>(
      "/vision/drink_detections", rclcpp::SensorDataQoS(),
      std::bind(&RestaurantTaskNode::onDrinkDetections, this, std::placeholders::_1),
      sub_opts);

    bag_sub_ = create_subscription<std_msgs::msg::String>(
      "/vision/bag_detections", rclcpp::SensorDataQoS(),
      std::bind(&RestaurantTaskNode::onBagDetections, this, std::placeholders::_1),
      sub_opts);

    // --------------------------------------------------------------------------
    // Action & Service Clients (Vision, Navigation, Arm)
    // --------------------------------------------------------------------------
    nav_action_client_ = rclcpp_action::create_client<NavigateToPose>(
      this, "navigate_to_pose", callback_group_);

    // Vision Services
    vision_wave_hand_client_ = create_client<Trigger>(
      "/vision/wave_hand", rclcpp::ServicesQoS(), callback_group_);
    vision_capture_face_client_ = create_client<Trigger>(
      "/vision/camera_capture_face", rclcpp::ServicesQoS(), callback_group_);

    // Navigation helper services (dari restaurant_node jika dijalankan bersamaan)
    go_to_order1_client_ = create_client<Trigger>(
      "/go_to_order1", rclcpp::ServicesQoS(), callback_group_);
    homing_client_ = create_client<Trigger>(
      "/homing", rclcpp::ServicesQoS(), callback_group_);

    // Arm Services
    arm_grab_simple_client_ = create_client<Trigger>(
      "/arm/ambil_tas_simple", rclcpp::ServicesQoS(), callback_group_);
    arm_release_client_ = create_client<Trigger>(
      "/arm/release_gripper", rclcpp::ServicesQoS(), callback_group_);
    arm_letakkan_client_ = create_client<Trigger>(
      "/arm/letakkan_tas", rclcpp::ServicesQoS(), callback_group_);
    arm_free_client_ = create_client<Trigger>(
      "/arm/free", rclcpp::ServicesQoS(), callback_group_);
    arm_relax_client_ = create_client<Trigger>(
      "/arm/relax", rclcpp::ServicesQoS(), callback_group_);

    // Timer siklus FSM (10 Hz / 100 ms)
    fsm_timer_ = create_wall_timer(
      100ms, std::bind(&RestaurantTaskNode::fsmTick, this), callback_group_);

    task_start_time_ = now();
    state_enter_time_ = now();
    initCurrentCustomer(1);

    RCLCPP_INFO(
      get_logger(),
      "\n============================================================\n"
      "  \033[1;32m[SYSTEM]\033[0m Restaurant Task Manager Started\n"
      "  \033[1;36m[ORIGIN]\033[0m x=%.2f, y=%.2f, yaw=%.2f\n"
      "  \033[1;33m[STATE]\033[0m  INIT\n"
      "============================================================",
      origin_x_, origin_y_, origin_yaw_);
  }

private:
  // ----------------------------------------------------------------------------
  // Konstanta Waktu & Timeout
  // ----------------------------------------------------------------------------
  static constexpr double GLOBAL_TIMEOUT_SEC     = 900.0;
  static constexpr double DIALOG_REPROMPT_SEC    = 12.0;
  static constexpr double WAVE_REPROMPT_SEC      = 10.0;
  static constexpr double TTS_MIN_TIMEOUT_SEC    = 4.0;
  static constexpr double TTS_SEC_PER_WORD       = 0.5;
  static constexpr double TTS_TIMEOUT_MARGIN_SEC = 2.0;
  static constexpr double NAV_RETRY_COOLDOWN_SEC = 2.0;

  // ----------------------------------------------------------------------------
  // Parameter & State Variables
  // ----------------------------------------------------------------------------
  double origin_x_{0.0};
  double origin_y_{0.0};
  double origin_yaw_{0.0};
  bool capture_tf_origin_on_init_{true};
  bool origin_initialized_{false};
  std::string global_frame_{"map"};
  std::string base_frame_{"base_link"};
  std::string order_frame_{"base_link"};
  double customer_standoff_dist_{0.85};
  double object_standoff_dist_{0.45};
  double nav_timeout_sec_{120.0};
  double order1_detect_timeout_sec_{8.0};
  int max_customers_{2};

  std::atomic<RestaurantState> state_{RestaurantState::INIT};
  std::atomic<bool> speaking_{false};
  std::atomic<bool> service_busy_{false};
  std::atomic<bool> nav_busy_{false};
  std::atomic<bool> in_callback_{false};

  // Data Pelanggan
  std::mutex customer_mutex_;
  std::vector<CustomerRecord> served_customers_;
  CustomerRecord current_customer_;
  int customer_idx_{1};

  // Data Deteksi Vision (Wave Hand, Person Position, Order Items)
  std::mutex vision_mutex_;
  bool wave_detected_{false};
  bool has_waving_person_coords_{false};
  double waving_person_x_{0.0};
  double waving_person_y_{0.0};
  double last_nav_goal_x_{0.0};
  double last_nav_goal_y_{0.0};
  double last_nav_goal_yaw_{0.0};
  std::vector<DetectedItem> detected_items_;

  // Data STT / Transkripsi Suara
  std::mutex stt_mutex_;
  std::string heard_utterance_;
  std::atomic<bool> has_new_utterance_{false};
  std::string last_utterance_;
  rclcpp::Time last_utterance_time_{0, 0, RCL_ROS_TIME};

  // Timers & Watchdogs
  rclcpp::Time task_start_time_;
  rclcpp::Time state_enter_time_;
  rclcpp::Time last_wave_prompt_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time tts_cooldown_until_{0, 0, RCL_ROS_TIME};
  rclcpp::Time tts_started_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_nav_fail_time_{0, 0, RCL_ROS_TIME};

  std::mutex tts_mutex_;
  double tts_timeout_sec_{TTS_MIN_TIMEOUT_SEC};
  std::function<void()> pending_tts_done_;

  std::mutex srv_mutex_;
  rclcpp::Time srv_started_time_{0, 0, RCL_ROS_TIME};
  double srv_timeout_sec_{30.0};
  std::string active_srv_name_;
  TriggerResult pending_srv_result_;

  GoalHandleNav::SharedPtr current_goal_handle_;

  // ROS 2 Interfaces
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp::TimerBase::SharedPtr fsm_timer_;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr tts_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr person_name_pub_;

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr tts_done_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr stt_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr emergency_stt_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr gesture_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr person_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr order_pos_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr drink_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr bag_sub_;

  rclcpp_action::Client<NavigateToPose>::SharedPtr nav_action_client_;
  TriggerClient::SharedPtr vision_wave_hand_client_;
  TriggerClient::SharedPtr vision_capture_face_client_;
  TriggerClient::SharedPtr go_to_order1_client_;
  TriggerClient::SharedPtr homing_client_;
  TriggerClient::SharedPtr arm_grab_simple_client_;
  TriggerClient::SharedPtr arm_release_client_;
  TriggerClient::SharedPtr arm_letakkan_client_;
  TriggerClient::SharedPtr arm_free_client_;
  TriggerClient::SharedPtr arm_relax_client_;

  // ============================================================================
  // HELPER STATE & MANAGEMENT
  // ============================================================================
  bool isBusy() const
  {
    return speaking_.load() || service_busy_.load() || nav_busy_.load() || in_callback_.load();
  }

  void publishStatus(const std::string & text)
  {
    if (status_pub_) {
      std_msgs::msg::String msg;
      msg.data = text;
      status_pub_->publish(msg);
    }
    RCLCPP_INFO(get_logger(), "\033[1;32m[RESTAURANT STATUS]\033[0m %s", text.c_str());
  }

  void initCurrentCustomer(int idx)
  {
    {
      std::lock_guard<std::mutex> lock(customer_mutex_);
      customer_idx_ = idx;
      current_customer_ = CustomerRecord{};
      current_customer_.id = idx;
    }
    {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      wave_detected_ = false;
      has_waving_person_coords_ = false;
      detected_items_.clear();
    }
    if (person_name_pub_) {
      std_msgs::msg::String msg;
      msg.data = "";
      person_name_pub_->publish(msg);
    }
  }

  void transitionTo(RestaurantState next_state, const std::string & reason)
  {
    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[FSM STATE]\033[0m %s \033[1;33m->\033[0m \033[1;32m%s\033[0m (Customer %d) | %s",
      stateToString(state_.load()).c_str(),
      stateToString(next_state).c_str(),
      customer_idx_,
      reason.c_str());

    if (next_state == RestaurantState::DETECT_WAVE_HAND) {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      wave_detected_ = false;
      last_wave_prompt_time_ = now();
    }
    if (next_state == RestaurantState::DETECT_AND_NAV_TO_ORDER1) {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      detected_items_.clear();
    }

    state_enter_time_ = now();
    has_new_utterance_ = false;
    state_ = next_state;
  }

  // ============================================================================
  // HELPER POSE & NAVIGATION (TF2 + Nav2 NavigateToPose / Services)
  // ============================================================================
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
      RCLCPP_DEBUG(
        get_logger(), "[TF] Lookup %s -> %s belum siap: %s",
        global_frame_.c_str(), base_frame_.c_str(), ex.what());
      return false;
    }
  }

  // Navigasi ke koordinat global (map_x, map_y, map_yaw) secara asinkron via Nav2
  void navigateToMapPoseAsync(
    double goal_x, double goal_y, double goal_yaw,
    std::function<void(bool)> on_finished)
  {
    {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      last_nav_goal_x_ = goal_x;
      last_nav_goal_y_ = goal_y;
      last_nav_goal_yaw_ = goal_yaw;
    }

    if (!nav_action_client_->wait_for_action_server(2s)) {
      RCLCPP_ERROR(get_logger(), "[Nav2] Action server 'navigate_to_pose' tidak tersedia!");
      last_nav_fail_time_ = now();
      if (on_finished) on_finished(false);
      return;
    }

    nav_busy_ = true;

    NavigateToPose::Goal goal;
    goal.pose.header.frame_id = global_frame_;
    goal.pose.header.stamp = now();
    goal.pose.pose.position.x = goal_x;
    goal.pose.pose.position.y = goal_y;
    goal.pose.pose.position.z = 0.0;
    goal.pose.pose.orientation = yawToQuaternion(goal_yaw);

    RCLCPP_INFO(
      get_logger(),
      "\033[1;34m[NAV2]\033[0m Sending Goal Pose: (x=%.2f, y=%.2f, yaw=%.1f deg)",
      goal_x, goal_y, goal_yaw * 180.0 / M_PI);

    auto send_opts = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    send_opts.goal_response_callback =
      [this, on_finished](const GoalHandleNav::SharedPtr & handle) {
        if (!handle) {
          RCLCPP_WARN(get_logger(), "[Nav2] Goal ditolak oleh server Nav2!");
          nav_busy_ = false;
          last_nav_fail_time_ = now();
          if (on_finished) on_finished(false);
        } else {
          current_goal_handle_ = handle;
          RCLCPP_INFO(get_logger(), "[Nav2] Goal diterima! Robot sedang bergerak...");
        }
      };

    send_opts.result_callback =
      [this, on_finished](const GoalHandleNav::WrappedResult & result) {
        in_callback_ = true;
        nav_busy_ = false;
        const bool ok = (result.code == rclcpp_action::ResultCode::SUCCEEDED);
        if (ok) {
          RCLCPP_INFO(get_logger(), "\033[1;32m[NAV2]\033[0m Robot tiba di tujuan!");
        } else {
          RCLCPP_WARN(
            get_logger(), "[Nav2] Navigasi gagal/dibatalkan (code=%d).",
            static_cast<int>(result.code));
          last_nav_fail_time_ = now();
        }
        if (on_finished) {
          on_finished(ok);
        }
        in_callback_ = false;
      };

    nav_action_client_->async_send_goal(goal, send_opts);
  }

  // Navigasi ke target (orang atau objek) dengan jarak henti aman (standoff distance)
  void navigateToRelativeOrMapTargetAsync(
    double target_x, double target_y, double standoff_dist,
    std::function<void(bool)> on_finished)
  {
    double rx = origin_x_, ry = origin_y_, ryaw = origin_yaw_;
    const bool has_tf = getRobotPose(rx, ry, ryaw);

    double map_tx = target_x;
    double map_ty = target_y;
    if (order_frame_ == "base_link" && has_tf) {
      map_tx = rx + target_x * std::cos(ryaw) - target_y * std::sin(ryaw);
      map_ty = ry + target_x * std::sin(ryaw) + target_y * std::cos(ryaw);
    }

    const double dx = map_tx - rx;
    const double dy = map_ty - ry;
    const double dist = std::hypot(dx, dy);
    const double angle = std::atan2(dy, dx);

    double goal_x = map_tx;
    double goal_y = map_ty;
    const double goal_yaw = angle;

    if (dist > standoff_dist) {
      goal_x = map_tx - standoff_dist * std::cos(angle);
      goal_y = map_ty - standoff_dist * std::sin(angle);
    } else {
      goal_x = rx;
      goal_y = ry;
    }

    navigateToMapPoseAsync(goal_x, goal_y, goal_yaw, std::move(on_finished));
  }

  // Navigasi ke pelanggan yang melambaikan tangan
  void navigateToWavingCustomer(std::function<void(bool)> on_finished)
  {
    // Prioritaskan service /go_to_order1 dari restaurant_node jika tersedia
    if (go_to_order1_client_ && go_to_order1_client_->service_is_ready()) {
      callTrigger(
        go_to_order1_client_, "/go_to_order1",
        [this, on_finished](bool ok, const std::string & msg) {
          if (!ok) {
            RCLCPP_WARN(get_logger(), "[NAV] /go_to_order1 gagal (%s), mencoba direct Nav2...", msg.c_str());
            double wx = 1.5, wy = 0.0;
            {
              std::lock_guard<std::mutex> lock(vision_mutex_);
              if (has_waving_person_coords_) {
                wx = waving_person_x_;
                wy = waving_person_y_;
              }
            }
            navigateToRelativeOrMapTargetAsync(wx, wy, customer_standoff_dist_, on_finished);
            return;
          }
          if (on_finished) on_finished(true);
        },
        nav_timeout_sec_);
      return;
    }

    double wx = 1.5, wy = 0.0;
    {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      if (has_waving_person_coords_) {
        wx = waving_person_x_;
        wy = waving_person_y_;
      }
    }
    navigateToRelativeOrMapTargetAsync(wx, wy, customer_standoff_dist_, std::move(on_finished));
  }

  // Navigasi kembali ke titik origin (barman place)
  void navigateToOrigin(std::function<void(bool)> on_finished)
  {
    if (homing_client_ && homing_client_->service_is_ready()) {
      callTrigger(
        homing_client_, "/homing",
        [this, on_finished](bool ok, const std::string &) {
          if (!ok) {
            navigateToMapPoseAsync(origin_x_, origin_y_, origin_yaw_, on_finished);
            return;
          }
          if (on_finished) on_finished(true);
        },
        nav_timeout_sec_);
      return;
    }
    navigateToMapPoseAsync(origin_x_, origin_y_, origin_yaw_, std::move(on_finished));
  }

  // ============================================================================
  // HELPER TEXT-TO-SPEECH (TTS) & SPEECH-TO-TEXT (STT) PARSING
  // ============================================================================
  static double estimateTtsTimeout(const std::string & text)
  {
    size_t words = 0;
    bool in_word = false;
    for (const char c : text) {
      if (std::isspace(static_cast<unsigned char>(c))) {
        in_word = false;
      } else if (!in_word) {
        in_word = true;
        ++words;
      }
    }
    return std::max(
      TTS_MIN_TIMEOUT_SEC,
      static_cast<double>(words) * TTS_SEC_PER_WORD + TTS_TIMEOUT_MARGIN_SEC);
  }

  void speakAsync(const std::string & text, std::function<void()> on_done = nullptr)
  {
    {
      std::lock_guard<std::mutex> lock(tts_mutex_);
      speaking_ = true;
      tts_started_time_ = now();
      tts_timeout_sec_ = estimateTtsTimeout(text);
      tts_cooldown_until_ = now() + rclcpp::Duration::from_seconds(0.5);
      pending_tts_done_ = std::move(on_done);
    }

    std_msgs::msg::String message;
    message.data = text;
    tts_pub_->publish(message);
    RCLCPP_INFO(
      get_logger(),
      "\033[1;35m[TTS SAY]\033[0m \"%s\" | Mic: \033[1;31m[MUTED]\033[0m",
      text.c_str());
  }

  void onTtsDone(const std_msgs::msg::String::SharedPtr = nullptr)
  {
    std::function<void()> callback;
    {
      std::lock_guard<std::mutex> lock(tts_mutex_);
      if (!speaking_.load()) {
        return;
      }
      in_callback_ = true;
      speaking_ = false;
      tts_cooldown_until_ = now() + rclcpp::Duration::from_seconds(0.4);
      callback = std::move(pending_tts_done_);
      pending_tts_done_ = nullptr;
    }

    RCLCPP_INFO(
      get_logger(),
      "\033[1;32m[TTS FINISHED]\033[0m Playback complete | Mic: \033[1;32m[LISTENING]\033[0m | State: \033[1;36m%s\033[0m",
      stateToString(state_.load()).c_str());

    if (callback) {
      callback();
    }
    in_callback_ = false;
  }

  bool canListen() const
  {
    return !isBusy() && now() >= tts_cooldown_until_;
  }

  static std::string normalize(const std::string & text)
  {
    std::string clean;
    clean.reserve(text.size());
    bool prev_space = true;
    for (unsigned char c : text) {
      if (std::isalnum(c)) {
        clean.push_back(static_cast<char>(std::tolower(c)));
        prev_space = false;
      } else if (!prev_space) {
        clean.push_back(' ');
        prev_space = true;
      }
    }
    if (!clean.empty() && clean.back() == ' ') {
      clean.pop_back();
    }
    return clean;
  }

  static std::string capitalizeWords(const std::string & text)
  {
    std::string out = text;
    bool cap_next = true;
    for (char & c : out) {
      if (std::isspace(static_cast<unsigned char>(c))) {
        cap_next = true;
      } else if (cap_next) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        cap_next = false;
      }
    }
    return out;
  }

  static bool containsWholePhrase(const std::string & norm_haystack, const std::string & phrase)
  {
    const std::string padded_h = " " + norm_haystack + " ";
    const std::string padded_p = " " + phrase + " ";
    return padded_h.find(padded_p) != std::string::npos;
  }

  static std::string trimString(const std::string & s)
  {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
  }

  // Ekstraksi nama pelanggan dari kalimat STT
  static std::string parseCustomerName(const std::string & raw_text)
  {
    const std::string norm = normalize(raw_text);
    if (norm.empty()) return "";

    static const std::vector<std::string> kIgnoredFillers = {
      "hello", "hi", "hey", "yes", "no", "okay", "ok", "what", "sorry",
      "pardon", "excuse me", "thank you", "thanks", "please", "robot"
    };
    for (const auto & filler : kIgnoredFillers) {
      if (norm == filler) return "";
    }

    static const std::vector<std::string> kKnownNames = {
      "adel", "angel", "axel", "charlie", "jane", "john", "jules", "morgan",
      "paris", "robin", "simone", "alex", "michael", "sarah", "david", "emma"
    };
    for (const auto & name : kKnownNames) {
      if (containsWholePhrase(norm, name)) {
        return capitalizeWords(name);
      }
    }

    static const std::vector<std::string> kPrefixes = {
      "my name is ", "i am ", "im ", "call me ", "this is ", "name is "
    };
    for (const auto & p : kPrefixes) {
      const auto pos = norm.find(p);
      if (pos != std::string::npos) {
        std::string rem = trimString(norm.substr(pos + p.size()));
        if (!rem.empty()) {
          std::stringstream ss(rem);
          std::string w1, w2;
          ss >> w1 >> w2;
          return capitalizeWords(w2.empty() ? w1 : (w1 + " " + w2));
        }
      }
    }

    std::stringstream ss(norm);
    std::vector<std::string> words;
    std::string w;
    while (ss >> w) {
      words.push_back(w);
    }
    if (!words.empty() && words.size() <= 2) {
      return capitalizeWords(norm);
    }
    return "";
  }

  // Ekstraksi 2 item pesanan ({order_1} dan {order_2}) dari kalimat pelanggan
  static bool parseCustomerOrders(
    const std::string & raw_text,
    std::string & order_1,
    std::string & order_2)
  {
    std::string norm = normalize(raw_text);
    if (norm.empty()) return false;

    // 1. Cek daftar item menu yang dikenal berdasarkan urutan kemunculan di kalimat
    static const std::vector<std::string> kKnownMenuItems = {
      "orange juice", "apple juice", "iced tea", "green tea", "mineral water",
      "cracker box", "potato chips", "coca cola",
      "pringles", "cheezit", "cracker", "chips", "snack",
      "coke", "cola", "pepsi", "sprite", "fanta",
      "water", "milk", "coffee", "tea", "juice", "soda", "bottle", "botol",
      "burger", "sandwich", "pizza", "cake", "donut", "apple", "banana", "orange"
    };

    struct FoundItem {
      size_t pos;
      std::string name;
    };
    std::vector<FoundItem> found;
    std::string masked = " " + norm + " ";

    for (const auto & item : kKnownMenuItems) {
      const std::string needle = " " + item + " ";
      const auto pos = masked.find(needle);
      if (pos != std::string::npos) {
        found.push_back({pos, capitalizeWords(item)});
        // Mask agar tidak terambil dua kali oleh sub-kata (misal "orange juice" & "juice")
        for (size_t i = pos + 1; i < pos + needle.size() - 1; ++i) {
          masked[i] = '_';
        }
      }
    }

    std::sort(found.begin(), found.end(), [](const FoundItem & a, const FoundItem & b) {
      return a.pos < b.pos;
    });

    if (found.size() >= 2) {
      order_1 = found[0].name;
      order_2 = found[1].name;
      return true;
    }

    // 2. Fallback: Pisahkan berdasarkan kata penghubung ("and", "with", "plus", "also", "dan")
    static const std::vector<std::string> kOrderPrefixes = {
      "i would like to order ", "i would like ", "i want to order ", "i want ",
      "can i have ", "may i have ", "please give me ", "give me ",
      "my order is ", "i order ", "order is ", "please "
    };
    for (const auto & prefix : kOrderPrefixes) {
      if (norm.find(prefix) == 0) {
        norm = trimString(norm.substr(prefix.size()));
        break;
      } else {
        const auto p = norm.find(prefix);
        if (p != std::string::npos) {
          norm = trimString(norm.substr(p + prefix.size()));
          break;
        }
      }
    }

    static const std::vector<std::string> kConnectors = {
      " and ", " with ", " plus ", " also ", " dan ", " sama "
    };
    for (const auto & conn : kConnectors) {
      const auto cpos = norm.find(conn);
      if (cpos != std::string::npos) {
        std::string left = trimString(norm.substr(0, cpos));
        std::string right = trimString(norm.substr(cpos + conn.size()));
        // Hilangkan artikel a / an / the di awal item
        for (std::string * part : {&left, &right}) {
          for (const char * art_cstr : {"a ", "an ", "the ", "one "}) {
            const std::string art(art_cstr);
            if (part->find(art) == 0) {
              *part = trimString(part->substr(art.size()));
            }
          }
        }
        if (!left.empty() && !right.empty()) {
          order_1 = capitalizeWords(left);
          order_2 = capitalizeWords(right);
          return true;
        }
      }
    }

    // Jika baru 1 item yang disebutkan, simpan ke slot yang masih kosong
    if (found.size() == 1) {
      if (order_1.empty()) {
        order_1 = found[0].name;
      } else if (order_2.empty() && found[0].name != order_1) {
        order_2 = found[0].name;
      }
      return !order_1.empty() && !order_2.empty();
    }

    return false;
  }

  // Pengenalan respon konfirmasi Yes / No (termasuk "yes robot" / "no robot")
  static int parseYesNoIntent(const std::string & raw_text)
  {
    const std::string norm = normalize(raw_text);
    if (norm.empty()) return 0;

    static const std::vector<std::string> kNoPhrases = {
      "no robot", "no", "nope", "nah", "not correct", "wrong", "false",
      "incorrect", "negative", "not yet", "wait", "belum", "tidak", "bukan"
    };
    static const std::vector<std::string> kYesPhrases = {
      "yes robot", "yes", "yeah", "yep", "correct", "right", "sure",
      "true", "exactly", "of course", "already", "done", "ya", "benar", "betul", "sudah"
    };

    for (const auto & phrase : kNoPhrases) {
      if (containsWholePhrase(norm, phrase)) {
        return -1;
      }
    }
    for (const auto & phrase : kYesPhrases) {
      if (containsWholePhrase(norm, phrase)) {
        return 1;
      }
    }
    return 0;
  }

  // ============================================================================
  // CALLBACK SUBSCRIPTION ROS 2 (STT & VISION)
  // ============================================================================
  void onTranscript(const std_msgs::msg::String::SharedPtr message)
  {
    if (!canListen() || !message) {
      return;
    }

    const std::string clean = normalize(message->data);
    if (clean.empty()) {
      return;
    }

    const auto current_time = now();
    {
      std::lock_guard<std::mutex> lock(stt_mutex_);
      if (clean == last_utterance_ && (current_time - last_utterance_time_).seconds() < 2.0) {
        return;
      }
      last_utterance_ = clean;
      last_utterance_time_ = current_time;
      heard_utterance_ = message->data;
      has_new_utterance_ = true;
    }

    RCLCPP_INFO(
      get_logger(),
      "\033[1;33m[STT HEARD]\033[0m \"%s\" | State: \033[1;36m%s\033[0m",
      message->data.c_str(),
      stateToString(state_.load()).c_str());
  }

  // Callback deteksi lambaian tangan / gesture dari Vision (/vision/gesture)
  void onGesture(const std_msgs::msg::String::SharedPtr message)
  {
    if (!message || message->data.empty()) return;
    const std::string gest = normalize(message->data);
    if (gest.empty() || gest == "none") return;

    if (gest.find("wave") != std::string::npos ||
        gest.find("left") != std::string::npos ||
        gest.find("right") != std::string::npos ||
        gest.find("raise") != std::string::npos ||
        gest.find("hand") != std::string::npos)
    {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      wave_detected_ = true;
    }
  }

  // Callback topik posisi pemesan (/vision/order)
  void onOrderTopic(const std_msgs::msg::String::SharedPtr message)
  {
    if (!message || message->data.empty()) return;
    double px = 0.0, py = 0.0;
    if (parseCoordinates(message->data, px, py)) {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      waving_person_x_ = px;
      waving_person_y_ = py;
      has_waving_person_coords_ = true;
      wave_detected_ = true;
    }
  }

  // Callback deteksi orang & ekstraksi koordinat 3D (/vision/person_label)
  void onPersonLabel(const std_msgs::msg::String::SharedPtr message)
  {
    if (!message || message->data.empty() ||
        message->data.find("NO_PERSON") != std::string::npos)
    {
      return;
    }

    double px = 0.0, py = 0.0;
    if (parseCoordinates(message->data, px, py)) {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      // Jangan menimpa koordinat orang yang melambai dari /vision/order.
      // /vision/person_label hanya menjadi fallback posisi awal.
      if (!has_waving_person_coords_) {
        waving_person_x_ = px;
        waving_person_y_ = py;
        has_waving_person_coords_ = true;
      }
    }
  }

  static std::vector<DetectedItem> parseVisionDetections(const std::string & data)
  {
    std::vector<DetectedItem> list;
    std::stringstream ss(data);
    std::string chunk;
    while (std::getline(ss, chunk, ';')) {
      if (chunk.empty()) continue;
      const auto colon = chunk.find(':');
      const auto bar = chunk.find('|');
      if (colon == std::string::npos || bar == std::string::npos || bar <= colon) continue;

      DetectedItem item;
      item.name = chunk.substr(0, colon);

      const auto second_bar = chunk.find('|', bar + 1);
      const std::string coord_str = (second_bar != std::string::npos)
        ? chunk.substr(bar + 1, second_bar - bar - 1)
        : chunk.substr(bar + 1);
      std::stringstream css(coord_str);
      std::string c_val;
      std::vector<double> coords;
      while (std::getline(css, c_val, ',')) {
        try { coords.push_back(std::stod(c_val)); } catch (...) {}
      }
      if (coords.size() >= 2 && std::isfinite(coords[0]) && std::isfinite(coords[1])) {
        item.x = coords[0];
        item.y = coords[1];
        item.z = (coords.size() >= 3 && std::isfinite(coords[2])) ? coords[2] : 0.0;
        item.has_3d = true;
      }
      list.push_back(item);
    }
    return list;
  }

  void onDrinkDetections(const std_msgs::msg::String::SharedPtr message)
  {
    if (!message || message->data.empty() || message->data.find("NONE") != std::string::npos) {
      return;
    }
    auto parsed = parseVisionDetections(message->data);
    if (!parsed.empty()) {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      detected_items_ = std::move(parsed);
    }
  }

  void onBagDetections(const std_msgs::msg::String::SharedPtr message)
  {
    if (!message || message->data.empty()) return;
    auto parsed = parseVisionDetections(message->data);
    if (!parsed.empty()) {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      if (detected_items_.empty()) {
        detected_items_ = std::move(parsed);
      }
    }
  }

  // Mencari posisi 3D dari {order_1} yang terdeteksi kamera
  bool findOrder1Position(const std::string & target_order_name, double & out_x, double & out_y)
  {
    std::lock_guard<std::mutex> lock(vision_mutex_);
    if (detected_items_.empty()) {
      return false;
    }

    const std::string target_norm = normalize(target_order_name);
    for (const auto & item : detected_items_) {
      if (!item.has_3d) continue;
      const std::string item_norm = normalize(item.name);
      if (!item_norm.empty() &&
          (item_norm == target_norm ||
           item_norm.find(target_norm) != std::string::npos ||
           target_norm.find(item_norm) != std::string::npos))
      {
        out_x = item.x;
        out_y = item.y;
        return true;
      }
    }

    // Fallback: ambil objek terdekat yang memiliki koordinat 3D valid
    for (const auto & item : detected_items_) {
      if (item.has_3d) {
        out_x = item.x;
        out_y = item.y;
        return true;
      }
    }
    return false;
  }

  // ============================================================================
  // HELPER SERVICE CLIENT NON-BLOCKING
  // ============================================================================
  void callTrigger(
    const TriggerClient::SharedPtr & client,
    const std::string & service_name,
    TriggerResult result,
    double timeout_sec = 15.0)
  {
    if (!client) {
      if (result) result(false, "Client is null");
      return;
    }

    if (!client->service_is_ready()) {
      if (!client->wait_for_service(1s)) {
        RCLCPP_WARN(
          get_logger(), "[SERVICE] Service %s belum tersedia.",
          service_name.c_str());
        if (result) result(false, "Service not ready");
        return;
      }
    }

    {
      std::lock_guard<std::mutex> lock(srv_mutex_);
      service_busy_ = true;
      srv_started_time_ = now();
      srv_timeout_sec_ = timeout_sec;
      active_srv_name_ = service_name;
      pending_srv_result_ = std::move(result);
    }

    auto request = std::make_shared<Trigger::Request>();
    client->async_send_request(
      request,
      [this, service_name](TriggerClient::SharedFuture future) {
        TriggerResult cb;
        {
          std::lock_guard<std::mutex> lock(srv_mutex_);
          if (!service_busy_.load() || active_srv_name_ != service_name) {
            return;
          }
          in_callback_ = true;
          service_busy_ = false;
          cb = std::move(pending_srv_result_);
          pending_srv_result_ = nullptr;
        }

        bool ok = false;
        std::string msg;
        try {
          const auto response = future.get();
          ok = response->success;
          msg = response->message;
        } catch (const std::exception & e) {
          ok = false;
          msg = e.what();
        }

        if (cb) {
          cb(ok, msg);
        }
        in_callback_ = false;
      });
  }

  // Memanggil service release gripper lalu mematikan torsi arm (/arm/free)
  void releaseGripperAndTorqueOff(std::function<void()> on_done)
  {
    TriggerClient::SharedPtr release_client = arm_release_client_;
    std::string release_srv_name = "/arm/release_gripper";
    if (!release_client || !release_client->service_is_ready()) {
      release_client = arm_letakkan_client_;
      release_srv_name = "/arm/letakkan_tas";
    }

    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[ARM]\033[0m Releasing gripper via %s...",
      release_srv_name.c_str());

    callTrigger(
      release_client, release_srv_name,
      [this, on_done](bool /*rel_ok*/, const std::string & rel_msg) {
        RCLCPP_INFO(get_logger(), "[ARM] Release gripper result: %s", rel_msg.c_str());

        TriggerClient::SharedPtr free_client = arm_free_client_;
        std::string free_srv_name = "/arm/free";
        if ((!free_client || !free_client->service_is_ready()) &&
            arm_relax_client_ && arm_relax_client_->service_is_ready())
        {
          free_client = arm_relax_client_;
          free_srv_name = "/arm/relax";
        }

        RCLCPP_INFO(
          get_logger(),
          "\033[1;36m[ARM]\033[0m Turning off arm torque via %s...",
          free_srv_name.c_str());

        callTrigger(
          free_client, free_srv_name,
          [this, on_done](bool /*free_ok*/, const std::string & free_msg) {
            RCLCPP_INFO(get_logger(), "[ARM] Torque off result: %s", free_msg.c_str());
            if (on_done) {
              on_done();
            }
          },
          10.0);
      },
      20.0);
  }

  // ============================================================================
  // MAIN FINITE STATE MACHINE (FSM TICK)
  // ============================================================================
  void fsmTick()
  {
    const RestaurantState current_state = state_.load();

    // 1. Global timeout watchdog
    if ((now() - task_start_time_).seconds() > GLOBAL_TIMEOUT_SEC &&
        current_state != RestaurantState::COMPLETED &&
        current_state != RestaurantState::ABORTED)
    {
      RCLCPP_ERROR(get_logger(), "Global restaurant task timeout reached.");
      state_ = RestaurantState::ABORTED;
      return;
    }

    // 2. Service timeout watchdog
    TriggerResult timed_out_srv_cb;
    std::string timed_out_srv_name;
    {
      std::lock_guard<std::mutex> lock(srv_mutex_);
      if (service_busy_.load() &&
          (now() - srv_started_time_).seconds() > srv_timeout_sec_)
      {
        service_busy_ = false;
        timed_out_srv_name = active_srv_name_;
        timed_out_srv_cb = std::move(pending_srv_result_);
        pending_srv_result_ = nullptr;
      }
    }
    if (timed_out_srv_cb) {
      RCLCPP_WARN(
        get_logger(),
        "Service %s timeout (> %.1f s), releasing lock.",
        timed_out_srv_name.c_str(), srv_timeout_sec_);
      timed_out_srv_cb(false, "Service response timeout");
      return;
    }

    // 3. TTS timeout watchdog
    bool tts_timed_out = false;
    double tts_timeout = 0.0;
    {
      std::lock_guard<std::mutex> lock(tts_mutex_);
      tts_timeout = tts_timeout_sec_;
      tts_timed_out = speaking_.load() &&
                      (now() - tts_started_time_).seconds() > tts_timeout_sec_;
    }
    if (tts_timed_out) {
      RCLCPP_WARN(
        get_logger(),
        "TTS timeout (> %.1f s, no /hri/tts_done received), releasing speaking lock.",
        tts_timeout);
      onTtsDone(nullptr);
      return;
    }

    if (isBusy()) {
      return;
    }

    switch (current_state) {
      // ========================================================================
      // POINT 1: Initial Initialization (origin_position, STT, and TTS engine)
      // ========================================================================
      case RestaurantState::INIT: {
        if (tts_pub_->get_subscription_count() == 0 &&
            (now() - state_enter_time_).seconds() < 2.0)
        {
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "[INIT] Menunggu STT & TTS engine siap...");
          return;
        }

        if (!origin_initialized_) {
          double rx = 0.0, ry = 0.0, ryaw = 0.0;
          if (capture_tf_origin_on_init_ && getRobotPose(rx, ry, ryaw)) {
            origin_x_ = rx;
            origin_y_ = ry;
            origin_yaw_ = ryaw;
            RCLCPP_INFO(
              get_logger(),
              "[INIT] Origin position captured from TF: (x=%.2f, y=%.2f, yaw=%.2f)",
              origin_x_, origin_y_, origin_yaw_);
          } else {
            RCLCPP_INFO(
              get_logger(),
              "[INIT] Using configured origin_position: (x=%.2f, y=%.2f, yaw=%.2f)",
              origin_x_, origin_y_, origin_yaw_);
          }
          origin_initialized_ = true;
        }

        served_customers_.clear();
        initCurrentCustomer(1);
        publishStatus("Initialized origin_position, STT and TTS engine. Starting Customer 1.");
        transitionTo(RestaurantState::ANNOUNCE_SERVER, "Initialization complete");
        break;
      }

      // ========================================================================
      // POINT 2: Announce Server & Detect Person Who Waves Hand -> zed/wave_hand
      // ========================================================================
      case RestaurantState::ANNOUNCE_SERVER:
        speakAsync(
          "Hello everyone, i'm barelang inari a server in this bar, if you would like to place an order, please wave your hand",
          [this]() {
            speakAsync(
              "i will detect who will order",
              [this]() {
                transitionTo(
                  RestaurantState::DETECT_WAVE_HAND,
                  "Waiting to detect waving customer");
              });
          });
        break;

      case RestaurantState::DETECT_WAVE_HAND: {
        bool waved = false;
        {
          std::lock_guard<std::mutex> lock(vision_mutex_);
          waved = wave_detected_;
        }

        if (waved) {
          speakAsync("i'm coming", [this]() {
            callTrigger(
              vision_wave_hand_client_, "/vision/wave_hand",
              [this](bool ok, const std::string & msg) {
                double px = 0.0, py = 0.0;
                if (parseCoordinates(msg, px, py)) {
                  std::lock_guard<std::mutex> lock(vision_mutex_);
                  waving_person_x_ = px;
                  waving_person_y_ = py;
                  has_waving_person_coords_ = true;
                }
                RCLCPP_INFO(
                  get_logger(),
                  "[VISION] Service /vision/wave_hand finished (ok=%s, msg='%s')",
                  ok ? "true" : "false", msg.c_str());
                transitionTo(
                  RestaurantState::NAV_TO_WAVING_CUSTOMER,
                  "Wave hand processed, navigating to customer");
              },
              10.0);
          });
        } else if (vision_wave_hand_client_ && vision_wave_hand_client_->service_is_ready() &&
                   (now() - state_enter_time_).seconds() >= 2.0)
        {
          // Jika deteksi wave disediakan langsung oleh service /vision/wave_hand
          callTrigger(
            vision_wave_hand_client_, "/vision/wave_hand",
            [this](bool ok, const std::string & msg) {
              if (ok) {
                double px = 0.0, py = 0.0;
                if (parseCoordinates(msg, px, py)) {
                  std::lock_guard<std::mutex> lock(vision_mutex_);
                  waving_person_x_ = px;
                  waving_person_y_ = py;
                  has_waving_person_coords_ = true;
                }
                speakAsync("i'm coming", [this]() {
                  transitionTo(
                    RestaurantState::NAV_TO_WAVING_CUSTOMER,
                    "Customer wave detected via /vision/wave_hand");
                });
              }
            },
            10.0);
        } else if ((now() - last_wave_prompt_time_).seconds() >= WAVE_REPROMPT_SEC) {
          last_wave_prompt_time_ = now();
          speakAsync("if you would like to place an order, please wave your hand");
        }
        break;
      }

      // ========================================================================
      // POINT 3: Navigate to Person That Waved Hand
      // ========================================================================
      case RestaurantState::NAV_TO_WAVING_CUSTOMER:
        if (last_nav_fail_time_.nanoseconds() > 0 &&
            (now() - last_nav_fail_time_).seconds() < NAV_RETRY_COOLDOWN_SEC)
        {
          return;
        }
        navigateToWavingCustomer([this](bool success) {
          if (!success) {
            RCLCPP_WARN(get_logger(), "[NAV] Gagal navigasi ke pelanggan, mencoba ulang...");
            return;
          }
          transitionTo(
            RestaurantState::SAVE_CUSTOMER_POSE_ASK_NAME,
            "Arrived at waving customer location");
        });
        break;

      // ========================================================================
      // POINT 4: Save Coordinate Location for Customer & Ask Name
      // ========================================================================
      case RestaurantState::SAVE_CUSTOMER_POSE_ASK_NAME: {
        double cx = 0.0, cy = 0.0, cyaw = 0.0;
        if (!getRobotPose(cx, cy, cyaw)) {
          std::lock_guard<std::mutex> v_lock(vision_mutex_);
          cx = last_nav_goal_x_;
          cy = last_nav_goal_y_;
          cyaw = last_nav_goal_yaw_;
        }
        {
          std::lock_guard<std::mutex> c_lock(customer_mutex_);
          current_customer_.map_x = cx;
          current_customer_.map_y = cy;
          current_customer_.map_yaw = cyaw;
          current_customer_.pose_saved = true;
        }
        RCLCPP_INFO(
          get_logger(),
          "\033[1;32m[CUSTOMER %d POSE SAVED]\033[0m x=%.2f, y=%.2f, yaw=%.2f",
          customer_idx_, cx, cy, cyaw);

        speakAsync(
          "Hello, Welcome to the bar!! Can you please tell your name?",
          [this]() {
            transitionTo(
              RestaurantState::WAIT_CUSTOMER_NAME,
              "Waiting for customer name response");
          });
        break;
      }

      case RestaurantState::WAIT_CUSTOMER_NAME:
        if (has_new_utterance_.exchange(false)) {
          std::string raw;
          {
            std::lock_guard<std::mutex> lock(stt_mutex_);
            raw = heard_utterance_;
          }
          const std::string parsed_name = parseCustomerName(raw);
          if (!parsed_name.empty()) {
            {
              std::lock_guard<std::mutex> lock(customer_mutex_);
              current_customer_.name = parsed_name;
            }
            if (person_name_pub_) {
              std_msgs::msg::String msg;
              msg.data = parsed_name;
              person_name_pub_->publish(msg);
            }
            RCLCPP_INFO(
              get_logger(),
              "\033[1;32m[CUSTOMER %d NAME]\033[0m \"%s\" (from \"%s\")",
              customer_idx_, parsed_name.c_str(), raw.c_str());
            transitionTo(
              RestaurantState::CAPTURE_CUSTOMER_FACE,
              "Customer name captured");
          } else {
            state_enter_time_ = now();
            speakAsync("Hello, Welcome to the bar!! Can you please tell your name?");
          }
        } else if ((now() - state_enter_time_).seconds() > DIALOG_REPROMPT_SEC) {
          state_enter_time_ = now();
          speakAsync("Hello, Welcome to the bar!! Can you please tell your name?");
        }
        break;

      // ========================================================================
      // POINT 5: Save Facial Data -> Call Service zed/camera_capture_face
      // ========================================================================
      case RestaurantState::CAPTURE_CUSTOMER_FACE: {
        std::string c_name;
        {
          std::lock_guard<std::mutex> lock(customer_mutex_);
          c_name = current_customer_.name;
        }
        speakAsync(
          "Hello " + c_name + "! please dont move, as i want to save your facial data",
          [this]() {
            callTrigger(
              vision_capture_face_client_, "/vision/camera_capture_face",
              [this](bool ok, const std::string & msg) {
                RCLCPP_INFO(
                  get_logger(),
                  "[VISION] /vision/camera_capture_face finished (ok=%s, msg='%s')",
                  ok ? "true" : "false", msg.c_str());
                transitionTo(
                  RestaurantState::ASK_CUSTOMER_ORDER,
                  "Facial data capture completed");
              },
              10.0);
          });
        break;
      }

      // ========================================================================
      // POINT 6: Take Order ({order_1} & {order_2}), Wait 0.5s, Confirm Yes/No
      // ========================================================================
      case RestaurantState::ASK_CUSTOMER_ORDER:
        speakAsync("it's finished, May i take your order please?", [this]() {
          transitionTo(
            RestaurantState::WAIT_CUSTOMER_ORDER,
            "Waiting for customer order");
        });
        break;

      case RestaurantState::WAIT_CUSTOMER_ORDER:
        if (has_new_utterance_.exchange(false)) {
          std::string raw;
          {
            std::lock_guard<std::mutex> lock(stt_mutex_);
            raw = heard_utterance_;
          }
          std::string o1, o2;
          {
            std::lock_guard<std::mutex> lock(customer_mutex_);
            o1 = current_customer_.order_1;
            o2 = current_customer_.order_2;
          }

          if (parseCustomerOrders(raw, o1, o2)) {
            {
              std::lock_guard<std::mutex> lock(customer_mutex_);
              current_customer_.order_1 = o1;
              current_customer_.order_2 = o2;
            }
            RCLCPP_INFO(
              get_logger(),
              "\033[1;32m[ORDER CAPTURED]\033[0m order_1=\"%s\", order_2=\"%s\"",
              o1.c_str(), o2.c_str());
            transitionTo(
              RestaurantState::PRE_CONFIRM_ORDER,
              "Both orders captured");
          } else if (!o1.empty() && o2.empty()) {
            {
              std::lock_guard<std::mutex> lock(customer_mutex_);
              current_customer_.order_1 = o1;
            }
            state_enter_time_ = now();
            speakAsync("What is your second order please?");
          } else {
            state_enter_time_ = now();
            speakAsync("May i take your order please?");
          }
        } else if ((now() - state_enter_time_).seconds() > DIALOG_REPROMPT_SEC) {
          state_enter_time_ = now();
          speakAsync("May i take your order please?");
        }
        break;

      case RestaurantState::PRE_CONFIRM_ORDER:
        speakAsync(
          "i will confirm your order, please answer with yes if the order has correct and answer with no, if the order is incorrect",
          [this]() {
            transitionTo(
              RestaurantState::WAIT_BEFORE_CONFIRM_SPEECH,
              "Waiting 0.5 seconds before stating order confirmation");
          });
        break;

      case RestaurantState::WAIT_BEFORE_CONFIRM_SPEECH:
        if ((now() - state_enter_time_).seconds() >= 0.5) {
          transitionTo(
            RestaurantState::SPEAK_ORDER_CONFIRMATION,
            "0.5s elapsed, speaking order confirmation");
        }
        break;

      case RestaurantState::SPEAK_ORDER_CONFIRMATION: {
        std::string o1, o2;
        {
          std::lock_guard<std::mutex> lock(customer_mutex_);
          o1 = current_customer_.order_1;
          o2 = current_customer_.order_2;
        }
        speakAsync(
          "your order is " + o1 + " and " + o2 + " is it correct?",
          [this]() {
            transitionTo(
              RestaurantState::WAIT_ORDER_CONFIRMATION,
              "Waiting for customer yes/no confirmation");
          });
        break;
      }

      case RestaurantState::WAIT_ORDER_CONFIRMATION:
        if (has_new_utterance_.exchange(false)) {
          std::string raw;
          {
            std::lock_guard<std::mutex> lock(stt_mutex_);
            raw = heard_utterance_;
          }
          const int ans = parseYesNoIntent(raw);
          if (ans == 1) {
            speakAsync(
              "alright thank you, i will come back to pick up the order you requested",
              [this]() {
                transitionTo(
                  RestaurantState::NAV_TO_ORIGIN_BARMAN,
                  "Order confirmed (YES), navigating to barman at origin");
              });
          } else if (ans == -1) {
            {
              std::lock_guard<std::mutex> lock(customer_mutex_);
              current_customer_.order_1.clear();
              current_customer_.order_2.clear();
            }
            speakAsync(
              "apologies for my mistake, may i ask about your order again?",
              [this]() {
                transitionTo(
                  RestaurantState::WAIT_CUSTOMER_ORDER,
                  "Order rejected (NO), returning to point 6");
              });
          } else {
            state_enter_time_ = now();
            speakAsync("Please answer with yes if the order has correct and answer with no, if the order is incorrect");
          }
        } else if ((now() - state_enter_time_).seconds() > DIALOG_REPROMPT_SEC) {
          state_enter_time_ = now();
          std::string o1, o2;
          {
            std::lock_guard<std::mutex> lock(customer_mutex_);
            o1 = current_customer_.order_1;
            o2 = current_customer_.order_2;
          }
          speakAsync("your order is " + o1 + " and " + o2 + " is it correct?");
        }
        break;

      // ========================================================================
      // POINT 7: Navigate to Origin (Barman Place)
      // ========================================================================
      case RestaurantState::NAV_TO_ORIGIN_BARMAN:
        if (last_nav_fail_time_.nanoseconds() > 0 &&
            (now() - last_nav_fail_time_).seconds() < NAV_RETRY_COOLDOWN_SEC)
        {
          return;
        }
        navigateToOrigin([this](bool success) {
          if (!success) {
            RCLCPP_WARN(get_logger(), "[NAV] Gagal kembali ke origin (barman), mencoba ulang...");
            return;
          }
          transitionTo(
            RestaurantState::ANNOUNCE_ORDER_TO_BARMAN,
            "Arrived at origin (barman place)");
        });
        break;

      // ========================================================================
      // POINT 8: Barman Interaction, Detect & Navigate to order_1, Grasp, Tray
      // ========================================================================
      case RestaurantState::ANNOUNCE_ORDER_TO_BARMAN: {
        std::string o1, o2;
        int c_idx = 1;
        {
          std::lock_guard<std::mutex> lock(customer_mutex_);
          o1 = current_customer_.order_1;
          o2 = current_customer_.order_2;
          c_idx = customer_idx_;
        }
        const std::string customer_ordinal = (c_idx == 1) ? "first" : "second";
        speakAsync(
          "The " + customer_ordinal + " customer order " + o1 + " and " + o2,
          [this, o1]() {
            speakAsync(
              "please place " + o1 + " in front of my gripper",
              [this]() {
                transitionTo(
                  RestaurantState::DETECT_AND_NAV_TO_ORDER1,
                  "Waiting to detect order_1 position");
              });
          });
        break;
      }

      case RestaurantState::DETECT_AND_NAV_TO_ORDER1: {
        std::string o1;
        {
          std::lock_guard<std::mutex> lock(customer_mutex_);
          o1 = current_customer_.order_1;
        }

        double ox = 0.0, oy = 0.0;
        if (findOrder1Position(o1, ox, oy)) {
          RCLCPP_INFO(
            get_logger(),
            "\033[1;32m[ORDER_1 DETECTED]\033[0m '%s' at (x=%.2f, y=%.2f). Navigating to position...",
            o1.c_str(), ox, oy);
          navigateToRelativeOrMapTargetAsync(
            ox, oy, object_standoff_dist_,
            [this](bool /*nav_ok*/) {
              transitionTo(
                RestaurantState::GRASP_ORDER1,
                "Reached order_1 position, calling /arm/ambil_tas_simple");
            });
        } else if ((now() - state_enter_time_).seconds() >= order1_detect_timeout_sec_) {
          RCLCPP_WARN(
            get_logger(),
            "[ORDER_1] Timeout %.1fs menunggu deteksi 3D '%s', langsung memanggil /arm/ambil_tas_simple di posisi saat ini.",
            order1_detect_timeout_sec_, o1.c_str());
          transitionTo(
            RestaurantState::GRASP_ORDER1,
            "Proceeding to grasp order_1 in front of gripper");
        }
        break;
      }

      case RestaurantState::GRASP_ORDER1:
        callTrigger(
          arm_grab_simple_client_, "/arm/ambil_tas_simple",
          [this](bool ok, const std::string & msg) {
            RCLCPP_INFO(
              get_logger(),
              "[ARM] /arm/ambil_tas_simple finished (ok=%s, msg='%s')",
              ok ? "true" : "false", msg.c_str());

            std::string o2;
            {
              std::lock_guard<std::mutex> lock(customer_mutex_);
              o2 = current_customer_.order_2;
            }
            speakAsync(
              "I've got it, for " + o2 + " please, place it on the tray in front of me",
              [this]() {
                transitionTo(
                  RestaurantState::WAIT_TRAY_SILENT_5S,
                  "Remaining silent for 5 seconds while order_2 is placed on tray");
              });
          },
          25.0);
        break;

      case RestaurantState::WAIT_TRAY_SILENT_5S:
        if ((now() - state_enter_time_).seconds() >= 5.0) {
          transitionTo(
            RestaurantState::ASK_TRAY_CONFIRMATION,
            "5 seconds silent wait elapsed");
        }
        break;

      case RestaurantState::ASK_TRAY_CONFIRMATION:
        speakAsync(
          "is the order had take in the tray, please answer with yes robot or no robot",
          [this]() {
            transitionTo(
              RestaurantState::WAIT_TRAY_CONFIRMATION,
              "Waiting for yes robot / no robot response");
          });
        break;

      case RestaurantState::WAIT_TRAY_CONFIRMATION:
        if (has_new_utterance_.exchange(false)) {
          std::string raw;
          {
            std::lock_guard<std::mutex> lock(stt_mutex_);
            raw = heard_utterance_;
          }
          const int ans = parseYesNoIntent(raw);
          if (ans == 1) {
            transitionTo(
              RestaurantState::NAV_DELIVER_TO_CUSTOMER,
              "Tray order confirmed (yes robot), proceeding to point 9");
          } else if (ans == -1) {
            transitionTo(
              RestaurantState::WAIT_TRAY_RETRY_3S,
              "Tray order not ready (no robot), waiting 3 seconds before repeating callback");
          } else {
            state_enter_time_ = now();
            speakAsync("is the order had take in the tray, please answer with yes robot or no robot");
          }
        } else if ((now() - state_enter_time_).seconds() > DIALOG_REPROMPT_SEC) {
          state_enter_time_ = now();
          speakAsync("is the order had take in the tray, please answer with yes robot or no robot");
        }
        break;

      case RestaurantState::WAIT_TRAY_RETRY_3S:
        if ((now() - state_enter_time_).seconds() >= 3.0) {
          transitionTo(
            RestaurantState::ASK_TRAY_CONFIRMATION,
            "3 seconds wait elapsed, repeating tray confirmation callback");
        }
        break;

      // ========================================================================
      // POINT 9: Navigate to Customer, Deliver Orders, Return to Origin
      // ========================================================================
      case RestaurantState::NAV_DELIVER_TO_CUSTOMER: {
        if (last_nav_fail_time_.nanoseconds() > 0 &&
            (now() - last_nav_fail_time_).seconds() < NAV_RETRY_COOLDOWN_SEC)
        {
          return;
        }
        double cx = 0.0, cy = 0.0, cyaw = 0.0;
        {
          std::lock_guard<std::mutex> lock(customer_mutex_);
          cx = current_customer_.map_x;
          cy = current_customer_.map_y;
          cyaw = current_customer_.map_yaw;
        }
        navigateToMapPoseAsync(cx, cy, cyaw, [this](bool success) {
          if (!success) {
            RCLCPP_WARN(get_logger(), "[NAV] Gagal navigasi ke posisi pelanggan, mencoba ulang...");
            return;
          }
          transitionTo(
            RestaurantState::GREET_DELIVERY_CUSTOMER,
            "Arrived at customer position for delivery");
        });
        break;
      }

      case RestaurantState::GREET_DELIVERY_CUSTOMER: {
        std::string c_name;
        {
          std::lock_guard<std::mutex> lock(customer_mutex_);
          c_name = current_customer_.name;
        }
        speakAsync(
          "Hello " + c_name + ", i have brought the order you requested",
          [this]() {
            transitionTo(
              RestaurantState::WAIT_1S_BEFORE_RELEASE_PROMPT,
              "Waiting 1 second before release prompt");
          });
        break;
      }

      case RestaurantState::WAIT_1S_BEFORE_RELEASE_PROMPT:
        if ((now() - state_enter_time_).seconds() >= 1.0) {
          transitionTo(
            RestaurantState::PROMPT_RELEASE_ORDER1,
            "1 second elapsed");
        }
        break;

      case RestaurantState::PROMPT_RELEASE_ORDER1: {
        std::string o1;
        {
          std::lock_guard<std::mutex> lock(customer_mutex_);
          o1 = current_customer_.order_1;
        }
        speakAsync(
          "for " + o1 + ", please hold out your hand beneath this order, i'm going to let go on one .. two .. three",
          [this]() {
            transitionTo(
              RestaurantState::RELEASE_GRIPPER_AND_TORQUE_OFF,
              "Releasing gripper and turning arm torque off");
          });
        break;
      }

      case RestaurantState::RELEASE_GRIPPER_AND_TORQUE_OFF:
        releaseGripperAndTorqueOff([this]() {
          transitionTo(
            RestaurantState::PROMPT_TAKE_ORDER2_TRAY,
            "Gripper released and arm torque disabled");
        });
        break;

      case RestaurantState::PROMPT_TAKE_ORDER2_TRAY: {
        std::string o2;
        {
          std::lock_guard<std::mutex> lock(customer_mutex_);
          o2 = current_customer_.order_2;
        }
        speakAsync(
          "for " + o2 + " please take it from the tray",
          [this]() {
            transitionTo(
              RestaurantState::WAIT_5S_CUSTOMER_TAKE_TRAY,
              "Waiting 5 seconds for customer to take order_2 from tray");
          });
        break;
      }

      case RestaurantState::WAIT_5S_CUSTOMER_TAKE_TRAY:
        if ((now() - state_enter_time_).seconds() >= 5.0) {
          speakAsync("thank you for your order", [this]() {
            transitionTo(
              RestaurantState::NAV_RETURN_ORIGIN_AFTER_SERVE,
              "Returning to origin_position after serving customer");
          });
        }
        break;

      // ========================================================================
      // POINT 9 (end), 10 & 11: Return to Origin -> Loop Customer 2 or Finish
      // ========================================================================
      case RestaurantState::NAV_RETURN_ORIGIN_AFTER_SERVE:
        if (last_nav_fail_time_.nanoseconds() > 0 &&
            (now() - last_nav_fail_time_).seconds() < NAV_RETRY_COOLDOWN_SEC)
        {
          return;
        }
        navigateToOrigin([this](bool success) {
          if (!success) {
            RCLCPP_WARN(get_logger(), "[NAV] Gagal kembali ke origin_position, mencoba ulang...");
            return;
          }

          int finished_idx = 1;
          {
            std::lock_guard<std::mutex> lock(customer_mutex_);
            served_customers_.push_back(current_customer_);
            finished_idx = customer_idx_;
          }

          if (finished_idx < max_customers_) {
            // Point 10: Looping back to Point 2 for second customer
            initCurrentCustomer(finished_idx + 1);
            transitionTo(
              RestaurantState::ANNOUNCE_SERVER,
              "Looping back to point 2 for second customer");
          } else {
            // Point 11: Point 9 finished for second customer and returned to origin
            transitionTo(
              RestaurantState::FINISH_TASK,
              "All customers served and returned to origin");
          }
        });
        break;

      case RestaurantState::FINISH_TASK:
        speakAsync("completed restaurant task", [this]() {
          publishStatus("completed restaurant task");
          transitionTo(RestaurantState::COMPLETED, "Restaurant task completed");
        });
        break;

      case RestaurantState::COMPLETED:
      case RestaurantState::ABORTED:
        break;
    }
  }
};

}  // namespace restaurant

// ==============================================================================
// 4. ENTRY POINT UTAMA
// ==============================================================================
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<restaurant::RestaurantTaskNode>();

  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
