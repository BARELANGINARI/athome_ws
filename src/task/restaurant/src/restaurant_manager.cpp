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
#include <iomanip>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
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
  FIND_AND_ALIGN_BARMAN,          // 7b. Cari & arahkan kamera ke barman (< 1.5 m, tengah frame)
  ANNOUNCE_ORDER_TO_BARMAN,       // 8a. "The first/second customer order..." -> "please place {order_1}..."
  DETECT_AND_NAV_TO_ORDER1,       // 8b. Deteksi posisi {order_1} & navigasi ke posisi tersebut
  GRASP_ORDER1,                   // 8c. Panggil srv /arm/ambil_order -> instruksi taruh {order_2} di tray
  WAIT_TRAY_SILENT_5S,            // 8d. Diam selama 5 detik menunggu {order_2} ditaruh di tray
  ASK_TRAY_CONFIRMATION,          // 8e. "is the order had take in the tray, please answer with yes robot or no robot"
  WAIT_TRAY_CONFIRMATION,         // 8f. Menunggu jawaban yes robot / no robot
  WAIT_TRAY_RETRY_3S,             // 8g. Jika no, tunggu 3 detik lagi lalu ulangi callback pertanyaan
  NAV_DELIVER_TO_CUSTOMER,        // 9a. Navigasi kembali ke koordinat pelanggan
  GREET_DELIVERY_CUSTOMER,        // 9b. "Hello {name}, i have brought the order you requested"
  WAIT_1S_BEFORE_RELEASE_PROMPT,  // 9c. Tunggu 1 detik
  PROMPT_RELEASE_ORDER1,          // 9d. Jalankan MOTION 1 (/arm/raising_arm) lalu prompt "for {order_1}..."
  RELEASE_GRIPPER_AND_TORQUE_OFF, // 9e. Jalankan MOTION 2 (/arm/release_gripper) lalu relax arm (/arm/relax)
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
    case RestaurantState::FIND_AND_ALIGN_BARMAN:          return "FIND_AND_ALIGN_BARMAN";
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
  std::vector<std::string> orders;
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
    // Baca Seluruh Parameter Konfigurasi (Wajib dari restaurant_params.yaml)
    // --------------------------------------------------------------------------
    origin_x_ = requireYamlParameter<double>("origin_x");
    origin_y_ = requireYamlParameter<double>("origin_y");
    origin_yaw_ = requireYamlParameter<double>("origin_yaw");
    capture_tf_origin_on_init_ = requireYamlParameter<bool>("capture_tf_origin_on_init");
    global_frame_ = requireYamlParameter<std::string>("global_frame");
    base_frame_ = requireYamlParameter<std::string>("base_frame");
    order_frame_ = requireYamlParameter<std::string>("order_frame");
    customer_standoff_dist_ = requireYamlParameter<double>("customer_standoff_distance");
    object_standoff_dist_ = requireYamlParameter<double>("object_standoff_distance");
    adaptive_standoff_enabled_ = requireYamlParameter<bool>("adaptive_standoff_enabled");
    adaptive_standoff_min_ = requireYamlParameter<double>("adaptive_standoff_min");
    adaptive_standoff_max_ = requireYamlParameter<double>("adaptive_standoff_max");
    adaptive_standoff_step_ = requireYamlParameter<double>("adaptive_standoff_step");
    costmap_obstacle_threshold_ = requireYamlParameter<int>("costmap_obstacle_threshold");
    costmap_topic_ = requireYamlParameter<std::string>("costmap_topic");
    robot_radius_ = requireYamlParameter<double>("robot_radius");
    nav_timeout_sec_ = requireYamlParameter<double>("nav_timeout_sec");
    order1_detect_timeout_sec_ = requireYamlParameter<double>("order1_detect_timeout_sec");
    max_customers_ = std::max(1, requireYamlParameter<int>("max_customers"));

    global_timeout_sec_ = requireYamlParameter<double>("global_timeout_sec");
    dialog_reprompt_sec_ = requireYamlParameter<double>("dialog_reprompt_sec");
    wave_reprompt_sec_ = requireYamlParameter<double>("wave_reprompt_sec");
    tts_min_timeout_sec_ = requireYamlParameter<double>("tts_min_timeout_sec");
    tts_sec_per_word_ = requireYamlParameter<double>("tts_sec_per_word");
    tts_timeout_margin_sec_ = requireYamlParameter<double>("tts_timeout_margin_sec");
    nav_retry_cooldown_sec_ = requireYamlParameter<double>("nav_retry_cooldown_sec");
    tray_wait_sec_ = std::max(0.0, requireYamlParameter<double>("tray_wait_sec"));
    tts_post_delay_sec_ = std::max(0.0, requireYamlParameter<double>("tts_post_delay_sec"));
    tts_timeout_sec_ = tts_min_timeout_sec_;

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
    hmi_customer_pub_ = create_publisher<std_msgs::msg::String>("/hmi/customer_display", 10);
    face_expression_pub_ = create_publisher<std_msgs::msg::String>("/robot/expression", 10);
    arm_target_pub_ = create_publisher<std_msgs::msg::String>("/arm/target_point", 10);
    object_pose_pub_ = create_publisher<std_msgs::msg::String>("/restaurant/detected_object_pose", 10);

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

    // Subscription /global_costmap/costmap untuk adaptive standoff
    costmap_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      costmap_topic_,
      rclcpp::QoS(1).reliable(),
      [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(costmap_mutex_);
        latest_costmap_ = msg;
      },
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
    search_client_ = create_client<Trigger>(
      "/search", rclcpp::ServicesQoS(), callback_group_);
    go_to_order_client_ = create_client<Trigger>(
      "/go_to_order", rclcpp::ServicesQoS(), callback_group_);
    go_to_order1_client_ = create_client<Trigger>(
      "/go_to_order1", rclcpp::ServicesQoS(), callback_group_);
    deliver_order_client_ = create_client<Trigger>(
      "/deliver_order", rclcpp::ServicesQoS(), callback_group_);
    homing_client_ = create_client<Trigger>(
      "/homing", rclcpp::ServicesQoS(), callback_group_);
    find_barman_client_ = create_client<Trigger>(
      "/find_barman", rclcpp::ServicesQoS(), callback_group_);

    // Arm Services
    arm_grab_simple_client_ = create_client<Trigger>(
      "/arm/ambil_order", rclcpp::ServicesQoS(), callback_group_);
    arm_raising_arm_client_ = create_client<Trigger>(
      "/arm/raising_arm", rclcpp::ServicesQoS(), callback_group_);
    arm_release_gripper_client_ = create_client<Trigger>(
      "/arm/release_gripper", rclcpp::ServicesQoS(), callback_group_);
    arm_free_client_ = create_client<Trigger>(
      "/arm/free", rclcpp::ServicesQoS(), callback_group_);
    arm_relax_client_ = create_client<Trigger>(
      "/arm/relax", rclcpp::ServicesQoS(), callback_group_);

    // Timer siklus FSM (10 Hz / 100 ms) menggunakan default MutuallyExclusive group
    fsm_timer_ = create_wall_timer(
      100ms, std::bind(&RestaurantTaskNode::fsmTick, this));

    task_start_time_ = now();
    state_enter_time_ = now();
    initCurrentCustomer(1);
    publishExpression(expressionForState(RestaurantState::INIT));

    RCLCPP_INFO(
      get_logger(),
      "\033[1;32m[RESTAURANT]\033[0m Task Manager started (Origin: %.2f, %.2f, %.2f)",
      origin_x_, origin_y_, origin_yaw_);
  }

private:
  template<typename T>
  T requireYamlParameter(const std::string & name)
  {
    try {
      rcl_interfaces::msg::ParameterDescriptor desc;
      desc.read_only = true;
      return declare_parameter<T>(name, desc);
    } catch (const std::exception & e) {
      RCLCPP_FATAL(
        get_logger(),
        "Parameter '%s' wajib diatur di file YAML (restaurant_params.yaml)! Error: %s",
        name.c_str(), e.what());
      throw;
    }
  }

  // ----------------------------------------------------------------------------
  // Parameter Waktu & Timeout (Diisi penuh dari restaurant_params.yaml)
  // ----------------------------------------------------------------------------
  double global_timeout_sec_{0.0};
  double dialog_reprompt_sec_{0.0};
  double wave_reprompt_sec_{0.0};
  double tts_min_timeout_sec_{0.0};
  double tts_sec_per_word_{0.0};
  double tts_timeout_margin_sec_{0.0};
  double nav_retry_cooldown_sec_{0.0};
  double tray_wait_sec_{0.0};
  double tts_post_delay_sec_{0.0};

  // ----------------------------------------------------------------------------
  // Parameter & State Variables (Diisi penuh dari restaurant_params.yaml)
  // ----------------------------------------------------------------------------
  double origin_x_{0.0};
  double origin_y_{0.0};
  double origin_yaw_{0.0};
  bool capture_tf_origin_on_init_{false};
  bool origin_initialized_{false};
  std::string global_frame_;
  std::string base_frame_;
  std::string order_frame_;
  double customer_standoff_dist_{0.0};
  double object_standoff_dist_{0.0};
  double nav_timeout_sec_{0.0};
  double order1_detect_timeout_sec_{0.0};
  int max_customers_{0};

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
  bool barman_briefed_{false};

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
  bool has_last_order1_3d_{false};
  double last_order1_x_{0.0};
  double last_order1_y_{0.0};
  double last_order1_z_{0.0};

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
  rclcpp::Time last_search_time_{0, 0, RCL_ROS_TIME};
  bool search_attempted_in_state_{false};
  rclcpp::Time tts_cooldown_until_{0, 0, RCL_ROS_TIME};
  rclcpp::Time tts_started_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_nav_fail_time_{0, 0, RCL_ROS_TIME};

  mutable std::mutex tts_mutex_;
  double tts_timeout_sec_{4.0};
  std::string active_tts_text_;
  std::string last_tts_text_norm_;
  rclcpp::Time last_tts_done_time_{0, 0, RCL_ROS_TIME};
  std::function<void()> pending_tts_done_;
  std::atomic<bool> tts_post_delay_active_{false};
  rclcpp::Time tts_post_delay_until_{0, 0, RCL_ROS_TIME};
  std::function<void()> pending_post_tts_cb_;

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
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr hmi_customer_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr face_expression_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr arm_target_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr object_pose_pub_;

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr tts_done_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr stt_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr emergency_stt_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr gesture_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr person_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr order_pos_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr drink_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr bag_sub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sub_;

  // Adaptive Standoff Settings & State
  bool adaptive_standoff_enabled_{true};
  double adaptive_standoff_min_{0.85};
  double adaptive_standoff_max_{1.40};
  double adaptive_standoff_step_{0.05};
  int costmap_obstacle_threshold_{60};
  std::string costmap_topic_{"/global_costmap/costmap"};
  double robot_radius_{0.24};
  std::mutex costmap_mutex_;
  nav_msgs::msg::OccupancyGrid::SharedPtr latest_costmap_;

  rclcpp_action::Client<NavigateToPose>::SharedPtr nav_action_client_;
  TriggerClient::SharedPtr vision_wave_hand_client_;
  TriggerClient::SharedPtr vision_capture_face_client_;
  TriggerClient::SharedPtr search_client_;
  TriggerClient::SharedPtr go_to_order_client_;
  TriggerClient::SharedPtr go_to_order1_client_;
  TriggerClient::SharedPtr deliver_order_client_;
  TriggerClient::SharedPtr homing_client_;
  TriggerClient::SharedPtr find_barman_client_;
  TriggerClient::SharedPtr arm_grab_simple_client_;
  TriggerClient::SharedPtr arm_raising_arm_client_;
  TriggerClient::SharedPtr arm_release_gripper_client_;
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
    RCLCPP_INFO(get_logger(), "\033[1;34m[RESTAURANT]\033[0m\n  Status: %s", text.c_str());
  }

  static std::string formatOrderList(const std::vector<std::string> & items)
  {
    if (items.empty()) return "";
    if (items.size() == 1) return items[0];
    if (items.size() == 2) return items[0] + " and " + items[1];
    std::string out = items[0];
    for (size_t i = 1; i + 1 < items.size(); ++i) {
      out += ", " + items[i];
    }
    out += ", and " + items.back();
    return out;
  }

  static std::string formatTrayOrders(const std::vector<std::string> & items)
  {
    if (items.size() < 2) return "";
    std::vector<std::string> tray_items(items.begin() + 1, items.end());
    return formatOrderList(tray_items);
  }

  void publishCustomerHmi(bool show)
  {
    if (!hmi_customer_pub_) return;
    std_msgs::msg::String msg;
    if (show) {
      std::string name, o1, o2, o3, all_orders;
      int c_idx = 1;
      {
        std::lock_guard<std::mutex> lock(customer_mutex_);
        name = current_customer_.name;
        if (current_customer_.orders.size() >= 1) o1 = current_customer_.orders[0];
        if (current_customer_.orders.size() >= 2) o2 = current_customer_.orders[1];
        if (current_customer_.orders.size() >= 3) o3 = current_customer_.orders[2];
        all_orders = formatOrderList(current_customer_.orders);
        c_idx = customer_idx_;
      }
      std::ostringstream ss;
      ss << "{\"action\":\"show\","
         << "\"customer_idx\":" << c_idx << ","
         << "\"name\":\"" << name << "\","
         << "\"order1\":\"" << o1 << "\","
         << "\"order2\":\"" << o2 << "\","
         << "\"order3\":\"" << o3 << "\","
         << "\"orders\":\"" << all_orders << "\","
         << "\"image_url\":\"captures/customer_latest.jpg\"}";
      msg.data = ss.str();
    } else {
      msg.data = "{\"action\":\"hide\"}";
    }
    hmi_customer_pub_->publish(msg);
    RCLCPP_INFO(get_logger(), "\033[1;34m[RESTAURANT]\033[0m\n  HMI: Order Card -> %s", show ? "SHOW" : "HIDE");
  }

  void publishExpression(const std::string & expr)
  {
    if (face_expression_pub_) {
      std_msgs::msg::String msg;
      msg.data = expr;
      face_expression_pub_->publish(msg);
      RCLCPP_INFO(get_logger(), "\033[1;34m[RESTAURANT]\033[0m\n  Face: Expression -> %s", expr.c_str());
    }
  }

  void initCurrentCustomer(int idx)
  {
    {
      std::lock_guard<std::mutex> lock(customer_mutex_);
      customer_idx_ = idx;
      current_customer_ = CustomerRecord{};
      current_customer_.id = idx;
      current_customer_.name = "Customer " + std::to_string(idx);
      current_customer_.orders.clear();
    }
    {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      wave_detected_ = false;
      has_waving_person_coords_ = false;
      detected_items_.clear();
      has_last_order1_3d_ = false;
      last_search_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
      search_attempted_in_state_ = false;
    }
    if (person_name_pub_) {
      std_msgs::msg::String msg;
      msg.data = "";
      person_name_pub_->publish(msg);
    }
  }

  static std::string expressionForState(RestaurantState state)
  {
    switch (state) {
      case RestaurantState::INIT:
        return "IDLE";
      case RestaurantState::DETECT_WAVE_HAND:
      case RestaurantState::FIND_AND_ALIGN_BARMAN:
      case RestaurantState::DETECT_AND_NAV_TO_ORDER1:
        return "SEARCHING";
      case RestaurantState::NAV_TO_WAVING_CUSTOMER:
      case RestaurantState::NAV_TO_ORIGIN_BARMAN:
      case RestaurantState::NAV_RETURN_ORIGIN_AFTER_SERVE:
        return "FOLLOWING";
      case RestaurantState::WAIT_CUSTOMER_NAME:
      case RestaurantState::WAIT_CUSTOMER_ORDER:
      case RestaurantState::WAIT_ORDER_CONFIRMATION:
      case RestaurantState::WAIT_TRAY_CONFIRMATION:
        return "LISTENING";
      case RestaurantState::WAIT_TRAY_SILENT_5S:
      case RestaurantState::WAIT_TRAY_RETRY_3S:
        return "CURIOUS";
      case RestaurantState::GRASP_ORDER1:
      case RestaurantState::RELEASE_GRIPPER_AND_TORQUE_OFF:
        return "WORKING";
      case RestaurantState::ANNOUNCE_SERVER:
      case RestaurantState::SAVE_CUSTOMER_POSE_ASK_NAME:
      case RestaurantState::CAPTURE_CUSTOMER_FACE:
      case RestaurantState::ASK_CUSTOMER_ORDER:
      case RestaurantState::PRE_CONFIRM_ORDER:
      case RestaurantState::WAIT_BEFORE_CONFIRM_SPEECH:
      case RestaurantState::SPEAK_ORDER_CONFIRMATION:
      case RestaurantState::ANNOUNCE_ORDER_TO_BARMAN:
      case RestaurantState::ASK_TRAY_CONFIRMATION:
      case RestaurantState::NAV_DELIVER_TO_CUSTOMER:
      case RestaurantState::GREET_DELIVERY_CUSTOMER:
      case RestaurantState::WAIT_1S_BEFORE_RELEASE_PROMPT:
      case RestaurantState::PROMPT_RELEASE_ORDER1:
      case RestaurantState::PROMPT_TAKE_ORDER2_TRAY:
        return "HAPPY";
      case RestaurantState::WAIT_5S_CUSTOMER_TAKE_TRAY:
      case RestaurantState::FINISH_TASK:
      case RestaurantState::COMPLETED:
        return "SUCCESS";
      case RestaurantState::ABORTED:
        return "LOST";
      default:
        return "IDLE";
    }
  }

  void transitionTo(RestaurantState next_state, const std::string & reason)
  {
    RCLCPP_INFO(
      get_logger(),
      "\033[1;34m[STATE]\033[0m %s -> %s (C%d: %s)",
      stateToString(state_.load()).c_str(),
      stateToString(next_state).c_str(),
      customer_idx_,
      reason.c_str());

    if (next_state == RestaurantState::DETECT_WAVE_HAND) {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      wave_detected_ = false;
      last_wave_prompt_time_ = now();
      last_search_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
      search_attempted_in_state_ = false;
    }
    if (next_state == RestaurantState::DETECT_AND_NAV_TO_ORDER1) {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      detected_items_.clear();
    }
    if (next_state == RestaurantState::NAV_DELIVER_TO_CUSTOMER) {
      publishCustomerHmi(false);
    }

    state_enter_time_ = now();
    has_new_utterance_ = false;
    state_ = next_state;

    // Sinkronisasi ekspresi mata robot ke HMI (athome_face_display)
    publishExpression(expressionForState(next_state));
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

    nav_busy_ = true;

    if (!nav_action_client_->wait_for_action_server(2s)) {
      nav_busy_ = false;
      RCLCPP_ERROR(get_logger(), "[Nav2] Action server 'navigate_to_pose' tidak tersedia!");
      last_nav_fail_time_ = now();
      if (on_finished) on_finished(false);
      return;
    }

    NavigateToPose::Goal goal;
    goal.pose.header.frame_id = global_frame_;
    goal.pose.header.stamp = now();
    goal.pose.pose.position.x = goal_x;
    goal.pose.pose.position.y = goal_y;
    goal.pose.pose.position.z = 0.0;
    goal.pose.pose.orientation = yawToQuaternion(goal_yaw);

    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[NAV]\033[0m Goal: (x=%.2f, y=%.2f, yaw=%.1f°)",
      goal_x, goal_y, goal_yaw * 180.0 / M_PI);

    auto send_opts = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    send_opts.goal_response_callback =
      [this, on_finished](const GoalHandleNav::SharedPtr & handle) {
        if (!handle) {
          RCLCPP_WARN(get_logger(), "\033[1;31m[NAV]\033[0m Goal rejected by Nav2");
          nav_busy_ = false;
          last_nav_fail_time_ = now();
          if (on_finished) on_finished(false);
        } else {
          current_goal_handle_ = handle;
          RCLCPP_INFO(get_logger(), "\033[1;34m[RESTAURANT]\033[0m\n  Nav: Goal accepted, navigating...");
        }
      };

    send_opts.result_callback =
      [this, on_finished](const GoalHandleNav::WrappedResult & result) {
        in_callback_ = true;
        nav_busy_ = false;
        const bool ok = (result.code == rclcpp_action::ResultCode::SUCCEEDED);
        if (ok) {
          RCLCPP_INFO(get_logger(), "\033[1;34m[RESTAURANT]\033[0m\n  Nav: Arrived safely at destination");
        } else {
          RCLCPP_WARN(
            get_logger(), "\033[1;33m[NAV]\033[0m Navigation failed/aborted (code %d)",
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

  // Mengambil nilai cell costmap pada koordinat peta global (map)
  int getCostmapValue(double map_x, double map_y) const
  {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex &>(costmap_mutex_));
    if (!latest_costmap_) {
      return -1;
    }
    const auto & info = latest_costmap_->info;
    const double res = info.resolution;
    if (res <= 1e-5) return -1;

    const int mx = static_cast<int>(std::floor((map_x - info.origin.position.x) / res));
    const int my = static_cast<int>(std::floor((map_y - info.origin.position.y) / res));

    if (mx < 0 || mx >= static_cast<int>(info.width) ||
        my < 0 || my >= static_cast<int>(info.height))
    {
      return 100; // Di luar jangkauan peta -> anggap rintangan
    }

    const size_t idx = static_cast<size_t>(my * info.width + mx);
    if (idx >= latest_costmap_->data.size()) {
      return 100;
    }
    return static_cast<int>(latest_costmap_->data[idx]);
  }

  // Memeriksa apakah suatu koordinat (map_x, map_y) beserta radius fisik robot aman dari dinding/rintangan costmap
  bool isPoseSafeInCostmap(
    double map_x, double map_y, double safety_radius, int cost_threshold, int & out_max_cost) const
  {
    out_max_cost = 0;
    {
      std::lock_guard<std::mutex> lock(const_cast<std::mutex &>(costmap_mutex_));
      if (!latest_costmap_) {
        // Jika costmap belum diterima, izinkan agar navigasi tetap berjalan
        return true;
      }
    }

    static const double angles[8] = {
      0.0, M_PI_4, M_PI_2, 3.0 * M_PI_4, M_PI, -3.0 * M_PI_4, -M_PI_2, -M_PI_4};

    int c = getCostmapValue(map_x, map_y);
    out_max_cost = std::max(out_max_cost, c);
    if (c < 0 || c >= cost_threshold) {
      return false;
    }

    for (int i = 0; i < 8; ++i) {
      const double px = map_x + safety_radius * std::cos(angles[i]);
      const double py = map_y + safety_radius * std::sin(angles[i]);
      c = getCostmapValue(px, py);
      out_max_cost = std::max(out_max_cost, c);
      if (c < 0 || c >= cost_threshold) {
        return false;
      }
    }
    return true;
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

    double chosen_standoff = standoff_dist;
    // Terapkan adaptive standoff jika navigasi ke orang (standoff >= 0.70m) dan adaptive aktif
    if (adaptive_standoff_enabled_ && standoff_dist >= 0.70) {
      double best_d = standoff_dist;
      int min_cost_seen = 999;
      bool found_safe = false;

      for (double d = adaptive_standoff_min_; d <= adaptive_standoff_max_ + 1e-4; d += adaptive_standoff_step_) {
        if (d >= dist - 0.20) {
          break;
        }

        const double cand_x = map_tx - d * std::cos(angle);
        const double cand_y = map_ty - d * std::sin(angle);

        int max_cost = 0;
        const bool safe = isPoseSafeInCostmap(
          cand_x, cand_y, robot_radius_ + 0.06, costmap_obstacle_threshold_, max_cost);

        if (max_cost < min_cost_seen) {
          min_cost_seen = max_cost;
          best_d = d;
        }

        if (safe) {
          chosen_standoff = d;
          found_safe = true;
          break;
        }
      }

      if (!found_safe) {
        chosen_standoff = best_d;
        RCLCPP_WARN(
          get_logger(),
          "\033[1;33m[Manager Adaptive Standoff]\033[0m Tidak ada sel bebas murni di rentang [%.2fm - %.2fm]. "
          "Menggunakan jarak dengan cost terendah: %.2fm (cost=%d).",
          adaptive_standoff_min_, adaptive_standoff_max_, chosen_standoff, min_cost_seen);
      } else if (std::abs(chosen_standoff - standoff_dist) > 0.02) {
        RCLCPP_INFO(
          get_logger(),
          "\033[1;33m[Manager Adaptive Standoff]\033[0m Titik henti standar (%.2fm) dekat dinding! "
          "Jarak disesuaikan ke \033[1;32m%.2fm\033[0m.",
          standoff_dist, chosen_standoff);
      }
    }

    double goal_x = map_tx;
    double goal_y = map_ty;
    const double goal_yaw = angle;

    if (dist > chosen_standoff) {
      goal_x = map_tx - chosen_standoff * std::cos(angle);
      goal_y = map_ty - chosen_standoff * std::sin(angle);
    } else {
      goal_x = rx;
      goal_y = ry;
    }

    navigateToMapPoseAsync(goal_x, goal_y, goal_yaw, std::move(on_finished));
  }

  // Navigasi ke pelanggan yang melambaikan tangan
  void navigateToWavingCustomer(std::function<void(bool)> on_finished)
  {
    // Prioritaskan service /go_to_order (atau /go_to_order1) dari restaurant_node jika tersedia
    auto active_order_client = (go_to_order_client_ && go_to_order_client_->service_is_ready())
      ? go_to_order_client_
      : ((go_to_order1_client_ && go_to_order1_client_->service_is_ready()) ? go_to_order1_client_ : nullptr);
    const std::string srv_name = (active_order_client == go_to_order_client_) ? "/go_to_order" : "/go_to_order1";

    if (active_order_client) {
      callTrigger(
        active_order_client, srv_name,
        [this, srv_name, on_finished](bool ok, const std::string & msg) {
          if (!ok) {
            RCLCPP_WARN(get_logger(), "[NAV] %s gagal (%s), mencoba direct Nav2...", srv_name.c_str(), msg.c_str());
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

  // Mengumumkan pelanggan ditemukan, menyegarkan koordinat via /vision/wave_hand, lalu lanjut navigasi
  void announceAndNavigateToWavingCustomer(const std::string & fallback_reason)
  {
    speakAsync("I see you, I am coming.", [this, fallback_reason]() {
      if (vision_wave_hand_client_ && vision_wave_hand_client_->service_is_ready()) {
        callTrigger(
          vision_wave_hand_client_, "/vision/wave_hand",
          [this](bool ok, const std::string & msg) {
            double px = 0.0, py = 0.0;
            if (parseCoordinates(msg, px, py)) {
              std::lock_guard<std::mutex> lock(vision_mutex_);
              waving_person_x_ = px;
              waving_person_y_ = -py;
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
      } else {
        transitionTo(RestaurantState::NAV_TO_WAVING_CUSTOMER, fallback_reason);
      }
    });
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
  double estimateTtsTimeout(const std::string & text) const
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
      tts_min_timeout_sec_,
      static_cast<double>(words) * tts_sec_per_word_ + tts_timeout_margin_sec_);
  }

  void speakAsync(const std::string & text, std::function<void()> on_done = nullptr)
  {
    {
      std::lock_guard<std::mutex> lock(tts_mutex_);
      speaking_ = true;
      tts_post_delay_active_ = false;
      pending_post_tts_cb_ = nullptr;
      tts_started_time_ = now();
      tts_timeout_sec_ = estimateTtsTimeout(text);
      tts_cooldown_until_ = now() + rclcpp::Duration::from_seconds(0.5);
      active_tts_text_ = text;
      pending_tts_done_ = std::move(on_done);
    }
    has_new_utterance_ = false;

    std_msgs::msg::String message;
    message.data = text;
    tts_pub_->publish(message);
    publishExpression("SPEAKING");
    RCLCPP_INFO(get_logger(), "\033[1;34m[RESTAURANT]\033[0m\n  TTS: \"%s\"", text.c_str());
  }

  void onTtsDone(const std_msgs::msg::String::SharedPtr msg = nullptr)
  {
    std::function<void()> callback;
    bool deferred = false;
    {
      std::lock_guard<std::mutex> lock(tts_mutex_);
      if (!speaking_.load()) {
        return;
      }
      speaking_ = false;
      const auto current_now = now();
      const std::string completed =
        (msg != nullptr && !msg->data.empty()) ? msg->data : active_tts_text_;
      if (!completed.empty()) {
        last_tts_text_norm_ = normalize(completed);
      }
      last_tts_done_time_ = current_now;
      active_tts_text_.clear();
      const auto min_cooldown_until = current_now + rclcpp::Duration::from_seconds(0.25);
      if (tts_post_delay_sec_ > 0.0) {
        tts_post_delay_active_ = true;
        tts_post_delay_until_ = current_now + rclcpp::Duration::from_seconds(tts_post_delay_sec_);
        tts_cooldown_until_ =
          (tts_post_delay_until_ > min_cooldown_until) ? tts_post_delay_until_ : min_cooldown_until;
        pending_post_tts_cb_ = std::move(pending_tts_done_);
        pending_tts_done_ = nullptr;
        deferred = true;
      } else {
        in_callback_ = true;
        tts_cooldown_until_ = min_cooldown_until;
        callback = std::move(pending_tts_done_);
        pending_tts_done_ = nullptr;
      }
    }

    RCLCPP_DEBUG(
      get_logger(),
      "[TTS] Done | State: %s | PostDelay: %.2fs",
      stateToString(state_.load()).c_str(),
      deferred ? tts_post_delay_sec_ : 0.0);

    publishExpression(expressionForState(state_.load()));

    if (deferred) {
      return;
    }

    if (callback) {
      callback();
    }
    in_callback_ = false;
  }

  bool isDialogWaitState(RestaurantState s) const
  {
    return s == RestaurantState::WAIT_CUSTOMER_ORDER ||
           s == RestaurantState::WAIT_ORDER_CONFIRMATION ||
           s == RestaurantState::WAIT_TRAY_CONFIRMATION;
  }

  bool canListen() const
  {
    if (isBusy() || tts_post_delay_active_.load()) {
      return false;
    }
    rclcpp::Time cooldown_until(0, 0, RCL_ROS_TIME);
    {
      std::lock_guard<std::mutex> lock(tts_mutex_);
      cooldown_until = tts_cooldown_until_;
    }
    return now() >= cooldown_until;
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

  static std::vector<std::string> splitWords(const std::string & norm)
  {
    std::vector<std::string> words;
    std::stringstream ss(norm);
    std::string w;
    while (ss >> w) {
      words.push_back(w);
    }
    return words;
  }

  static bool isNameStopWord(const std::string & w)
  {
    static const std::vector<std::string> kStopWords = {
      "my", "name", "is", "am", "im", "are", "you", "can", "call", "me",
      "this", "that", "the", "hello", "hi", "hey", "welcome", "to", "bar",
      "please", "tell", "your", "yes", "no", "robot", "thanks", "thank",
      "sorry", "okay", "ok", "what", "pardon", "uh", "um", "er", "ah", "oh",
      "hmm", "huh", "actually", "well", "so", "s", "m", "and", "or", "correct"
    };
    return std::find(kStopWords.begin(), kStopWords.end(), w) != kStopWords.end();
  }

  static bool isLikelyNoiseOrPromptEcho(const std::string & norm)
  {
    if (norm.size() < 2) {
      return true;
    }
    const bool all_digits_or_space = std::all_of(norm.begin(), norm.end(), [](unsigned char c) {
      return std::isdigit(c) || std::isspace(c);
    });
    if (all_digits_or_space) {
      return true;
    }

    static const std::vector<std::string> kPromptEchoPhrases = {
      "may i take your order", "please say your items", "separated by and",
      "what else would you like", "is that correct", "please say yes robot",
      "yes robot or no robot", "yes or no", "placed on my tray",
      "please wave your hand", "please tell me your order", "let us try again"
    };
    for (const auto & p : kPromptEchoPhrases) {
      if (containsWholePhrase(norm, p)) {
        return true;
      }
    }

    static const std::vector<std::string> kExactNoiseHallucinations = {
      "hello", "hi", "hey", "okay", "ok", "what", "sorry", "pardon",
      "excuse me", "thank you", "thank you very much", "thank you so much",
      "thanks", "thanks for watching", "thanks for listening", "please", "robot",
      "uh", "um", "er", "ah", "oh", "hmm", "huh", "mm", "mhm",
      "you", "bye", "music", "applause", "laughter", "silence", "blank audio",
      "subtitles by", "please subscribe"
    };
    for (const auto & item : kExactNoiseHallucinations) {
      if (norm == item) {
        return true;
      }
    }

    static const std::vector<std::string> kPureFillers = {
      "uh", "um", "er", "ah", "oh", "hmm", "huh", "mm", "mhm",
      "okay", "ok", "thanks", "thank", "you", "please", "sorry",
      "hello", "hi", "hey", "robot"
    };
    const std::vector<std::string> words = splitWords(norm);
    if (words.empty()) {
      return true;
    }
    const bool all_fillers = std::all_of(words.begin(), words.end(), [](const std::string & w) {
      return std::find(kPureFillers.begin(), kPureFillers.end(), w) != kPureFillers.end();
    });
    return all_fillers;
  }

  bool isTtsEcho(const std::string & norm_text) const
  {
    std::string last_norm;
    rclcpp::Time last_done(0, 0, RCL_ROS_TIME);
    {
      std::lock_guard<std::mutex> lock(tts_mutex_);
      last_norm = last_tts_text_norm_;
      last_done = last_tts_done_time_;
    }
    if (last_norm.empty() || last_done.nanoseconds() == 0 ||
        (now() - last_done).seconds() > 2.5)
    {
      return false;
    }
    const RestaurantState st = state_.load();
    if ((st == RestaurantState::WAIT_ORDER_CONFIRMATION ||
         st == RestaurantState::WAIT_TRAY_CONFIRMATION) &&
        parseYesNoIntent(norm_text) != 0 &&
        !isLikelyNoiseOrPromptEcho(norm_text))
    {
      return false;
    }

    const std::vector<std::string> tokens = splitWords(norm_text);
    const size_t min_tokens = (st == RestaurantState::WAIT_CUSTOMER_ORDER) ? 4 : 3;
    if (tokens.size() < min_tokens) {
      return false;
    }

    size_t overlap = 0;
    bool has_novel_content = false;
    for (const auto & tok : tokens) {
      if (containsWholePhrase(last_norm, tok)) {
        ++overlap;
      } else if (!isNameStopWord(tok)) {
        has_novel_content = true;
      }
    }

    if (containsWholePhrase(last_norm, norm_text)) {
      return true;
    }
    const double overlap_ratio = static_cast<double>(overlap) / static_cast<double>(tokens.size());
    return overlap_ratio >= 0.65 && !has_novel_content;
  }

  // Ekstraksi nama pelanggan dari kalimat STT
  static std::string parseCustomerName(const std::string & raw_text)
  {
    const std::string norm = normalize(raw_text);
    if (norm.empty() || isLikelyNoiseOrPromptEcho(norm) ||
        norm == "yes" || norm == "no" || norm == "yes robot" || norm == "no robot")
    {
      return "";
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

    auto stripLeadingFillers = [](std::string s) -> std::string {
      static const std::vector<std::string> kLeadFillers = {
        "is ", "a ", "an ", "the ", "uh ", "um ", "er ", "ah ", "oh ",
        "hmm ", "well ", "so ", "actually "
      };
      bool stripped = true;
      while (stripped) {
        stripped = false;
        while (!s.empty() && s.front() == ' ') {
          s.erase(s.begin());
        }
        for (const auto & lf : kLeadFillers) {
          if (s.rfind(lf, 0) == 0) {
            s.erase(0, lf.size());
            stripped = true;
            break;
          }
        }
      }
      return s;
    };

    auto stripTrailingFillers = [](std::string s) -> std::string {
      static const std::vector<std::string> kTrailFillers = {
        " please", " robot", " thank you", " thanks",
        " uh", " um", " okay", " ok", " sorry"
      };
      bool stripped = true;
      while (stripped) {
        stripped = false;
        while (!s.empty() && s.back() == ' ') {
          s.pop_back();
        }
        for (const auto & tf : kTrailFillers) {
          if (s.size() >= tf.size() &&
              s.compare(s.size() - tf.size(), tf.size(), tf) == 0)
          {
            s.erase(s.size() - tf.size());
            stripped = true;
            break;
          }
        }
      }
      while (!s.empty() && s.back() == ' ') {
        s.pop_back();
      }
      return s;
    };

    auto isValidNameFallback = [&](const std::string & rem) -> bool {
      if (rem.size() < 2) {
        return false;
      }
      const std::vector<std::string> rem_words = splitWords(rem);
      if (rem_words.empty() || rem_words.size() > 2) {
        return false;
      }
      const bool all_digits = std::all_of(rem.begin(), rem.end(), [](unsigned char c) {
        return std::isdigit(c) || std::isspace(c);
      });
      if (all_digits) {
        return false;
      }
      return std::any_of(rem_words.begin(), rem_words.end(), [](const std::string & tok) {
        return !isNameStopWord(tok);
      });
    };

    static const std::vector<std::string> kPrefixes = {
      "you can call me ", "my name is ", "may name is ", "mai name is ",
      "my name s ", "mai name s ", "my name ", "name is ", "name s ",
      "ame is ", "me is ", "call me ", "this is ",
      "i am ", "iam ", "iyem ", "i m ", "im ", "name ", "is "
    };
    for (const auto & p : kPrefixes) {
      size_t search_from = 0;
      while (search_from < norm.size()) {
        const auto pos = norm.find(p, search_from);
        if (pos == std::string::npos) {
          break;
        }
        if (pos == 0 || norm[pos - 1] == ' ') {
          std::string rem = stripTrailingFillers(
            stripLeadingFillers(norm.substr(pos + p.size())));
          const std::vector<std::string> rem_words = splitWords(rem);
          if (!rem_words.empty()) {
            const std::string cand = (rem_words.size() >= 2)
              ? (rem_words[0] + " " + rem_words[1])
              : rem_words[0];
            if (isValidNameFallback(cand)) {
              return capitalizeWords(cand);
            }
          }
        }
        search_from = pos + 1;
      }
    }

    const std::vector<std::string> words = splitWords(norm);
    if (!words.empty() && words.size() <= 2) {
      std::string rem = stripTrailingFillers(stripLeadingFillers(norm));
      if (isValidNameFallback(rem)) {
        return capitalizeWords(rem);
      }
    }
    return "";
  }

  static size_t levenshteinDistance(const std::string & a, const std::string & b)
  {
    const size_t n = a.size();
    const size_t m = b.size();
    if (n == 0) return m;
    if (m == 0) return n;

    std::vector<size_t> prev(m + 1);
    std::vector<size_t> curr(m + 1);
    for (size_t j = 0; j <= m; ++j) prev[j] = j;
    for (size_t i = 1; i <= n; ++i) {
      curr[0] = i;
      for (size_t j = 1; j <= m; ++j) {
        const size_t cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
        curr[j] = std::min({prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost});
      }
      prev.swap(curr);
    }
    return prev[m];
  }

  static size_t maxFuzzyDistance(size_t len)
  {
    if (len < 5) return 0;
    if (len <= 6) return 1;
    return 2;
  }

  static bool isOrderStopWord(const std::string & w)
  {
    static const std::vector<std::string> kStopWords = {
      "i", "would", "like", "want", "order", "ordered", "my", "is", "are",
      "can", "may", "have", "get", "give", "me", "please", "robot",
      "and", "with", "plus", "also", "the", "a", "an", "one", "two", "three",
      "some", "for", "of", "to", "hello", "hi", "hey", "yes", "no",
      "thanks", "thank", "you", "sorry", "okay", "ok", "uh", "um",
      "then", "items", "item"
    };
    return std::find(kStopWords.begin(), kStopWords.end(), w) != kStopWords.end();
  }

  // Ekstraksi 2-3 item pesanan dari kalimat pelanggan (exact + alias fonetik + fuzzy + fallback)
  static bool parseCustomerOrders(
    const std::string & raw_text,
    std::vector<std::string> & orders)
  {
    const std::string norm = normalize(raw_text);
    if (norm.empty() || isLikelyNoiseOrPromptEcho(norm) ||
        norm == "yes" || norm == "no" || norm == "yes robot" || norm == "no robot")
    {
      return false;
    }

    auto addUnique = [](std::vector<std::string> & dst, const std::string & item) {
      if (item.empty() || dst.size() >= 3) return;
      const std::string item_norm = normalize(item);
      if (item_norm.empty()) return;
      for (const auto & existing : dst) {
        if (normalize(existing) == item_norm) return;
      }
      dst.push_back(item);
    };

    auto formatMenuItem = [](const std::string & norm_item) -> std::string {
      if (norm_item == "cheez it" || norm_item == "cheezit") return "Cheez-It";
      if (norm_item == "seven up") return "Seven Up";
      return capitalizeWords(norm_item);
    };

    // 1. Kamus item makanan/minuman standar RoboCup (urutan: frase panjang lebih dulu)
    static const std::vector<std::string> kKnownMenuItems = {
      "chocolate milk", "strawberry juice", "orange juice", "apple juice",
      "grape juice", "mango juice", "pineapple juice", "tropical juice",
      "sparkling water", "mineral water", "iced coffee", "black coffee",
      "iced tea", "green tea", "lemon tea", "red wine", "white wine",
      "coca cola", "root beer", "seven up", "potato chips", "cracker box",
      "sugar box", "tomato soup", "mustard bottle", "cheez it", "ice cream",
      "hot dog",
      "coke", "cola", "pepsi", "sprite", "fanta", "beer", "wine",
      "water", "milk", "coffee", "tea", "juice", "soda", "lemonade", "smoothie",
      "pringles", "crackers", "cracker", "biscuits", "biscuit",
      "cookies", "cookie", "cornflakes", "cereal", "popcorn", "pretzels",
      "oreo", "chips", "snack", "candy", "chocolate", "sugar",
      "sandwich", "burger", "pizza", "pasta", "noodles", "ramen", "soup",
      "bread", "muffin", "cupcake", "donut", "cake", "tuna", "spam",
      "cheese", "butter", "jam", "honey", "ketchup", "mustard",
      "apple", "banana", "orange", "pear", "peach", "plum",
      "lemon", "lime", "strawberry", "grapes", "tomato", "salad"
    };

    struct PhoneticAlias {
      const char * alias;
      const char * canonical;
    };
    static const std::vector<PhoneticAlias> kMenuAliases = {
      {"cheese it", "Cheez-It"}, {"cheezit", "Cheez-It"},
      {"cheesit", "Cheez-It"}, {"cheese its", "Cheez-It"},
      {"cook", "Coke"}, {"koke", "Coke"}, {"cock", "Coke"}, {"coca", "Coke"},
      {"springles", "Pringles"}, {"pringle", "Pringles"}, {"bringles", "Pringles"},
      {"ice tea", "Iced Tea"}, {"ice coffee", "Iced Coffee"},
      {"orange choose", "Orange Juice"}, {"apple choose", "Apple Juice"},
      {"grape choose", "Grape Juice"},
      {"watter", "Water"}, {"kofi", "Coffee"}, {"copi", "Coffee"},
      {"spread", "Sprite"}, {"fenta", "Fanta"}, {"pepsy", "Pepsi"},
      {"sugarbox", "Sugar Box"}, {"crackerbox", "Cracker Box"},
      {"corn flakes", "Cornflakes"}, {"crisps", "Chips"},
      {"doughnut", "Donut"}, {"7 up", "Seven Up"}, {"7up", "Seven Up"},
      {"choc milk", "Chocolate Milk"}, {"choco milk", "Chocolate Milk"},
      {"biscuit", "Biscuits"}, {"cookie", "Cookies"},
      {"noodle", "Noodles"}, {"cheetos", "Chips"}
    };

    struct FoundItem {
      size_t pos;
      std::string name;
    };
    std::vector<FoundItem> found;
    std::string masked = " " + norm + " ";

    auto maskMatch = [&](const std::string & needle, const std::string & canonical) {
      size_t search_from = 0;
      while (search_from < masked.size()) {
        const auto pos = masked.find(needle, search_from);
        if (pos == std::string::npos) break;
        found.push_back({pos, canonical});
        for (size_t i = pos + 1; i + 1 < pos + needle.size(); ++i) {
          masked[i] = '_';
        }
        search_from = pos + needle.size() - 1;
      }
    };

    for (const auto & item : kKnownMenuItems) {
      maskMatch(" " + item + " ", formatMenuItem(item));
    }

    // 2. Alias fonetik pada sisa token yang belum ter-mask
    for (const auto & entry : kMenuAliases) {
      maskMatch(" " + std::string(entry.alias) + " ", entry.canonical);
    }

    std::sort(found.begin(), found.end(), [](const FoundItem & a, const FoundItem & b) {
      return a.pos < b.pos;
    });
    for (const auto & f : found) {
      addUnique(orders, f.name);
    }

    // 3. Fuzzy matching (Levenshtein) untuk token tersisa jika belum lengkap (< 3 item)
    if (orders.size() < 3) {
      std::string unmasked = masked;
      for (char & c : unmasked) {
        if (c == '_') c = ' ';
      }
      const std::vector<std::string> rem_tokens = splitWords(unmasked);
      std::vector<std::string> active_tokens;
      for (const auto & t : rem_tokens) {
        if (!isOrderStopWord(t)) {
          active_tokens.push_back(t);
        }
      }

      if (!active_tokens.empty()) {
        // Cek jendela 2 kata lalu 1 kata
        for (size_t k : {size_t{2}, size_t{1}}) {
          if (active_tokens.size() < k) continue;
          for (size_t i = 0; i + k <= active_tokens.size() && orders.size() < 3; ++i) {
            std::string win = active_tokens[i];
            if (k == 2) win += " " + active_tokens[i + 1];

            std::string best_cand;
            size_t best_dist = 999;
            for (const auto & cand : kKnownMenuItems) {
              if (splitWords(cand).size() != k) continue;
              const size_t max_d = maxFuzzyDistance(cand.size());
              if (max_d == 0) continue;
              const size_t len_diff = (win.size() > cand.size())
                ? (win.size() - cand.size())
                : (cand.size() - win.size());
              if (len_diff > max_d) continue;
              const size_t d = levenshteinDistance(win, cand);
              if (d <= max_d &&
                  (d < best_dist || (d == best_dist && cand.size() > best_cand.size())))
              {
                best_dist = d;
                best_cand = cand;
              }
            }
            if (!best_cand.empty()) {
              addUnique(orders, formatMenuItem(best_cand));
            }
          }
        }
      }
    }

    if (orders.size() >= 2) {
      return true;
    }

    // 4. Fallback delimiter splitting untuk item di luar kamus (2-3 item dipisah koma / "and")
    std::string raw_lower = raw_text;
    std::transform(raw_lower.begin(), raw_lower.end(), raw_lower.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    for (char & c : raw_lower) {
      if (c == ';') c = ',';
    }

    std::vector<std::string> segments = {raw_lower};
    for (const std::string_view delim : {",", " and ", " with ", " plus ", " also ", " dan ", " sama "}) {
      std::vector<std::string> next_segs;
      for (const auto & seg : segments) {
        size_t start = 0;
        while (start < seg.size()) {
          const auto pos = seg.find(delim, start);
          if (pos == std::string::npos) {
            next_segs.push_back(seg.substr(start));
            break;
          }
          next_segs.push_back(seg.substr(start, pos - start));
          start = pos + delim.size();
        }
      }
      segments = std::move(next_segs);
    }

    if (segments.size() >= 2) {
      static const std::vector<std::string> kOrderPrefixes = {
        "i would like to order ", "i would like ", "i want to order ", "i want ",
        "can i have ", "may i have ", "please give me ", "give me ",
        "my order is ", "i order ", "order is ", "please "
      };
      static const std::vector<std::string> kTrailFillers = {
        " please", " robot", " thank you", " thanks"
      };
      for (const auto & raw_seg : segments) {
        std::string seg = normalize(raw_seg);
        if (seg.empty()) continue;
        for (const auto & prefix : kOrderPrefixes) {
          const auto p = seg.find(prefix);
          if (p != std::string::npos && (p == 0 || seg[p - 1] == ' ')) {
            seg = trimString(seg.substr(p + prefix.size()));
            break;
          }
        }
        for (const std::string_view art : {"a ", "an ", "the ", "one ", "some "}) {
          if (seg.rfind(art, 0) == 0) {
            seg = trimString(seg.substr(art.size()));
          }
        }
        for (const auto & tf : kTrailFillers) {
          if (seg.size() >= tf.size() &&
              seg.compare(seg.size() - tf.size(), tf.size(), tf) == 0)
          {
            seg = trimString(seg.substr(0, seg.size() - tf.size()));
          }
        }
        const std::vector<std::string> sw = splitWords(seg);
        if (seg.size() >= 2 && !sw.empty() && sw.size() <= 3 &&
            std::any_of(sw.begin(), sw.end(), [](const std::string & w) { return !isOrderStopWord(w); }))
        {
          addUnique(orders, formatMenuItem(seg));
        }
      }
    }

    return orders.size() >= 2;
  }

  // Pengenalan respon konfirmasi Yes / No (termasuk "yes robot" / "no robot")
  static int parseYesNoIntent(const std::string & raw_text)
  {
    const std::string norm = normalize(raw_text);
    if (norm.empty()) return 0;

    static const std::vector<std::string> kNoPhrases = {
      "no robot", "o robot", "no", "nope", "nah", "not correct", "wrong", "false",
      "incorrect", "negative", "not yet", "wait", "belum", "tidak", "bukan"
    };
    static const std::vector<std::string> kYesPhrases = {
      "yes robot", "es robot", "yes", "yeah", "yep", "correct", "right", "sure",
      "true", "exactly", "of course", "already", "done", "ya", "benar", "betul", "sudah"
    };

    bool has_no = false;
    for (const auto & phrase : kNoPhrases) {
      if (containsWholePhrase(norm, phrase)) {
        has_no = true;
        break;
      }
    }
    bool has_yes = false;
    for (const auto & phrase : kYesPhrases) {
      if (containsWholePhrase(norm, phrase)) {
        has_yes = true;
        break;
      }
    }
    if (has_no && has_yes) {
      return 0;
    }
    if (has_no) {
      return -1;
    }
    if (has_yes) {
      return 1;
    }
    return 0;
  }

  // ============================================================================
  // CALLBACK SUBSCRIPTION ROS 2 (STT & VISION)
  // ============================================================================
  void onTranscript(const std_msgs::msg::String::SharedPtr message)
  {
    if (!message || !isDialogWaitState(state_.load())) {
      return;
    }
    if (!canListen()) {
      RCLCPP_DEBUG(
        get_logger(),
        "[STT] Ignored while busy/post-delay: \"%s\"",
        message->data.c_str());
      return;
    }

    const std::string clean = normalize(message->data);
    if (clean.empty()) {
      return;
    }
    if (isTtsEcho(clean)) {
      RCLCPP_INFO(
        get_logger(),
        "\033[2m[STT IGNORED]\033[0m TTS echo: \"%s\"",
        message->data.c_str());
      return;
    }
    if (isLikelyNoiseOrPromptEcho(clean)) {
      RCLCPP_INFO(
        get_logger(),
        "\033[2m[STT IGNORED]\033[0m Noise/filler: \"%s\"",
        message->data.c_str());
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
      "\033[1;33m[STT]\033[0m \"%s\"",
      message->data.c_str());
  }

  // Callback deteksi lambaian tangan / gesture dari Vision (/vision/gesture)
  void onGesture(const std_msgs::msg::String::SharedPtr message)
  {
    if (!message || message->data.empty()) return;
    const std::string gest = normalize(message->data);
    if (gest.empty() || gest == "none") return;

    if (gest.find("wave") != std::string::npos ||
        gest.find("raise") != std::string::npos)
    {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      if (has_waving_person_coords_) {
        wave_detected_ = true;
      }
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
      waving_person_y_ = -py; // Negasikan kembali agar sesuai standar ROS base_link (+Y kiri, -Y kanan)
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
        waving_person_y_ = -py;
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
        item.y = -coords[1]; // Negasikan sumbu Y dari athome_vision agar sesuai base_link (+Y kiri, -Y kanan)
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
      if (object_pose_pub_) {
        std::ostringstream ss;
        bool first = true;
        for (const auto & it : detected_items_) {
          if (!it.has_3d) continue;
          if (!first) ss << "; ";
          first = false;
          ss << it.name << ": x=" << std::fixed << std::setprecision(2) << it.x
             << ", y=" << it.y << ", depth=" << it.z;
        }
        if (!ss.str().empty()) {
          std_msgs::msg::String pmsg;
          pmsg.data = ss.str();
          object_pose_pub_->publish(pmsg);
        }
      }
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
  bool findOrder1Position(
    const std::string & target_order_name, double & out_x, double & out_y, double & out_z)
  {
    std::lock_guard<std::mutex> lock(vision_mutex_);
    if (detected_items_.empty()) {
      return false;
    }

    auto stripSpaces = [](std::string s) {
      s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); }), s.end());
      return s;
    };

    const std::string target_norm = normalize(target_order_name);
    const std::string target_stripped = stripSpaces(target_norm);

    for (const auto & item : detected_items_) {
      if (!item.has_3d) continue;
      const std::string item_norm = normalize(item.name);
      const std::string item_stripped = stripSpaces(item_norm);

      if (!item_norm.empty() &&
          (item_norm == target_norm ||
           item_norm.find(target_norm) != std::string::npos ||
           target_norm.find(item_norm) != std::string::npos ||
           (!target_stripped.empty() && (item_stripped == target_stripped ||
            item_stripped.find(target_stripped) != std::string::npos ||
            target_stripped.find(item_stripped) != std::string::npos))))
      {
        out_x = item.x;
        out_y = item.y;
        out_z = item.z;
        return true;
      }
    }

    // Fallback: ambil objek terdekat yang memiliki koordinat 3D valid
    for (const auto & item : detected_items_) {
      if (item.has_3d) {
        out_x = item.x;
        out_y = item.y;
        out_z = item.z;
        return true;
      }
    }
    return false;
  }

  bool findOrder1Position(const std::string & target_order_name, double & out_x, double & out_y)
  {
    double unused_z = 0.0;
    return findOrder1Position(target_order_name, out_x, out_y, unused_z);
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

    service_busy_ = true;

    if (!client->service_is_ready()) {
      if (!client->wait_for_service(1s)) {
        service_busy_ = false;
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

  // Memanggil service MOTION 2 (/arm/release_gripper) lalu relax arm (/arm/relax)
  void releaseGripperAndTorqueOff(std::function<void()> on_done)
  {
    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[ARM]\033[0m Releasing gripper via /arm/release_gripper...");

    callTrigger(
      arm_release_gripper_client_, "/arm/release_gripper",
      [this, on_done](bool /*rel_ok*/, const std::string & rel_msg) {
        RCLCPP_INFO(get_logger(), "\033[1;34m[RESTAURANT]\033[0m\n  Arm: Release gripper: %s", rel_msg.c_str());

        TriggerClient::SharedPtr relax_client = arm_relax_client_;
        std::string relax_srv_name = "/arm/relax";
        if ((!relax_client || !relax_client->service_is_ready()) &&
            arm_free_client_ && arm_free_client_->service_is_ready())
        {
          relax_client = arm_free_client_;
          relax_srv_name = "/arm/free";
        }

        callTrigger(
          relax_client, relax_srv_name,
          [this, on_done](bool /*relax_ok*/, const std::string & relax_msg) {
            RCLCPP_INFO(get_logger(), "\033[1;34m[RESTAURANT]\033[0m\n  Arm: Relax gripper: %s", relax_msg.c_str());
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
    if ((now() - task_start_time_).seconds() > global_timeout_sec_ &&
        current_state != RestaurantState::COMPLETED &&
        current_state != RestaurantState::ABORTED)
    {
      RCLCPP_ERROR(get_logger(), "Global restaurant task timeout reached.");
      transitionTo(RestaurantState::ABORTED, "Global restaurant task timeout reached");
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

    // 4. Non-blocking post-TTS delay sebelum callback on_done / state berikutnya berjalan
    if (tts_post_delay_active_.load()) {
      std::function<void()> post_tts_cb;
      {
        std::lock_guard<std::mutex> lock(tts_mutex_);
        if (tts_post_delay_active_.load()) {
          if (now() < tts_post_delay_until_) {
            return;
          }
          tts_post_delay_active_ = false;
          post_tts_cb = std::move(pending_post_tts_cb_);
          pending_post_tts_cb_ = nullptr;
          if (post_tts_cb) {
            in_callback_ = true;
          }
        }
      }
      if (post_tts_cb) {
        post_tts_cb();
        in_callback_ = false;
        return;
      }
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
        barman_briefed_ = false;
        initCurrentCustomer(1);
        publishStatus("Initialized origin_position, STT and TTS engine. Starting Customer 1.");
        transitionTo(RestaurantState::ANNOUNCE_SERVER, "Initialization complete");
        break;
      }

      // ========================================================================
      // POINT 2: Announce Server & Detect Person Who Waves Hand -> zed/wave_hand
      // ========================================================================
      case RestaurantState::ANNOUNCE_SERVER: {
        std::string announce_msg;
        {
          std::lock_guard<std::mutex> lock(customer_mutex_);
          if (customer_idx_ == 1) {
            announce_msg =
              "Hello everyone, I am Barelang Inari, your server. If you would like to order, please wave your hand.";
          } else {
            announce_msg = "If anyone else would like to order, please wave your hand.";
          }
        }
        speakAsync(announce_msg, [this]() {
          transitionTo(
            RestaurantState::DETECT_WAVE_HAND,
            "Waiting to detect waving customer");
        });
        break;
      }

      case RestaurantState::DETECT_WAVE_HAND: {
        bool waved = false;
        {
          std::lock_guard<std::mutex> lock(vision_mutex_);
          waved = wave_detected_;
        }

        if (waved) {
          announceAndNavigateToWavingCustomer(
            "Wave hand detected via vision topic, navigating to customer");
        } else if (search_client_ && search_client_->service_is_ready() &&
                   (now() - state_enter_time_).seconds() >= 1.0 &&
                   (!search_attempted_in_state_ ||
                    (now() - last_search_time_).seconds() >= wave_reprompt_sec_))
        {
          search_attempted_in_state_ = true;
          last_search_time_ = now();
          RCLCPP_INFO(
            get_logger(),
            "\033[1;36m[SEARCH]\033[0m Calling /search service for Customer %d...",
            customer_idx_);
          callTrigger(
            search_client_, "/search",
            [this](bool ok, const std::string & msg) {
              last_search_time_ = now();
              RCLCPP_INFO(
                get_logger(),
                "[SEARCH] Service /search finished (ok=%s, msg='%s')",
                ok ? "true" : "false", msg.c_str());
              if (ok) {
                announceAndNavigateToWavingCustomer(
                  "Customer found via /search, navigating to customer");
              } else {
                last_wave_prompt_time_ = now();
                speakAsync("If you would like to order, please wave your hand.");
              }
            },
            40.0);
        } else if ((!search_client_ || !search_client_->service_is_ready()) &&
                   vision_wave_hand_client_ && vision_wave_hand_client_->service_is_ready() &&
                   (now() - state_enter_time_).seconds() >= 2.0)
        {
          // Fallback jika /search tidak tersedia: deteksi langsung via /vision/wave_hand
          callTrigger(
            vision_wave_hand_client_, "/vision/wave_hand",
            [this](bool ok, const std::string & msg) {
              if (ok) {
                double px = 0.0, py = 0.0;
                if (parseCoordinates(msg, px, py)) {
                  std::lock_guard<std::mutex> lock(vision_mutex_);
                  waving_person_x_ = px;
                  waving_person_y_ = -py;
                  has_waving_person_coords_ = true;
                }
                speakAsync("I see you, I am coming.", [this]() {
                  transitionTo(
                    RestaurantState::NAV_TO_WAVING_CUSTOMER,
                    "Customer wave detected via /vision/wave_hand");
                });
              }
            },
            10.0);
        } else if ((now() - last_wave_prompt_time_).seconds() >= wave_reprompt_sec_) {
          last_wave_prompt_time_ = now();
          speakAsync("If you would like to order, please wave your hand.");
        }
        break;
      }

      // ========================================================================
      // POINT 3: Navigate to Person That Waved Hand
      // ========================================================================
      case RestaurantState::NAV_TO_WAVING_CUSTOMER:
        if (last_nav_fail_time_.nanoseconds() > 0 &&
            (now() - last_nav_fail_time_).seconds() < nav_retry_cooldown_sec_)
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
      // POINT 4: Save Coordinate Location for Customer (skip ask name & capture face)
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

        transitionTo(
          RestaurantState::ASK_CUSTOMER_ORDER,
          "Customer pose saved; proceeding directly to take order");
        break;
      }

      case RestaurantState::WAIT_CUSTOMER_NAME:
      case RestaurantState::CAPTURE_CUSTOMER_FACE:
        transitionTo(
          RestaurantState::ASK_CUSTOMER_ORDER,
          "Name and face capture skipped; proceeding to take order");
        break;

      // ========================================================================
      // POINT 6: Take Order (2-3 items) & Confirm Yes/No
      // ========================================================================
      case RestaurantState::ASK_CUSTOMER_ORDER:
        speakAsync(
          "Hello, welcome! May I take your order? Please say your items, separated by and.",
          [this]() {
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
          std::vector<std::string> orders;
          {
            std::lock_guard<std::mutex> lock(customer_mutex_);
            orders = current_customer_.orders;
          }

          if (parseCustomerOrders(raw, orders)) {
            {
              std::lock_guard<std::mutex> lock(customer_mutex_);
              current_customer_.orders = orders;
            }
            RCLCPP_INFO(
              get_logger(),
              "\033[1;32m[ORDER CAPTURED]\033[0m (%zu items): %s",
              orders.size(), formatOrderList(orders).c_str());
            transitionTo(
              RestaurantState::SPEAK_ORDER_CONFIRMATION,
              "Orders captured");
          } else if (orders.size() == 1) {
            {
              std::lock_guard<std::mutex> lock(customer_mutex_);
              current_customer_.orders = orders;
            }
            speakAsync(
              "I got " + orders[0] + ". What else would you like to order?",
              [this]() { state_enter_time_ = now(); });
          } else {
            speakAsync(
              "Sorry, please tell me your order, separated by and.",
              [this]() { state_enter_time_ = now(); });
          }
        } else if ((now() - state_enter_time_).seconds() > dialog_reprompt_sec_) {
          speakAsync(
            "May I take your order? Please say your items separated by and.",
            [this]() { state_enter_time_ = now(); });
        }
        break;

      case RestaurantState::PRE_CONFIRM_ORDER:
      case RestaurantState::WAIT_BEFORE_CONFIRM_SPEECH:
        transitionTo(
          RestaurantState::SPEAK_ORDER_CONFIRMATION,
          "Proceeding directly to order confirmation");
        break;

      case RestaurantState::SPEAK_ORDER_CONFIRMATION: {
        std::vector<std::string> orders;
        {
          std::lock_guard<std::mutex> lock(customer_mutex_);
          orders = current_customer_.orders;
        }
        speakAsync(
          "Your order is " + formatOrderList(orders) +
          ". Is that correct? Please say yes robot, or no robot.",
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
            publishCustomerHmi(true);
            speakAsync(
              "Thank you. I will go to the bar to get your order.",
              [this]() {
                transitionTo(
                  RestaurantState::NAV_TO_ORIGIN_BARMAN,
                  "Order confirmed (YES), navigating to barman at origin");
              });
          } else if (ans == -1) {
            {
              std::lock_guard<std::mutex> lock(customer_mutex_);
              current_customer_.orders.clear();
            }
            speakAsync(
              "Sorry, let us try again. Please tell me your order, separated by and.",
              [this]() {
                transitionTo(
                  RestaurantState::WAIT_CUSTOMER_ORDER,
                  "Order rejected (NO), returning to take order");
              });
          } else {
            speakAsync(
              "Please say yes robot, or no robot.",
              [this]() { state_enter_time_ = now(); });
          }
        } else if ((now() - state_enter_time_).seconds() > dialog_reprompt_sec_) {
          std::vector<std::string> orders;
          {
            std::lock_guard<std::mutex> lock(customer_mutex_);
            orders = current_customer_.orders;
          }
          speakAsync(
            "Your order is " + formatOrderList(orders) +
            ". Is that correct? Please say yes robot, or no robot.",
            [this]() { state_enter_time_ = now(); });
        }
        break;

      // ========================================================================
      // POINT 7: Navigate to Origin (Barman Place)
      // ========================================================================
      case RestaurantState::NAV_TO_ORIGIN_BARMAN:
        if (last_nav_fail_time_.nanoseconds() > 0 &&
            (now() - last_nav_fail_time_).seconds() < nav_retry_cooldown_sec_)
        {
          return;
        }
        navigateToOrigin([this](bool success) {
          if (!success) {
            RCLCPP_WARN(get_logger(), "[NAV] Gagal kembali ke origin (barman), mencoba ulang...");
            return;
          }
          transitionTo(
            RestaurantState::FIND_AND_ALIGN_BARMAN,
            "Arrived at origin (barman place), searching & aligning to barman");
        });
        break;

      case RestaurantState::FIND_AND_ALIGN_BARMAN:
        if (find_barman_client_ && find_barman_client_->service_is_ready()) {
          callTrigger(
            find_barman_client_, "/find_barman",
            [this](bool ok, const std::string & msg) {
              RCLCPP_INFO(
                get_logger(),
                "[BARMAN] /find_barman finished (ok=%s, msg='%s')",
                ok ? "true" : "false", msg.c_str());
              transitionTo(
                RestaurantState::ANNOUNCE_ORDER_TO_BARMAN,
                ok ? "Barman aligned within < 1.5 m" : "Barman align fallback");
            },
            20.0);
        } else {
          RCLCPP_WARN(
            get_logger(),
            "[BARMAN] Service /find_barman belum tersedia, langsung menuju ANNOUNCE_ORDER_TO_BARMAN.");
          transitionTo(
            RestaurantState::ANNOUNCE_ORDER_TO_BARMAN,
            "Proceeding without /find_barman service");
        }
        break;

      // ========================================================================
      // POINT 8: Barman Interaction, Detect & Navigate to order_1, Grasp, Tray
      // ========================================================================
      case RestaurantState::ANNOUNCE_ORDER_TO_BARMAN: {
        std::vector<std::string> orders;
        int c_idx = 1;
        bool need_briefing = false;
        {
          std::lock_guard<std::mutex> lock(customer_mutex_);
          orders = current_customer_.orders;
          c_idx = customer_idx_;
          need_briefing = !barman_briefed_;
          barman_briefed_ = true;
        }
        const std::string o1 = orders.empty() ? "the first item" : orders[0];
        const std::string order_list = formatOrderList(orders);
        const std::string customer_ordinal = (c_idx == 1) ? "first" : "second";

        auto announce_order = [this, customer_ordinal, order_list, o1]() {
          speakAsync(
            "The " + customer_ordinal + " customer ordered " + order_list +
            ". Please place " + o1 + " in front of my gripper.",
            [this]() {
              transitionTo(
                RestaurantState::GRASP_ORDER1,
                "Order_1 requested in front of gripper, proceeding directly to grasp");
            });
        };

        if (need_briefing) {
          speakAsync(
            "Hello Barman. Please place the first item in front of my gripper, and the remaining items on my tray. "
            "When I ask if the tray is ready, please say yes robot, or no robot.",
            [announce_order]() { announce_order(); });
        } else {
          announce_order();
        }
        break;
      }

      case RestaurantState::DETECT_AND_NAV_TO_ORDER1: {
        std::string o1;
        {
          std::lock_guard<std::mutex> lock(customer_mutex_);
          if (!current_customer_.orders.empty()) {
            o1 = current_customer_.orders[0];
          }
        }

        double ox = 0.0, oy = 0.0, oz = 0.0;
        if (findOrder1Position(o1, ox, oy, oz)) {
          {
            std::lock_guard<std::mutex> lock(vision_mutex_);
            last_order1_x_ = ox;
            last_order1_y_ = oy;
            last_order1_z_ = oz;
            has_last_order1_3d_ = true;
          }
          RCLCPP_INFO(
            get_logger(),
            "\033[1;32m[ORDER_1 DETECTED]\033[0m '%s' at (x=%.2f, y=%.2f, z=%.2f). Navigating to position...",
            o1.c_str(), ox, oy, oz);
          navigateToRelativeOrMapTargetAsync(
            ox, oy, object_standoff_dist_,
            [this](bool /*nav_ok*/) {
              transitionTo(
                RestaurantState::GRASP_ORDER1,
                "Reached order_1 position, calling /arm/ambil_order");
            });
        } else if ((now() - state_enter_time_).seconds() >= order1_detect_timeout_sec_) {
          RCLCPP_WARN(
            get_logger(),
            "[ORDER_1] Timeout %.1fs menunggu deteksi 3D '%s', langsung memanggil /arm/ambil_order di posisi saat ini.",
            order1_detect_timeout_sec_, o1.c_str());
          transitionTo(
            RestaurantState::GRASP_ORDER1,
            "Proceeding to grasp order_1 in front of gripper");
        }
        break;
      }

      case RestaurantState::GRASP_ORDER1: {
        std::string o1;
        {
          std::lock_guard<std::mutex> lock(customer_mutex_);
          if (!current_customer_.orders.empty()) {
            o1 = current_customer_.orders[0];
          }
        }
        double tx = 0.0, ty = 0.0, tz = 0.0;
        bool has_target_3d = findOrder1Position(o1, tx, ty, tz);
        if (!has_target_3d) {
          std::lock_guard<std::mutex> lock(vision_mutex_);
          if (has_last_order1_3d_) {
            tx = last_order1_x_;
            ty = last_order1_y_;
            tz = last_order1_z_;
            has_target_3d = true;
          }
        }
        if (has_target_3d && arm_target_pub_) {
          std_msgs::msg::String target_msg;
          std::ostringstream tss;
          tss << tx << "," << ty << "," << tz;
          target_msg.data = tss.str();
          arm_target_pub_->publish(target_msg);
          RCLCPP_INFO(
            get_logger(),
            "\033[1;36m[ARM TARGET]\033[0m Published /arm/target_point: (%s)",
            target_msg.data.c_str());
        }

        callTrigger(
          arm_grab_simple_client_, "/arm/ambil_order",
          [this](bool ok, const std::string & msg) {
            RCLCPP_INFO(
              get_logger(),
              "[ARM] /arm/ambil_order finished (ok=%s, msg='%s')",
              ok ? "true" : "false", msg.c_str());

            std::vector<std::string> orders;
            {
              std::lock_guard<std::mutex> lock(customer_mutex_);
              orders = current_customer_.orders;
            }
            const std::string tray_str = formatTrayOrders(orders);
            speakAsync(
              "Thank you. Please place " + tray_str + " on the tray in front of me.",
              [this]() {
                transitionTo(
                  RestaurantState::WAIT_TRAY_SILENT_5S,
                  "Remaining silent for 5 seconds while remaining items are placed on tray");
              });
          },
          25.0);
        break;
      }

      case RestaurantState::WAIT_TRAY_SILENT_5S:
        if ((now() - state_enter_time_).seconds() >= tray_wait_sec_) {
          transitionTo(
            RestaurantState::ASK_TRAY_CONFIRMATION,
            "Silent tray wait elapsed");
        }
        break;

      case RestaurantState::ASK_TRAY_CONFIRMATION: {
        std::vector<std::string> orders;
        {
          std::lock_guard<std::mutex> lock(customer_mutex_);
          orders = current_customer_.orders;
        }
        const std::string tray_str = formatTrayOrders(orders);
        const std::string prompt = (orders.size() > 2)
          ? ("Are " + tray_str + " placed on my tray? Please say yes robot, or no robot.")
          : ("Is " + tray_str + " placed on my tray? Please say yes robot, or no robot.");
        speakAsync(
          prompt,
          [this]() {
            transitionTo(
              RestaurantState::WAIT_TRAY_CONFIRMATION,
              "Waiting for yes robot / no robot response");
          });
        break;
      }

      case RestaurantState::WAIT_TRAY_CONFIRMATION:
        if (has_new_utterance_.exchange(false)) {
          std::string raw;
          {
            std::lock_guard<std::mutex> lock(stt_mutex_);
            raw = heard_utterance_;
          }
          const int ans = parseYesNoIntent(raw);
          if (ans == 1) {
            publishCustomerHmi(false);
            publishExpression("HAPPY");
            speakAsync(
              "Thank you. I will deliver the order to the customer.",
              [this]() {
                transitionTo(
                  RestaurantState::NAV_DELIVER_TO_CUSTOMER,
                  "Tray order confirmed (yes robot), proceeding to point 9");
              });
          } else if (ans == -1) {
            transitionTo(
              RestaurantState::WAIT_TRAY_RETRY_3S,
              "Tray order not ready (no robot), waiting 3 seconds before repeating callback");
          } else {
            speakAsync(
              "Please say yes robot, or no robot.",
              [this]() { state_enter_time_ = now(); });
          }
        } else if ((now() - state_enter_time_).seconds() > dialog_reprompt_sec_) {
          std::vector<std::string> orders;
          {
            std::lock_guard<std::mutex> lock(customer_mutex_);
            orders = current_customer_.orders;
          }
          const std::string tray_str = formatTrayOrders(orders);
          const std::string prompt = (orders.size() > 2)
            ? ("Are " + tray_str + " placed on my tray? Please say yes robot, or no robot.")
            : ("Is " + tray_str + " placed on my tray? Please say yes robot, or no robot.");
          speakAsync(
            prompt,
            [this]() { state_enter_time_ = now(); });
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
            (now() - last_nav_fail_time_).seconds() < nav_retry_cooldown_sec_)
        {
          return;
        }

        // Prioritaskan pemanggilan service /deliver_order dari restaurant_node
        if (deliver_order_client_ && deliver_order_client_->service_is_ready()) {
          callTrigger(
            deliver_order_client_, "/deliver_order",
            [this](bool ok, const std::string & msg) {
              if (!ok) {
                RCLCPP_WARN(
                  get_logger(),
                  "[NAV] Service /deliver_order gagal (%s), fallback ke koordinat internal...",
                  msg.c_str());
                double cx = 0.0, cy = 0.0, cyaw = 0.0;
                {
                  std::lock_guard<std::mutex> lock(customer_mutex_);
                  cx = current_customer_.map_x;
                  cy = current_customer_.map_y;
                  cyaw = current_customer_.map_yaw;
                }
                navigateToMapPoseAsync(cx, cy, cyaw, [this](bool success) {
                  if (!success) {
                    RCLCPP_WARN(get_logger(), "[NAV] Gagal navigasi fallback ke posisi pelanggan, mencoba ulang...");
                    return;
                  }
                  transitionTo(
                    RestaurantState::GREET_DELIVERY_CUSTOMER,
                    "Arrived at customer position for delivery (fallback)");
                });
                return;
              }
              transitionTo(
                RestaurantState::GREET_DELIVERY_CUSTOMER,
                "Arrived at customer position for delivery via /deliver_order");
            },
            nav_timeout_sec_);
          break;
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

      case RestaurantState::GREET_DELIVERY_CUSTOMER:
        speakAsync(
          "Hello, here is your order.",
          [this]() {
            transitionTo(
              RestaurantState::WAIT_1S_BEFORE_RELEASE_PROMPT,
              "Waiting 1 second before release prompt");
          });
        break;

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
          o1 = current_customer_.orders.empty() ? "this item" : current_customer_.orders[0];
        }
        RCLCPP_INFO(
          get_logger(),
          "\033[1;36m[ARM]\033[0m Raising arm via /arm/raising_arm...");
        callTrigger(
          arm_raising_arm_client_, "/arm/raising_arm",
          [this, o1](bool raise_ok, const std::string & raise_msg) {
            RCLCPP_INFO(
              get_logger(),
              "\033[1;34m[RESTAURANT]\033[0m\n  Arm: Raising arm (ok=%s): %s",
              raise_ok ? "true" : "false", raise_msg.c_str());
            speakAsync(
              "For " + o1 + ", please hold your hand under my gripper. Releasing on one, two, three.",
              [this]() {
                transitionTo(
                  RestaurantState::RELEASE_GRIPPER_AND_TORQUE_OFF,
                  "Releasing gripper and relaxing arm");
              });
          },
          20.0);
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
        std::vector<std::string> orders;
        {
          std::lock_guard<std::mutex> lock(customer_mutex_);
          orders = current_customer_.orders;
        }
        const std::string tray_str = formatTrayOrders(orders);
        const std::string pronoun = (orders.size() > 2) ? "them" : "it";
        speakAsync(
          "For " + tray_str + ", please take " + pronoun + " from my tray.",
          [this]() {
            transitionTo(
              RestaurantState::WAIT_5S_CUSTOMER_TAKE_TRAY,
              "Waiting 5 seconds for customer to take remaining items from tray");
          });
        break;
      }

      case RestaurantState::WAIT_5S_CUSTOMER_TAKE_TRAY:
        if ((now() - state_enter_time_).seconds() >= 5.0) {
          publishCustomerHmi(false);
          speakAsync("Enjoy your order, thank you.", [this]() {
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
            (now() - last_nav_fail_time_).seconds() < nav_retry_cooldown_sec_)
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
        publishCustomerHmi(false);
        speakAsync("Restaurant task complete.", [this]() {
          publishStatus("Restaurant task complete.");
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
  setenv("RCUTILS_CONSOLE_OUTPUT_FORMAT", "[{severity}] {message}", 1);
  rclcpp::init(argc, argv);
  auto node = std::make_shared<restaurant::RestaurantTaskNode>();

  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
