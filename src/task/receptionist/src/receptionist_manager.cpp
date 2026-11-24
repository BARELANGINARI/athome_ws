// ==============================================================================
// Barelang Inari - RoboCup@Home
// Package  : receptionist
// Node     : receptionist_manager (ReceptionistTaskNode)
//
// Overview:
//   Main Finite State Machine (FSM) orchestrating the RoboCup@Home Receptionist
//   task for two sequential guests (Guest 1 and Guest 2).
//
// High-Level Task Flow (per guest cycle):
//   1. Entrance Door : Navigate to entrance_door, wait/align to guest, capture
//                      name & interest via STT, confirm via Yes/No, and collect
//                      visual attributes (gender, shirt color, hair color).
//   2. Beverage Area : Navigate to beverage_area, snapshot visible drinks on
//                      the table, ask guest's favorite drink (with left/right
//                      rotation for HRI alignment), and report its table side.
//   3. Living Room   : Escort guest to living_room (describing seated guests on
//                      the way), scan/lock person & chair headings, introduce the
//                      new guest to the Host (and Guest 1 <-> Guest 2 on cycle 2),
//                      and announce shared interests.
//   4. Seat Offering : Search for an empty chair, point to it with the robotic
//                      arm, release arm torque (/arm/free), wait for the guest
//                      to sit, clear living-room locks, then loop or finish.
//
// Concurrency & Safety Model:
//   - Spun with MultiThreadedExecutor (2 threads): sensor/service callbacks use
//     a Reentrant callback group while fsmTick() runs on a 10 Hz timer (100 ms).
//   - Shared state is guarded by dedicated mutexes: guest_mutex_, vision_mutex_,
//     stt_mutex_, tts_mutex_, and srv_mutex_.
//   - Watchdogs in fsmTick() prevent hangs from missing /hri/tts_done events,
//     unresponsive Trigger services, or global task timeouts.
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
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"

using namespace std::chrono_literals;

// ==============================================================================
// 1. FINITE STATE MACHINE (FSM) STATE DEFINITIONS
// ==============================================================================
enum class ReceptionistState {
  // Startup: wait for required navigation & TTS endpoints, initialize Guest 1
  INIT,
  // Phase 1 — Entrance Door: wait for guest, align, scan attributes at 1.5m, approach to 0.7m, capture & confirm profile
  NAV_TO_DOOR,
  WAIT_GUEST,
  ALIGN_GUEST,
  SCAN_GUEST_ATTRIBUTES,
  APPROACH_GUEST,
  ASK_NAME,
  WAIT_NAME,
  ASK_INTEREST,
  WAIT_INTEREST,
  CONFIRM_GUEST_INFO,
  WAIT_CONFIRM_GUEST_INFO,
  // Phase 2 — Beverage Area: snapshot drinks, ask favorite drink, locate side
  NAV_TO_BEVERAGE,
  ASK_DRINK,
  WAIT_DRINK,
  CHECK_BEVERAGE,
  // Phase 3 & 4 — Living Room: scan room, introduce Host/Guests, check similarity
  NAV_TO_LIVING_ROOM,
  SCAN_LIVING_ROOM,
  FACE_AND_INTRODUCE_HOST,
  INTRODUCE_G1_TO_G2,
  INTRODUCE_G2_TO_G1,
  EVALUATE_SIMILARITY,
  RETURN_TO_LIVING_ROOM,
  // Phase 5 — Seat Offering: find empty seat, point with arm, confirm seated
  SEARCH_CHAIR,
  OFFER_SEAT,
  WAIT_GUEST_SIT,
  // Phase 6 — Terminal States
  FINISH_TASK,
  COMPLETED,
  ABORTED
};

/// Convert a ReceptionistState enum value to a human-readable string for logs.

static std::string stateToString(ReceptionistState state)
{
  switch (state) {
    case ReceptionistState::INIT:                    return "INIT";
    case ReceptionistState::NAV_TO_DOOR:             return "NAV_TO_DOOR";
    case ReceptionistState::WAIT_GUEST:              return "WAIT_GUEST";
    case ReceptionistState::ALIGN_GUEST:             return "ALIGN_GUEST";
    case ReceptionistState::SCAN_GUEST_ATTRIBUTES:   return "SCAN_GUEST_ATTRIBUTES";
    case ReceptionistState::APPROACH_GUEST:          return "APPROACH_GUEST";
    case ReceptionistState::ASK_NAME:                return "ASK_NAME";
    case ReceptionistState::WAIT_NAME:               return "WAIT_NAME";
    case ReceptionistState::ASK_INTEREST:            return "ASK_INTEREST";
    case ReceptionistState::WAIT_INTEREST:           return "WAIT_INTEREST";
    case ReceptionistState::CONFIRM_GUEST_INFO:      return "CONFIRM_GUEST_INFO";
    case ReceptionistState::WAIT_CONFIRM_GUEST_INFO: return "WAIT_CONFIRM_GUEST_INFO";
    case ReceptionistState::NAV_TO_BEVERAGE:         return "NAV_TO_BEVERAGE";
    case ReceptionistState::ASK_DRINK:               return "ASK_DRINK";
    case ReceptionistState::WAIT_DRINK:              return "WAIT_DRINK";
    case ReceptionistState::CHECK_BEVERAGE:          return "CHECK_BEVERAGE";
    case ReceptionistState::NAV_TO_LIVING_ROOM:      return "NAV_TO_LIVING_ROOM";
    case ReceptionistState::SCAN_LIVING_ROOM:        return "SCAN_LIVING_ROOM";
    case ReceptionistState::FACE_AND_INTRODUCE_HOST: return "FACE_AND_INTRODUCE_HOST";
    case ReceptionistState::INTRODUCE_G1_TO_G2:      return "INTRODUCE_G1_TO_G2";
    case ReceptionistState::INTRODUCE_G2_TO_G1:      return "INTRODUCE_G2_TO_G1";
    case ReceptionistState::EVALUATE_SIMILARITY:     return "EVALUATE_SIMILARITY";
    case ReceptionistState::RETURN_TO_LIVING_ROOM:   return "RETURN_TO_LIVING_ROOM";
    case ReceptionistState::SEARCH_CHAIR:            return "SEARCH_CHAIR";
    case ReceptionistState::OFFER_SEAT:              return "OFFER_SEAT";
    case ReceptionistState::WAIT_GUEST_SIT:          return "WAIT_GUEST_SIT";
    case ReceptionistState::FINISH_TASK:             return "FINISH_TASK";
    case ReceptionistState::COMPLETED:               return "COMPLETED";
    case ReceptionistState::ABORTED:                 return "ABORTED";
    default:                                         return "UNKNOWN";
  }
}

// ==============================================================================
// 2. DATA STRUCTURES (HOST / GUEST PROFILES & VISION DETECTIONS)
// ==============================================================================
/// Stores spoken preferences and fused visual appearance attributes for a person.
struct PersonProfile {
  std::string id;                              ///< Identifier ("host", "guest_1", "guest_2")
  std::string name;                            ///< Recognized guest/host name
  std::string drink;                           ///< Favorite beverage captured via STT
  std::string interest;                        ///< Main hobby/interest captured via STT
  std::string gender{"unknown"};               ///< "male", "female", or "unknown" (mode-voted)
  std::string shirt_color{"casual attire"};    ///< Dominant upper-body clothing description
  std::string hair_color{"dark hair"};         ///< Dominant hair color description
  std::vector<std::string> attributes;         ///< Combined attribute tags for logging/display
};

/// Represents a detected object (drink or fallback object) from vision topics.
struct DetectedObject {
  std::string name;  ///< Raw class label from YOLO / vision detector
  int bbox_x{0};     ///< 2D bounding box top-left X in image pixels
  int bbox_y{0};     ///< 2D bounding box top-left Y in image pixels
  int bbox_w{0};     ///< 2D bounding box width in image pixels
  int bbox_h{0};     ///< 2D bounding box height in image pixels
  double x{0.0};     ///< 3D X coordinate relative to vision frame (meters)
  double y{0.0};     ///< 3D Y coordinate relative to vision frame (meters)
  double z{0.0};     ///< 3D Z coordinate / depth (meters)
};

// ==============================================================================
// 3. MAIN ROS 2 NODE: ReceptionistTaskNode
// ==============================================================================
/// Coordinates Receptionist navigation, perception, HRI speech, and arm actions.
class ReceptionistTaskNode : public rclcpp::Node {
public:
  using Trigger = std_srvs::srv::Trigger;
  using TriggerClient = rclcpp::Client<Trigger>;
  using TriggerResult = std::function<void(bool, const std::string &)>;

  ReceptionistTaskNode()
  : Node("receptionist_manager")
  {
    // Parameter Host & Konfigurasi Task (WAJIB dari file YAML receptionist_params.yaml, read-only)
    host_.id = "host";
    host_.name = requireYamlParameter<std::string>("host.name");
    host_.drink = requireYamlParameter<std::string>("host.drink");
    host_.interest = requireYamlParameter<std::string>("host.interest");
    host_.gender = requireYamlParameter<std::string>("host.gender");
    host_.shirt_color = requireYamlParameter<std::string>("host.shirt_color");
    host_.hair_color = requireYamlParameter<std::string>("host.hair_color");
    host_.attributes = {
      host_.gender,
      host_.shirt_color + " shirt",
      host_.hair_color + " hair"
    };
    camera_width_ = requireYamlParameter<int>("camera_width");

    global_timeout_sec_ = requireYamlParameter<double>("global_timeout_sec");
    dialog_reprompt_sec_ = requireYamlParameter<double>("dialog_reprompt_sec");
    wait_guest_prompt_sec_ = requireYamlParameter<double>("wait_guest_prompt_sec");
    vision_initial_settle_sec_ = requireYamlParameter<double>("vision_initial_settle_sec");
    min_sit_wait_sec_ = requireYamlParameter<double>("min_sit_wait_sec");
    person_freshness_sec_ = requireYamlParameter<double>("person_freshness_sec");
    tts_min_timeout_sec_ = requireYamlParameter<double>("tts_min_timeout_sec");
    tts_sec_per_word_ = requireYamlParameter<double>("tts_sec_per_word");
    tts_timeout_margin_sec_ = requireYamlParameter<double>("tts_timeout_margin_sec");
    nav_retry_cooldown_sec_ = requireYamlParameter<double>("nav_retry_cooldown_sec");
    tts_timeout_sec_ = tts_min_timeout_sec_;

    callback_group_ = create_callback_group(
      rclcpp::CallbackGroupType::Reentrant);

    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.callback_group = callback_group_;

    // --------------------------------------------------------------------------
    // Publisher & Subscriber HRI (TTS, STT, & Task Status)
    // --------------------------------------------------------------------------
    tts_pub_ = create_publisher<std_msgs::msg::String>("/hri/tts_say", 10);
    status_pub_ = create_publisher<std_msgs::msg::String>("/receptionist/status", 10);
    person_name_pub_ = create_publisher<std_msgs::msg::String>("/vision/set_person_name", 10);
    guest_1_name_pub_ = create_publisher<std_msgs::msg::String>("/receptionist/guest_1_name", 10);
    guest_2_name_pub_ = create_publisher<std_msgs::msg::String>("/receptionist/guest_2_name", 10);
    face_expression_pub_ = create_publisher<std_msgs::msg::String>("/robot/expression", 10);

    tts_done_sub_ = create_subscription<std_msgs::msg::String>(
      "/hri/tts_done", 10,
      std::bind(&ReceptionistTaskNode::onTtsDone, this, std::placeholders::_1),
      sub_opts);

    stt_sub_ = create_subscription<std_msgs::msg::String>(
      "/hri/stt_transcript", 10,
      std::bind(&ReceptionistTaskNode::onTranscript, this, std::placeholders::_1),
      sub_opts);

    // --------------------------------------------------------------------------
    // Subscriber Vision (Static RGB-D ZED Camera)
    // --------------------------------------------------------------------------
    person_sub_ = create_subscription<std_msgs::msg::String>(
      "/vision/person_label", rclcpp::SensorDataQoS(),
      std::bind(&ReceptionistTaskNode::onPersonLabel, this, std::placeholders::_1),
      sub_opts);

    drink_sub_ = create_subscription<std_msgs::msg::String>(
      "/vision/drink_detections", rclcpp::SensorDataQoS(),
      std::bind(&ReceptionistTaskNode::onDrinkDetections, this, std::placeholders::_1),
      sub_opts);

    // Fallback ke /vision/bag_detections (tidak menghapus latest_drinks_ jika kosong)
    object_sub_ = create_subscription<std_msgs::msg::String>(
      "/vision/bag_detections", rclcpp::SensorDataQoS(),
      std::bind(&ReceptionistTaskNode::onBagDetectionsFallback, this, std::placeholders::_1),
      sub_opts);

    attributes_sub_ = create_subscription<std_msgs::msg::String>(
      "/vision/person_attributes", 10,
      std::bind(&ReceptionistTaskNode::onPersonAttributes, this, std::placeholders::_1),
      sub_opts);

    seat_sub_ = create_subscription<std_msgs::msg::String>(
      "/vision/empty_seat", 10,
      std::bind(&ReceptionistTaskNode::onEmptySeat, this, std::placeholders::_1),
      sub_opts);

    // --------------------------------------------------------------------------
    // Service Clients: Navigation (receptionist_node) & Arm (athome_arm)
    // --------------------------------------------------------------------------
    nav_door_client_ = create_client<Trigger>(
      "/receptionist/go_to_door", rclcpp::ServicesQoS(), callback_group_);
    nav_beverage_client_ = create_client<Trigger>(
      "/receptionist/go_to_beverage_area", rclcpp::ServicesQoS(), callback_group_);
    nav_living_room_client_ = create_client<Trigger>(
      "/receptionist/go_to_living_room", rclcpp::ServicesQoS(), callback_group_);

    search_chair_client_ = create_client<Trigger>(
      "/search_chair", rclcpp::ServicesQoS(), callback_group_);
    face_host_client_ = create_client<Trigger>(
      "/receptionist/host", rclcpp::ServicesQoS(), callback_group_);
    face_guest1_client_ = create_client<Trigger>(
      "/receptionist/guest1", rclcpp::ServicesQoS(), callback_group_);
    adjust_guest_client_ = create_client<Trigger>(
      "/receptionist/adjust_guest", rclcpp::ServicesQoS(), callback_group_);
    approach_guest_client_ = create_client<Trigger>(
      "/receptionist/approach_guest", rclcpp::ServicesQoS(), callback_group_);
    rotate_left_client_ = create_client<Trigger>(
      "/receptionist/left", rclcpp::ServicesQoS(), callback_group_);
    rotate_right_client_ = create_client<Trigger>(
      "/receptionist/right", rclcpp::ServicesQoS(), callback_group_);
    scan_living_room_client_ = create_client<Trigger>(
      "/receptionist/scan_living_room", rclcpp::ServicesQoS(), callback_group_);
    clear_locks_client_ = create_client<Trigger>(
      "/receptionist/clear_living_room_locks", rclcpp::ServicesQoS(), callback_group_);
    reset_living_room_yaw_client_ = create_client<Trigger>(
      "/receptionist/reset_living_room_yaw", rclcpp::ServicesQoS(), callback_group_);

    arm_ambil_tas_client_ = create_client<Trigger>(
      "/arm/ambil_tas", rclcpp::ServicesQoS(), callback_group_);
    arm_ambil_tas_simple_client_ = create_client<Trigger>(
      "/arm/ambil_tas_simple", rclcpp::ServicesQoS(), callback_group_);

    arm_point_position_client_ = create_client<Trigger>(
      "/arm/point_position", rclcpp::ServicesQoS(), callback_group_);
    arm_point_to_chair_client_ = create_client<Trigger>(
      "/arm/point_to_chair", rclcpp::ServicesQoS(), callback_group_);
    arm_free_client_ = create_client<Trigger>(
      "/arm/free", rclcpp::ServicesQoS(), callback_group_);

    // Timer siklus FSM (10 Hz / 100 ms)
    fsm_timer_ = create_wall_timer(
      100ms, std::bind(&ReceptionistTaskNode::fsmTick, this));

    task_start_time_ = now();
    state_enter_time_ = now();
    last_wait_prompt_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);

    guests_.clear();
    initCurrentGuest(1);

    RCLCPP_INFO(get_logger(),
      "\033[1;32m[RECEPTIONIST]\033[0m Task Manager started (Host: %s, Drink: %s)",
      host_.name.c_str(), host_.drink.c_str());
  }

private:
  /// Declares a read-only parameter without a C++ default so its value must be
  /// supplied exclusively via YAML (receptionist_params.yaml).
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
        "Parameter '%s' wajib diatur di file YAML (receptionist_params.yaml)! Error: %s",
        name.c_str(), e.what());
      throw;
    }
  }

  // ----------------------------------------------------------------------------
  // Parameter Waktu & Timeout (Diisi penuh dari YAML saat startup)
  // ----------------------------------------------------------------------------
  double global_timeout_sec_{0.0};
  double dialog_reprompt_sec_{0.0};
  double wait_guest_prompt_sec_{0.0};
  double vision_initial_settle_sec_{0.0};
  double min_sit_wait_sec_{0.0};
  double person_freshness_sec_{0.0};
  double tts_min_timeout_sec_{0.0};
  double tts_sec_per_word_{0.0};
  double tts_timeout_margin_sec_{0.0};
  double nav_retry_cooldown_sec_{0.0};

  // ----------------------------------------------------------------------------
  // Variabel State & Memory Store Host / Guests
  // ----------------------------------------------------------------------------
  std::atomic<ReceptionistState> state_{ReceptionistState::INIT};
  std::atomic<bool> speaking_{false};
  std::atomic<bool> service_busy_{false};
  std::atomic<bool> in_callback_{false};
  std::atomic<bool> person_seen_{false};
  std::atomic<int> nearest_person_id_{-1};

  std::mutex guest_mutex_;
  PersonProfile host_;
  std::vector<PersonProfile> guests_;
  PersonProfile current_guest_;
  int guest_idx_{1};
  ReceptionistState after_return_living_room_state_{ReceptionistState::SEARCH_CHAIR};
  bool return_yaw_only_{false};
  int camera_width_{0};

  // Data Persepsi Vision
  std::mutex vision_mutex_;
  std::vector<DetectedObject> latest_drinks_;
  rclcpp::Time last_drink_msg_time_{0, 0, RCL_ROS_TIME};
  std::vector<DetectedObject> table_drinks_snapshot_;
  bool drinks_detected_on_arrival_{false};
  bool beverage_snapshot_locked_{false};
  std::vector<std::string> latest_attributes_;
  std::unordered_map<std::string, int> attr_gender_counts_;
  std::unordered_map<std::string, int> attr_shirt_counts_;
  std::unordered_map<std::string, int> attr_hair_counts_;
  bool attributes_locked_from_scan_{false};
  std::string active_vision_reg_name_;
  std::string empty_seat_position_{"center"};
  bool has_explicit_empty_seat_{false};
  int nearest_person_bbox_x_{0};
  int nearest_person_bbox_w_{0};
  int nearest_person_bbox_y_{0};
  int nearest_person_bbox_h_{0};

  // Data STT / Dialog
  std::mutex stt_mutex_;
  std::string heard_utterance_;
  std::atomic<bool> has_new_utterance_{false};
  std::string last_utterance_;
  rclcpp::Time last_utterance_time_{0, 0, RCL_ROS_TIME};

  // Timing, TTS, & Service Watchdog
  rclcpp::Time task_start_time_;
  rclcpp::Time state_enter_time_;
  rclcpp::Time last_wait_prompt_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_person_seen_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time tts_cooldown_until_{0, 0, RCL_ROS_TIME};
  rclcpp::Time tts_started_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_nav_fail_time_{0, 0, RCL_ROS_TIME};

  std::mutex tts_mutex_;
  double tts_timeout_sec_{4.0};
  std::function<void()> pending_tts_done_;

  std::mutex srv_mutex_;
  rclcpp::Time srv_started_time_{0, 0, RCL_ROS_TIME};
  double srv_timeout_sec_{120.0};
  std::string active_srv_name_;
  TriggerResult pending_srv_result_;

  // Interfaces ROS 2
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp::TimerBase::SharedPtr fsm_timer_;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr tts_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr person_name_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr guest_1_name_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr guest_2_name_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr face_expression_pub_;

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr tts_done_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr stt_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr person_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr drink_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr object_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr attributes_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr seat_sub_;

  TriggerClient::SharedPtr nav_door_client_;
  TriggerClient::SharedPtr nav_beverage_client_;
  TriggerClient::SharedPtr nav_living_room_client_;
  TriggerClient::SharedPtr search_chair_client_;
  TriggerClient::SharedPtr face_host_client_;
  TriggerClient::SharedPtr face_guest1_client_;
  TriggerClient::SharedPtr adjust_guest_client_;
  TriggerClient::SharedPtr approach_guest_client_;
  TriggerClient::SharedPtr rotate_left_client_;
  TriggerClient::SharedPtr rotate_right_client_;
  TriggerClient::SharedPtr scan_living_room_client_;
  TriggerClient::SharedPtr clear_locks_client_;
  TriggerClient::SharedPtr reset_living_room_yaw_client_;

  TriggerClient::SharedPtr arm_ambil_tas_client_;
  TriggerClient::SharedPtr arm_ambil_tas_simple_client_;
  TriggerClient::SharedPtr arm_point_position_client_;
  TriggerClient::SharedPtr arm_point_to_chair_client_;
  TriggerClient::SharedPtr arm_free_client_;

  // ============================================================================
  // SUB-ROUTINES & CONCURRENCY GATES
  // ============================================================================
  /// Returns true while TTS, an async service call, or its completion callback
  /// is active so fsmTick() does not re-trigger the current state prematurely.
  bool isBusy() const
  {
    return speaking_.load() || service_busy_.load() || in_callback_.load();
  }

  /// Ignore vision updates during navigation and terminal states so objects or
  /// bystanders passed in transit never pollute guest/table/seat caches.
  bool isVisionActive() const
  {
    const ReceptionistState current = state_.load();
    return current != ReceptionistState::INIT &&
           current != ReceptionistState::NAV_TO_DOOR &&
           current != ReceptionistState::NAV_TO_BEVERAGE &&
           current != ReceptionistState::NAV_TO_LIVING_ROOM &&
           current != ReceptionistState::RETURN_TO_LIVING_ROOM &&
           current != ReceptionistState::COMPLETED &&
           current != ReceptionistState::ABORTED;
  }

  void publishTaskStatus(const std::string & status_text)
  {
    if (status_pub_) {
      std_msgs::msg::String msg;
      msg.data = status_text;
      status_pub_->publish(msg);
    }
    RCLCPP_INFO(get_logger(), "\033[1;36m[RECEPTIONIST]\033[0m\n  Status: %s", status_text.c_str());
  }

  /// Publishes a face expression command to /robot/expression (athome_face).
  void publishExpression(const std::string & expr)
  {
    if (face_expression_pub_) {
      std_msgs::msg::String msg;
      msg.data = expr;
      face_expression_pub_->publish(msg);
      RCLCPP_INFO(get_logger(), "\033[1;36m[RECEPTIONIST]\033[0m\n  Face: Expression -> %s", expr.c_str());
    }
  }

  /// Maps each FSM state to its default /robot/expression state.
  static std::string expressionForState(ReceptionistState state)
  {
    switch (state) {
      case ReceptionistState::INIT:
      case ReceptionistState::COMPLETED:
        return "IDLE";
      case ReceptionistState::NAV_TO_DOOR:
      case ReceptionistState::NAV_TO_BEVERAGE:
      case ReceptionistState::NAV_TO_LIVING_ROOM:
      case ReceptionistState::RETURN_TO_LIVING_ROOM:
        return "FOLLOWING";
      case ReceptionistState::WAIT_GUEST:
      case ReceptionistState::SCAN_GUEST_ATTRIBUTES:
      case ReceptionistState::CHECK_BEVERAGE:
      case ReceptionistState::SCAN_LIVING_ROOM:
      case ReceptionistState::SEARCH_CHAIR:
        return "SEARCHING";
      case ReceptionistState::ALIGN_GUEST:
      case ReceptionistState::APPROACH_GUEST:
      case ReceptionistState::ASK_NAME:
      case ReceptionistState::ASK_INTEREST:
      case ReceptionistState::CONFIRM_GUEST_INFO:
      case ReceptionistState::ASK_DRINK:
      case ReceptionistState::FACE_AND_INTRODUCE_HOST:
      case ReceptionistState::INTRODUCE_G1_TO_G2:
      case ReceptionistState::INTRODUCE_G2_TO_G1:
        return "GREETING";
      case ReceptionistState::WAIT_NAME:
      case ReceptionistState::WAIT_INTEREST:
      case ReceptionistState::WAIT_CONFIRM_GUEST_INFO:
      case ReceptionistState::WAIT_DRINK:
        return "LISTENING";
      case ReceptionistState::EVALUATE_SIMILARITY:
        return "HAPPY";
      case ReceptionistState::OFFER_SEAT:
        return "WORKING";
      case ReceptionistState::WAIT_GUEST_SIT:
        return "CURIOUS";
      case ReceptionistState::FINISH_TASK:
        return "SUCCESS";
      case ReceptionistState::ABORTED:
        return "LOST";
      default:
        return "IDLE";
    }
  }

  /// Publishes registered guest names at 2 Hz for HMI/vision synchronization.
  void publishGuestNames()
  {
    std::string g1;
    std::string g2;
    {
      std::lock_guard<std::mutex> lock(guest_mutex_);
      if (guests_.size() >= 1 && !guests_[0].name.empty()) {
        g1 = guests_[0].name;
      } else if (guest_idx_ == 1 && !current_guest_.name.empty()) {
        g1 = current_guest_.name;
      }

      if (guests_.size() >= 2 && !guests_[1].name.empty()) {
        g2 = guests_[1].name;
      } else if (guest_idx_ == 2 && !current_guest_.name.empty()) {
        g2 = current_guest_.name;
      }
    }

    if (guest_1_name_pub_) {
      std_msgs::msg::String msg;
      msg.data = g1;
      guest_1_name_pub_->publish(msg);
    }
    if (guest_2_name_pub_) {
      std_msgs::msg::String msg;
      msg.data = g2;
      guest_2_name_pub_->publish(msg);
    }
  }

  /// Resets all profile fields to neutral defaults for guest slot `idx`.
  void resetGuestProfileFields(PersonProfile & profile, int idx)
  {
    profile = PersonProfile{};
    profile.id = "guest_" + std::to_string(idx);
    profile.gender = "unknown";
    profile.shirt_color = "casual attire";
    profile.hair_color = "dark hair";
  }

  /// Initializes state, vision caches, and attribute voting histograms for a new guest cycle (1 or 2).
  void initCurrentGuest(int idx)
  {
    {
      std::lock_guard<std::mutex> lock(guest_mutex_);
      guest_idx_ = idx;
      resetGuestProfileFields(current_guest_, idx);
    }

    if (person_name_pub_) {
      std_msgs::msg::String m;
      m.data = "";
      person_name_pub_->publish(m);
    }

    person_seen_ = false;
    drinks_detected_on_arrival_ = false;
    {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      table_drinks_snapshot_.clear();
      beverage_snapshot_locked_ = false;
      latest_drinks_.clear();
      latest_attributes_.clear();
      attr_gender_counts_.clear();
      attr_shirt_counts_.clear();
      attr_hair_counts_.clear();
      attributes_locked_from_scan_ = false;
      active_vision_reg_name_.clear();
      has_explicit_empty_seat_ = false;
      nearest_person_bbox_w_ = 0;
    }
  }

  /// Transitions the FSM to `next_state`, clears stale STT flags, and resets
  /// state-specific vision caches when entering observation states.
  void transitionTo(ReceptionistState next_state, const std::string & reason)
  {
    RCLCPP_INFO(
      get_logger(),
      "\033[1;34m[STATE]\033[0m %s -> %s (G%d: %s)",
      stateToString(state_.load()).c_str(),
      stateToString(next_state).c_str(),
      guest_idx_,
      reason.c_str());

    // Reset cache vision saat memasuki state yang membutuhkan pengamatan baru
    if (next_state == ReceptionistState::WAIT_GUEST ||
        next_state == ReceptionistState::WAIT_GUEST_SIT ||
        next_state == ReceptionistState::ASK_DRINK ||
        next_state == ReceptionistState::OFFER_SEAT ||
        next_state == ReceptionistState::INTRODUCE_G1_TO_G2)
    {
      person_seen_ = false;
      last_wait_prompt_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
      std::lock_guard<std::mutex> lock(vision_mutex_);
      if (next_state == ReceptionistState::ASK_DRINK) {
        table_drinks_snapshot_.clear();
        drinks_detected_on_arrival_ = false;
        beverage_snapshot_locked_ = false;
        latest_drinks_.clear();
      }
      if (next_state == ReceptionistState::OFFER_SEAT) {
        has_explicit_empty_seat_ = false;
      }
    }

    state_enter_time_ = now();
    has_new_utterance_ = false;
    state_ = next_state;
    publishExpression(expressionForState(next_state));
  }

  /// Navigates to a named waypoint ("entrance_door", "beverage_area", or "living_room")
  /// via the corresponding receptionist_node Trigger service and starts nav retry cooldown on failure.
  void navigateTo(
    const std::string & waypoint_name,
    std::function<void(bool)> on_finished)
  {
    TriggerClient::SharedPtr client;
    std::string srv_name;

    if (waypoint_name == "entrance_door" || waypoint_name == "door") {
      client = nav_door_client_;
      srv_name = "/receptionist/go_to_door";
    } else if (waypoint_name == "beverage_area") {
      client = nav_beverage_client_;
      srv_name = "/receptionist/go_to_beverage_area";
    } else {
      client = nav_living_room_client_;
      srv_name = "/receptionist/go_to_living_room";
    }

    RCLCPP_INFO(get_logger(), "\033[1;36m[RECEPTIONIST]\033[0m\n  Nav: Navigating to '%s' (%s)",
      waypoint_name.c_str(), srv_name.c_str());

    const rclcpp::Time nav_start = now();

    callTrigger(
      client, srv_name,
      [this, waypoint_name, srv_name, nav_start, on_finished](bool success, const std::string & msg) {
        const double elapsed = (now() - nav_start).seconds();
        if (!success) {
          RCLCPP_ERROR(
            get_logger(),
            "\033[1;31m[NAV]\033[0m Navigation to '%s' FAILED (%s)",
            waypoint_name.c_str(), msg.c_str());
          last_nav_fail_time_ = now();
        } else {
          if (elapsed < 0.5) {
            RCLCPP_WARN(
              get_logger(),
              "\033[1;33m[NAV]\033[0m Service %s returned TRUE in %.2fs (< 0.5s)",
              srv_name.c_str(), elapsed);
          }
          RCLCPP_INFO(
            get_logger(),
            "\033[1;32m[NAV]\033[0m Reached '%s' in %.1fs",
            waypoint_name.c_str(), elapsed);
        }
        if (on_finished) {
          on_finished(success);
        }
      },
      120.0);
  }

  /// Points toward the empty seat using /arm/point_to_chair, falling back to
  /// /arm/point_position or /arm/ambil_tas_simple if the primary service is unavailable.
  void pointToSeatWithArm(std::function<void(bool)> on_finished)
  {
    TriggerClient::SharedPtr client = arm_point_to_chair_client_;
    std::string srv_name = "/arm/point_to_chair";

    if ((!client || !client->service_is_ready()) &&
        arm_point_position_client_ && arm_point_position_client_->service_is_ready())
    {
      client = arm_point_position_client_;
      srv_name = "/arm/point_position";
    } else if ((!client || !client->service_is_ready()) &&
        arm_ambil_tas_simple_client_ && arm_ambil_tas_simple_client_->service_is_ready())
    {
      client = arm_ambil_tas_simple_client_;
      srv_name = "/arm/ambil_tas_simple";
    }

    RCLCPP_INFO(get_logger(), "\033[1;36m[RECEPTIONIST]\033[0m\n  Arm: Point seat (%s)", srv_name.c_str());

    callTrigger(
      client, srv_name,
      [this, srv_name, on_finished](bool success, const std::string & msg) {
        if (success) {
          RCLCPP_INFO(
            get_logger(),
            "\033[1;36m[RECEPTIONIST]\033[0m\n  Arm: Point seat done");
        } else {
          RCLCPP_WARN(
            get_logger(),
            "\033[1;33m[ARM]\033[0m Point seat warning: %s",
            msg.c_str());
        }
        if (on_finished) {
          on_finished(success);
        }
      },
      15.0);
  }

  // Mematikan daya / torsi servo lengan robot (mode free)
  void freeArm(std::function<void(bool)> on_finished = nullptr)
  {
    TriggerClient::SharedPtr client = arm_free_client_;
    std::string srv_name = "/arm/free";

    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[ARM]\033[0m Releasing arm torque (free mode) via %s...",
      srv_name.c_str());

    callTrigger(
      client, srv_name,
      [this, srv_name, on_finished](bool success, const std::string & msg) {
        if (success) {
          RCLCPP_INFO(
            get_logger(),
            "\033[1;32m[ARM]\033[0m Arm free mode succeeded: %s", msg.c_str());
        } else {
          RCLCPP_WARN(
            get_logger(),
            "\033[1;33m[ARM]\033[0m Arm free mode finished with warning: %s", msg.c_str());
        }
        if (on_finished) {
          on_finished(success);
        }
      },
      5.0);
  }

  // Mencari kursi kosong dan menyelaraskan arah hadap robot via /search_chair
  void searchChair(std::function<void(bool, const std::string &)> on_finished)
  {
    TriggerClient::SharedPtr client = search_chair_client_;
    std::string srv_name = "/search_chair";

    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[CHAIR]\033[0m Calling %s to search and align to empty chair...",
      srv_name.c_str());

    callTrigger(
      client, srv_name,
      [this, srv_name, on_finished](bool success, const std::string & msg) {
        if (success) {
          RCLCPP_INFO(
            get_logger(),
            "\033[1;32m[CHAIR]\033[0m Search chair (%s) succeeded: %s",
            srv_name.c_str(), msg.c_str());
        } else {
          RCLCPP_WARN(
            get_logger(),
            "\033[1;33m[CHAIR]\033[0m Search chair (%s) finished with warning/failure: %s",
            srv_name.c_str(), msg.c_str());
        }
        if (on_finished) {
          on_finished(success, msg);
        }
      },
      45.0);
  }

  // Mencari dan menyelaraskan arah hadap robot ke Host via /receptionist/host
  void faceHost(std::function<void(bool, const std::string &)> on_finished)
  {
    TriggerClient::SharedPtr client = face_host_client_;
    std::string srv_name = "/receptionist/host";

    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[HOST]\033[0m Calling %s to search and align facing Host...",
      srv_name.c_str());

    callTrigger(
      client, srv_name,
      [this, srv_name, on_finished](bool success, const std::string & msg) {
        if (success) {
          RCLCPP_INFO(
            get_logger(),
            "\033[1;32m[HOST]\033[0m Face Host (%s) succeeded: %s",
            srv_name.c_str(), msg.c_str());
        } else {
          RCLCPP_WARN(
            get_logger(),
            "\033[1;33m[HOST]\033[0m Face Host (%s) finished with warning/failure: %s",
            srv_name.c_str(), msg.c_str());
        }
        if (on_finished) {
          on_finished(success, msg);
        }
      },
      35.0);
  }

  // Mencari dan menyelaraskan arah hadap robot ke Guest 1 via /receptionist/guest1
  void faceGuest1(std::function<void(bool, const std::string &)> on_finished)
  {
    TriggerClient::SharedPtr client = face_guest1_client_;
    std::string srv_name = "/receptionist/guest1";

    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[GUEST 1]\033[0m Calling %s to search and align facing Guest 1...",
      srv_name.c_str());

    callTrigger(
      client, srv_name,
      [this, srv_name, on_finished](bool success, const std::string & msg) {
        if (success) {
          RCLCPP_INFO(
            get_logger(),
            "\033[1;32m[GUEST 1]\033[0m Face Guest 1 (%s) succeeded: %s",
            srv_name.c_str(), msg.c_str());
        } else {
          RCLCPP_WARN(
            get_logger(),
            "\033[1;33m[GUEST 1]\033[0m Face Guest 1 (%s) finished with warning/failure: %s",
            srv_name.c_str(), msg.c_str());
        }
        if (on_finished) {
          on_finished(success, msg);
        }
      },
      35.0);
  }

  // Memindai ruang tamu saat tiba dan mengunci sudut yaw Host / Guest 1 / Kursi Kosong
  void scanLivingRoom(std::function<void(bool, const std::string &)> on_finished)
  {
    const std::string srv_name = "/receptionist/scan_living_room";
    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[LIVING ROOM]\033[0m Scanning living room via %s to lock person & seat positions...",
      srv_name.c_str());

    callTrigger(
      scan_living_room_client_, srv_name,
      [this, on_finished](bool success, const std::string & msg) {
        if (success) {
          RCLCPP_INFO(get_logger(), "\033[1;32m[LIVING ROOM]\033[0m Lock ready: %s", msg.c_str());
        } else {
          RCLCPP_WARN(get_logger(), "\033[1;33m[LIVING ROOM]\033[0m Scan lock warning: %s", msg.c_str());
        }
        if (on_finished) on_finished(success, msg);
      },
      25.0);
  }

  // Menghapus kunci posisi sementara ruang tamu setelah tamu mendapat kursi
  void clearLivingRoomLocks(std::function<void()> on_finished)
  {
    const std::string srv_name = "/receptionist/clear_living_room_locks";
    callTrigger(
      clear_locks_client_, srv_name,
      [this, on_finished](bool /*success*/, const std::string & msg) {
        RCLCPP_INFO(get_logger(), "\033[1;36m[LIVING ROOM]\033[0m %s", msg.c_str());
        if (on_finished) on_finished();
      },
      5.0);
  }

  // Memutar robot di tempat kembali ke yaw living_room via /receptionist/reset_living_room_yaw
  void resetLivingRoomYaw(std::function<void(bool, const std::string &)> on_finished)
  {
    const std::string srv_name = "/receptionist/reset_living_room_yaw";
    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[LIVING ROOM]\033[0m Calling %s to reset robot yaw in-place...",
      srv_name.c_str());

    callTrigger(
      reset_living_room_yaw_client_, srv_name,
      [this, srv_name, on_finished](bool success, const std::string & msg) {
        if (success) {
          RCLCPP_INFO(
            get_logger(),
            "\033[1;32m[LIVING ROOM]\033[0m Reset yaw (%s) succeeded: %s",
            srv_name.c_str(), msg.c_str());
        } else {
          RCLCPP_WARN(
            get_logger(),
            "\033[1;33m[LIVING ROOM]\033[0m Reset yaw (%s) warning/failure: %s",
            srv_name.c_str(), msg.c_str());
        }
        if (on_finished) {
          on_finished(success, msg);
        }
      },
      10.0);
  }

  // Merotasi robot ke kiri via /receptionist/left
  void rotateLeft(std::function<void(bool, const std::string &)> on_finished)
  {
    TriggerClient::SharedPtr client = rotate_left_client_;
    std::string srv_name = "/receptionist/left";

    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[ROTATE]\033[0m Calling %s to rotate robot left...",
      srv_name.c_str());

    callTrigger(
      client, srv_name,
      [this, srv_name, on_finished](bool success, const std::string & msg) {
        if (success) {
          RCLCPP_INFO(
            get_logger(),
            "\033[1;32m[ROTATE]\033[0m %s succeeded: %s",
            srv_name.c_str(), msg.c_str());
        } else {
          RCLCPP_WARN(
            get_logger(),
            "\033[1;33m[ROTATE]\033[0m %s finished with warning/failure: %s",
            srv_name.c_str(), msg.c_str());
        }
        if (on_finished) {
          on_finished(success, msg);
        }
      },
      20.0);
  }

  // Merotasi robot kembali ke kanan via /receptionist/right
  void rotateRight(std::function<void(bool, const std::string &)> on_finished)
  {
    TriggerClient::SharedPtr client = rotate_right_client_;
    std::string srv_name = "/receptionist/right";

    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[ROTATE]\033[0m Calling %s to rotate robot right...",
      srv_name.c_str());

    callTrigger(
      client, srv_name,
      [this, srv_name, on_finished](bool success, const std::string & msg) {
        if (success) {
          RCLCPP_INFO(
            get_logger(),
            "\033[1;32m[ROTATE]\033[0m %s succeeded: %s",
            srv_name.c_str(), msg.c_str());
        } else {
          RCLCPP_WARN(
            get_logger(),
            "\033[1;33m[ROTATE]\033[0m %s finished with warning/failure: %s",
            srv_name.c_str(), msg.c_str());
        }
        if (on_finished) {
          on_finished(success, msg);
        }
      },
      20.0);
  }

  // Menyelaraskan arah hadap robot ke tamu di pintu dan mundur hingga jarak 1.5m
  void adjustGuestDistance(std::function<void(bool, const std::string &)> on_finished)
  {
    TriggerClient::SharedPtr client = adjust_guest_client_;
    std::string srv_name = "/receptionist/adjust_guest";

    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[GUEST]\033[0m Calling %s to align heading and adjust distance to guest...",
      srv_name.c_str());

    callTrigger(
      client, srv_name,
      [this, srv_name, on_finished](bool success, const std::string & msg) {
        if (success) {
          RCLCPP_INFO(
            get_logger(),
            "\033[1;32m[GUEST]\033[0m Adjust guest distance succeeded: %s",
            msg.c_str());
        } else {
          RCLCPP_WARN(
            get_logger(),
            "\033[1;33m[GUEST]\033[0m Adjust guest distance finished with warning/failure: %s",
            msg.c_str());
        }
        if (on_finished) {
          on_finished(success, msg);
        }
      },
      25.0);
  }

  // Mendekat kembali ke tamu hingga jarak 0.7m untuk dialog suara (STT) & face recognition
  void approachGuest(std::function<void(bool, const std::string &)> on_finished)
  {
    TriggerClient::SharedPtr client = approach_guest_client_;
    std::string srv_name = "/receptionist/approach_guest";

    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[GUEST]\033[0m Calling %s to approach guest to 0.7m...",
      srv_name.c_str());

    callTrigger(
      client, srv_name,
      [this, srv_name, on_finished](bool success, const std::string & msg) {
        if (success) {
          RCLCPP_INFO(
            get_logger(),
            "\033[1;32m[GUEST]\033[0m Approach guest succeeded: %s",
            msg.c_str());
        } else {
          RCLCPP_WARN(
            get_logger(),
            "\033[1;33m[GUEST]\033[0m Approach guest finished with warning/failure: %s",
            msg.c_str());
        }
        if (on_finished) {
          on_finished(success, msg);
        }
      },
      25.0);
  }

  // Menentukan posisi horizontal (left, center, right) pada frame kamera statis
  std::string calculateHorizontalPosition(int bbox_x, int bbox_w) const
  {
    const double center_x = static_cast<double>(bbox_x) + static_cast<double>(bbox_w) * 0.5;
    const double w = (camera_width_ > 0) ? static_cast<double>(camera_width_) : 672.0;
    const double mid = w * 0.5;
    const double deadband = w * 0.12; // ±12% dari garis tengah kamera (~80 px pada 672px)
    if (center_x < mid - deadband) {
      return "left";
    } else if (center_x > mid + deadband) {
      return "right";
    }
    return "center";
  }

  /// Maps brand names and synonyms (e.g., "coke", "pepsi", "mineral water") to
  /// canonical YOLO detector labels ("soda", "water") for matching table drinks.
  static std::string canonicalDrinkName(const std::string & raw_name)
  {
    const std::string n = normalize(raw_name);
    if (n == "coke" || n == "cola" || n == "pepsi" || n == "sprite" || n == "fanta" || n == "soda") {
      return "soda";
    }
    if (n == "water" || n == "mineral water" || n == "mineral_water" ||
        n == "botol" || n == "bottle")
    {
      return "water";
    }
    return n;
  }

  void mergeDrinksIntoSnapshotLocked(const std::vector<DetectedObject> & items)
  {
    for (const auto & item : items) {
      const std::string canon = canonicalDrinkName(item.name);
      if (canon.empty()) continue;
      bool updated = false;
      for (auto & existing : table_drinks_snapshot_) {
        if (canonicalDrinkName(existing.name) == canon) {
          existing = item;
          updated = true;
          break;
        }
      }
      if (!updated) {
        table_drinks_snapshot_.push_back(item);
      }
    }
  }

  /// Locks and freezes the beverage table snapshot upon arriving at the beverage
  /// area so turning left/right toward the guest does not overwrite table detections.
  size_t snapshotBeveragesOnTable()
  {
    std::lock_guard<std::mutex> lock(vision_mutex_);
    beverage_snapshot_locked_ = true;
    mergeDrinksIntoSnapshotLocked(latest_drinks_);
    drinks_detected_on_arrival_ = !table_drinks_snapshot_.empty();
    return table_drinks_snapshot_.size();
  }

  /// Computes which side ("left", "center", or "right") holds `favorite_drink`
  /// by deduplicating canonical labels, sorting left-to-right by bbox X, and
  /// falling back to the closest visible drink if no exact label match exists.
  std::string locateBeverageSide(const std::string & favorite_drink, bool & found_match)
  {
    std::lock_guard<std::mutex> lock(vision_mutex_);
    found_match = false;
    const std::string target_raw = normalize(favorite_drink);
    const std::string target = canonicalDrinkName(favorite_drink);

    const auto & raw_list = !table_drinks_snapshot_.empty()
      ? table_drinks_snapshot_
      : latest_drinks_;

    // Deduplikasi per jenis minuman lalu urutkan dari kiri ke kanan berdasarkan posisi bbox
    std::vector<DetectedObject> drinks_list;
    for (const auto & item : raw_list) {
      const std::string canon = canonicalDrinkName(item.name);
      if (canon.empty()) continue;
      bool exists = false;
      for (auto & d : drinks_list) {
        if (canonicalDrinkName(d.name) == canon) {
          d = item;
          exists = true;
          break;
        }
      }
      if (!exists) {
        drinks_list.push_back(item);
      }
    }

    auto centerX = [](const DetectedObject & d) {
      return static_cast<double>(d.bbox_x) + static_cast<double>(d.bbox_w) * 0.5;
    };
    std::sort(drinks_list.begin(), drinks_list.end(),
      [&](const DetectedObject & a, const DetectedObject & b) {
        return centerX(a) < centerX(b);
      });

    const double cam_w = (camera_width_ > 0) ? static_cast<double>(camera_width_) : 672.0;
    const double cam_center = cam_w * 0.5;

    auto sideByIndex = [&](size_t idx) -> std::string {
      // 1. Jika hanya ada 2 minuman: mutlak gunakan "left" dan "right"
      if (drinks_list.size() == 2) {
        return (idx == 0) ? "left" : "right";
      }

      // 2. Jika ada banyak (>= 3 minuman):
      //    Jelaskan posisi tiap minuman, dan "center" hanya digunakan jika benar-benar di tengah!
      if (drinks_list.size() >= 3) {
        if (idx == 0) return "left";
        if (idx == drinks_list.size() - 1) return "right";

        // Untuk item di antara indeks ujung:
        const double cx = centerX(drinks_list[idx]);
        const double cx_min = centerX(drinks_list.front());
        const double cx_max = centerX(drinks_list.back());
        const double span = cx_max - cx_min;
        const double cluster_center = (cx_min + cx_max) * 0.5;

        // "center" hanya digunakan jika benar-benar di tengah (posisi relatif di tengah deretan minuman
        // atau berada dekat sumbu simetri kamera)
        const double ratio = (span > 1.0) ? ((cx - cx_min) / span) : 0.5;
        const bool is_centered_in_cluster = (ratio >= 0.28 && ratio <= 0.72) &&
                                            (std::abs(cx - cluster_center) <= span * 0.22);
        const bool is_centered_in_camera = (std::abs(cx - cam_center) <= cam_w * 0.12);

        if (is_centered_in_cluster || is_centered_in_camera) {
          return "center";
        }

        // Jika tidak benar-benar di tengah, jelaskan apakah di sisi kiri atau kanan
        return (cx < cluster_center) ? "left" : "right";
      }

      // 3. Jika hanya ada 1 minuman: tentukan berdasarkan sumbu tengah kamera
      const double cx = centerX(drinks_list[idx]);
      const double deadband = cam_w * 0.10;
      if (cx < cam_center - deadband) {
        return "left";
      } else if (cx > cam_center + deadband) {
        return "right";
      }
      return "center";
    };

    for (size_t i = 0; i < drinks_list.size(); ++i) {
      const auto & drink = drinks_list[i];
      const std::string d_raw = normalize(drink.name);
      const std::string d_canon = canonicalDrinkName(drink.name);
      const bool match =
        (!d_canon.empty() && !target.empty() && d_canon == target) ||
        (!d_raw.empty() && !target_raw.empty() &&
         (d_raw == target_raw || d_raw.find(target_raw) != std::string::npos ||
          target_raw.find(d_raw) != std::string::npos));
      if (match) {
        found_match = true;
        const std::string side = sideByIndex(i);
        RCLCPP_INFO(
          get_logger(),
          "\033[1;32m[BEVERAGE]\033[0m Cocok item '%s' di sisi %s (rank %zu/%zu, cx=%.0f, 3D: x=%.2f m, y=%.2f m)",
          drink.name.c_str(), side.c_str(), i + 1, drinks_list.size(),
          centerX(drink), drink.x, drink.y);
        return side;
      }
    }

    if (!drinks_list.empty()) {
      found_match = true;
      const auto & first = drinks_list.front();
      const std::string side = sideByIndex(0);
      RCLCPP_INFO(
        get_logger(),
        "\033[1;33m[BEVERAGE]\033[0m Fallback item terdekat '%s' di sisi %s (3D: x=%.2f m, y=%.2f m)",
        first.name.c_str(), side.c_str(), first.x, first.y);
      return side;
    }

    return "center";
  }


  /// Aggregates mode (most frequent) gender, shirt color, and hair color from
  /// /vision/person_attributes frames accumulated during the door dialog, falling
  /// back to the latest snapshot if any histogram is empty.
  void updateCurrentGuestVisualAttributes(int person_id)
  {
    std::vector<std::string> raw_attrs;
    std::unordered_map<std::string, int> gender_counts;
    std::unordered_map<std::string, int> shirt_counts;
    std::unordered_map<std::string, int> hair_counts;
    {
      std::lock_guard<std::mutex> v_lock(vision_mutex_);
      if (attributes_locked_from_scan_) {
        return;
      }
      raw_attrs = latest_attributes_;
      gender_counts = attr_gender_counts_;
      shirt_counts = attr_shirt_counts_;
      hair_counts = attr_hair_counts_;
    }

    auto pickMode = [](
      const std::unordered_map<std::string, int> & counts,
      const std::string & skip_if_other_exists = "") -> std::string
    {
      std::string best_val;
      int best_cnt = 0;
      for (const auto & kv : counts) {
        if (!skip_if_other_exists.empty() && kv.first == skip_if_other_exists) {
          continue;
        }
        if (kv.second > best_cnt) {
          best_cnt = kv.second;
          best_val = kv.first;
        }
      }
      if (best_val.empty() && !skip_if_other_exists.empty()) {
        auto it = counts.find(skip_if_other_exists);
        if (it != counts.end() && it->second > 0) {
          best_val = it->first;
        }
      }
      return best_val;
    };

    RCLCPP_INFO(
      get_logger(),
      "\033[1;32m[VISION]\033[0m Extracting visual attributes (direct snapshot) for person_id=%d",
      person_id);

    std::lock_guard<std::mutex> g_lock(guest_mutex_);
    std::vector<std::string> attrs;

    // Ekstrak langsung dari snapshot atribut terbaru pada jarak 1.5m (tanpa voting)
    std::string snap_gender, snap_shirt, snap_hair;
    for (const auto & a : raw_attrs) {
      std::string lower_a = a;
      std::transform(lower_a.begin(), lower_a.end(), lower_a.begin(), ::tolower);
      if (lower_a == "male" || lower_a == "female") {
        snap_gender = lower_a;
      } else if (lower_a.find("shirt") != std::string::npos ||
                 lower_a.find("clothes") != std::string::npos ||
                 lower_a.find("attire") != std::string::npos)
      {
        snap_shirt = a;
      } else if (lower_a.find("hair") != std::string::npos) {
        snap_hair = a;
      }
    }

    if (!snap_gender.empty()) {
      current_guest_.gender = snap_gender;
    } else {
      const std::string mode_gender = pickMode(gender_counts);
      if (!mode_gender.empty()) current_guest_.gender = mode_gender;
    }

    if (!snap_shirt.empty() && snap_shirt != "casual attire") {
      current_guest_.shirt_color = snap_shirt;
    } else {
      const std::string mode_shirt = pickMode(shirt_counts, "casual attire");
      if (!mode_shirt.empty()) current_guest_.shirt_color = mode_shirt;
      else if (!snap_shirt.empty()) current_guest_.shirt_color = snap_shirt;
    }

    if (!snap_hair.empty()) {
      current_guest_.hair_color = snap_hair;
    } else {
      const std::string mode_hair = pickMode(hair_counts);
      if (!mode_hair.empty()) current_guest_.hair_color = mode_hair;
    }

    if (!current_guest_.gender.empty() && current_guest_.gender != "unknown") {
      attrs.push_back(current_guest_.gender);
    }
    if (!current_guest_.shirt_color.empty()) {
      attrs.push_back(current_guest_.shirt_color);
    }
    if (!current_guest_.hair_color.empty()) {
      attrs.push_back(current_guest_.hair_color);
    }
    if (attrs.empty()) {
      attrs.push_back("casual attire");
    }
    current_guest_.attributes = attrs;
    RCLCPP_INFO(
      get_logger(),
      "\033[1;32m[VISION]\033[0m Guest %d Visual Profile (Mode): Gender='%s', Shirt='%s', Hair='%s'",
      guest_idx_, current_guest_.gender.c_str(), current_guest_.shirt_color.c_str(), current_guest_.hair_color.c_str());
  }

  /// Formats a grammatically consistent English appearance sentence for TTS
  /// (defaults "unknown" gender to neutral "person" + "They are...").
  static std::string formatPersonDescription(const PersonProfile & person)
  {
    std::string gender = (person.gender.empty() || person.gender == "unknown")
      ? "person" : person.gender;
    std::string shirt = person.shirt_color.empty() ? "casual attire" : person.shirt_color;
    std::string hair = person.hair_color.empty() ? "dark hair" : person.hair_color;

    std::string pronoun = "They";
    std::string verb = "are";
    std::string has_verb = "have";
    if (gender == "male") {
      pronoun = "He";
      verb = "is";
      has_verb = "has";
    } else if (gender == "female") {
      pronoun = "She";
      verb = "is";
      has_verb = "has";
    }

    if (shirt.find("shirt") == std::string::npos &&
        shirt.find("attire") == std::string::npos &&
        shirt.find("clothes") == std::string::npos)
    {
      shirt += " shirt";
    }
    if (hair.find("hair") == std::string::npos) {
      hair += " hair";
    }

    return pronoun + " " + verb + " a " + gender + ", wearing a " + shirt + ", and " + has_verb + " " + hair;
  }

  // Deteksi orang terdekat di depan robot (dari ZED /zed/person_label)
  bool isPersonInFront() const
  {
    return person_seen_.load() &&
           (now() - last_person_seen_time_).seconds() < person_freshness_sec_;
  }

  // ============================================================================
  // HELPER TEXT-TO-SPEECH (TTS) & SPEECH-TO-TEXT (STT) SLOT PARSING
  // ============================================================================
  /// Estimates a dynamic watchdog timeout for a TTS utterance based on word count.
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

  /// Publishes text to /hri/tts_say asynchronously, locks speaking_ & STT cooldown,
  /// and stores `on_done` to run when /hri/tts_done arrives (or TTS watchdog fires).
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
    publishExpression("SPEAKING");
    RCLCPP_INFO(get_logger(), "\033[1;36m[RECEPTIONIST]\033[0m\n  TTS: \"%s\"", text.c_str());
  }

  /// Callback invoked when /hri/tts_done is received (or by fsmTick TTS watchdog).
  /// Releases speaking_, applies a short anti-echo cooldown, then runs pending callback.
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

    RCLCPP_DEBUG(
      get_logger(),
      "[TTS] Done | State: %s",
      stateToString(state_.load()).c_str());

    publishExpression(expressionForState(state_.load()));

    if (callback) {
      callback();
    }
    in_callback_ = false;
  }

  /// Prevents STT echo capture while the robot is busy or within post-TTS cooldown.
  bool canListen() const
  {
    return !isBusy() && now() >= tts_cooldown_until_;
  }

  /// Lowercases text and collapses non-alphanumeric punctuation into single spaces.
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

  // Pencocokan frase utuh (word-boundary) agar tidak salah tangkap substring
  static bool containsWholePhrase(const std::string & norm_haystack, const std::string & phrase)
  {
    const std::string padded_h = " " + norm_haystack + " ";
    const std::string padded_p = " " + phrase + " ";
    return padded_h.find(padded_p) != std::string::npos;
  }

  /// Extracts a slot value ("person_name", "interest", or "drink") from raw STT text:
  /// 1) Filters common greeting/filler words.
  /// 2) Matches known RoboCup vocabulary with word boundaries (longer phrases first).
  /// 3) Falls back to stripping common carrier phrases (e.g. "my name is ...").
  static std::string parseSlotIntent(const std::string & raw_text, const std::string & slot)
  {
    const std::string norm = normalize(raw_text);
    if (norm.empty()) {
      return "";
    }

    static const std::vector<std::string> kIgnoredFillers = {
      "hello", "hi", "hey", "yes", "no", "okay", "ok", "what", "sorry",
      "pardon", "excuse me", "thank you", "thanks", "please", "robot"
    };
    for (const auto & filler : kIgnoredFillers) {
      if (norm == filler) {
        return "";
      }
    }

    auto countWords = [](const std::string & s) -> size_t {
      std::stringstream ss(s);
      std::string w;
      size_t count = 0;
      while (ss >> w) {
        ++count;
      }
      return count;
    };

    auto stripPrefixes = [&](const std::vector<std::string> & prefixes) -> std::string {
      for (const auto & p : prefixes) {
        const auto pos = norm.find(p);
        if (pos != std::string::npos) {
          std::string rem = norm.substr(pos + p.size());
          while (!rem.empty() && rem.front() == ' ') {
            rem.erase(rem.begin());
          }
          if (!rem.empty() && countWords(rem) <= 3) {
            return capitalizeWords(rem);
          }
        }
      }
      if (countWords(norm) <= 2) {
        return capitalizeWords(norm);
      }
      return "";
    };

    if (slot == "person_name") {
      static const std::vector<std::string> kKnownNames = {
        "adel", "angel", "axel", "charlie", "jane", "john", "jules", "morgan", "ahmad", "rizal"
        "paris", "robin", "simone", "alex", "michael", "sarah", "david", "emma", "maria", "muhammad",
        "budi", "abdul", "jefrey", "jefri", "hezekiel"
      };
      for (const auto & name : kKnownNames) {
        if (containsWholePhrase(norm, name)) {
          return capitalizeWords(name);
        }
      }
      return stripPrefixes({
        "my name is ", "i am ", "im ", "i m ", "iam ", "iyem ", "name s ", "may name is ", "mai name is ", "you can call me ", "my name s ", "mai name s ","call me ", "this is ", "name is "
      });
    }

    if (slot == "drink") {
      // Frase spesifik (2 kata) diperiksa lebih dulu sebelum kata tunggal ("tea", "juice")
      static const std::vector<std::string> kKnownDrinks = {
        "orange juice", "apple juice", "iced tea", "green tea",
        "coke", "cola", "pepsi", "sprite", "fanta",
        "mineral water", "milk", "water", "coffee", "tea", "juice", "soda",
        "botol", "bottle", "cracker", "cracker box", "snack",
        "pringles", "sugar", "sugar box", "cheezit", "cheez-it", "chips", "milk"
      };
      for (const auto & drink : kKnownDrinks) {
        if (containsWholePhrase(norm, drink) &&
            !containsWholePhrase(norm, "not " + drink) &&
            !containsWholePhrase(norm, "dont like " + drink) &&
            !containsWholePhrase(norm, "do not like " + drink))
        {
          return capitalizeWords(drink);
        }
      }
      return stripPrefixes({
        "my favorite drink is ", "i would like ", "i like ", "i want ",
        "i prefer ", "favorite drink is ", "drink is "
      });
    }

    if (slot == "interest") {
      static const std::vector<std::string> kKnownInterests = {
        "artificial intelligence", "robotics", "reading", "music", "sports",
        "cooking", "gaming", "traveling", "photography", "movies", "swimming",
        "football", "basketball", "coding", "programming", "dancing", "painting",
        "program", "dancing", "dance", "drawing", "playing football", "playing badminton",
        "playing tennis", "playing golf", "playing", "boxing"
      };
      for (const auto & interest : kKnownInterests) {
        if (containsWholePhrase(norm, interest)) {
          return capitalizeWords(interest);
        }
      }
      return stripPrefixes({
        "my main interest is ", "my interest is ", "my hobby is ",
        "i am interested in ", "i like ", "i love ", "i enjoy "
      });
    }

    return "";
  }

  // Pengenalan respon konfirmasi (Yes / No) dengan pencocokan word-boundary
  static int parseYesNoIntent(const std::string & raw_text)
  {
    const std::string norm = normalize(raw_text);
    if (norm.empty()) return 0;

    static const std::vector<std::string> kNoPhrases = {
      "no", "no robot", "nope", "not correct", "wrong", "false",
    };
    static const std::vector<std::string> kYesPhrases = {
      "yes", "yes robot", "yeah", "yep", "correct", "right", "sure", "true", "exactly",
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
  // CALLBACK SUBSCRIPTION ROS 2
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
      "\033[1;33m[STT]\033[0m \"%s\"",
      message->data.c_str());
  }

  // Callback deteksi orang dari ZED (/zed/person_label)
  void onPersonLabel(const std_msgs::msg::String::SharedPtr message)
  {
    if (!isVisionActive() || !message || message->data.empty() ||
        message->data.find("NO_PERSON") != std::string::npos)
    {
      return;
    }

    person_seen_ = true;
    last_person_seen_time_ = now();

    std::lock_guard<std::mutex> lock(vision_mutex_);
    const std::string & data = message->data;
    const auto sep1 = data.find(':');
    const auto bar1 = data.find('|');
    if (sep1 != std::string::npos && bar1 != std::string::npos && bar1 > sep1) {
      const std::string bbox_str = data.substr(sep1 + 1, bar1 - sep1 - 1);
      std::stringstream ss(bbox_str);
      std::string item;
      std::vector<int> rect;
      while (std::getline(ss, item, ',')) {
        try { rect.push_back(std::stoi(item)); } catch (...) {}
      }
      if (rect.size() >= 4) {
        nearest_person_bbox_x_ = rect[0];
        nearest_person_bbox_y_ = rect[1];
        nearest_person_bbox_w_ = rect[2];
        nearest_person_bbox_h_ = rect[3];
      }
    }

    const auto id_pos = data.find("id=");
    if (id_pos != std::string::npos) {
      try {
        nearest_person_id_ = std::stoi(data.substr(id_pos + 3));
      } catch (...) {
        nearest_person_id_ = 0;
      }
    } else {
      nearest_person_id_ = 0;
    }
  }

  static std::vector<DetectedObject> parseDetectionString(const std::string & data)
  {
    std::vector<DetectedObject> list;
    std::stringstream ss(data);
    std::string chunk;
    while (std::getline(ss, chunk, ';')) {
      if (chunk.empty()) continue;
      const auto colon = chunk.find(':');
      const auto bar = chunk.find('|');
      if (colon == std::string::npos || bar == std::string::npos || bar <= colon) continue;

      DetectedObject obj;
      obj.name = chunk.substr(0, colon);

      std::stringstream bss(chunk.substr(colon + 1, bar - colon - 1));
      std::string val;
      std::vector<int> rect;
      while (std::getline(bss, val, ',')) {
        try { rect.push_back(std::stoi(val)); } catch (...) {}
      }
      if (rect.size() >= 4) {
        obj.bbox_x = rect[0];
        obj.bbox_y = rect[1];
        obj.bbox_w = rect[2];
        obj.bbox_h = rect[3];
      }

      // Parse 3D coordinates (x, y, z)
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
      if (coords.size() >= 3) {
        obj.x = coords[0];
        obj.y = coords[1];
        obj.z = coords[2];
      }

      list.push_back(obj);
    }
    return list;
  }

  // Callback utama deteksi minuman (/zed/drink_detections)
  void onDrinkDetections(const std_msgs::msg::String::SharedPtr message)
  {
    if (!isVisionActive() || !message) return;

    std::lock_guard<std::mutex> lock(vision_mutex_);
    last_drink_msg_time_ = now();
    if (message->data.empty() || message->data.find("NONE") != std::string::npos) {
      latest_drinks_.clear();
      return;
    }
    latest_drinks_ = parseDetectionString(message->data);
    if (state_.load() == ReceptionistState::ASK_DRINK && !beverage_snapshot_locked_) {
      mergeDrinksIntoSnapshotLocked(latest_drinks_);
    }
    for (const auto & item : latest_drinks_) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "\033[1;36m[DRINK_DET]\033[0m Terdeteksi '%s' pada posisi 3D (x=%.2f m, y=%.2f m, z=%.2f m)",
        item.name.c_str(), item.x, item.y, item.z);
    }
  }

  // Callback fallback (/zed/bag_detections): HANYA digunakan jika /zed/drink_detections
  // tidak aktif, dan TIDAK menghapus latest_drinks_ jika kosong.
  void onBagDetectionsFallback(const std_msgs::msg::String::SharedPtr message)
  {
    if (!isVisionActive() || !message || message->data.empty()) return;

    std::lock_guard<std::mutex> lock(vision_mutex_);
    if (last_drink_msg_time_.nanoseconds() > 0 &&
        (now() - last_drink_msg_time_).seconds() < 3.0)
    {
      return; // Abaikan karena topic utama /zed/drink_detections sedang aktif
    }
    auto parsed = parseDetectionString(message->data);
    if (!parsed.empty()) {
      latest_drinks_ = std::move(parsed);
    }
  }

  // Callback atribut visual tamu (/vision/person_attributes)
  void onPersonAttributes(const std_msgs::msg::String::SharedPtr message)
  {
    if (!isVisionActive() || !message || message->data.empty()) return;
    // Hanya terima update atribut visual saat robot berhadapan langsung dengan tamu di pintu
    // agar tidak tertimpa background meja minuman atau host di ruang tamu.
    const ReceptionistState cur = state_.load();
    if (cur != ReceptionistState::WAIT_GUEST &&
        cur != ReceptionistState::ALIGN_GUEST &&
        cur != ReceptionistState::SCAN_GUEST_ATTRIBUTES &&
        cur != ReceptionistState::APPROACH_GUEST &&
        cur != ReceptionistState::ASK_NAME &&
        cur != ReceptionistState::WAIT_NAME &&
        cur != ReceptionistState::ASK_INTEREST &&
        cur != ReceptionistState::WAIT_INTEREST &&
        cur != ReceptionistState::CONFIRM_GUEST_INFO &&
        cur != ReceptionistState::WAIT_CONFIRM_GUEST_INFO)
    {
      return;
    }
    std::lock_guard<std::mutex> lock(vision_mutex_);
    std::vector<std::string> parsed;
    std::stringstream ss(message->data);
    std::string item;
    while (std::getline(ss, item, ',')) {
      size_t first = item.find_first_not_of(" \t\r\n");
      if (first != std::string::npos) {
        size_t last = item.find_last_not_of(" \t\r\n");
        parsed.push_back(item.substr(first, last - first + 1));
      }
    }
    if (!parsed.empty()) {
      latest_attributes_ = parsed;
      // Akumulasi frekuensi saat fase scan 1.5m (SCAN_GUEST_ATTRIBUTES) dan dialog tamu di pintu
      if (cur == ReceptionistState::SCAN_GUEST_ATTRIBUTES ||
          cur == ReceptionistState::ASK_NAME ||
          cur == ReceptionistState::WAIT_NAME ||
          cur == ReceptionistState::ASK_INTEREST ||
          cur == ReceptionistState::WAIT_INTEREST ||
          cur == ReceptionistState::CONFIRM_GUEST_INFO ||
          cur == ReceptionistState::WAIT_CONFIRM_GUEST_INFO)
      {
        for (const auto & a : parsed) {
          std::string lower_a = a;
          std::transform(lower_a.begin(), lower_a.end(), lower_a.begin(), ::tolower);
          if (lower_a == "male" || lower_a == "female") {
            attr_gender_counts_[lower_a]++;
          } else if (lower_a.find("shirt") != std::string::npos ||
                     lower_a.find("clothes") != std::string::npos ||
                     lower_a.find("attire") != std::string::npos)
          {
            attr_shirt_counts_[a]++;
          } else if (lower_a.find("hair") != std::string::npos) {
            attr_hair_counts_[a]++;
          }
        }
      }
    }
  }

  // Callback deteksi kursi kosong (/vision/empty_seat: "side|x,y,z|depth" atau "left", "center", "right")
  void onEmptySeat(const std_msgs::msg::String::SharedPtr message)
  {
    if (!isVisionActive() || !message || message->data.empty()) return;
    std::lock_guard<std::mutex> lock(vision_mutex_);
    const std::string raw = normalize(message->data);
    if (raw == "none" || raw.empty()) {
      has_explicit_empty_seat_ = false;
      return;
    }
    const auto pipe1 = raw.find('|');
    std::string side_str = (pipe1 != std::string::npos) ? raw.substr(0, pipe1) : raw;

    // Jika pesan menyertakan koordinat 3D: side|x,y,z|depth
    if (pipe1 != std::string::npos) {
      const auto pipe2 = raw.find('|', pipe1 + 1);
      std::string xyz_str = (pipe2 != std::string::npos)
        ? raw.substr(pipe1 + 1, pipe2 - pipe1 - 1)
        : raw.substr(pipe1 + 1);
      double px = 0.0, py = 0.0, pz = 0.0;
      if (xyz_str.find("nan") == std::string::npos &&
          std::sscanf(xyz_str.c_str(), "%lf,%lf,%lf", &px, &py, &pz) == 3) {
        // Koreksi mirror / inversi sumbu Y dari vision: negasikan kembali (-py)
        // persis seperti di restaurant_node (p.y = -py) agar sesuai standar ROS (+Y Kiri, -Y Kanan)
        const double corrected_y = -py;
        side_str = (corrected_y >= 0.20) ? "left" : (corrected_y <= -0.20 ? "right" : "center");
      } else {
        if (side_str == "left") side_str = "right";
        else if (side_str == "right") side_str = "left";
      }
    } else {
      if (side_str == "left") side_str = "right";
      else if (side_str == "right") side_str = "left";
    }

    empty_seat_position_ = side_str;
    has_explicit_empty_seat_ = true;
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "\033[1;32m[EMPTY_SEAT]\033[0m Kursi kosong terdeteksi di sisi: %s",
      empty_seat_position_.c_str());
  }

  // ============================================================================
  // HELPER SERVICE CLIENT NON-BLOCKING (TANPA DETACHED THREAD / MEMORY SAFE)
  // ============================================================================
  /// Sends an asynchronous std_srvs::srv::Trigger request guarded by srv_mutex_
  /// and a software timeout watchdog in fsmTick() (no detached threads).
  void callTrigger(
    const TriggerClient::SharedPtr & client,
    const std::string & service_name,
    TriggerResult result,
    double timeout_sec = 10.0)
  {
    if (!client) {
      if (result) result(false, "Client is null");
      return;
    }

    if (!client->service_is_ready()) {
      if (!client->wait_for_service(1s)) {
        RCLCPP_WARN(
          get_logger(), "Service %s belum tersedia.",
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
            return; // Sudah di-handle oleh timeout watchdog
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

  // ============================================================================
  // MAIN FINITE STATE MACHINE (10 Hz FSM TICK & WATCHDOGS)
  // ============================================================================
  /// Periodic 10 Hz FSM loop:
  ///   1) Publishes guest names at 2 Hz.
  ///   2) Enforces global, service-call, and TTS timeout watchdogs.
  ///   3) Skips state execution while isBusy() is true.
  ///   4) Dispatches the active state handler.
  void fsmTick()
  {
    const ReceptionistState current_state = state_.load();

    // Periodically publish registered guest names (2 Hz / every 500 ms)
    static int guest_pub_counter = 0;
    if (++guest_pub_counter >= 5) {
      guest_pub_counter = 0;
      publishGuestNames();
    }

    // 1. Global timeout watchdog
    if ((now() - task_start_time_).seconds() > global_timeout_sec_ &&
        current_state != ReceptionistState::COMPLETED &&
        current_state != ReceptionistState::ABORTED)
    {
      RCLCPP_ERROR(get_logger(), "Global receptionist task timeout reached.");
      state_ = ReceptionistState::ABORTED;
      publishExpression("LOST");
      return;
    }

    // 2. Service timeout watchdog (mencegah hang jika service server macet)
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
        "Service %s timeout (> %.1f s), releasing service lock.",
        timed_out_srv_name.c_str(), srv_timeout_sec_);
      timed_out_srv_cb(false, "Service response timeout");
      return;
    }

    // 3. TTS watchdog jika /hri/tts_done tidak diterima
    bool tts_timed_out = false;
    double tts_timeout = 0.0;
    {
      std::lock_guard<std::mutex> lock(tts_mutex_);
      tts_timeout = tts_timeout_sec_;
      tts_timed_out = speaking_.load() && (now() - tts_started_time_).seconds() > tts_timeout_sec_;
    }
    if (tts_timed_out) {
      RCLCPP_WARN(
        get_logger(),
        "TTS timeout (> %.1f s, no /hri/tts_done received), releasing speaking lock.",
        tts_timeout);
      onTtsDone(nullptr);
      return;
    }

    // Pastikan robot telah selesai bergerak atau berbicara sebelum memulai aksi berikutnya
    if (isBusy()) {
      return;
    }

    switch (current_state) {
      // ------------------------------------------------------------------------
      // INIT: Tunggu service /receptionist/go_to_door siap lalu mulai tugas
      // ------------------------------------------------------------------------
      case ReceptionistState::INIT:
        if (!nav_door_client_ || !nav_door_client_->service_is_ready()) {
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Waiting for '/receptionist/go_to_door' service to be ready...");
          return;
        }
        if (tts_pub_->get_subscription_count() == 0 &&
            (now() - state_enter_time_).seconds() < 1.5)
        {
          return;
        }
        RCLCPP_INFO(get_logger(), "\033[1;36m[RECEPTIONIST]\033[0m\n  Status: Starting task");
        guests_.clear();
        initCurrentGuest(1);
        publishGuestNames();
        transitionTo(ReceptionistState::NAV_TO_DOOR, "Start cycle for Guest 1: navigating to door");
        break;

      // ------------------------------------------------------------------------
      // STATE 1: NAVIGASI KE PINTU & DETEKSI TAMU
      // ------------------------------------------------------------------------
      case ReceptionistState::NAV_TO_DOOR:
        if (last_nav_fail_time_.nanoseconds() > 0 &&
            (now() - last_nav_fail_time_).seconds() < nav_retry_cooldown_sec_)
        {
          return;
        }
        speakAsync("Now I will go to the door to wait for the guest.", [this]() {
          navigateTo("entrance_door", [this](bool success) {
            if (!success) {
              return; // Tetap di NAV_TO_DOOR dan retry setelah cooldown
            }
            transitionTo(ReceptionistState::WAIT_GUEST, "Arrived at entrance_door");
          });
        });
        break;

      case ReceptionistState::WAIT_GUEST:
        // Beri waktu kamera ZED settle setelah tiba di pintu
        if ((now() - state_enter_time_).seconds() < vision_initial_settle_sec_) {
          return;
        }
        if (isPersonInFront()) {
          transitionTo(
            ReceptionistState::ALIGN_GUEST,
            "Guest detected at entrance; aligning heading and adjusting distance to 1.5m");
        } else if (last_wait_prompt_time_.nanoseconds() == 0 ||
                   (now() - last_wait_prompt_time_).seconds() >= wait_guest_prompt_sec_)
        {
          last_wait_prompt_time_ = now();
          speakAsync("I am waiting for a guest; please stand in front of me.");
        }
        break;

      case ReceptionistState::ALIGN_GUEST:
        adjustGuestDistance([this](bool /*success*/, const std::string & /*msg*/) {
          std::string temp_name;
          {
            std::lock_guard<std::mutex> lock(vision_mutex_);
            latest_attributes_.clear();
            attr_gender_counts_.clear();
            attr_shirt_counts_.clear();
            attr_hair_counts_.clear();
            attributes_locked_from_scan_ = false;
            temp_name = "__pending_guest_" + std::to_string(guest_idx_) + "__";
            active_vision_reg_name_ = temp_name;
          }
          if (person_name_pub_) {
            std_msgs::msg::String m;
            m.data = temp_name;
            person_name_pub_->publish(m);
          }
          transitionTo(
            ReceptionistState::SCAN_GUEST_ATTRIBUTES,
            "Stepped back to 1.5m; scanning face, gender, shirt color, and hair color silently");
        });
        break;

      case ReceptionistState::SCAN_GUEST_ATTRIBUTES:
        if ((now() - state_enter_time_).seconds() < 2.5) {
          return;
        }
        updateCurrentGuestVisualAttributes(nearest_person_id_.load());
        {
          std::lock_guard<std::mutex> lock(vision_mutex_);
          attributes_locked_from_scan_ = true;
        }
        {
          std::lock_guard<std::mutex> lock(guest_mutex_);
          if (guests_.size() < static_cast<size_t>(guest_idx_)) {
            guests_.push_back(current_guest_);
          } else if (guest_idx_ >= 1) {
            guests_[static_cast<size_t>(guest_idx_ - 1)] = current_guest_;
          }
        }
        transitionTo(
          ReceptionistState::APPROACH_GUEST,
          "Face & attributes locked at 1.5m; approaching guest to 0.7m for name");
        break;

      case ReceptionistState::APPROACH_GUEST:
        approachGuest([this](bool /*success*/, const std::string & /*msg*/) {
          speakAsync(
            "Hello!. I am Barelang Inari from Politeknik Negeri Batam; I am your robot receptionist.",
            [this]() {
              transitionTo(ReceptionistState::ASK_NAME, "Approached guest to 0.7m and greeted");
            });
        });
        break;

      case ReceptionistState::ASK_NAME:
        speakAsync("May I please have your name?", [this]() {
          transitionTo(ReceptionistState::WAIT_NAME, "Waiting for guest name");
        });
        break;

      case ReceptionistState::WAIT_NAME:
        if (has_new_utterance_.exchange(false)) {
          std::string raw;
          {
            std::lock_guard<std::mutex> lock(stt_mutex_);
            raw = heard_utterance_;
          }
          const std::string parsed_name = parseSlotIntent(raw, "person_name");
          if (!parsed_name.empty()) {
            {
              std::lock_guard<std::mutex> lock(guest_mutex_);
              current_guest_.name = parsed_name;
              if (guests_.size() < static_cast<size_t>(guest_idx_)) {
                guests_.push_back(current_guest_);
              } else if (guest_idx_ >= 1) {
                guests_[static_cast<size_t>(guest_idx_ - 1)].name = parsed_name;
              }
            }
            if (person_name_pub_) {
              std_msgs::msg::String m;
              std::string old_reg_name;
              {
                std::lock_guard<std::mutex> v_lock(vision_mutex_);
                old_reg_name = active_vision_reg_name_;
                active_vision_reg_name_ = parsed_name;
              }
              m.data = (!old_reg_name.empty() && old_reg_name != parsed_name)
                ? ("rename:" + old_reg_name + ":" + parsed_name)
                : parsed_name;
              person_name_pub_->publish(m);
            }
            publishGuestNames();
            RCLCPP_INFO(
              get_logger(),
              "\033[1;32m[DIALOG NAME]\033[0m Guest %d Name -> \033[1;32m\"%s\"\033[0m (from \"%s\")",
              guest_idx_, parsed_name.c_str(), raw.c_str());
            updateCurrentGuestVisualAttributes(nearest_person_id_.load());
            {
              std::lock_guard<std::mutex> lock(guest_mutex_);
              if (guest_idx_ >= 1 && guests_.size() >= static_cast<size_t>(guest_idx_)) {
                guests_[static_cast<size_t>(guest_idx_ - 1)] = current_guest_;
              }
            }
            transitionTo(ReceptionistState::ASK_INTEREST, "Guest name captured and stored");
          } else {
            RCLCPP_WARN(
              get_logger(),
              "\033[1;33m[DIALOG NAME]\033[0m Unrecognized input \"%s\", asking again.",
              raw.c_str());
            state_enter_time_ = now();
            speakAsync("I am sorry, I did not catch your name. May I please have your name?");
          }
        } else if ((now() - state_enter_time_).seconds() > dialog_reprompt_sec_) {
          state_enter_time_ = now();
          speakAsync("May I please have your name?");
        }
        break;

      case ReceptionistState::ASK_INTEREST: {
        std::string g_name;
        {
          std::lock_guard<std::mutex> lock(guest_mutex_);
          g_name = current_guest_.name;
        }
        speakAsync(
          "Nice to meet you " + g_name +
          ". What is your interesting or hobby?",
          [this]() {
            transitionTo(ReceptionistState::WAIT_INTEREST, "Waiting for guest interest");
          });
        break;
      }

      case ReceptionistState::WAIT_INTEREST:
        if (has_new_utterance_.exchange(false)) {
          std::string raw;
          {
            std::lock_guard<std::mutex> lock(stt_mutex_);
            raw = heard_utterance_;
          }
          const std::string parsed_interest = parseSlotIntent(raw, "interest");
          if (!parsed_interest.empty()) {
            {
              std::lock_guard<std::mutex> lock(guest_mutex_);
              current_guest_.interest = parsed_interest;
            }
            updateCurrentGuestVisualAttributes(nearest_person_id_.load());
            {
              std::lock_guard<std::mutex> lock(guest_mutex_);
              if (guest_idx_ >= 1 && guests_.size() >= static_cast<size_t>(guest_idx_)) {
                guests_[static_cast<size_t>(guest_idx_ - 1)] = current_guest_;
              }
            }
            RCLCPP_INFO(
              get_logger(),
              "\033[1;32m[DIALOG INTEREST]\033[0m Guest %d Interest -> \033[1;32m\"%s\"\033[0m (from \"%s\")",
              guest_idx_, parsed_interest.c_str(), raw.c_str());
            transitionTo(ReceptionistState::CONFIRM_GUEST_INFO, "Guest interest captured; confirming info");
          } else {
            RCLCPP_WARN(
              get_logger(),
              "\033[1;33m[DIALOG INTEREST]\033[0m Unrecognized input \"%s\", asking again.",
              raw.c_str());
            state_enter_time_ = now();
            speakAsync("I am sorry. What is your interesting or hobby?");
          }
        } else if ((now() - state_enter_time_).seconds() > dialog_reprompt_sec_) {
          state_enter_time_ = now();
          speakAsync("Could you please tell me your interesting or hobby?");
        }
        break;

      case ReceptionistState::CONFIRM_GUEST_INFO: {
        std::string g_name, g_interest;
        {
          std::lock_guard<std::mutex> lock(guest_mutex_);
          g_name = current_guest_.name;
          g_interest = current_guest_.interest;
        }
        const std::string confirm_text =
          "Hello " + g_name + ", you have an interest in " +
          g_interest + ". Is that correct? Please say yes robot or no robot";

        speakAsync(confirm_text, [this]() {
          transitionTo(ReceptionistState::WAIT_CONFIRM_GUEST_INFO, "Waiting for confirmation (yes/no)");
        });
        break;
      }

      case ReceptionistState::WAIT_CONFIRM_GUEST_INFO:
        if (has_new_utterance_.exchange(false)) {
          std::string raw;
          {
            std::lock_guard<std::mutex> lock(stt_mutex_);
            raw = heard_utterance_;
          }
          const int ans = parseYesNoIntent(raw);
          if (ans == 1) {
            updateCurrentGuestVisualAttributes(nearest_person_id_.load());
            {
              std::lock_guard<std::mutex> lock(guest_mutex_);
              if (guest_idx_ >= 1 && guests_.size() >= static_cast<size_t>(guest_idx_)) {
                guests_[static_cast<size_t>(guest_idx_ - 1)] = current_guest_;
              }
            }
            RCLCPP_INFO(
              get_logger(),
              "\033[1;32m[DIALOG CONFIRM]\033[0m Heard: \"%s\" | Decision: \033[1;32mYES\033[0m (proceeding to beverage area)",
              raw.c_str());
            transitionTo(ReceptionistState::NAV_TO_BEVERAGE, "Guest info confirmed, proceeding to beverage area");
          } else if (ans == -1) {
            RCLCPP_WARN(
              get_logger(),
              "\033[1;31m[DIALOG CONFIRM]\033[0m Heard: \"%s\" | Decision: \033[1;31mNO\033[0m (restarting name and interest)",
              raw.c_str());
            bool keep_locked_attrs = false;
            {
              std::lock_guard<std::mutex> v_lock(vision_mutex_);
              keep_locked_attrs = attributes_locked_from_scan_;
            }
            {
              std::lock_guard<std::mutex> lock(guest_mutex_);
              if (keep_locked_attrs) {
                current_guest_.name = "";
                current_guest_.interest = "";
                current_guest_.drink = "";
              } else {
                resetGuestProfileFields(current_guest_, guest_idx_);
              }
              if (guests_.size() >= static_cast<size_t>(guest_idx_)) {
                guests_[static_cast<size_t>(guest_idx_ - 1)].name = "";
              }
            }
            if (person_name_pub_ && !keep_locked_attrs) {
              std_msgs::msg::String m;
              m.data = "";
              person_name_pub_->publish(m);
            }
            publishGuestNames();
            speakAsync("I am sorry. Let us start over. Can you tell me your name?", [this]() {
              transitionTo(ReceptionistState::WAIT_NAME, "Restarted name and interest capture");
            });
          } else {
            RCLCPP_WARN(
              get_logger(),
              "\033[1;33m[DIALOG CONFIRM]\033[0m Heard: \"%s\" | Decision: \033[1;33mUNRECOGNIZED\033[0m (reprompting yes/no)",
              raw.c_str());
            state_enter_time_ = now();
            speakAsync("Please say yes robot if it is, or no robot if it is not.");
          }
        } else if ((now() - state_enter_time_).seconds() > dialog_reprompt_sec_) {
          RCLCPP_INFO(
            get_logger(),
            "\033[1;33m[DIALOG CONFIRM]\033[0m Confirmation timeout (%.1fs without speech), reprompting.",
            dialog_reprompt_sec_);
          state_enter_time_ = now();
          speakAsync("Please say yes robot, or no robot.");
        }
        break;

      // ------------------------------------------------------------------------
      // STATE 2: BEVERAGE AREA & DRINK CHECK
      // ------------------------------------------------------------------------
      case ReceptionistState::NAV_TO_BEVERAGE:
        if (last_nav_fail_time_.nanoseconds() > 0 &&
            (now() - last_nav_fail_time_).seconds() < nav_retry_cooldown_sec_)
        {
          return;
        }
        speakAsync("Please follow me to the beverage area.", [this]() {
          navigateTo("beverage_area", [this](bool success) {
            if (!success) {
              return; // Retry navigasi ke beverage_area setelah cooldown
            }
            transitionTo(ReceptionistState::ASK_DRINK, "Arrived at beverage_area");
          });
        });
        break;

      case ReceptionistState::ASK_DRINK: {
        // Wait for ZED RGB-D camera to stabilize after navigation stops at the table
        if ((now() - state_enter_time_).seconds() < vision_initial_settle_sec_) {
          return;
        }

        speakAsync("We've already arrived in Beverage Area, please stand on my left side.", [this](){
          // Freeze table drink detections BEFORE rotating left toward the guest
          const size_t drink_count = snapshotBeveragesOnTable();
          std::string table_status;

          if (!drinks_detected_on_arrival_ || drink_count == 0) {
            table_status = "There are no drinks on this table.";
          } else {
            table_status = "There are " + std::to_string(drink_count) + " drinks on the table.";
          }

          // Announce table drinks, rotate left to face the standing guest, then ask favorite drink
          speakAsync(table_status, [this]() {
            rotateLeft([this](bool /*ok*/, const std::string & /*msg*/) {
              speakAsync("please tell me, what is your favorite drink?", [this]() {
                transitionTo(ReceptionistState::WAIT_DRINK, "Rotated left and asked for favorite drink");
              });
            });
          });
        });
        break;
      }

      case ReceptionistState::WAIT_DRINK:
        if (has_new_utterance_.exchange(false)) {
          std::string raw;
          {
            std::lock_guard<std::mutex> lock(stt_mutex_);
            raw = heard_utterance_;
          }
          const std::string parsed_drink = parseSlotIntent(raw, "drink");
          if (!parsed_drink.empty()) {
            bool need_fallback_attrs = false;
            {
              std::lock_guard<std::mutex> lock(guest_mutex_);
              need_fallback_attrs =
                (current_guest_.shirt_color.empty() || current_guest_.shirt_color == "casual attire");
            }
            if (need_fallback_attrs) {
              updateCurrentGuestVisualAttributes(nearest_person_id_.load());
            }
            {
              std::lock_guard<std::mutex> lock(guest_mutex_);
              current_guest_.drink = parsed_drink;
              if (guests_.size() < static_cast<size_t>(guest_idx_)) {
                guests_.push_back(current_guest_);
              } else if (guest_idx_ >= 1) {
                guests_[static_cast<size_t>(guest_idx_ - 1)] = current_guest_;
              }
            }
            RCLCPP_INFO(
              get_logger(),
              "\033[1;32m[DIALOG DRINK]\033[0m Guest %d Drink -> \033[1;32m\"%s\"\033[0m (from \"%s\")",
              guest_idx_, parsed_drink.c_str(), raw.c_str());

            rotateRight([this](bool /*ok*/, const std::string & /*msg*/) {
              transitionTo(ReceptionistState::CHECK_BEVERAGE, "Favorite drink captured and rotated right");
            });
          } else {
            RCLCPP_WARN(
              get_logger(),
              "\033[1;33m[DIALOG DRINK]\033[0m Unrecognized input \"%s\", asking again.",
              raw.c_str());
            state_enter_time_ = now();
            speakAsync("I am sorry, I did not catch that. What is your favorite drink?");
          }
        } else if ((now() - state_enter_time_).seconds() > dialog_reprompt_sec_) {
          state_enter_time_ = now();
          speakAsync("I am sorry, What is your favorite drink?");
        }
        break;

      case ReceptionistState::CHECK_BEVERAGE: {
        std::string fav_drink;
        {
          std::lock_guard<std::mutex> lock(guest_mutex_);
          fav_drink = current_guest_.drink;
        }
        std::string report;
        if (drinks_detected_on_arrival_) {
          bool found_match = false;
          const std::string side = locateBeverageSide(fav_drink, found_match);
          if (found_match) {
            report = "Your favorite drink is located on the " + side + " side.";
          } else {
            report = "The drink you like is not available on this table.";
          }
        } else {
          report = "The drink you like is not available here.";
        }

        speakAsync(report, [this]() {
          transitionTo(ReceptionistState::NAV_TO_LIVING_ROOM, "Beverage location announced");
        });
        break;
      }

      // ------------------------------------------------------------------------
      // STATE 3: ESCORT TO LIVING ROOM (DESKRIPSI HOST / GUEST 1)
      // ------------------------------------------------------------------------
      case ReceptionistState::NAV_TO_LIVING_ROOM: {
        if (last_nav_fail_time_.nanoseconds() > 0 &&
            (now() - last_nav_fail_time_).seconds() < nav_retry_cooldown_sec_)
        {
          return;
        }

        std::string escort_msg =
          "We are moving to the living room; please follow me slowly from behind.";

        {
          std::lock_guard<std::mutex> lock(guest_mutex_);
          if (guest_idx_ == 1) {
            escort_msg += " A guest is waiting in the living room.";
          } else if (guest_idx_ == 2 && !guests_.empty()) {
            escort_msg +=
              " The first guest, " + guests_[0].name +
              ", is in the living room. " +
              formatPersonDescription(guests_[0]) + ".";
          }
        }

        // jalankan TTS tanpa callback
        speakAsync(escort_msg);

        // langsung jalankan navigasi secara bersamaan
        navigateTo("living_room", [this](bool success) {
          if (!success) {
            return; //retry di tick berikutnya setelah TTS dan cooldown selesai
          }
          transitionTo(
            ReceptionistState::SCAN_LIVING_ROOM,
            "Arrived at living_room; scanning & locking person/chair positions");
        });
        break;

      }

      case ReceptionistState::SCAN_LIVING_ROOM: {
        if ((now() - state_enter_time_).seconds() < vision_initial_settle_sec_) {
          return;
        }
        // Pre-scan the living room to lock yaw angles for Host, Guest 1, and empty seats
        // so subsequent introductions and seat pointing can align quickly.
        scanLivingRoom([this](bool /*success*/, const std::string & /*msg*/) {
          transitionTo(
            ReceptionistState::FACE_AND_INTRODUCE_HOST,
            "Living room scan locked; facing Host to introduce guest");
        });
        break;
      }

      case ReceptionistState::FACE_AND_INTRODUCE_HOST:
        faceHost([this](bool success, const std::string & /*msg*/) {
          PersonProfile cg;
          int g_idx = 1;
          {
            std::lock_guard<std::mutex> lock(guest_mutex_);
            cg = current_guest_;
            g_idx = guest_idx_;
          }

          if (g_idx == 1) {
            // Host is pre-registered from host.jpg / launch params.
            // Do not blindly send set_person_name here to avoid accidentally capturing Guest 1 who is standing nearby.
            RCLCPP_INFO(
              get_logger(),
              "\033[1;32m[HOST]\033[0m Host '%s' verified from pre-registered profile.",
              host_.name.c_str());
          }

          if (!success) {
            RCLCPP_WARN(
              get_logger(),
              "\033[1;33m[LIVING ROOM]\033[0m Face Host gagal/timeout, tetap melanjutkan perkenalan.");
          }

          const std::string intro_guest_to_host =
            "Hello, this is our guest, " + cg.name + ". " +
            formatPersonDescription(cg) + ". Their favorite drink is " +
            cg.drink + ", and their interest is " + cg.interest + ".";

          const std::string intro_host_to_guest =
            cg.name + ", this is our host, " + host_.name + ". " +
            formatPersonDescription(host_) + ". Their favorite drink is " +
            host_.drink + ", and their interest is " + host_.interest + ".";

          speakAsync(intro_guest_to_host, [this, g_idx, intro_host_to_guest]() {
            speakAsync(intro_host_to_guest, [this, g_idx]() {
              if (g_idx == 1) {
                after_return_living_room_state_ = ReceptionistState::SEARCH_CHAIR;
                return_yaw_only_ = false;
                transitionTo(
                  ReceptionistState::RETURN_TO_LIVING_ROOM,
                  "Guest and Host introduced; returning to living_room before searching chair");
              } else {
                after_return_living_room_state_ = ReceptionistState::INTRODUCE_G1_TO_G2;
                return_yaw_only_ = true;
                transitionTo(
                  ReceptionistState::RETURN_TO_LIVING_ROOM,
                  "Guest and Host introduced; resetting yaw to living_room before guest introductions");
              }
            });
          });
        });
        break;

      // ------------------------------------------------------------------------
      // STATE 4: INTRODUCTIONS & SHARED INTERESTS (SAAT BERSAMA TAMU 2)
      // ------------------------------------------------------------------------
      case ReceptionistState::INTRODUCE_G1_TO_G2: {
        {
          std::lock_guard<std::mutex> lock(guest_mutex_);
          if (guests_.size() < 2) {
            RCLCPP_ERROR(
              get_logger(),
              "Insufficient guest profiles (%zu < 2) for introduction!",
              guests_.size());
            transitionTo(ReceptionistState::SEARCH_CHAIR, "Skipped introductions due to missing guest data");
            break;
          }
        }

        faceGuest1([this](bool /*success*/, const std::string & /*msg*/) {
          PersonProfile guest1, guest2;
          {
            std::lock_guard<std::mutex> lock(guest_mutex_);
            if (guests_.size() < 2) {
              transitionTo(ReceptionistState::SEARCH_CHAIR, "Skipped introductions due to missing guest data");
              return;
            }
            guest1 = guests_[0];
            guest2 = guests_[1];
          }

          const std::string text =
            guest1.name + ", I would like to introduce " + guest2.name + ". " +
            formatPersonDescription(guest2) + ". " +
            "Their favorite drink is " + guest2.drink +
            ", and their interest is " + guest2.interest + ".";

          speakAsync(text, [this]() {
            transitionTo(
              ReceptionistState::INTRODUCE_G2_TO_G1,
              "Introduced Guest 2 to Guest 1");
          });
        });
        break;
      }

      case ReceptionistState::INTRODUCE_G2_TO_G1: {
        PersonProfile guest1, guest2;
        {
          std::lock_guard<std::mutex> lock(guest_mutex_);
          if (guests_.size() < 2) {
            transitionTo(ReceptionistState::SEARCH_CHAIR, "Skipped introduction due to missing guest data");
            break;
          }
          guest1 = guests_[0];
          guest2 = guests_[1];
        }

        const std::string text =
          guest2.name + ", this is " + guest1.name + ". " +
          formatPersonDescription(guest1) + ". " +
          "Their favorite drink is " + guest1.drink +
          ", and their interest is " + guest1.interest + ".";

        speakAsync(text, [this]() {
          transitionTo(
            ReceptionistState::EVALUATE_SIMILARITY,
            "Introduced Guest 1 to Guest 2");
        });
        break;
      }

      case ReceptionistState::EVALUATE_SIMILARITY: {
        std::vector<PersonProfile> all_people;
        {
          std::lock_guard<std::mutex> lock(guest_mutex_);
          if (guests_.size() < 2) {
            transitionTo(ReceptionistState::SEARCH_CHAIR, "Skipped similarity evaluation");
            break;
          }
          all_people = {host_, guests_[0], guests_[1]};
        }

        std::string similarity_speech;
        const std::string h_int = normalize(all_people[0].interest);
        const std::string g1_int = normalize(all_people[1].interest);
        const std::string g2_int = normalize(all_people[2].interest);

        if (!h_int.empty() && h_int == g1_int && g1_int == g2_int) {
          similarity_speech =
            "Interestingly, " + all_people[0].name + ", " + all_people[1].name + ", and " +
            all_people[2].name + " all share the same interest in " +
            all_people[1].interest + "!";
        } else {
          for (size_t i = 0; i < all_people.size(); ++i) {
            for (size_t j = i + 1; j < all_people.size(); ++j) {
              const std::string int_i = normalize(all_people[i].interest);
              const std::string int_j = normalize(all_people[j].interest);
              if (!int_i.empty() && int_i == int_j) {
                similarity_speech =
                  "Interestingly, both " + all_people[i].name + " and " +
                  all_people[j].name + " share the same interest in " +
                  all_people[i].interest + "!";
                break;
              }
            }
            if (!similarity_speech.empty()) {
              break;
            }
          }
        }

        if (!similarity_speech.empty()) {
          speakAsync(similarity_speech, [this]() {
            after_return_living_room_state_ = ReceptionistState::SEARCH_CHAIR;
            transitionTo(
              ReceptionistState::RETURN_TO_LIVING_ROOM,
              "Shared interest announced; returning to living_room before searching chair");
          });
        } else {
          after_return_living_room_state_ = ReceptionistState::SEARCH_CHAIR;
          transitionTo(
            ReceptionistState::RETURN_TO_LIVING_ROOM,
            "No identical interest found; returning to living_room before searching chair");
        }
        break;
      }

      // ------------------------------------------------------------------------
      // STATE 4.5: RETURN TO LIVING_ROOM CENTER REFERENCE AFTER INTRODUCTIONS
      // Re-centers base pose at the living_room waypoint before rotating to the
      // next person or searching for an empty chair (after_return_living_room_state_).
      // ------------------------------------------------------------------------
      case ReceptionistState::RETURN_TO_LIVING_ROOM: {
        const bool yaw_only = return_yaw_only_;
        return_yaw_only_ = false;

        if (yaw_only) {
          resetLivingRoomYaw([this](bool success, const std::string & /*msg*/) {
            if (!success) {
              RCLCPP_WARN(
                get_logger(),
                "\033[1;33m[LIVING ROOM]\033[0m reset_living_room_yaw setelah perkenalan gagal/timeout, tetap lanjut ke %s.",
                stateToString(after_return_living_room_state_).c_str());
            }
            transitionTo(
              after_return_living_room_state_,
              "Reset yaw to living_room after introduction");
          });
        } else {
          navigateTo("living_room", [this](bool success) {
            if (!success) {
              RCLCPP_WARN(
                get_logger(),
                "\033[1;33m[LIVING ROOM]\033[0m go_to_living_room setelah perkenalan gagal/timeout, tetap lanjut ke %s.",
                stateToString(after_return_living_room_state_).c_str());
            }
            transitionTo(
              after_return_living_room_state_,
              "Returned to living_room after introduction");
          });
        }
        break;
      }

      // ------------------------------------------------------------------------
      // STATE 5: CARI KURSI KOSONG DENGAN BERPUTAR & ALIGNMENT (/search_chair)
      // ------------------------------------------------------------------------
      case ReceptionistState::SEARCH_CHAIR:
        searchChair([this](bool /*success*/, const std::string & /*msg*/) {
          transitionTo(
            ReceptionistState::OFFER_SEAT,
            "Chair search finished; proceeding to offer seat");
        });
        break;

      // ------------------------------------------------------------------------
      // STATE 5: DETEKSI KURSI KOSONG, SERVO (/arm/ambil_tas), & TUNGGU DUDUK
      // ------------------------------------------------------------------------
      case ReceptionistState::OFFER_SEAT: {
        // Tunggu kamera ZED settle setelah tiba di living room
        if ((now() - state_enter_time_).seconds() < vision_initial_settle_sec_) {
          return;
        }

        pointToSeatWithArm([this](bool /*success*/) {
          speakAsync(
            "please take a seat in the empty chair on the side that I am indicating.",
            [this]() {
              // Setelah selesai menunjuk dan menawarkan kursi, matikan daya lengan (mode free)
              freeArm([this](bool /*ok*/) {
                transitionTo(
                  ReceptionistState::WAIT_GUEST_SIT,
                  "Waiting for guest to occupy the indicated chair");
              });
            });
        });
        break;
      }

      case ReceptionistState::WAIT_GUEST_SIT: {
        // FIX: Beri waktu minimal (min_sit_wait_sec_ = 4.0s) agar tamu berjalan ke kursi,
        // sehingga Host yang sudah duduk tidak langsung memicu state ini selesai dalam 100ms.
        const double wait_elapsed = (now() - state_enter_time_).seconds();
        if (wait_elapsed < min_sit_wait_sec_) {
          return;
        }

        if (isPersonInFront() || wait_elapsed > 10.0) {
          std::string cg_name;
          {
            std::lock_guard<std::mutex> lock(guest_mutex_);
            cg_name = current_guest_.name;
          }
          publishTaskStatus(
            "SUCCESS: Guest " + std::to_string(guest_idx_) + " (" +
            cg_name + ") occupied the indicated chair.");

          // Release locked living-room yaw targets before starting Guest 2 or completing
          const int finished_guest_idx = guest_idx_;
          clearLivingRoomLocks([this, finished_guest_idx]() {
            if (finished_guest_idx == 1) {
              initCurrentGuest(2);
              transitionTo(
                ReceptionistState::NAV_TO_DOOR,
                "Guest 1 seated confirmed; cleared locks and returning to door for Guest 2");
            } else {
              transitionTo(
                ReceptionistState::FINISH_TASK,
                "Guest 2 seated confirmed; cleared locks and completing task");
            }
          });
        } else {
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 3000,
            "\033[1;33m[VISION WAIT]\033[0m Waiting for Guest %d to occupy the indicated chair...",
            guest_idx_);
        }
        break;
      }

      // ------------------------------------------------------------------------
      // STATE 6: PENYELESAIAN TUGAS RECEPTIONIST
      // ------------------------------------------------------------------------
      case ReceptionistState::FINISH_TASK:
        speakAsync("The second guest is now seated in the provided spot.", [this]() {
          speakAsync("Receptionist task complete.", [this]() {
            publishTaskStatus("SUCCESS: Receptionist task complete.");
            RCLCPP_INFO(get_logger(), "\033[1;36m[RECEPTIONIST]\033[0m\n  Status: Task completed successfully");
            transitionTo(ReceptionistState::COMPLETED, "Receptionist task complete");
          });
        });
        break;

      case ReceptionistState::COMPLETED:
      case ReceptionistState::ABORTED:
        break;
    }
  }
};

// ==============================================================================
// 4. MAIN ENTRY POINT
// ==============================================================================
int main(int argc, char ** argv)
{
  setenv("RCUTILS_CONSOLE_OUTPUT_FORMAT", "[{severity}] {message}", 1);
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<ReceptionistTaskNode>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception & e) {
    RCLCPP_FATAL(
      rclcpp::get_logger("receptionist_manager"),
      "Gagal menjalankan receptionist_manager (pastikan file YAML parameter dimuat): %s",
      e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}