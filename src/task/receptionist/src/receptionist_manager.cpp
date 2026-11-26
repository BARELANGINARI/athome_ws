// ==============================================================================
// Barelang Inari - RoboCup@Home 2025 (Sec. 5.3 Receptionist)
// Package  : receptionist
// Node     : receptionist_manager (ReceptionistTaskNode)
//
// Alur Strategi (< 6:00 menit, tanpa dynamic gaze & tanpa buka pintu):
//   Guest 1 : Entrance Door (align -> scan 3 atribut konsensus -> approach ->
//             name & interest + confirm) -> Beverage Area (snapshot -> drink +
//             confirm -> posisi kiri/tengah/kanan) -> Living Room (scan ->
//             search_chair -> offer_seat -> wait_sit -> clear_locks)
//   Guest 2 : Entrance Door -> Beverage Area -> ucapkan deskripsi visual Guest 1
//             -> Living Room -> scan -> search_chair -> offer_seat -> wait_sit
//             -> clear_locks -> reset_yaw -> RESCAN (antisipasi switching places)
//             -> INTRODUCE_G1_TO_G2 -> INTRODUCE_G2_TO_G1 -> EVALUATE_SIMILARITY
//             -> clear_locks -> FINISH_TASK
// ==============================================================================

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
  INIT,
  // Phase 1 — Entrance Door
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
  // Phase 2 — Beverage Area
  NAV_TO_BEVERAGE,
  ASK_DRINK,
  WAIT_DRINK,
  CONFIRM_DRINK,
  WAIT_CONFIRM_DRINK,
  CHECK_BEVERAGE,
  // Phase 3 — Living Room & Seat Offering
  NAV_TO_LIVING_ROOM,
  SCAN_LIVING_ROOM,
  SEARCH_CHAIR,
  OFFER_SEAT,
  WAIT_GUEST_SIT,
  RETURN_TO_LIVING_ROOM,
  // Phase 4 — Post-Seating Introductions & Similarity (Guest 2 cycle)
  INTRODUCE_G1_TO_G2,
  INTRODUCE_G2_TO_G1,
  EVALUATE_SIMILARITY,
  // Phase 5 — Terminal States
  FINISH_TASK,
  COMPLETED,
  ABORTED
};

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
    case ReceptionistState::CONFIRM_DRINK:           return "CONFIRM_DRINK";
    case ReceptionistState::WAIT_CONFIRM_DRINK:      return "WAIT_CONFIRM_DRINK";
    case ReceptionistState::CHECK_BEVERAGE:          return "CHECK_BEVERAGE";
    case ReceptionistState::NAV_TO_LIVING_ROOM:      return "NAV_TO_LIVING_ROOM";
    case ReceptionistState::SCAN_LIVING_ROOM:        return "SCAN_LIVING_ROOM";
    case ReceptionistState::SEARCH_CHAIR:            return "SEARCH_CHAIR";
    case ReceptionistState::OFFER_SEAT:              return "OFFER_SEAT";
    case ReceptionistState::WAIT_GUEST_SIT:          return "WAIT_GUEST_SIT";
    case ReceptionistState::RETURN_TO_LIVING_ROOM:   return "RETURN_TO_LIVING_ROOM";
    case ReceptionistState::INTRODUCE_G1_TO_G2:      return "INTRODUCE_G1_TO_G2";
    case ReceptionistState::INTRODUCE_G2_TO_G1:      return "INTRODUCE_G2_TO_G1";
    case ReceptionistState::EVALUATE_SIMILARITY:     return "EVALUATE_SIMILARITY";
    case ReceptionistState::FINISH_TASK:             return "FINISH_TASK";
    case ReceptionistState::COMPLETED:               return "COMPLETED";
    case ReceptionistState::ABORTED:                 return "ABORTED";
    default:                                         return "UNKNOWN";
  }
}

// ==============================================================================
// 2. DATA STRUCTURES
// ==============================================================================
struct PersonProfile {
  std::string id;
  std::string name;
  std::string drink;
  std::string interest;
  std::string gender{"unknown"};
  std::string shirt_color;
  std::string hair_color;
  std::string pants_color;
  bool has_hat{false};
  bool has_glasses{false};
  std::vector<std::string> attributes;
};

struct DetectedObject {
  std::string name;
  int bbox_x{0};
  int bbox_y{0};
  int bbox_w{0};
  int bbox_h{0};
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

// ==============================================================================
// 3. MAIN ROS 2 NODE: ReceptionistTaskNode
// ==============================================================================
class ReceptionistTaskNode : public rclcpp::Node {
public:
  using Trigger = std_srvs::srv::Trigger;
  using TriggerClient = rclcpp::Client<Trigger>;
  using TriggerResult = std::function<void(bool, const std::string &)>;

  ReceptionistTaskNode()
  : Node("receptionist_manager")
  {
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
    nav_service_timeout_sec_ = requireYamlParameter<double>("nav_service_timeout_sec");
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

    auto loadSortNormList = [this](const std::string & param_name) {
      std::vector<std::string> raw = requireYamlParameter<std::vector<std::string>>(param_name);
      std::vector<std::string> out;
      out.reserve(raw.size());
      for (const auto & item : raw) {
        const std::string n = normalize(item);
        if (!n.empty()) out.push_back(n);
      }
      std::sort(out.begin(), out.end(), [](const std::string & a, const std::string & b) {
        return a.size() > b.size();
      });
      return out;
    };
    known_names_ = loadSortNormList("known_names");
    known_drinks_ = loadSortNormList("known_drinks");
    known_interests_ = loadSortNormList("known_interests");

    callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.callback_group = callback_group_;

    // Publishers & Subscribers HRI
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

    // Subscribers Vision
    person_sub_ = create_subscription<std_msgs::msg::String>(
      "/vision/person_label", rclcpp::SensorDataQoS(),
      std::bind(&ReceptionistTaskNode::onPersonLabel, this, std::placeholders::_1),
      sub_opts);

    drink_sub_ = create_subscription<std_msgs::msg::String>(
      "/vision/drink_detections", rclcpp::SensorDataQoS(),
      std::bind(&ReceptionistTaskNode::onDrinkDetections, this, std::placeholders::_1),
      sub_opts);

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

    // Service Clients: Navigation & Arm
    nav_door_client_ = create_client<Trigger>(
      "/receptionist/go_to_door", rclcpp::ServicesQoS(), callback_group_);
    nav_beverage_client_ = create_client<Trigger>(
      "/receptionist/go_to_beverage_area", rclcpp::ServicesQoS(), callback_group_);
    nav_living_room_client_ = create_client<Trigger>(
      "/receptionist/go_to_living_room", rclcpp::ServicesQoS(), callback_group_);

    search_chair_client_ = create_client<Trigger>(
      "/search_chair", rclcpp::ServicesQoS(), callback_group_);
    face_guest1_client_ = create_client<Trigger>(
      "/receptionist/guest1", rclcpp::ServicesQoS(), callback_group_);
    face_guest2_client_ = create_client<Trigger>(
      "/receptionist/guest2", rclcpp::ServicesQoS(), callback_group_);
    adjust_guest_client_ = create_client<Trigger>(
      "/receptionist/adjust_guest", rclcpp::ServicesQoS(), callback_group_);
    approach_guest_client_ = create_client<Trigger>(
      "/receptionist/approach_guest", rclcpp::ServicesQoS(), callback_group_);
    rotate_left_client_ = create_client<Trigger>(
      "/receptionist/left", rclcpp::ServicesQoS(), callback_group_);
    scan_living_room_client_ = create_client<Trigger>(
      "/receptionist/scan_living_room", rclcpp::ServicesQoS(), callback_group_);
    clear_locks_client_ = create_client<Trigger>(
      "/receptionist/clear_living_room_locks", rclcpp::ServicesQoS(), callback_group_);
    reset_living_room_yaw_client_ = create_client<Trigger>(
      "/receptionist/reset_living_room_yaw", rclcpp::ServicesQoS(), callback_group_);

    arm_ambil_tas_simple_client_ = create_client<Trigger>(
      "/arm/ambil_tas_simple", rclcpp::ServicesQoS(), callback_group_);
    arm_point_position_client_ = create_client<Trigger>(
      "/arm/point_position", rclcpp::ServicesQoS(), callback_group_);
    arm_point_to_chair_client_ = create_client<Trigger>(
      "/arm/point_to_chair", rclcpp::ServicesQoS(), callback_group_);
    arm_free_client_ = create_client<Trigger>(
      "/arm/free", rclcpp::ServicesQoS(), callback_group_);

    fsm_timer_ = create_wall_timer(
      100ms, std::bind(&ReceptionistTaskNode::fsmTick, this));

    task_start_time_ = now();
    state_enter_time_ = now();
    last_wait_prompt_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);

    guests_.clear();
    initCurrentGuest(1);

    RCLCPP_INFO(get_logger(),
      "\033[1;32m[RECEPTIONIST]\033[0m Task Manager started (Host: %s, Drink: %s, Interest: %s)",
      host_.name.c_str(), host_.drink.c_str(), host_.interest.c_str());
  }

private:
  // Ambang konsensus atribut visual untuk menghindari penalti -30 per atribut salah
  static constexpr int kMinAttrSamples = 3;
  static constexpr double kMinAttrConsensusRatio = 0.65;

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

  // Parameter Waktu & Timeout (dari YAML)
  double global_timeout_sec_{0.0};
  double nav_service_timeout_sec_{50.0};
  double dialog_reprompt_sec_{0.0};
  double wait_guest_prompt_sec_{0.0};
  double vision_initial_settle_sec_{0.0};
  double min_sit_wait_sec_{0.0};
  double person_freshness_sec_{0.0};
  double tts_min_timeout_sec_{0.0};
  double tts_sec_per_word_{0.0};
  double tts_timeout_margin_sec_{0.0};
  double nav_retry_cooldown_sec_{0.0};

  // State & Profile Store
  struct CallbackScope {
    std::atomic<int> & counter;
    explicit CallbackScope(std::atomic<int> & c) : counter(c) { counter.fetch_add(1); }
    ~CallbackScope() { counter.fetch_sub(1); }
  };
  struct CallbackExit {
    std::atomic<int> & counter;
    ~CallbackExit() { counter.fetch_sub(1); }
  };

  std::atomic<ReceptionistState> state_{ReceptionistState::INIT};
  std::atomic<bool> speaking_{false};
  std::atomic<bool> service_busy_{false};
  std::atomic<int> in_callback_depth_{0};
  std::atomic<bool> person_seen_{false};
  std::atomic<bool> standing_guest_seen_{false};
  std::atomic<int> nearest_person_id_{-1};

  std::mutex guest_mutex_;
  PersonProfile host_;
  std::vector<PersonProfile> guests_;
  PersonProfile current_guest_;
  int guest_idx_{1};
  int search_chair_retries_{0};
  int face_g1_retries_{0};
  int face_g2_retries_{0};
  std::vector<std::string> known_names_;
  std::vector<std::string> known_drinks_;
  std::vector<std::string> known_interests_;
  ReceptionistState after_return_living_room_state_{ReceptionistState::SEARCH_CHAIR};
  ReceptionistState after_scan_living_room_state_{ReceptionistState::SEARCH_CHAIR};
  bool return_yaw_only_{false};
  int camera_width_{0};

  // Vision Perception Data
  mutable std::mutex vision_mutex_;
  std::vector<DetectedObject> latest_drinks_;
  rclcpp::Time last_drink_msg_time_{0, 0, RCL_ROS_TIME};
  std::vector<DetectedObject> table_drinks_snapshot_;
  bool drinks_detected_on_arrival_{false};
  bool beverage_snapshot_locked_{false};
  std::vector<std::string> latest_attributes_;
  std::unordered_map<std::string, int> attr_gender_counts_;
  std::unordered_map<std::string, int> attr_shirt_counts_;
  std::unordered_map<std::string, int> attr_hair_counts_;
  std::unordered_map<std::string, int> attr_pants_counts_;
  int attr_hat_true_count_{0};
  int attr_glasses_true_count_{0};
  int attr_total_frames_{0};
  bool attributes_locked_from_scan_{false};
  std::string active_vision_reg_name_;
  std::string empty_seat_position_{"center"};
  bool has_explicit_empty_seat_{false};
  int nearest_person_bbox_x_{0};
  int nearest_person_bbox_w_{0};
  int nearest_person_bbox_y_{0};
  int nearest_person_bbox_h_{0};

  // STT / Dialog Data
  std::mutex stt_mutex_;
  std::string heard_utterance_;
  std::atomic<bool> has_new_utterance_{false};
  std::string last_utterance_;
  rclcpp::Time last_utterance_time_{0, 0, RCL_ROS_TIME};

  // Timing & Watchdogs
  rclcpp::Time task_start_time_;
  rclcpp::Time state_enter_time_;
  rclcpp::Time last_wait_prompt_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_person_seen_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_standing_guest_seen_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time tts_cooldown_until_{0, 0, RCL_ROS_TIME};
  rclcpp::Time tts_started_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_nav_fail_time_{0, 0, RCL_ROS_TIME};

  mutable std::mutex tts_mutex_;
  double tts_timeout_sec_{4.0};
  std::string active_tts_text_;
  std::string last_tts_text_norm_;
  rclcpp::Time last_tts_done_time_{0, 0, RCL_ROS_TIME};
  std::function<void()> pending_tts_done_;

  std::mutex srv_mutex_;
  rclcpp::Time srv_started_time_{0, 0, RCL_ROS_TIME};
  double srv_timeout_sec_{120.0};
  std::string active_srv_name_;
  TriggerResult pending_srv_result_;

  // ROS 2 Interfaces
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
  TriggerClient::SharedPtr face_guest1_client_;
  TriggerClient::SharedPtr face_guest2_client_;
  TriggerClient::SharedPtr adjust_guest_client_;
  TriggerClient::SharedPtr approach_guest_client_;
  TriggerClient::SharedPtr rotate_left_client_;
  TriggerClient::SharedPtr scan_living_room_client_;
  TriggerClient::SharedPtr clear_locks_client_;
  TriggerClient::SharedPtr reset_living_room_yaw_client_;

  TriggerClient::SharedPtr arm_ambil_tas_simple_client_;
  TriggerClient::SharedPtr arm_point_position_client_;
  TriggerClient::SharedPtr arm_point_to_chair_client_;
  TriggerClient::SharedPtr arm_free_client_;

  // ============================================================================
  // CONCURRENCY GATES & STATUS HELPERS
  // ============================================================================
  bool isBusy() const
  {
    return speaking_.load() || service_busy_.load() || in_callback_depth_.load() > 0;
  }

  static bool isTerminalState(ReceptionistState s)
  {
    return s == ReceptionistState::COMPLETED || s == ReceptionistState::ABORTED;
  }

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

  void publishExpression(const std::string & expr)
  {
    if (face_expression_pub_) {
      std_msgs::msg::String msg;
      msg.data = expr;
      face_expression_pub_->publish(msg);
    }
  }

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
      case ReceptionistState::CONFIRM_DRINK:
      case ReceptionistState::INTRODUCE_G1_TO_G2:
      case ReceptionistState::INTRODUCE_G2_TO_G1:
        return "GREETING";
      case ReceptionistState::WAIT_NAME:
      case ReceptionistState::WAIT_INTEREST:
      case ReceptionistState::WAIT_CONFIRM_GUEST_INFO:
      case ReceptionistState::WAIT_DRINK:
      case ReceptionistState::WAIT_CONFIRM_DRINK:
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

  void resetGuestProfileFields(PersonProfile & profile, int idx)
  {
    profile = PersonProfile{};
    profile.id = "guest_" + std::to_string(idx);
    profile.gender = "unknown";
    profile.shirt_color.clear();
    profile.hair_color.clear();
    profile.pants_color.clear();
    profile.has_hat = false;
    profile.has_glasses = false;
  }

  void initCurrentGuest(int idx)
  {
    {
      std::lock_guard<std::mutex> lock(guest_mutex_);
      guest_idx_ = idx;
      resetGuestProfileFields(current_guest_, idx);
    }
    search_chair_retries_ = 0;
    face_g1_retries_ = 0;
    face_g2_retries_ = 0;
    after_scan_living_room_state_ = ReceptionistState::SEARCH_CHAIR;

    if (person_name_pub_) {
      std_msgs::msg::String m;
      m.data = "";
      person_name_pub_->publish(m);
    }

    person_seen_ = false;
    standing_guest_seen_ = false;
    {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      drinks_detected_on_arrival_ = false;
      table_drinks_snapshot_.clear();
      beverage_snapshot_locked_ = false;
      latest_drinks_.clear();
      latest_attributes_.clear();
      attr_gender_counts_.clear();
      attr_shirt_counts_.clear();
      attr_hair_counts_.clear();
      attr_pants_counts_.clear();
      attr_hat_true_count_ = 0;
      attr_glasses_true_count_ = 0;
      attr_total_frames_ = 0;
      attributes_locked_from_scan_ = false;
      active_vision_reg_name_.clear();
      has_explicit_empty_seat_ = false;
      nearest_person_bbox_w_ = 0;
    }
  }

  void transitionTo(ReceptionistState next_state, const std::string & reason)
  {
    const ReceptionistState cur = state_.load();
    if (isTerminalState(cur)) {
      RCLCPP_WARN(
        get_logger(),
        "\033[1;33m[STATE]\033[0m Ignoring transition %s -> %s (already terminal)",
        stateToString(cur).c_str(),
        stateToString(next_state).c_str());
      return;
    }

    RCLCPP_INFO(
      get_logger(),
      "\033[1;34m[STATE]\033[0m %s -> %s (G%d: %s)",
      stateToString(cur).c_str(),
      stateToString(next_state).c_str(),
      guest_idx_,
      reason.c_str());

    if (next_state == ReceptionistState::WAIT_GUEST ||
        next_state == ReceptionistState::WAIT_GUEST_SIT ||
        next_state == ReceptionistState::ASK_DRINK ||
        next_state == ReceptionistState::OFFER_SEAT ||
        next_state == ReceptionistState::INTRODUCE_G1_TO_G2)
    {
      person_seen_ = false;
      standing_guest_seen_ = false;
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

  // ============================================================================
  // NAVIGATION, ARM & ALIGNMENT SERVICE WRAPPERS
  // ============================================================================
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

    RCLCPP_INFO(get_logger(), "\033[1;36m[NAV]\033[0m Navigating to '%s' (%s)",
      waypoint_name.c_str(), srv_name.c_str());

    const rclcpp::Time nav_start = now();
    callTrigger(
      client, srv_name,
      [this, waypoint_name, srv_name, nav_start, on_finished](bool success, const std::string & msg) {
        const double elapsed = (now() - nav_start).seconds();
        if (!success) {
          RCLCPP_ERROR(
            get_logger(), "\033[1;31m[NAV]\033[0m Navigation to '%s' FAILED (%s)",
            waypoint_name.c_str(), msg.c_str());
          last_nav_fail_time_ = now();
        } else {
          if (elapsed < 0.5) {
            RCLCPP_WARN(
              get_logger(), "\033[1;33m[NAV]\033[0m %s returned TRUE in %.2fs (< 0.5s)",
              srv_name.c_str(), elapsed);
          }
          RCLCPP_INFO(
            get_logger(), "\033[1;32m[NAV]\033[0m Reached '%s' in %.1fs",
            waypoint_name.c_str(), elapsed);
        }
        if (on_finished) on_finished(success);
      },
      nav_service_timeout_sec_);
  }

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

    RCLCPP_INFO(get_logger(), "\033[1;36m[ARM]\033[0m Point seat (%s)", srv_name.c_str());
    callTrigger(
      client, srv_name,
      [this, on_finished](bool success, const std::string & msg) {
        if (!success) {
          RCLCPP_WARN(get_logger(), "\033[1;33m[ARM]\033[0m Point seat warning: %s", msg.c_str());
        }
        if (on_finished) on_finished(success);
      },
      15.0);
  }

  void freeArm(std::function<void(bool)> on_finished = nullptr)
  {
    callTrigger(
      arm_free_client_, "/arm/free",
      [this, on_finished](bool success, const std::string & msg) {
        if (!success) {
          RCLCPP_WARN(get_logger(), "\033[1;33m[ARM]\033[0m Free mode warning: %s", msg.c_str());
        }
        if (on_finished) on_finished(success);
      },
      5.0);
  }

  void searchChair(std::function<void(bool, const std::string &)> on_finished)
  {
    callTrigger(
      search_chair_client_, "/search_chair",
      [this, on_finished](bool success, const std::string & msg) {
        RCLCPP_INFO(get_logger(), "[CHAIR] %s (%s)", success ? "OK" : "WARN", msg.c_str());
        if (on_finished) on_finished(success, msg);
      },
      45.0);
  }

  void faceGuest1(std::function<void(bool, const std::string &)> on_finished)
  {
    callTrigger(
      face_guest1_client_, "/receptionist/guest1",
      [this, on_finished](bool success, const std::string & msg) {
        RCLCPP_INFO(get_logger(), "[GUEST 1] %s (%s)", success ? "OK" : "WARN", msg.c_str());
        if (on_finished) on_finished(success, msg);
      },
      35.0);
  }

  void faceGuest2(std::function<void(bool, const std::string &)> on_finished)
  {
    callTrigger(
      face_guest2_client_, "/receptionist/guest2",
      [this, on_finished](bool success, const std::string & msg) {
        RCLCPP_INFO(get_logger(), "[GUEST 2] %s (%s)", success ? "OK" : "WARN", msg.c_str());
        if (on_finished) on_finished(success, msg);
      },
      25.0);
  }

  void scanLivingRoom(std::function<void(bool, const std::string &)> on_finished)
  {
    callTrigger(
      scan_living_room_client_, "/receptionist/scan_living_room",
      [this, on_finished](bool success, const std::string & msg) {
        RCLCPP_INFO(get_logger(), "[LIVING ROOM SCAN] %s (%s)", success ? "OK" : "WARN", msg.c_str());
        if (on_finished) on_finished(success, msg);
      },
      25.0);
  }

  void clearLivingRoomLocks(std::function<void()> on_finished)
  {
    callTrigger(
      clear_locks_client_, "/receptionist/clear_living_room_locks",
      [this, on_finished](bool /*success*/, const std::string & msg) {
        RCLCPP_INFO(get_logger(), "[LIVING ROOM] Clear locks: %s", msg.c_str());
        if (on_finished) on_finished();
      },
      5.0);
  }

  void resetLivingRoomYaw(std::function<void(bool, const std::string &)> on_finished)
  {
    callTrigger(
      reset_living_room_yaw_client_, "/receptionist/reset_living_room_yaw",
      [this, on_finished](bool success, const std::string & msg) {
        RCLCPP_INFO(get_logger(), "[LIVING ROOM YAW] %s (%s)", success ? "OK" : "WARN", msg.c_str());
        if (on_finished) on_finished(success, msg);
      },
      10.0);
  }

  void rotateLeft(std::function<void(bool, const std::string &)> on_finished)
  {
    callTrigger(
      rotate_left_client_, "/receptionist/left",
      [this, on_finished](bool success, const std::string & msg) {
        RCLCPP_INFO(get_logger(), "[ROTATE LEFT] %s (%s)", success ? "OK" : "WARN", msg.c_str());
        if (on_finished) on_finished(success, msg);
      },
      20.0);
  }

  void adjustGuestDistance(std::function<void(bool, const std::string &)> on_finished)
  {
    callTrigger(
      adjust_guest_client_, "/receptionist/adjust_guest",
      [this, on_finished](bool success, const std::string & msg) {
        RCLCPP_INFO(get_logger(), "[ADJUST GUEST] %s (%s)", success ? "OK" : "WARN", msg.c_str());
        if (on_finished) on_finished(success, msg);
      },
      25.0);
  }

  void approachGuest(std::function<void(bool, const std::string &)> on_finished)
  {
    callTrigger(
      approach_guest_client_, "/receptionist/approach_guest",
      [this, on_finished](bool success, const std::string & msg) {
        RCLCPP_INFO(get_logger(), "[APPROACH GUEST] %s (%s)", success ? "OK" : "WARN", msg.c_str());
        if (on_finished) on_finished(success, msg);
      },
      25.0);
  }

  // ============================================================================
  // BEVERAGE & VISUAL ATTRIBUTE HELPERS
  // ============================================================================
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
    if (n == "cheezit" || n == "cheez it") {
      return "cheez it";
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

  size_t snapshotBeveragesOnTable()
  {
    std::lock_guard<std::mutex> lock(vision_mutex_);
    beverage_snapshot_locked_ = true;
    mergeDrinksIntoSnapshotLocked(latest_drinks_);
    drinks_detected_on_arrival_ = !table_drinks_snapshot_.empty();
    return table_drinks_snapshot_.size();
  }

  std::string locateBeverageSide(const std::string & favorite_drink, bool & found_match)
  {
    std::lock_guard<std::mutex> lock(vision_mutex_);
    found_match = false;
    const std::string target_raw = normalize(favorite_drink);
    const std::string target = canonicalDrinkName(favorite_drink);

    const auto & raw_list = !table_drinks_snapshot_.empty()
      ? table_drinks_snapshot_
      : latest_drinks_;

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
      if (!exists) drinks_list.push_back(item);
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
      if (drinks_list.size() == 2) {
        return (idx == 0) ? "left" : "right";
      }
      if (drinks_list.size() >= 3) {
        if (idx == 0) return "left";
        if (idx == drinks_list.size() - 1) return "right";

        const double cx = centerX(drinks_list[idx]);
        const double cx_min = centerX(drinks_list.front());
        const double cx_max = centerX(drinks_list.back());
        const double span = cx_max - cx_min;
        const double cluster_center = (cx_min + cx_max) * 0.5;
        const double ratio = (span > 1.0) ? ((cx - cx_min) / span) : 0.5;
        const bool is_centered_in_cluster = (ratio >= 0.28 && ratio <= 0.72) &&
                                            (std::abs(cx - cluster_center) <= span * 0.22);
        const bool is_centered_in_camera = (std::abs(cx - cam_center) <= cam_w * 0.12);
        if (is_centered_in_cluster || is_centered_in_camera) {
          return "center";
        }
        return (cx < cluster_center) ? "left" : "right";
      }
      const double cx = centerX(drinks_list[idx]);
      const double deadband = cam_w * 0.10;
      if (cx < cam_center - deadband) return "left";
      if (cx > cam_center + deadband) return "right";
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
          "\033[1;32m[BEVERAGE]\033[0m Match '%s' on %s (rank %zu/%zu, cx=%.0f)",
          drink.name.c_str(), side.c_str(), i + 1, drinks_list.size(), centerX(drink));
        return side;
      }
    }
    return "center";
  }

  /// Hanya menerima atribut yang lolos syarat konsensus (total >= kMinAttrSamples
  /// & rasio dominan >= kMinAttrConsensusRatio) agar tidak memicu penalti -30 pts.
  void updateCurrentGuestVisualAttributes(int person_id)
  {
    std::unordered_map<std::string, int> gender_counts;
    std::unordered_map<std::string, int> shirt_counts;
    std::unordered_map<std::string, int> hair_counts;
    std::unordered_map<std::string, int> pants_counts;
    int hat_cnt = 0;
    int glasses_cnt = 0;
    int total_frames = 0;
    {
      std::lock_guard<std::mutex> v_lock(vision_mutex_);
      if (attributes_locked_from_scan_) {
        return;
      }
      gender_counts = attr_gender_counts_;
      shirt_counts = attr_shirt_counts_;
      hair_counts = attr_hair_counts_;
      pants_counts = attr_pants_counts_;
      hat_cnt = attr_hat_true_count_;
      glasses_cnt = attr_glasses_true_count_;
      total_frames = attr_total_frames_;
    }

    auto pickConsensus = [](
      const std::unordered_map<std::string, int> & counts,
      int & out_best_cnt,
      int & out_total) -> std::string
    {
      out_best_cnt = 0;
      out_total = 0;
      std::string best_val;
      for (const auto & kv : counts) {
        out_total += kv.second;
        if (kv.second > out_best_cnt) {
          out_best_cnt = kv.second;
          best_val = kv.first;
        }
      }
      if (out_total < kMinAttrSamples || out_best_cnt == 0) {
        return "";
      }
      const double ratio = static_cast<double>(out_best_cnt) / static_cast<double>(out_total);
      return (ratio >= kMinAttrConsensusRatio) ? best_val : "";
    };

    int g_cnt = 0, g_tot = 0;
    int s_cnt = 0, s_tot = 0;
    int h_cnt = 0, h_tot = 0;
    int p_cnt = 0, p_tot = 0;
    const std::string consensus_gender = pickConsensus(gender_counts, g_cnt, g_tot);
    const std::string consensus_shirt = pickConsensus(shirt_counts, s_cnt, s_tot);
    const std::string consensus_hair = pickConsensus(hair_counts, h_cnt, h_tot);
    const std::string consensus_pants = pickConsensus(pants_counts, p_cnt, p_tot);
    const bool consensus_hat =
      (total_frames >= kMinAttrSamples && hat_cnt >= kMinAttrSamples &&
       (static_cast<double>(hat_cnt) / static_cast<double>(total_frames)) >= kMinAttrConsensusRatio);
    const bool consensus_glasses =
      (total_frames >= kMinAttrSamples && glasses_cnt >= kMinAttrSamples &&
       (static_cast<double>(glasses_cnt) / static_cast<double>(total_frames)) >= kMinAttrConsensusRatio);

    std::lock_guard<std::mutex> g_lock(guest_mutex_);
    current_guest_.gender = consensus_gender.empty() ? "unknown" : consensus_gender;
    current_guest_.shirt_color = consensus_shirt;
    current_guest_.hair_color = consensus_hair;
    current_guest_.pants_color = consensus_pants;
    current_guest_.has_hat = consensus_hat;
    current_guest_.has_glasses = consensus_glasses;

    std::vector<std::string> attrs;
    if (current_guest_.gender == "male" || current_guest_.gender == "female") {
      attrs.push_back(current_guest_.gender);
    }
    if (!current_guest_.shirt_color.empty()) {
      attrs.push_back(current_guest_.shirt_color);
    }
    if (!current_guest_.pants_color.empty()) {
      attrs.push_back(current_guest_.pants_color);
    }
    if (!current_guest_.hair_color.empty()) {
      attrs.push_back(current_guest_.hair_color);
    }
    if (current_guest_.has_hat) {
      attrs.push_back("wearing hat");
    }
    if (current_guest_.has_glasses) {
      attrs.push_back("wearing glasses");
    }
    current_guest_.attributes = attrs;

    RCLCPP_INFO(
      get_logger(),
      "\033[1;32m[VISION]\033[0m G%d (id=%d) Consensus Profile -> "
      "Gender='%s' (%d/%d), Shirt='%s' (%d/%d), Pants='%s' (%d/%d), "
      "Hair='%s' (%d/%d), Hat=%s (%d/%d), Glasses=%s (%d/%d)",
      guest_idx_, person_id,
      current_guest_.gender.c_str(), g_cnt, g_tot,
      current_guest_.shirt_color.c_str(), s_cnt, s_tot,
      current_guest_.pants_color.c_str(), p_cnt, p_tot,
      current_guest_.hair_color.c_str(), h_cnt, h_tot,
      current_guest_.has_hat ? "yes" : "no", hat_cnt, total_frames,
      current_guest_.has_glasses ? "yes" : "no", glasses_cnt, total_frames);
  }

  static std::string formatPersonDescription(const PersonProfile & person)
  {
    const bool has_gender = !isPlaceholderAttr(person.gender) &&
                            (person.gender == "male" || person.gender == "female");
    const bool has_shirt = !isPlaceholderAttr(person.shirt_color);
    const bool has_pants = !isPlaceholderAttr(person.pants_color);
    const bool has_hair = !isPlaceholderAttr(person.hair_color);
    const bool has_hat = person.has_hat;
    const bool has_glasses = person.has_glasses;
    if (!has_gender && !has_shirt && !has_pants && !has_hair && !has_hat && !has_glasses) {
      return "";
    }

    std::string pronoun = "They";
    std::string verb = "are";
    if (has_gender && person.gender == "male") {
      pronoun = "He";
      verb = "is";
    } else if (has_gender && person.gender == "female") {
      pronoun = "She";
      verb = "is";
    }

    std::vector<std::string> parts;
    if (has_gender) {
      parts.push_back(person.gender);
    }

    std::vector<std::string> wearing_items;
    if (has_shirt) {
      std::string shirt = person.shirt_color;
      if (shirt.find("shirt") == std::string::npos &&
          shirt.find("attire") == std::string::npos &&
          shirt.find("clothes") == std::string::npos)
      {
        shirt += " shirt";
      }
      const std::string shirt_phrase =
        (shirt.find("attire") != std::string::npos || shirt.find("clothes") != std::string::npos)
        ? shirt : ("a " + shirt);
      wearing_items.push_back(shirt_phrase);
    }
    if (has_pants) {
      std::string pants = person.pants_color;
      if (pants.find("pants") == std::string::npos &&
          pants.find("trousers") == std::string::npos &&
          pants.find("jeans") == std::string::npos &&
          pants.find("shorts") == std::string::npos)
      {
        pants += " pants";
      }
      wearing_items.push_back(pants);
    }
    if (has_hat) {
      wearing_items.push_back("a hat");
    }
    if (has_glasses) {
      wearing_items.push_back("glasses");
    }
    if (!wearing_items.empty()) {
      std::string wearing_str = wearing_items[0];
      if (wearing_items.size() == 2) {
        wearing_str += " and " + wearing_items[1];
      } else if (wearing_items.size() >= 3) {
        for (size_t i = 1; i + 1 < wearing_items.size(); ++i) {
          wearing_str += ", " + wearing_items[i];
        }
        wearing_str += ", and " + wearing_items.back();
      }
      parts.push_back("wearing " + wearing_str);
    }

    if (has_hair) {
      std::string hair = person.hair_color;
      if (hair.find("hair") == std::string::npos) {
        hair += " hair";
      }
      parts.push_back("with " + hair);
    }

    std::string out = pronoun + " " + verb + " " + parts[0];
    for (size_t i = 1; i < parts.size(); ++i) {
      out += ", " + parts[i];
    }
    return out;
  }

  bool isPersonInFront() const
  {
    if (!person_seen_.load()) return false;
    rclcpp::Time seen_t(0, 0, RCL_ROS_TIME);
    {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      seen_t = last_person_seen_time_;
    }
    return seen_t.nanoseconds() > 0 &&
           (now() - seen_t).seconds() < person_freshness_sec_;
  }

  bool isStandingGuestInFront() const
  {
    if (!standing_guest_seen_.load()) return false;
    rclcpp::Time seen_t(0, 0, RCL_ROS_TIME);
    {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      seen_t = last_standing_guest_seen_time_;
    }
    return seen_t.nanoseconds() > 0 &&
           (now() - seen_t).seconds() < person_freshness_sec_;
  }

  // ============================================================================
  // TTS & STT / NLU HELPERS
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
    RCLCPP_INFO(get_logger(), "\033[1;36m[RECEPTIONIST]\033[0m\n  TTS: \"%s\"", text.c_str());
  }

  void onTtsDone(const std_msgs::msg::String::SharedPtr msg = nullptr)
  {
    std::function<void()> callback;
    {
      std::lock_guard<std::mutex> lock(tts_mutex_);
      if (!speaking_.load()) {
        return;
      }
      if (msg != nullptr && !msg->data.empty() && !active_tts_text_.empty() &&
          normalize(msg->data) != normalize(active_tts_text_))
      {
        RCLCPP_WARN(
          get_logger(),
          "[TTS] Ignoring stale /hri/tts_done (got \"%s\", expected \"%s\")",
          msg->data.c_str(), active_tts_text_.c_str());
        return;
      }
      in_callback_depth_.fetch_add(1);
      speaking_ = false;
      const std::string completed =
        (msg != nullptr && !msg->data.empty()) ? msg->data : active_tts_text_;
      if (!completed.empty()) {
        last_tts_text_norm_ = normalize(completed);
      }
      last_tts_done_time_ = now();
      active_tts_text_.clear();
      tts_cooldown_until_ = now() + rclcpp::Duration::from_seconds(0.25);
      callback = std::move(pending_tts_done_);
      pending_tts_done_ = nullptr;
    }

    CallbackExit cb_exit{in_callback_depth_};
    if (isTerminalState(state_.load())) {
      return;
    }
    publishExpression(expressionForState(state_.load()));
    if (callback) {
      callback();
    }
  }

  bool isDialogWaitState(ReceptionistState s) const
  {
    return s == ReceptionistState::WAIT_NAME ||
           s == ReceptionistState::WAIT_INTEREST ||
           s == ReceptionistState::WAIT_CONFIRM_GUEST_INFO ||
           s == ReceptionistState::WAIT_DRINK ||
           s == ReceptionistState::WAIT_CONFIRM_DRINK;
  }

  bool canListen() const
  {
    if (isBusy()) {
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

  static bool isPlaceholderAttr(const std::string & raw)
  {
    const std::string n = normalize(raw);
    return n.empty() || n == "unknown" || n == "person" || n == "casual attire" || n == "dark hair";
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

  static size_t maxFuzzyDistance(size_t candidate_len)
  {
    if (candidate_len < 5) return 0;
    if (candidate_len <= 6) return 1;
    return 2;
  }

  static bool isCarrierStopWord(const std::string & w)
  {
    static const std::vector<std::string> kStopWords = {
      "my", "name", "is", "am", "im", "are", "you", "can", "call", "me",
      "this", "that", "the", "favorite", "favourite", "drink", "would",
      "like", "want", "prefer", "main", "interest", "interested", "hobby",
      "love", "enjoy", "not", "dont", "don", "hello", "hi", "hey",
      "please", "thanks", "thank", "then", "your", "answer", "with", "say",
      "sorry", "okay", "ok", "what", "pardon", "uh", "um", "er", "ah", "oh",
      "hmm", "huh", "yes", "no", "robot", "actually", "well", "so", "here",
      "there", "s", "m"
    };
    return std::find(kStopWords.begin(), kStopWords.end(), w) != kStopWords.end();
  }

  static bool isLikelyNoiseOrPromptEcho(const std::string & norm)
  {
    if (norm.size() < 2) return true;
    const bool all_digits_or_space = std::all_of(norm.begin(), norm.end(), [](unsigned char c) {
      return std::isdigit(c) || std::isspace(c);
    });
    if (all_digits_or_space) return true;

    static const std::vector<std::string> kPromptEchoPhrases = {
      "please answer", "answer with", "please say", "then your", "then the",
      "is that correct", "what is your", "nice to meet you", "let us try again",
      "beverage area", "stand on my left", "follow me",
      "yes robot or no robot", "robot yes or robot no",
      "yes correct or no wrong", "yes or no"
    };
    for (const auto & p : kPromptEchoPhrases) {
      if (containsWholePhrase(norm, p)) return true;
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
      if (norm == item) return true;
    }

    static const std::vector<std::string> kPureFillers = {
      "uh", "um", "er", "ah", "oh", "hmm", "huh", "mm", "mhm",
      "okay", "ok", "thanks", "thank", "you", "please", "sorry",
      "hello", "hi", "hey", "robot"
    };
    const std::vector<std::string> words = splitWords(norm);
    if (words.empty()) return true;
    return std::all_of(words.begin(), words.end(), [](const std::string & w) {
      return std::find(kPureFillers.begin(), kPureFillers.end(), w) != kPureFillers.end();
    });
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
    if ((state_.load() == ReceptionistState::WAIT_CONFIRM_GUEST_INFO ||
         state_.load() == ReceptionistState::WAIT_CONFIRM_DRINK) &&
        parseYesNoIntent(norm_text) != 0 &&
        !isLikelyNoiseOrPromptEcho(norm_text))
    {
      return false;
    }

    const std::vector<std::string> tokens = splitWords(norm_text);
    if (tokens.size() < 3) return false;

    size_t overlap = 0;
    bool has_novel_content = false;
    for (const auto & tok : tokens) {
      if (containsWholePhrase(last_norm, tok)) {
        ++overlap;
      } else if (!isCarrierStopWord(tok)) {
        has_novel_content = true;
      }
    }

    if (containsWholePhrase(last_norm, norm_text)) return true;
    const double overlap_ratio = static_cast<double>(overlap) / static_cast<double>(tokens.size());
    return overlap_ratio >= 0.65 && !has_novel_content;
  }

  static std::string findBestFuzzyMatch(
    const std::vector<std::string> & tokens,
    const std::vector<std::string> & candidates,
    const std::function<bool(const std::string &)> & is_excluded = nullptr)
  {
    if (tokens.empty()) return "";

    std::string best_candidate;
    size_t best_dist = 999;

    auto considerMatch = [&](const std::string & window, const std::string & target,
                             const std::string & candidate, size_t max_d) {
      if (max_d == 0) return;
      if (is_excluded && is_excluded(window)) return;
      const size_t len_diff = (window.size() > target.size())
        ? (window.size() - target.size())
        : (target.size() - window.size());
      if (len_diff > max_d) return;
      const size_t d = levenshteinDistance(window, target);
      if (d <= max_d &&
          (d < best_dist || (d == best_dist && candidate.size() > best_candidate.size())))
      {
        best_dist = d;
        best_candidate = candidate;
      }
    };

    for (const auto & candidate : candidates) {
      const size_t max_d = maxFuzzyDistance(candidate.size());
      const std::vector<std::string> c_words = splitWords(candidate);
      const size_t k = c_words.size();
      if (k == 0) continue;

      if (max_d > 0 && tokens.size() >= k) {
        for (size_t i = 0; i + k <= tokens.size(); ++i) {
          if (k == 1 && isCarrierStopWord(tokens[i])) continue;
          std::string window = tokens[i];
          for (size_t j = 1; j < k; ++j) {
            window.push_back(' ');
            window += tokens[i + j];
          }
          considerMatch(window, candidate, candidate, max_d);
        }
      }

      if (k >= 2) {
        std::string compact;
        compact.reserve(candidate.size());
        for (const auto & cw : c_words) compact += cw;
        const size_t compact_max_d = maxFuzzyDistance(compact.size());
        if (compact_max_d > 0) {
          for (const auto & tok : tokens) {
            if (isCarrierStopWord(tok)) continue;
            considerMatch(tok, compact, candidate, compact_max_d);
          }
        }
      }
    }
    return best_candidate;
  }

  std::string parseSlotIntent(const std::string & raw_text, const std::string & slot) const
  {
    const std::string norm = normalize(raw_text);
    const std::vector<std::string> tokens = splitWords(norm);
    if (norm.empty() || isLikelyNoiseOrPromptEcho(norm) ||
        (tokens.size() <= 2 && parseYesNoIntent(norm) != 0))
    {
      return "";
    }

    struct PhoneticAlias {
      const char * alias;
      const char * canonical;
    };

    auto isNegated = [&](const std::string & phrase) -> bool {
      return containsWholePhrase(norm, "not " + phrase) ||
             containsWholePhrase(norm, "dont like " + phrase) ||
             containsWholePhrase(norm, "don t like " + phrase) ||
             containsWholePhrase(norm, "do not like " + phrase);
    };

    auto countWords = [](const std::string & s) -> size_t {
      std::stringstream ss(s);
      std::string w;
      size_t count = 0;
      while (ss >> w) ++count;
      return count;
    };

    auto stripLeadingFillers = [](std::string s) -> std::string {
      static const std::vector<std::string> kLeadFillers = {
        "is ", "a ", "an ", "the ", "uh ", "um ", "er ", "ah ", "oh ",
        "hmm ", "well ", "so ", "actually ", "like "
      };
      bool stripped = true;
      while (stripped) {
        stripped = false;
        while (!s.empty() && s.front() == ' ') s.erase(s.begin());
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
        while (!s.empty() && s.back() == ' ') s.pop_back();
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
      while (!s.empty() && s.back() == ' ') s.pop_back();
      return s;
    };

    auto isValidFallbackValue = [&](const std::string & rem) -> bool {
      if (rem.size() < 2) return false;
      const size_t wc = countWords(rem);
      if (wc < 1 || wc > 3) return false;
      const bool all_digits = std::all_of(rem.begin(), rem.end(), [](unsigned char c) {
        return std::isdigit(c) || std::isspace(c);
      });
      if (all_digits) return false;
      const std::vector<std::string> rem_words = splitWords(rem);
      return std::any_of(rem_words.begin(), rem_words.end(), [](const std::string & tok) {
        return !isCarrierStopWord(tok);
      });
    };

    auto stripPrefixes = [&](const std::vector<std::string> & prefixes) -> std::string {
      for (const auto & p : prefixes) {
        size_t search_from = 0;
        while (search_from < norm.size()) {
          const auto pos = norm.find(p, search_from);
          if (pos == std::string::npos) break;
          if (pos == 0 || norm[pos - 1] == ' ') {
            std::string rem = stripTrailingFillers(
              stripLeadingFillers(norm.substr(pos + p.size())));
            if (isValidFallbackValue(rem)) {
              return capitalizeWords(rem);
            }
          }
          search_from = pos + 1;
        }
      }
      if (countWords(norm) <= 2) {
        std::string rem = stripTrailingFillers(stripLeadingFillers(norm));
        if (isValidFallbackValue(rem)) {
          return capitalizeWords(rem);
        }
      }
      return "";
    };

    if (slot == "person_name") {
      static const std::vector<PhoneticAlias> kNameAliases = {
        {"jefrey", "Jeffrey"}, {"jefri", "Jeffrey"}, {"geoffrey", "Jeffrey"},
        {"jeffry", "Jeffrey"}, {"jefery", "Jeffrey"},
        {"hezekial", "Hezekiel"}, {"ezekiel", "Hezekiel"},
        {"charly", "Charlie"},
        {"micheal", "Michael"}, {"mikael", "Michael"},
        {"sara", "Sarah"},
        {"mohammad", "Muhammad"}, {"muhamad", "Muhammad"},
        {"ahmed", "Ahmad"}, {"robyn", "Robin"},
        {"simona", "Simone"}, {"adele", "Adel"}, {"jule", "Jules"}
      };
      for (const auto & name : known_names_) {
        if (containsWholePhrase(norm, name)) return capitalizeWords(name);
      }
      for (const auto & entry : kNameAliases) {
        if (containsWholePhrase(norm, entry.alias)) return entry.canonical;
      }
      const std::string fuzzy = findBestFuzzyMatch(tokens, known_names_);
      if (!fuzzy.empty()) return capitalizeWords(fuzzy);
      return stripPrefixes({
        "you can call me ", "my name is ", "may name is ", "mai name is ",
        "my name s ", "mai name s ", "my name ", "name is ", "name s ",
        "ame is ", "me is ", "call me ", "this is ",
        "i am ", "iam ", "iyem ", "i m ", "im ", "name ", "is "
      });
    }

    if (slot == "drink") {
      auto formatDrink = [](const std::string & norm_drink) -> std::string {
        if (norm_drink == "cheez it" || norm_drink == "cheezit") return "Cheez-It";
        return capitalizeWords(norm_drink);
      };
      static const std::vector<PhoneticAlias> kDrinkAliases = {
        {"cheese it", "Cheez-It"}, {"cheezit", "Cheez-It"},
        {"cheesit", "Cheez-It"}, {"cheese its", "Cheez-It"},
        {"cook", "Coke"}, {"koke", "Coke"}, {"cock", "Coke"},
        {"coca cola", "Coke"}, {"coca", "Coke"},
        {"springles", "Pringles"}, {"pringle", "Pringles"}, {"bringles", "Pringles"},
        {"ice tea", "Iced Tea"},
        {"orange choose", "Orange Juice"}, {"apple choose", "Apple Juice"},
        {"watter", "Water"}, {"kofi", "Coffee"}, {"copi", "Coffee"},
        {"spread", "Sprite"}, {"fenta", "Fanta"}, {"pepsy", "Pepsi"},
        {"crackers", "Cracker Box"}
      };
      for (const auto & drink : known_drinks_) {
        if (containsWholePhrase(norm, drink) && !isNegated(drink)) return formatDrink(drink);
      }
      for (const auto & entry : kDrinkAliases) {
        if (containsWholePhrase(norm, entry.alias) && !isNegated(entry.alias)) {
          return entry.canonical;
        }
      }
      const std::string fuzzy = findBestFuzzyMatch(tokens, known_drinks_, isNegated);
      if (!fuzzy.empty()) return formatDrink(fuzzy);
      return stripPrefixes({
        "my favorite drink is ", "my favourite drink is ",
        "favorite drink is ", "favourite drink is ",
        "vorite drink is ", "rite drink is ",
        "my favorite drink ", "my favourite drink ",
        "favorite drink ", "favourite drink ",
        "i would like ", "would like ", "drink is ", "ink is ",
        "i prefer ", "i like ", "i want ",
        "prefer ", "drink ", "like ", "want ", "is "
      });
    }

    if (slot == "interest") {
      static const std::vector<PhoneticAlias> kInterestAliases = {
        {"robotic", "Robotics"}, {"robots", "Robotics"},
        {"programing", "Programming"}, {"program", "Programming"},
        {"fotball", "Football"}, {"badminton", "Playing Badminton"},
        {"tenis", "Playing Tennis"}, {"basket ball", "Basketball"},
        {"photograph", "Photography"}, {"cook", "Cooking"},
        {"game", "Gaming"}, {"games", "Gaming"},
        {"swim", "Swimming"}, {"movie", "Movies"},
        {"read", "Reading"}, {"travel", "Traveling"},
        {"paint", "Painting"}, {"draw", "Drawing"},
        {"dance", "Dancing"}, {"sport", "Sports"}, {"box", "Boxing"}
      };
      for (const auto & interest : known_interests_) {
        if (containsWholePhrase(norm, interest)) return capitalizeWords(interest);
      }
      for (const auto & entry : kInterestAliases) {
        if (containsWholePhrase(norm, entry.alias)) return entry.canonical;
      }
      const std::string fuzzy = findBestFuzzyMatch(tokens, known_interests_);
      if (!fuzzy.empty()) return capitalizeWords(fuzzy);
      return stripPrefixes({
        "my main interest is ", "main interest is ",
        "i am interested in ", "im interested in ", "i m interested in ",
        "interested in ", "terested in ",
        "my interest is ", "my hobby is ",
        "interest is ", "terest is ", "rest is ", "est is ",
        "hobby is ", "obby is ",
        "my interest ", "my hobby ", "interest ", "hobby ",
        "i like ", "i love ", "i enjoy ",
        "like ", "love ", "enjoy ", "is "
      });
    }
    return "";
  }

  static std::string canonicalInterestCategory(const std::string & raw)
  {
    const std::string n = normalize(raw);
    if (n.empty()) return "";
    if (n.find("football") != std::string::npos || n.find("badminton") != std::string::npos ||
        n.find("tennis") != std::string::npos || n.find("golf") != std::string::npos ||
        n.find("basketball") != std::string::npos || n.find("swim") != std::string::npos ||
        n.find("boxing") != std::string::npos || n == "sports")
    {
      return "sports";
    }
    if (n == "coding" || n == "programming") return "programming";
    if (n == "robotics" || n == "artificial intelligence") return "robotics and AI";
    if (n == "painting" || n == "drawing") return "art";
    return n;
  }

  /// Pengenalan konfirmasi Yes/No tahan variasi STT dan aman terhadap negasi ("not correct")
  static int parseYesNoIntent(const std::string & raw_text)
  {
    const std::string norm = normalize(raw_text);
    if (norm.empty()) return 0;

    static const std::vector<std::string> kNoPhrases = {
      "no", "nope", "nah", "not correct", "not right", "wrong", "false",
      "incorrect", "negative", "robot no", "no robot", "bot no", "obot no"
    };
    static const std::vector<std::string> kYesPhrases = {
      "yes", "yeah", "yep", "yup", "ya", "yess", "correct", "right",
      "affirmative", "robot yes", "yes robot", "bot yes", "obot yes", "es robot"
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
        // Cegah "not correct" / "not right" ikut terhitung sebagai YES
        if ((phrase == "correct" || phrase == "right") &&
            containsWholePhrase(norm, "not " + phrase))
        {
          continue;
        }
        has_yes = true;
        break;
      }
    }

    if (has_no && has_yes) return 0;
    if (has_no) return -1;
    if (has_yes) return 1;
    return 0;
  }

  // ============================================================================
  // ROS 2 SUBSCRIPTION CALLBACKS
  // ============================================================================
  void onTranscript(const std_msgs::msg::String::SharedPtr message)
  {
    if (!canListen() || !message || !isDialogWaitState(state_.load())) {
      return;
    }

    const std::string clean = normalize(message->data);
    if (clean.empty()) return;

    if (isTtsEcho(clean)) {
      RCLCPP_INFO(get_logger(), "\033[2m[STT IGNORED]\033[0m TTS echo: \"%s\"", message->data.c_str());
      return;
    }
    if (isLikelyNoiseOrPromptEcho(clean)) {
      RCLCPP_INFO(get_logger(), "\033[2m[STT IGNORED]\033[0m Noise/filler: \"%s\"", message->data.c_str());
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
    RCLCPP_INFO(get_logger(), "\033[1;33m[STT]\033[0m \"%s\"", message->data.c_str());
  }

  void onPersonLabel(const std_msgs::msg::String::SharedPtr message)
  {
    if (!isVisionActive() || !message || message->data.empty() ||
        message->data.find("NO_PERSON") != std::string::npos)
    {
      return;
    }

    const auto rx_time = now();
    {
      std::lock_guard<std::mutex> lock(vision_mutex_);
      last_person_seen_time_ = rx_time;
    }
    person_seen_ = true;

    {
      bool has_standing_guest = false;
      std::string host_lower = host_.name;
      std::transform(host_lower.begin(), host_lower.end(), host_lower.begin(), ::tolower);
      std::stringstream pss(message->data);
      std::string pitem;
      while (std::getline(pss, pitem, ';')) {
        if (pitem.empty()) continue;
        std::string pitem_lower = pitem;
        std::transform(pitem_lower.begin(), pitem_lower.end(), pitem_lower.begin(), ::tolower);
        const auto colon_pos = pitem_lower.find(':');
        const std::string plabel_lower = (colon_pos != std::string::npos) ? pitem_lower.substr(0, colon_pos) : pitem_lower;
        const bool is_seated = (pitem_lower.find("sitting=1") != std::string::npos ||
                                plabel_lower.find("seated") != std::string::npos ||
                                plabel_lower.find("sitting") != std::string::npos ||
                                plabel_lower.find("duduk") != std::string::npos);
        const bool is_host = (!host_lower.empty() && plabel_lower.find(host_lower) != std::string::npos);
        const bool is_standing_3d = (pitem_lower.find("standing=1") != std::string::npos);
        bool is_tall_bbox = false;
        const auto bbar = pitem.find('|');
        if (colon_pos != std::string::npos && bbar != std::string::npos && bbar > colon_pos) {
          int bx = 0, by = 0, bw = 0, bh = 0;
          if (std::sscanf(pitem.substr(colon_pos + 1, bbar - colon_pos - 1).c_str(), "%d,%d,%d,%d", &bx, &by, &bw, &bh) == 4) {
            is_tall_bbox = (bw > 0 && bh >= static_cast<int>(bw * 1.70));
          }
        }
        bool has_lower_kpt = false;
        const auto kpos = pitem_lower.find("|kpt=");
        if (kpos != std::string::npos) {
          std::vector<double> kc;
          kc.reserve(68);
          const size_t kstart = kpos + 5;
          const size_t kend = pitem.find('|', kstart);
          std::stringstream kss(
            (kend != std::string::npos) ? pitem.substr(kstart, kend - kstart) : pitem.substr(kstart));
          std::string tok;
          while (std::getline(kss, tok, ',')) {
            try { kc.push_back(std::stod(tok)); } catch (...) { kc.push_back(-1.0); }
          }
          auto kv = [&](size_t idx) -> bool {
            return (idx * 2 + 1) < kc.size() && kc[idx * 2] >= 0.0 && kc[idx * 2 + 1] >= 0.0;
          };
          has_lower_kpt = (kv(18) || kv(22)) && (kv(19) || kv(23));
        }
        const bool is_fullbody_standing = !is_seated && (is_standing_3d || (has_lower_kpt && is_tall_bbox));
        if (is_fullbody_standing && !is_host) {
          has_standing_guest = true;
          break;
        }
      }
      if (has_standing_guest) {
        {
          std::lock_guard<std::mutex> lock(vision_mutex_);
          last_standing_guest_seen_time_ = rx_time;
        }
        standing_guest_seen_ = true;
      }
    }

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
      try { nearest_person_id_ = std::stoi(data.substr(id_pos + 3)); }
      catch (...) { nearest_person_id_ = 0; }
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
  }

  void onBagDetectionsFallback(const std_msgs::msg::String::SharedPtr message)
  {
    if (!isVisionActive() || !message || message->data.empty()) return;

    std::lock_guard<std::mutex> lock(vision_mutex_);
    if (last_drink_msg_time_.nanoseconds() > 0 &&
        (now() - last_drink_msg_time_).seconds() < 3.0)
    {
      return;
    }
    auto parsed = parseDetectionString(message->data);
    if (!parsed.empty()) {
      latest_drinks_ = std::move(parsed);
    }
  }

  void onPersonAttributes(const std_msgs::msg::String::SharedPtr message)
  {
    if (!isVisionActive() || !message || message->data.empty()) return;
    const ReceptionistState cur = state_.load();
    if (cur != ReceptionistState::SCAN_GUEST_ATTRIBUTES &&
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
    if (attributes_locked_from_scan_) return;

    std::vector<std::string> parsed;
    std::stringstream ss(message->data);
    std::string item;
    while (std::getline(ss, item, ',')) {
      const size_t first = item.find_first_not_of(" \t\r\n");
      if (first != std::string::npos) {
        const size_t last = item.find_last_not_of(" \t\r\n");
        parsed.push_back(item.substr(first, last - first + 1));
      }
    }
    if (!parsed.empty()) {
      latest_attributes_ = parsed;
      attr_total_frames_++;
      bool frame_hat = false;
      bool frame_glasses = false;
      for (const auto & a : parsed) {
        if (isPlaceholderAttr(a)) continue;
        std::string lower_a = a;
        std::transform(lower_a.begin(), lower_a.end(), lower_a.begin(), ::tolower);
        if (lower_a == "male" || lower_a == "female") {
          attr_gender_counts_[lower_a]++;
        } else if (lower_a.find("shirt") != std::string::npos ||
                   lower_a.find("clothes") != std::string::npos ||
                   lower_a.find("attire") != std::string::npos)
        {
          attr_shirt_counts_[a]++;
        } else if (lower_a.find("pants") != std::string::npos ||
                   lower_a.find("trousers") != std::string::npos ||
                   lower_a.find("jeans") != std::string::npos)
        {
          attr_pants_counts_[a]++;
        } else if (lower_a.find("hair") != std::string::npos) {
          attr_hair_counts_[a]++;
        } else if (lower_a.find("hat") != std::string::npos ||
                   lower_a.find("cap") != std::string::npos)
        {
          if (lower_a.find("no ") == std::string::npos) {
            frame_hat = true;
          }
        } else if (lower_a.find("glasses") != std::string::npos ||
                   lower_a.find("eyewear") != std::string::npos)
        {
          if (lower_a.find("no ") == std::string::npos) {
            frame_glasses = true;
          }
        }
      }
      if (frame_hat) attr_hat_true_count_++;
      if (frame_glasses) attr_glasses_true_count_++;
    }
  }

  void onEmptySeat(const std_msgs::msg::String::SharedPtr message)
  {
    if (!isVisionActive() || !message || message->data.empty()) return;
    std::lock_guard<std::mutex> lock(vision_mutex_);
    std::string raw = message->data;
    std::transform(raw.begin(), raw.end(), raw.begin(), ::tolower);
    while (!raw.empty() && std::isspace(static_cast<unsigned char>(raw.front()))) raw.erase(raw.begin());
    while (!raw.empty() && std::isspace(static_cast<unsigned char>(raw.back()))) raw.pop_back();

    if (raw == "none" || raw.empty()) {
      has_explicit_empty_seat_ = false;
      return;
    }
    const auto pipe1 = raw.find('|');
    std::string side_str = (pipe1 != std::string::npos) ? raw.substr(0, pipe1) : raw;

    if (pipe1 != std::string::npos) {
      const auto pipe2 = raw.find('|', pipe1 + 1);
      std::string xyz_str = (pipe2 != std::string::npos)
        ? raw.substr(pipe1 + 1, pipe2 - pipe1 - 1)
        : raw.substr(pipe1 + 1);
      double px = 0.0, py = 0.0, pz = 0.0;
      if (xyz_str.find("nan") == std::string::npos &&
          std::sscanf(xyz_str.c_str(), "%lf,%lf,%lf", &px, &py, &pz) == 3)
      {
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
  }

  // ============================================================================
  // NON-BLOCKING TRIGGER SERVICE CLIENT
  // ============================================================================
  void callTrigger(
    const TriggerClient::SharedPtr & client,
    const std::string & service_name,
    TriggerResult result,
    double timeout_sec = 10.0)
  {
    if (!client) {
      if (result && !isTerminalState(state_.load())) {
        CallbackScope cb_scope(in_callback_depth_);
        result(false, "Client is null");
      }
      return;
    }

    if (!client->service_is_ready()) {
      RCLCPP_WARN(get_logger(), "Service %s belum tersedia.", service_name.c_str());
      if (result && !isTerminalState(state_.load())) {
        CallbackScope cb_scope(in_callback_depth_);
        result(false, "Service not ready");
      }
      return;
    }

    if (isTerminalState(state_.load())) {
      return;
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
          in_callback_depth_.fetch_add(1);
          service_busy_ = false;
          cb = std::move(pending_srv_result_);
          pending_srv_result_ = nullptr;
        }

        CallbackExit cb_exit{in_callback_depth_};
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

        if (cb && !isTerminalState(state_.load())) cb(ok, msg);
      });
  }

  // ============================================================================
  // MAIN FINITE STATE MACHINE (10 Hz)
  // ============================================================================
  void fsmTick()
  {
    const ReceptionistState current_state = state_.load();

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
      {
        std::lock_guard<std::mutex> lock(tts_mutex_);
        speaking_ = false;
        active_tts_text_.clear();
        pending_tts_done_ = nullptr;
      }
      {
        std::lock_guard<std::mutex> lock(srv_mutex_);
        service_busy_ = false;
        active_srv_name_.clear();
        pending_srv_result_ = nullptr;
      }
      publishExpression("LOST");
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
        get_logger(), "Service %s timeout (> %.1f s), releasing lock.",
        timed_out_srv_name.c_str(), srv_timeout_sec_);
      CallbackScope cb_scope(in_callback_depth_);
      timed_out_srv_cb(false, "Service response timeout");
      return;
    }

    // 3. TTS timeout watchdog
    bool tts_timed_out = false;
    double tts_timeout = 0.0;
    {
      std::lock_guard<std::mutex> lock(tts_mutex_);
      tts_timeout = tts_timeout_sec_;
      tts_timed_out = speaking_.load() && (now() - tts_started_time_).seconds() > tts_timeout_sec_;
    }
    if (tts_timed_out) {
      RCLCPP_WARN(
        get_logger(), "TTS timeout (> %.1f s, no /hri/tts_done), releasing speaking lock.",
        tts_timeout);
      onTtsDone(nullptr);
      return;
    }

    if (isBusy()) return;

    switch (current_state) {
      // ------------------------------------------------------------------------
      // INIT
      // ------------------------------------------------------------------------
      case ReceptionistState::INIT:
        if (!nav_door_client_ || !nav_door_client_->service_is_ready()) {
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Waiting for '/receptionist/go_to_door' service...");
          return;
        }
        if (tts_pub_->get_subscription_count() == 0 &&
            (now() - state_enter_time_).seconds() < 10.0)
        {
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Waiting for TTS node (/hri/tts_say subscriber)...");
          return;
        }
        guests_.clear();
        initCurrentGuest(1);
        publishGuestNames();
        transitionTo(ReceptionistState::NAV_TO_DOOR, "Start cycle for Guest 1");
        break;

      // ------------------------------------------------------------------------
      // PHASE 1: ENTRANCE DOOR
      // ------------------------------------------------------------------------
      case ReceptionistState::NAV_TO_DOOR:
        if (last_nav_fail_time_.nanoseconds() > 0 &&
            (now() - last_nav_fail_time_).seconds() < nav_retry_cooldown_sec_)
        {
          return;
        }
        speakAsync("I am going to the door.", [this]() {
          navigateTo("entrance_door", [this](bool success) {
            if (!success) return;
            transitionTo(ReceptionistState::WAIT_GUEST, "Arrived at entrance_door");
          });
        });
        break;

      case ReceptionistState::WAIT_GUEST:
        if ((now() - state_enter_time_).seconds() < vision_initial_settle_sec_) {
          return;
        }
        if (isStandingGuestInFront()) {
          // FIX: hanya satu transisi ke ALIGN_GUEST agar adjustGuestDistance() benar-benar jalan
          transitionTo(
            ReceptionistState::ALIGN_GUEST,
            "Standing guest detected; aligning heading and distance");
        } else if (last_wait_prompt_time_.nanoseconds() == 0 ||
                   (now() - last_wait_prompt_time_).seconds() >= wait_guest_prompt_sec_)
        {
          last_wait_prompt_time_ = now();
          speakAsync("Please stand in front of me.");
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
            attr_pants_counts_.clear();
            attr_hat_true_count_ = 0;
            attr_glasses_true_count_ = 0;
            attr_total_frames_ = 0;
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
            "Aligned to guest; scanning face & visual attributes silently");
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
          "Attributes locked; approaching guest to 0.7m for HRI");
        break;

      case ReceptionistState::APPROACH_GUEST:
        approachGuest([this](bool /*success*/, const std::string & /*msg*/) {
          speakAsync(
            "Hello, I am Barelang Inari, your receptionist.",
            [this]() {
              transitionTo(ReceptionistState::ASK_NAME, "Approached guest and greeted");
            });
        });
        break;

      case ReceptionistState::ASK_NAME:
        speakAsync("What is your name? Please say: my name is, then your name.", [this]() {
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
              get_logger(), "\033[1;32m[DIALOG NAME]\033[0m G%d -> \"%s\" (raw=\"%s\")",
              guest_idx_, parsed_name.c_str(), raw.c_str());
            transitionTo(ReceptionistState::ASK_INTEREST, "Guest name captured");
          } else {
            RCLCPP_WARN(get_logger(), "[DIALOG NAME] Unrecognized \"%s\"", raw.c_str());
            state_enter_time_ = now();
            speakAsync("Sorry, please say: my name is, then your name.");
          }
        } else if ((now() - state_enter_time_).seconds() > dialog_reprompt_sec_) {
          state_enter_time_ = now();
          speakAsync("What is your name? Please say: my name is, then your name.");
        }
        break;

      case ReceptionistState::ASK_INTEREST: {
        std::string g_name;
        {
          std::lock_guard<std::mutex> lock(guest_mutex_);
          g_name = current_guest_.name;
        }
        speakAsync(
          "Nice to meet you, " + g_name +
          ". What is your interest? Please say: my interest is, then your interest.",
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
              if (guest_idx_ >= 1 && guests_.size() >= static_cast<size_t>(guest_idx_)) {
                guests_[static_cast<size_t>(guest_idx_ - 1)] = current_guest_;
              }
            }
            RCLCPP_INFO(
              get_logger(), "\033[1;32m[DIALOG INTEREST]\033[0m G%d -> \"%s\" (raw=\"%s\")",
              guest_idx_, parsed_interest.c_str(), raw.c_str());
            transitionTo(ReceptionistState::CONFIRM_GUEST_INFO, "Guest interest captured");
          } else {
            RCLCPP_WARN(get_logger(), "[DIALOG INTEREST] Unrecognized \"%s\"", raw.c_str());
            state_enter_time_ = now();
            speakAsync("Sorry, please say: my interest is, then your interest.");
          }
        } else if ((now() - state_enter_time_).seconds() > dialog_reprompt_sec_) {
          state_enter_time_ = now();
          speakAsync("Sorry, please say: my interest is, then your interest.");
        }
        break;

      case ReceptionistState::CONFIRM_GUEST_INFO: {
        std::string g_name, g_interest;
        {
          std::lock_guard<std::mutex> lock(guest_mutex_);
          g_name = current_guest_.name;
          g_interest = current_guest_.interest;
        }
        speakAsync(
          g_name + ", your interest is " + g_interest +
          ". Is that correct? Please say yes, correct, or no, wrong.",
          [this]() {
            transitionTo(ReceptionistState::WAIT_CONFIRM_GUEST_INFO, "Waiting for guest info confirm");
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
            RCLCPP_INFO(get_logger(), "\033[1;32m[DIALOG CONFIRM INFO]\033[0m \"%s\" -> YES", raw.c_str());
            transitionTo(ReceptionistState::NAV_TO_BEVERAGE, "Guest info confirmed");
          } else if (ans == -1) {
            RCLCPP_WARN(get_logger(), "\033[1;31m[DIALOG CONFIRM INFO]\033[0m \"%s\" -> NO", raw.c_str());
            bool keep_locked_attrs = false;
            {
              std::lock_guard<std::mutex> v_lock(vision_mutex_);
              keep_locked_attrs = attributes_locked_from_scan_;
            }
            {
              std::lock_guard<std::mutex> lock(guest_mutex_);
              if (keep_locked_attrs) {
                current_guest_.name.clear();
                current_guest_.interest.clear();
                current_guest_.drink.clear();
              } else {
                resetGuestProfileFields(current_guest_, guest_idx_);
              }
              if (guests_.size() >= static_cast<size_t>(guest_idx_)) {
                guests_[static_cast<size_t>(guest_idx_ - 1)] = current_guest_;
              }
            }
            if (person_name_pub_ && !keep_locked_attrs) {
              std_msgs::msg::String m;
              m.data = "";
              person_name_pub_->publish(m);
            }
            publishGuestNames();
            speakAsync("Sorry, let us try again. Please say: my name is, then your name.", [this]() {
              transitionTo(ReceptionistState::WAIT_NAME, "Restarted name & interest");
            });
          } else {
            RCLCPP_WARN(get_logger(), "\033[1;33m[DIALOG CONFIRM INFO]\033[0m \"%s\" -> UNRECOGNIZED", raw.c_str());
            state_enter_time_ = now();
            speakAsync("Please say yes, correct, or no, wrong.");
          }
        } else if ((now() - state_enter_time_).seconds() > dialog_reprompt_sec_) {
          state_enter_time_ = now();
          speakAsync("Please say yes, correct, or no, wrong.");
        }
        break;

      // ------------------------------------------------------------------------
      // PHASE 2: BEVERAGE AREA
      // ------------------------------------------------------------------------
      case ReceptionistState::NAV_TO_BEVERAGE:
        if (last_nav_fail_time_.nanoseconds() > 0 &&
            (now() - last_nav_fail_time_).seconds() < nav_retry_cooldown_sec_)
        {
          return;
        }
        speakAsync("Please follow me to the beverage area.", [this]() {
          navigateTo("beverage_area", [this](bool success) {
            if (!success) return;
            transitionTo(ReceptionistState::ASK_DRINK, "Arrived at beverage_area");
          });
        });
        break;

      case ReceptionistState::ASK_DRINK: {
        if ((now() - state_enter_time_).seconds() < vision_initial_settle_sec_) {
          return;
        }
        // Snapshot minuman meja sebelum berputar menghadap tamu di sebelah kiri
        speakAsync("This is the beverage area. Please stand on my left.", [this]() {
          const size_t drink_count = snapshotBeveragesOnTable();
          RCLCPP_INFO(get_logger(), "[BEVERAGE] Table snapshot locked: %zu drink(s)", drink_count);
          rotateLeft([this](bool /*ok*/, const std::string & /*msg*/) {
            speakAsync(
              "What is your favorite drink? Please say: my favorite drink is, then the drink.",
              [this]() {
                transitionTo(ReceptionistState::WAIT_DRINK, "Rotated left and asked favorite drink");
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
              get_logger(), "\033[1;32m[DIALOG DRINK]\033[0m G%d -> \"%s\" (raw=\"%s\")",
              guest_idx_, parsed_drink.c_str(), raw.c_str());
            transitionTo(ReceptionistState::CONFIRM_DRINK, "Favorite drink captured");
          } else {
            RCLCPP_WARN(get_logger(), "[DIALOG DRINK] Unrecognized \"%s\"", raw.c_str());
            state_enter_time_ = now();
            speakAsync("Sorry, please say: my favorite drink is, then the drink.");
          }
        } else if ((now() - state_enter_time_).seconds() > dialog_reprompt_sec_) {
          state_enter_time_ = now();
          speakAsync("Sorry, please say: my favorite drink is, then the drink.");
        }
        break;

      case ReceptionistState::CONFIRM_DRINK: {
        std::string g_drink;
        {
          std::lock_guard<std::mutex> lock(guest_mutex_);
          g_drink = current_guest_.drink;
        }
        speakAsync(
          "Your favorite drink is " + g_drink +
          ". Is that correct? Please say yes, correct, or no, wrong.",
          [this]() {
            transitionTo(ReceptionistState::WAIT_CONFIRM_DRINK, "Waiting for drink confirmation");
          });
        break;
      }

      case ReceptionistState::WAIT_CONFIRM_DRINK:
        if (has_new_utterance_.exchange(false)) {
          std::string raw;
          {
            std::lock_guard<std::mutex> lock(stt_mutex_);
            raw = heard_utterance_;
          }
          const int ans = parseYesNoIntent(raw);
          if (ans == 1) {
            RCLCPP_INFO(get_logger(), "\033[1;32m[DIALOG CONFIRM DRINK]\033[0m \"%s\" -> YES", raw.c_str());
            transitionTo(ReceptionistState::CHECK_BEVERAGE, "Drink confirmed");
          } else if (ans == -1) {
            RCLCPP_WARN(get_logger(), "\033[1;31m[DIALOG CONFIRM DRINK]\033[0m \"%s\" -> NO", raw.c_str());
            {
              std::lock_guard<std::mutex> lock(guest_mutex_);
              current_guest_.drink.clear();
              if (guest_idx_ >= 1 && guests_.size() >= static_cast<size_t>(guest_idx_)) {
                guests_[static_cast<size_t>(guest_idx_ - 1)].drink.clear();
              }
            }
            speakAsync("Sorry, let us try again. Please say: my favorite drink is, then the drink.", [this]() {
              transitionTo(ReceptionistState::WAIT_DRINK, "Restarted favorite drink capture");
            });
          } else {
            RCLCPP_WARN(get_logger(), "\033[1;33m[DIALOG CONFIRM DRINK]\033[0m \"%s\" -> UNRECOGNIZED", raw.c_str());
            state_enter_time_ = now();
            speakAsync("Please say yes, correct, or no, wrong.");
          }
        } else if ((now() - state_enter_time_).seconds() > dialog_reprompt_sec_) {
          RCLCPP_INFO(
            get_logger(), "[DIALOG CONFIRM DRINK] Timeout (%.1fs), reprompting.",
            dialog_reprompt_sec_);
          state_enter_time_ = now();
          speakAsync("Please say yes, correct, or no, wrong.");
        }
        break;

      case ReceptionistState::CHECK_BEVERAGE: {
        std::string fav_drink;
        {
          std::lock_guard<std::mutex> lock(guest_mutex_);
          fav_drink = current_guest_.drink;
        }
        bool has_arrival_drinks = false;
        {
          std::lock_guard<std::mutex> v_lock(vision_mutex_);
          has_arrival_drinks = drinks_detected_on_arrival_;
        }
        std::string report;
        if (has_arrival_drinks) {
          bool found_match = false;
          const std::string side = locateBeverageSide(fav_drink, found_match);
          if (found_match) {
            report = "Your " + fav_drink + " is on the " + side + " of the table.";
          } else {
            report = "Sorry, " + fav_drink + " is not available on the table.";
          }
        } else {
          report = "Sorry, " + fav_drink + " is not available on the table.";
        }

        speakAsync(report, [this]() {
          transitionTo(ReceptionistState::NAV_TO_LIVING_ROOM, "Beverage location announced");
        });
        break;
      }

      // ------------------------------------------------------------------------
      // PHASE 3: ESCORT TO LIVING ROOM & OFFER SEAT FIRST
      // ------------------------------------------------------------------------
      case ReceptionistState::NAV_TO_LIVING_ROOM: {
        if (last_nav_fail_time_.nanoseconds() > 0 &&
            (now() - last_nav_fail_time_).seconds() < nav_retry_cooldown_sec_)
        {
          return;
        }

        std::string escort_msg = "Please follow me to the living room.";
        {
          std::lock_guard<std::mutex> lock(guest_mutex_);
          if (guest_idx_ == 2 && !guests_.empty()) {
            // Optional Goal 2: deskripsikan Guest 1 ke Guest 2 SEBELUM tiba di living room
            const std::string g1_desc = formatPersonDescription(guests_[0]);
            escort_msg += " The first guest is " + guests_[0].name + ".";
            if (!g1_desc.empty()) {
              escort_msg += " " + g1_desc + ".";
            }
          }
        }

        speakAsync(escort_msg, [this]() {
          navigateTo("living_room", [this](bool success) {
            if (!success) return;
            search_chair_retries_ = 0;
            face_g1_retries_ = 0;
            face_g2_retries_ = 0;
            // Langsung cari kursi setelah scan ruang tamu (intro baru setelah kedua tamu duduk)
            after_scan_living_room_state_ = ReceptionistState::SEARCH_CHAIR;
            transitionTo(
              ReceptionistState::SCAN_LIVING_ROOM,
              "Arrived at living_room; scanning room before offering seat");
          });
        });
        break;
      }

      case ReceptionistState::SCAN_LIVING_ROOM: {
        if ((now() - state_enter_time_).seconds() < vision_initial_settle_sec_) {
          return;
        }
        const ReceptionistState next_after_scan = after_scan_living_room_state_;
        scanLivingRoom([this, next_after_scan](bool /*success*/, const std::string & /*msg*/) {
          transitionTo(
            next_after_scan,
            "Living room scan finished -> " + stateToString(next_after_scan));
        });
        break;
      }

      case ReceptionistState::SEARCH_CHAIR:
        searchChair([this](bool success, const std::string & /*msg*/) {
          if (!success && search_chair_retries_ < 1) {
            search_chair_retries_++;
            after_return_living_room_state_ = ReceptionistState::SEARCH_CHAIR;
            return_yaw_only_ = true;
            transitionTo(
              ReceptionistState::RETURN_TO_LIVING_ROOM,
              "Chair search failed; resetting yaw for 1x retry");
            return;
          }
          transitionTo(ReceptionistState::OFFER_SEAT, "Chair aligned; offering seat");
        });
        break;

      case ReceptionistState::OFFER_SEAT:
        if ((now() - state_enter_time_).seconds() < vision_initial_settle_sec_) {
          return;
        }
        pointToSeatWithArm([this](bool /*success*/) {
          speakAsync("Please sit on the seat I am pointing at.", [this]() {
            freeArm([this](bool /*ok*/) {
              transitionTo(ReceptionistState::WAIT_GUEST_SIT, "Waiting for guest to sit");
            });
          });
        });
        break;

      case ReceptionistState::WAIT_GUEST_SIT: {
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

          const int finished_guest_idx = guest_idx_;
          clearLivingRoomLocks([this, finished_guest_idx]() {
            if (finished_guest_idx == 1) {
              // Guest 1 duduk -> langsung jemput Guest 2 (hemat waktu < 6 menit)
              initCurrentGuest(2);
              transitionTo(
                ReceptionistState::NAV_TO_DOOR,
                "Guest 1 seated; returning to door for Guest 2");
            } else {
              // Guest 2 duduk -> reset yaw & scan ulang living room (antisipasi switching places)
              // sebelum memperkenalkan Guest 1 <-> Guest 2
              after_scan_living_room_state_ = ReceptionistState::INTRODUCE_G1_TO_G2;
              after_return_living_room_state_ = ReceptionistState::SCAN_LIVING_ROOM;
              return_yaw_only_ = true;
              transitionTo(
                ReceptionistState::RETURN_TO_LIVING_ROOM,
                "Both guests seated; resetting yaw to rescan room before introductions");
            }
          });
        } else {
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 3000,
            "[VISION WAIT] Waiting for Guest %d to occupy the indicated chair...", guest_idx_);
        }
        break;
      }

      case ReceptionistState::RETURN_TO_LIVING_ROOM: {
        const bool yaw_only = return_yaw_only_;
        return_yaw_only_ = false;

        if (yaw_only) {
          resetLivingRoomYaw([this](bool /*success*/, const std::string & /*msg*/) {
            transitionTo(
              after_return_living_room_state_,
              "Reset yaw to living_room -> " + stateToString(after_return_living_room_state_));
          });
        } else {
          navigateTo("living_room", [this](bool /*success*/) {
            transitionTo(
              after_return_living_room_state_,
              "Returned to living_room -> " + stateToString(after_return_living_room_state_));
          });
        }
        break;
      }

      // ------------------------------------------------------------------------
      // PHASE 4: INTRODUCE BOTH SEATED GUESTS TO EACH OTHER & SIMILARITY BONUS
      // ------------------------------------------------------------------------
      case ReceptionistState::INTRODUCE_G1_TO_G2: {
        {
          std::lock_guard<std::mutex> lock(guest_mutex_);
          if (guests_.size() < 2) {
            RCLCPP_ERROR(get_logger(), "Insufficient guest profiles (%zu < 2)!", guests_.size());
            clearLivingRoomLocks([this]() {
              transitionTo(ReceptionistState::FINISH_TASK, "Skipped introductions (missing profile)");
            });
            break;
          }
        }

        faceGuest1([this](bool success, const std::string & /*msg*/) {
          if (!success && face_g1_retries_ < 1) {
            face_g1_retries_++;
            after_return_living_room_state_ = ReceptionistState::INTRODUCE_G1_TO_G2;
            return_yaw_only_ = true;
            transitionTo(
              ReceptionistState::RETURN_TO_LIVING_ROOM,
              "Face Guest 1 failed; resetting yaw for 1x retry");
            return;
          }

          PersonProfile guest1, guest2;
          {
            std::lock_guard<std::mutex> lock(guest_mutex_);
            if (guests_.size() < 2) {
              clearLivingRoomLocks([this]() {
                transitionTo(ReceptionistState::FINISH_TASK, "Missing guest profile");
              });
              return;
            }
            guest1 = guests_[0];
            guest2 = guests_[1];
          }

          const std::string text =
            guest1.name + ", this is " + guest2.name + ". " +
            guest2.name + "'s favorite drink is " + guest2.drink +
            ", and interest is " + guest2.interest + ".";

          speakAsync(text, [this]() {
            transitionTo(ReceptionistState::INTRODUCE_G2_TO_G1, "Introduced Guest 2 to Guest 1");
          });
        });
        break;
      }

      case ReceptionistState::INTRODUCE_G2_TO_G1: {
        {
          std::lock_guard<std::mutex> lock(guest_mutex_);
          if (guests_.size() < 2) {
            clearLivingRoomLocks([this]() {
              transitionTo(ReceptionistState::FINISH_TASK, "Skipped intro (missing profile)");
            });
            break;
          }
        }

        faceGuest2([this](bool success, const std::string & /*msg*/) {
          if (!success && face_g2_retries_ < 1) {
            face_g2_retries_++;
            after_return_living_room_state_ = ReceptionistState::INTRODUCE_G2_TO_G1;
            return_yaw_only_ = true;
            transitionTo(
              ReceptionistState::RETURN_TO_LIVING_ROOM,
              "Face Guest 2 failed; resetting yaw for 1x retry");
            return;
          }

          PersonProfile guest1, guest2;
          {
            std::lock_guard<std::mutex> lock(guest_mutex_);
            if (guests_.size() < 2) {
              clearLivingRoomLocks([this]() {
                transitionTo(ReceptionistState::FINISH_TASK, "Missing guest profile");
              });
              return;
            }
            guest1 = guests_[0];
            guest2 = guests_[1];
          }

          const std::string text =
            guest2.name + ", this is " + guest1.name + ". " +
            guest1.name + "'s favorite drink is " + guest1.drink +
            ", and interest is " + guest1.interest + ".";

          speakAsync(text, [this]() {
            transitionTo(ReceptionistState::EVALUATE_SIMILARITY, "Introduced Guest 1 to Guest 2");
          });
        });
        break;
      }

      case ReceptionistState::EVALUATE_SIMILARITY: {
        std::vector<PersonProfile> all_people;
        {
          std::lock_guard<std::mutex> lock(guest_mutex_);
          if (guests_.size() < 2) {
            clearLivingRoomLocks([this]() {
              transitionTo(ReceptionistState::FINISH_TASK, "Skipped similarity evaluation");
            });
            break;
          }
          all_people = {host_, guests_[0], guests_[1]};
        }

        std::string similarity_speech;
        const std::string h_int = normalize(all_people[0].interest);
        const std::string g1_int = normalize(all_people[1].interest);
        const std::string g2_int = normalize(all_people[2].interest);
        const std::string h_cat = canonicalInterestCategory(all_people[0].interest);
        const std::string g1_cat = canonicalInterestCategory(all_people[1].interest);
        const std::string g2_cat = canonicalInterestCategory(all_people[2].interest);

        if (!h_int.empty() && h_int == g1_int && g1_int == g2_int) {
          similarity_speech =
            all_people[0].name + ", " + all_people[1].name + ", and " +
            all_people[2].name + " share the same interest: " +
            all_people[1].interest + ".";
        } else if (!h_cat.empty() && h_cat == g1_cat && g1_cat == g2_cat) {
          similarity_speech =
            all_people[0].name + ", " + all_people[1].name + ", and " +
            all_people[2].name + " all like " + h_cat + ".";
        } else {
          for (size_t i = 0; i < all_people.size(); ++i) {
            for (size_t j = i + 1; j < all_people.size(); ++j) {
              const std::string int_i = normalize(all_people[i].interest);
              const std::string int_j = normalize(all_people[j].interest);
              if (!int_i.empty() && int_i == int_j) {
                similarity_speech =
                  all_people[i].name + " and " + all_people[j].name +
                  " share the same interest: " + all_people[i].interest + ".";
                break;
              }
              const std::string cat_i = canonicalInterestCategory(all_people[i].interest);
              const std::string cat_j = canonicalInterestCategory(all_people[j].interest);
              if (!cat_i.empty() && cat_i == cat_j) {
                similarity_speech =
                  all_people[i].name + " and " + all_people[j].name +
                  " both like " + cat_i + ".";
                break;
              }
            }
            if (!similarity_speech.empty()) break;
          }
        }

        auto finishAfterSimilarity = [this]() {
          clearLivingRoomLocks([this]() {
            transitionTo(ReceptionistState::FINISH_TASK, "Introductions & similarity completed");
          });
        };

        if (!similarity_speech.empty()) {
          speakAsync(similarity_speech, finishAfterSimilarity);
        } else {
          finishAfterSimilarity();
        }
        break;
      }

      // ------------------------------------------------------------------------
      // PHASE 5: FINISH
      // ------------------------------------------------------------------------
      case ReceptionistState::FINISH_TASK:
        speakAsync("All guests are seated and introduced. Task complete.", [this]() {
          publishTaskStatus("SUCCESS: Receptionist task complete.");
          transitionTo(ReceptionistState::COMPLETED, "Receptionist task complete");
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