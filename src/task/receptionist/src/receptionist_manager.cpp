// ==============================================================================
// Barelang Inari - RoboCup@Home 2025
// Package  : receptionist
// Node     : receptionist_manager (ReceptionistTaskNode)
// Deskripsi: State Machine (FSM) utama untuk tugas Receptionist (Static Head).
//            Subsystems: ASR/TTS (athome_hri), Vision (ZED Static RGB-D),
//            Nav2 (receptionist_node / navigate_to_pose), Arm (athome_arm).
// edit     : 4 november 2026
// ==============================================================================

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "msgs/msg/person_detected.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"

using namespace std::chrono_literals;

// ==============================================================================
// 1. DEFINISI STATE FINITE STATE MACHINE (FSM)
// ==============================================================================
enum class ReceptionistState {
  INIT,
  NAV_TO_DOOR,
  WAIT_GUEST,
  ASK_NAME,
  WAIT_NAME,
  ASK_INTEREST,
  WAIT_INTEREST,
  NAV_TO_BEVERAGE,
  ASK_DRINK,
  WAIT_DRINK,
  CHECK_BEVERAGE,
  NAV_TO_LIVING_ROOM,
  OFFER_SEAT,
  WAIT_GUEST_SIT,
  NAV_TO_LIVING_ROOM_CENTER,
  INTRODUCE_G1_TO_G2,
  INTRODUCE_G2_TO_G1,
  EVALUATE_SIMILARITY,
  FINISH_TASK,
  COMPLETED,
  ABORTED
};

static std::string stateToString(ReceptionistState state)
{
  switch (state) {
    case ReceptionistState::INIT:                      return "INIT";
    case ReceptionistState::NAV_TO_DOOR:               return "NAV_TO_DOOR";
    case ReceptionistState::WAIT_GUEST:                return "WAIT_GUEST";
    case ReceptionistState::ASK_NAME:                  return "ASK_NAME";
    case ReceptionistState::WAIT_NAME:                 return "WAIT_NAME";
    case ReceptionistState::ASK_INTEREST:              return "ASK_INTEREST";
    case ReceptionistState::WAIT_INTEREST:             return "WAIT_INTEREST";
    case ReceptionistState::NAV_TO_BEVERAGE:           return "NAV_TO_BEVERAGE";
    case ReceptionistState::ASK_DRINK:                 return "ASK_DRINK";
    case ReceptionistState::WAIT_DRINK:                return "WAIT_DRINK";
    case ReceptionistState::CHECK_BEVERAGE:            return "CHECK_BEVERAGE";
    case ReceptionistState::NAV_TO_LIVING_ROOM:        return "NAV_TO_LIVING_ROOM";
    case ReceptionistState::OFFER_SEAT:                return "OFFER_SEAT";
    case ReceptionistState::WAIT_GUEST_SIT:            return "WAIT_GUEST_SIT";
    case ReceptionistState::NAV_TO_LIVING_ROOM_CENTER: return "NAV_TO_LIVING_ROOM_CENTER";
    case ReceptionistState::INTRODUCE_G1_TO_G2:        return "INTRODUCE_G1_TO_G2";
    case ReceptionistState::INTRODUCE_G2_TO_G1:        return "INTRODUCE_G2_TO_G1";
    case ReceptionistState::EVALUATE_SIMILARITY:       return "EVALUATE_SIMILARITY";
    case ReceptionistState::FINISH_TASK:               return "FINISH_TASK";
    case ReceptionistState::COMPLETED:                 return "COMPLETED";
    case ReceptionistState::ABORTED:                   return "ABORTED";
    default:                                           return "UNKNOWN";
  }
}

// ==============================================================================
// 2. STRUKTUR DATA HOST, GUEST, DAN DETEKSI VISION
// ==============================================================================
struct PersonProfile {
  std::string id;
  std::string name;
  std::string drink;
  std::string interest;
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
// 3. KELAS UTAMA: ReceptionistTaskNode
// ==============================================================================
class ReceptionistTaskNode : public rclcpp::Node {
public:
  using Trigger = std_srvs::srv::Trigger;
  using TriggerClient = rclcpp::Client<Trigger>;
  using TriggerResult = std::function<void(bool, const std::string &)>;

  ReceptionistTaskNode()
  : Node("receptionist_manager")
  {
    // Parameter Host & Konfigurasi Task
    declare_parameter<std::string>("host.name", "yesaya");
    declare_parameter<std::string>("host.drink", "coffee");
    declare_parameter<std::string>("host.interest", "Robotics");
    declare_parameter<int>("camera_width", 672);

    host_.id = "host";
    host_.name = get_parameter("host.name").as_string();
    host_.drink = get_parameter("host.drink").as_string();
    host_.interest = get_parameter("host.interest").as_string();
    camera_width_ = get_parameter("camera_width").as_int();

    callback_group_ = create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.callback_group = callback_group_;

    // --------------------------------------------------------------------------
    // Publisher & Subscriber HRI (TTS & STT)
    // --------------------------------------------------------------------------
    tts_pub_ = create_publisher<std_msgs::msg::String>("/hri/tts_say", 10);
    sound_event_pub_ = create_publisher<std_msgs::msg::String>("/hri/sound_event", 10);
    guest_name_pub_ = create_publisher<std_msgs::msg::String>("/hri/guest_name", 10);

    tts_done_sub_ = create_subscription<std_msgs::msg::String>(
      "/hri/tts_done", 10,
      std::bind(&ReceptionistTaskNode::onTtsDone, this, std::placeholders::_1),
      sub_opts);

    stt_sub_ = create_subscription<std_msgs::msg::String>(
      "/hri/stt_transcript", 10,
      std::bind(&ReceptionistTaskNode::onTranscript, this, std::placeholders::_1),
      sub_opts);

    // --------------------------------------------------------------------------
    // Subscriber Vision (person_det C++ & Static RGB-D ZED Camera)
    // --------------------------------------------------------------------------
    person_detected_sub_ = create_subscription<msgs::msg::PersonDetected>(
      "/vision/detected_person", 10,
      std::bind(&ReceptionistTaskNode::onPersonDetected, this, std::placeholders::_1),
      sub_opts);

    person_sub_ = create_subscription<std_msgs::msg::String>(
      "/zed/person_label", rclcpp::SensorDataQoS(),
      std::bind(&ReceptionistTaskNode::onPersonLabel, this, std::placeholders::_1),
      sub_opts);

    drink_sub_ = create_subscription<std_msgs::msg::String>(
      "/zed/drink_detections", rclcpp::SensorDataQoS(),
      std::bind(&ReceptionistTaskNode::onDrinkDetections, this, std::placeholders::_1),
      sub_opts);

    // Fallback ke /zed/bag_detections jika node vision mempublish objek di topic tersebut
    object_sub_ = create_subscription<std_msgs::msg::String>(
      "/zed/bag_detections", rclcpp::SensorDataQoS(),
      std::bind(&ReceptionistTaskNode::onDrinkDetections, this, std::placeholders::_1),
      sub_opts);

    attributes_sub_ = create_subscription<std_msgs::msg::String>(
      "/zed/person_attributes", 10,
      std::bind(&ReceptionistTaskNode::onPersonAttributes, this, std::placeholders::_1),
      sub_opts);

    seat_sub_ = create_subscription<std_msgs::msg::String>(
      "/zed/empty_seat", 10,
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
    nav_living_center_client_ = create_client<Trigger>(
      "/receptionist/go_to_living_room_center", rclcpp::ServicesQoS(), callback_group_);

    arm_point_seat_client_ = create_client<Trigger>(
      "/arm/point_seat", rclcpp::ServicesQoS(), callback_group_);
    arm_play_motion_client_ = create_client<Trigger>(
      "/arm/play_motion", rclcpp::ServicesQoS(), callback_group_);

    // Timer siklus FSM (10 Hz / 100 ms)
    fsm_timer_ = create_wall_timer(
      100ms, std::bind(&ReceptionistTaskNode::fsmTick, this), callback_group_);

    task_start_time_ = now();
    state_enter_time_ = now();

    initCurrentGuest(1);

    RCLCPP_INFO(get_logger(),
      "\n============================================================\n"
      "  \033[1;32m[SYSTEM]\033[0m Receptionist Task Manager Started\n"
      "  \033[1;36m[HOST]\033[0m   %s | Drink: %s | Interest: %s\n"
      "  \033[1;33m[STATE]\033[0m  INIT\n"
      "============================================================",
      host_.name.c_str(), host_.drink.c_str(), host_.interest.c_str());
  }

private:
  // ----------------------------------------------------------------------------
  // Konstanta Waktu & Parameter
  // ----------------------------------------------------------------------------
  static constexpr double GLOBAL_TIMEOUT_SEC     = 600.0;
  static constexpr double DIALOG_TIMEOUT_SEC     = 15.0;
  static constexpr double SIT_WAIT_TIMEOUT_SEC   = 10.0;
  static constexpr double TTS_MIN_TIMEOUT_SEC    = 4.0;
  static constexpr double TTS_SEC_PER_WORD       = 0.5;
  static constexpr double TTS_TIMEOUT_MARGIN_SEC = 2.0;

  // ----------------------------------------------------------------------------
  // Variabel State & Memory Store Host / Guests
  // ----------------------------------------------------------------------------
  std::atomic<ReceptionistState> state_{ReceptionistState::INIT};
  std::atomic<bool> busy_{false};
  std::atomic<bool> speaking_{false};
  std::atomic<bool> person_seen_{false};
  std::atomic<int> nearest_person_id_{-1};

  PersonProfile host_;
  std::vector<PersonProfile> guests_;
  PersonProfile current_guest_;
  int guest_idx_{1};
  int camera_width_{672};

  // Data Persepsi Vision
  std::mutex vision_mutex_;
  std::vector<DetectedObject> latest_drinks_;
  std::vector<std::string> latest_attributes_{
    "Dark shirt", "Black hair", "Standing posture", "No glasses"
  };
  std::string empty_seat_position_{"center"};
  double nearest_person_dist_{999.0};
  int nearest_person_bbox_y_{0};

  // Data STT / Dialog
  std::mutex stt_mutex_;
  std::string heard_utterance_;
  std::atomic<bool> has_new_utterance_{false};
  std::string last_utterance_;
  rclcpp::Time last_utterance_time_{0, 0, RCL_ROS_TIME};

  // Timing & TTS
  rclcpp::Time task_start_time_;
  rclcpp::Time state_enter_time_;
  rclcpp::Time last_person_seen_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time tts_cooldown_until_{0, 0, RCL_ROS_TIME};
  rclcpp::Time tts_started_time_{0, 0, RCL_ROS_TIME};

  std::mutex tts_mutex_;
  double tts_timeout_sec_{TTS_MIN_TIMEOUT_SEC};
  std::function<void()> pending_tts_done_;

  // Interfaces ROS 2
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp::TimerBase::SharedPtr fsm_timer_;
  // Data Persepsi person_det (Pose, Gender, Shirt Color, Re-ID)
  std::string latest_person_name_{""};
  std::string latest_person_gender_{"unknown"};
  std::string latest_person_shirt_{"unknown"};

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr tts_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr sound_event_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr guest_name_pub_;
  rclcpp::Subscription<msgs::msg::PersonDetected>::SharedPtr person_detected_sub_;
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
  TriggerClient::SharedPtr nav_living_center_client_;

  TriggerClient::SharedPtr arm_point_seat_client_;
  TriggerClient::SharedPtr arm_play_motion_client_;

  // ============================================================================
  // SUB-ROUTINES / HELPER FUNCTIONS (SESUAI PSEUDOCODE)
  // ============================================================================
  void initCurrentGuest(int idx)
  {
    guest_idx_ = idx;
    current_guest_ = PersonProfile{};
    current_guest_.id = "guest_" + std::to_string(idx);
    current_guest_.name.clear();
    current_guest_.drink.clear();
    current_guest_.interest.clear();
    current_guest_.attributes.clear();
  }

  void transitionTo(ReceptionistState next_state, const std::string & reason)
  {
    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[FSM STATE]\033[0m %s \033[1;33m->\033[0m \033[1;32m%s\033[0m (Guest %d) | %s",
      stateToString(state_.load()).c_str(),
      stateToString(next_state).c_str(),
      guest_idx_,
      reason.c_str());

    state_enter_time_ = now();
    has_new_utterance_ = false;
    state_ = next_state;
  }

  // Navigasi ke waypoint menggunakan service dari receptionist_node
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
    } else if (waypoint_name == "living_room") {
      client = nav_living_room_client_;
      srv_name = "/receptionist/go_to_living_room";
    } else if (waypoint_name == "living_room_center") {
      client = nav_living_center_client_;
      srv_name = "/receptionist/go_to_living_room_center";
    } else {
      client = nav_living_room_client_;
      srv_name = "/receptionist/go_to_living_room";
    }

    RCLCPP_INFO(get_logger(), "\033[1;34m[NAV2]\033[0m Navigating to '%s' via %s...",
      waypoint_name.c_str(), srv_name.c_str());

    callTrigger(
      client, srv_name,
      [this, waypoint_name, on_finished](bool success, const std::string & msg) {
        if (!success) {
          RCLCPP_WARN(
            get_logger(),
            "[NAV2] Navigation to '%s' reported failure (%s), continuing state machine.",
            waypoint_name.c_str(), msg.c_str());
        } else {
          RCLCPP_INFO(get_logger(), "[NAV2] Reached '%s': %s", waypoint_name.c_str(), msg.c_str());
        }
        on_finished(success);
      },
      120s);
  }

  // Menentukan posisi horizontal (left, center, right) pada frame kamera statis
  std::string calculateHorizontalPosition(int bbox_x, int bbox_w) const
  {
    const double center_x = static_cast<double>(bbox_x) + static_cast<double>(bbox_w) * 0.5;
    const double third = static_cast<double>(camera_width_) / 3.0;
    if (center_x < third) {
      return "left";
    } else if (center_x > 2.0 * third) {
      return "right";
    }
    return "center";
  }

  // Memindai meja minuman dan menentukan ketersediaan serta posisi horizontal
  std::pair<bool, std::string> detectAndCheckBeverage(const std::string & favorite_drink)
  {
    std::lock_guard<std::mutex> lock(vision_mutex_);
    const std::string target = normalize(favorite_drink);

    for (const auto & drink : latest_drinks_) {
      const std::string d_name = normalize(drink.name);
      if (!d_name.empty() &&
          (d_name == target || d_name.find(target) != std::string::npos ||
           target.find(d_name) != std::string::npos))
      {
        return {true, calculateHorizontalPosition(drink.bbox_x, drink.bbox_w)};
      }
    }

    // Jika objek terdeteksi sebagai generic bottle/cup di meja minuman
    if (!latest_drinks_.empty()) {
      const auto & first = latest_drinks_.front();
      if (first.name == "bottle" || first.name == "cup" || first.name == "drink") {
        return {true, calculateHorizontalPosition(first.bbox_x, first.bbox_w)};
      }
    }

    return {false, "not available"};
  }

  // Mengekstrak 4 atribut visual Tamu 1 untuk dideskripsikan ke Tamu 2 (+120 pts)
  std::vector<std::string> extractVisualAttributes(int person_id)
  {
    std::lock_guard<std::mutex> lock(vision_mutex_);
    RCLCPP_INFO(
      get_logger(),
      "\033[1;32m[VISION]\033[0m Extracting visual attributes for person_id=%d",
      person_id);

    std::vector<std::string> attrs;
    if (!latest_person_shirt_.empty() && latest_person_shirt_ != "unknown") {
      attrs.push_back(latest_person_shirt_ + " shirt");
    }
    if (!latest_person_gender_.empty() && latest_person_gender_ != "unknown") {
      attrs.push_back(latest_person_gender_);
    }
    for (const auto & a : latest_attributes_) {
      if (std::find(attrs.begin(), attrs.end(), a) == attrs.end()) {
        attrs.push_back(a);
      }
    }
    while (attrs.size() < 4) {
      attrs.push_back("Casual attire");
    }
    if (attrs.size() > 4) {
      attrs.resize(4);
    }
    return attrs;
  }

  // Deteksi orang terdekat di depan base robot
  bool isPersonInFront() const
  {
    return person_seen_.load() && (now() - last_person_seen_time_).seconds() < 2.5;
  }

  // ============================================================================
  // HELPER TEXT-TO-SPEECH (TTS) & SPEECH-TO-TEXT (STT) SLOT PARSING
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
      busy_ = true;
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
      speaking_ = false;
      busy_ = false;
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
  }

  bool canListen() const
  {
    return !speaking_.load() && now() >= tts_cooldown_until_;
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

  // Ekstraksi slot dari kalimat STT ("person_name", "interest", "drink")
  static std::string parseSlotIntent(const std::string & raw_text, const std::string & slot)
  {
    const std::string norm = normalize(raw_text);
    if (norm.empty()) {
      return "";
    }

    auto stripPrefixes = [&](const std::vector<std::string> & prefixes) -> std::string {
      for (const auto & p : prefixes) {
        const auto pos = norm.find(p);
        if (pos != std::string::npos) {
          std::string rem = norm.substr(pos + p.size());
          while (!rem.empty() && rem.front() == ' ') {
            rem.erase(rem.begin());
          }
          if (!rem.empty()) {
            return capitalizeWords(rem);
          }
        }
      }
      return capitalizeWords(norm);
    };

    if (slot == "person_name") {
      static const std::vector<std::string> kKnownNames = {
        "adel", "angel", "axel", "charlie", "jane", "john", "jules", "morgan",
        "paris", "robin", "simone", "alex", "michael", "sarah", "david", "emma"
      };
      for (const auto & name : kKnownNames) {
        if (norm.find(name) != std::string::npos) {
          return capitalizeWords(name);
        }
      }
      return stripPrefixes({
        "my name is ", "i am ", "im ", "call me ", "this is ", "name is "
      });
    }

    if (slot == "drink") {
      static const std::vector<std::string> kKnownDrinks = {
        "coke", "cola", "pepsi", "sprite", "fanta", "orange juice", "apple juice",
        "milk", "water", "iced tea", "green tea", "coffee", "tea", "juice", "soda"
      };
      for (const auto & drink : kKnownDrinks) {
        if (norm.find(drink) != std::string::npos) {
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
        "robotics", "artificial intelligence", "reading", "music", "sports",
        "cooking", "gaming", "traveling", "photography", "movies", "swimming",
        "football", "basketball", "coding", "programming", "dancing", "painting"
      };
      for (const auto & interest : kKnownInterests) {
        if (norm.find(interest) != std::string::npos) {
          return capitalizeWords(interest);
        }
      }
      return stripPrefixes({
        "my main interest is ", "my interest is ", "my hobby is ",
        "i am interested in ", "i like ", "i love ", "i enjoy "
      });
    }

    return capitalizeWords(norm);
  }

  // ============================================================================
  // CALLBACK SUBSCRIPTION ROS 2
  // ============================================================================
  void onTranscript(const std_msgs::msg::String::SharedPtr message)
  {
    if (!canListen()) {
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

  // Callback deteksi orang dari person_det (/vision/detected_person)
  void onPersonDetected(const msgs::msg::PersonDetected::SharedPtr message)
  {
    if (!message) return;
    person_seen_ = true;
    last_person_seen_time_ = now();

    std::lock_guard<std::mutex> lock(vision_mutex_);
    latest_person_name_ = message->name;
    latest_person_gender_ = message->gender;
    latest_person_shirt_ = message->shirt_color;

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "\033[1;35m[PERSON_DET]\033[0m Detected: name='%s', gender='%s', shirt='%s'",
      message->name.c_str(), message->gender.c_str(), message->shirt_color.c_str());
  }

  void triggerDoorbell()
  {
    if (!sound_event_pub_) return;
    std_msgs::msg::String msg;
    msg.data = "doorbell";
    sound_event_pub_->publish(msg);
    RCLCPP_INFO(get_logger(), "\033[1;32m[HRI]\033[0m Triggered 'doorbell' on /hri/sound_event for person_det");
  }

  void registerGuestName(const std::string & name)
  {
    if (!guest_name_pub_ || name.empty()) return;
    std_msgs::msg::String msg;
    msg.data = name;
    guest_name_pub_->publish(msg);
    RCLCPP_INFO(get_logger(), "\033[1;32m[HRI]\033[0m Registered guest '%s' on /hri/guest_name for Face Re-ID", name.c_str());
  }

  // Callback deteksi orang dari ZED (/zed/person_label)
  // Format: "operator:x,y,w,h|x,y,z|id=0|kpt=...;person:..."
  void onPersonLabel(const std_msgs::msg::String::SharedPtr message)
  {
    if (!message || message->data.empty() || message->data.find("NO_PERSON") != std::string::npos) {
      person_seen_ = false;
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
        nearest_person_bbox_y_ = rect[1];
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

  // Callback deteksi minuman/objek di meja (/zed/drink_detections atau /zed/bag_detections)
  // Format yang didukung: "coke:x,y,w,h|X,Y,Z|score;sprite:x,y,w,h|..."
  void onDrinkDetections(const std_msgs::msg::String::SharedPtr message)
  {
    std::lock_guard<std::mutex> lock(vision_mutex_);
    latest_drinks_.clear();
    if (!message || message->data.empty()) {
      return;
    }

    std::stringstream ss(message->data);
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
      latest_drinks_.push_back(obj);
    }
  }

  // Callback atribut visual tamu (/zed/person_attributes, dipisahkan koma/titik koma)
  void onPersonAttributes(const std_msgs::msg::String::SharedPtr message)
  {
    if (!message || message->data.empty()) return;
    std::lock_guard<std::mutex> lock(vision_mutex_);
    std::vector<std::string> parsed;
    std::stringstream ss(message->data);
    std::string item;
    while (std::getline(ss, item, ',')) {
      if (!item.empty()) {
        parsed.push_back(item);
      }
    }
    if (!parsed.empty()) {
      latest_attributes_ = parsed;
    }
  }

  // Callback deteksi kursi kosong (/zed/empty_seat: "left", "center", "right")
  void onEmptySeat(const std_msgs::msg::String::SharedPtr message)
  {
    if (!message || message->data.empty()) return;
    std::lock_guard<std::mutex> lock(vision_mutex_);
    empty_seat_position_ = message->data;
  }

  // ============================================================================
  // HELPER SERVICE CLIENT NON-BLOCKING
  // ============================================================================
  void callTrigger(
    const TriggerClient::SharedPtr & client,
    const std::string & service_name,
    TriggerResult result,
    std::chrono::seconds timeout = 3s)
  {
    std::thread([this, client, service_name, result, timeout]() {
      if (!client) {
        result(false, "Client is null");
        return;
      }
      if (!client->wait_for_service(timeout)) {
        RCLCPP_WARN(
          get_logger(), "Service %s belum tersedia setelah timeout %ld detik.",
          service_name.c_str(), static_cast<long>(timeout.count()));
        result(false, "Service not ready (timeout)");
        return;
      }

      auto request = std::make_shared<Trigger::Request>();
      client->async_send_request(
        request,
        [result](TriggerClient::SharedFuture future) {
          try {
            const auto response = future.get();
            result(response->success, response->message);
          } catch (const std::exception & e) {
            result(false, e.what());
          }
        });
    }).detach();
  }

  // ============================================================================
  // MAIN FINITE STATE MACHINE (FSM TICK)
  // ============================================================================
  void fsmTick()
  {
    const ReceptionistState current_state = state_.load();

    // Global timeout watchdog
    if ((now() - task_start_time_).seconds() > GLOBAL_TIMEOUT_SEC &&
        current_state != ReceptionistState::COMPLETED &&
        current_state != ReceptionistState::ABORTED)
    {
      RCLCPP_ERROR(get_logger(), "Global receptionist task timeout reached.");
      state_ = ReceptionistState::ABORTED;
      return;
    }

    // TTS watchdog jika /hri/tts_done tidak diterima
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
        "TTS timeout (> %.1f s, no /hri/tts_done received), releasing busy lock.",
        tts_timeout);
      onTtsDone(nullptr);
    }

    if (busy_.load()) {
      return;
    }

    switch (current_state) {
      // ------------------------------------------------------------------------
      // INIT: Tunggu service /receptionist/go_to_door siap lalu langsung navigasi ke pintu
      // ------------------------------------------------------------------------
      case ReceptionistState::INIT:
        if (!nav_door_client_ || !nav_door_client_->service_is_ready()) {
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Waiting for '/receptionist/go_to_door' service to be ready...");
          return;
        }
        RCLCPP_INFO(get_logger(), "=== STARTING RECEPTIONIST TASK ===");
        initCurrentGuest(1);
        transitionTo(ReceptionistState::NAV_TO_DOOR, "Start cycle for Guest 1: navigating to door");
        break;

      // ------------------------------------------------------------------------
      // STATE 1: GREETING AT ENTRANCE DOOR
      // ------------------------------------------------------------------------
      case ReceptionistState::NAV_TO_DOOR:
        busy_ = true;
        navigateTo("entrance_door", [this](bool) {
          triggerDoorbell();
          transitionTo(ReceptionistState::WAIT_GUEST, "Arrived at entrance_door");
          busy_ = false;
        });
        break;

      case ReceptionistState::WAIT_GUEST:
        if (isPersonInFront() || (now() - state_enter_time_).seconds() > 8.0) {
          speakAsync("Hello! Welcome to the party. I am Barelang Inari from Batam State Polytechic, here im is your robot receptionist.", [this]() {
            transitionTo(ReceptionistState::ASK_NAME, "Guest greeted at entrance");
          });
        }
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
            current_guest_.name = parsed_name;
            RCLCPP_INFO(get_logger(), "[DIALOG] Guest %d Name: %s",
              guest_idx_, current_guest_.name.c_str());
            registerGuestName(current_guest_.name);
            transitionTo(ReceptionistState::ASK_INTEREST, "Guest name captured");
          }
        } else if ((now() - state_enter_time_).seconds() > DIALOG_TIMEOUT_SEC) {
          current_guest_.name = (guest_idx_ == 1) ? "Charlie" : "Robin";
          RCLCPP_WARN(get_logger(), "[DIALOG] Name timeout, fallback to %s",
            current_guest_.name.c_str());
          registerGuestName(current_guest_.name);
          transitionTo(ReceptionistState::ASK_INTEREST, "Guest name fallback on timeout");
        }
        break;

      case ReceptionistState::ASK_INTEREST:
        speakAsync(
          "Nice to meet you " + current_guest_.name +
          ". What is your main interest or hobby?",
          [this]() {
            transitionTo(ReceptionistState::WAIT_INTEREST, "Waiting for guest interest");
          });
        break;

      case ReceptionistState::WAIT_INTEREST:
        if (has_new_utterance_.exchange(false)) {
          std::string raw;
          {
            std::lock_guard<std::mutex> lock(stt_mutex_);
            raw = heard_utterance_;
          }
          const std::string parsed_interest = parseSlotIntent(raw, "interest");
          if (!parsed_interest.empty()) {
            current_guest_.interest = parsed_interest;
            RCLCPP_INFO(get_logger(), "[DIALOG] Guest %d Interest: %s",
              guest_idx_, current_guest_.interest.c_str());

            if (guest_idx_ == 1) {
              current_guest_.attributes = extractVisualAttributes(nearest_person_id_.load());
            }
            transitionTo(ReceptionistState::NAV_TO_BEVERAGE, "Guest interest captured");
          }
        } else if ((now() - state_enter_time_).seconds() > DIALOG_TIMEOUT_SEC) {
          current_guest_.interest = "Robotics";
          RCLCPP_WARN(get_logger(), "[DIALOG] Interest timeout, fallback to %s",
            current_guest_.interest.c_str());
          if (guest_idx_ == 1) {
            current_guest_.attributes = extractVisualAttributes(nearest_person_id_.load());
          }
          transitionTo(ReceptionistState::NAV_TO_BEVERAGE, "Guest interest fallback on timeout");
        }
        break;

      // ------------------------------------------------------------------------
      // STATE 2: BEVERAGE AREA & DRINK CHECK
      // ------------------------------------------------------------------------
      case ReceptionistState::NAV_TO_BEVERAGE:
        speakAsync("Please follow me to the beverage area.", [this]() {
          busy_ = true;
          navigateTo("beverage_area", [this](bool) {
            transitionTo(ReceptionistState::ASK_DRINK, "Arrived at beverage_area");
            busy_ = false;
          });
        });
        break;

      case ReceptionistState::ASK_DRINK:
        speakAsync(current_guest_.name + ", what is your favorite drink?", [this]() {
          transitionTo(ReceptionistState::WAIT_DRINK, "Waiting for favorite drink");
        });
        break;

      case ReceptionistState::WAIT_DRINK:
        if (has_new_utterance_.exchange(false)) {
          std::string raw;
          {
            std::lock_guard<std::mutex> lock(stt_mutex_);
            raw = heard_utterance_;
          }
          const std::string parsed_drink = parseSlotIntent(raw, "drink");
          if (!parsed_drink.empty()) {
            current_guest_.drink = parsed_drink;
            RCLCPP_INFO(get_logger(), "[DIALOG] Guest %d Drink: %s",
              guest_idx_, current_guest_.drink.c_str());
            transitionTo(ReceptionistState::CHECK_BEVERAGE, "Favorite drink captured");
          }
        } else if ((now() - state_enter_time_).seconds() > DIALOG_TIMEOUT_SEC) {
          current_guest_.drink = (guest_idx_ == 1) ? "Juice" : "Coffee";
          RCLCPP_WARN(get_logger(), "[DIALOG] Drink timeout, fallback to %s",
            current_guest_.drink.c_str());
          transitionTo(ReceptionistState::CHECK_BEVERAGE, "Drink fallback on timeout");
        }
        break;

      case ReceptionistState::CHECK_BEVERAGE: {
        const auto [found, position] = detectAndCheckBeverage(current_guest_.drink);
        std::string report;
        if (found) {
          report = "Your favorite drink, " + current_guest_.drink +
                   ", is available on the " + position + " side of the table.";
        } else {
          report = "I am sorry, " + current_guest_.drink +
                   " is currently not available on the table.";
        }

        guests_.push_back(current_guest_);

        speakAsync(report, [this]() {
          transitionTo(ReceptionistState::NAV_TO_LIVING_ROOM, "Beverage check completed");
        });
        break;
      }

      // ------------------------------------------------------------------------
      // STATE 3: ESCORT TO LIVING ROOM & OFFER SEAT
      // ------------------------------------------------------------------------
      case ReceptionistState::NAV_TO_LIVING_ROOM: {
        std::string escort_msg = "Now, let me escort you to the living room.";

        // [BONUS] Ceritakan deskripsi Tamu 1 saat mengantar Tamu 2 (+120 pts)
        if (guest_idx_ == 2 && !guests_.empty()) {
          std::string guest1_attrs;
          for (size_t i = 0; i < guests_[0].attributes.size(); ++i) {
            if (i > 0) guest1_attrs += ", ";
            guest1_attrs += guests_[0].attributes[i];
          }
          escort_msg += " By the way, Guest 1, " + guests_[0].name +
                        ", is already in the living room. Their visual attributes are: " +
                        guest1_attrs + ".";
        }

        speakAsync(escort_msg, [this]() {
          busy_ = true;
          navigateTo("living_room", [this](bool) {
            transitionTo(ReceptionistState::OFFER_SEAT, "Arrived at living_room");
            busy_ = false;
          });
        });
        break;
      }

      case ReceptionistState::OFFER_SEAT: {
        std::string seat_pos;
        {
          std::lock_guard<std::mutex> lock(vision_mutex_);
          seat_pos = empty_seat_position_;
        }
        RCLCPP_INFO(
          get_logger(),
          "\033[1;32m[VISION]\033[0m Empty seat detected at '%s', pointing arm...",
          seat_pos.c_str());

        busy_ = true;
        callTrigger(
          arm_point_seat_client_, "/arm/point_seat",
          [this](bool, const std::string &) {
            speakAsync(
              current_guest_.name +
              ", please feel free to take a seat at the location I am pointing to.",
              [this]() {
                transitionTo(ReceptionistState::WAIT_GUEST_SIT, "Waiting for guest to sit");
              });
          },
          3s);
        break;
      }

      case ReceptionistState::WAIT_GUEST_SIT:
        if ((now() - state_enter_time_).seconds() >= SIT_WAIT_TIMEOUT_SEC) {
          if (guest_idx_ == 1) {
            initCurrentGuest(2);
            transitionTo(ReceptionistState::NAV_TO_DOOR, "Guest 1 seated, starting Guest 2 cycle");
          } else {
            transitionTo(
              ReceptionistState::NAV_TO_LIVING_ROOM_CENTER,
              "Guest 2 seated, proceeding to introductions");
          }
        }
        break;

      // ------------------------------------------------------------------------
      // STATE 4: INTRODUCTIONS (PERKENALAN ANTAR TAMU)
      // ------------------------------------------------------------------------
      case ReceptionistState::NAV_TO_LIVING_ROOM_CENTER:
        busy_ = true;
        navigateTo("living_room_center", [this](bool) {
          transitionTo(
            ReceptionistState::INTRODUCE_G1_TO_G2,
            "Arrived at living_room_center");
          busy_ = false;
        });
        break;

      case ReceptionistState::INTRODUCE_G1_TO_G2: {
        const auto & guest1 = guests_[0];
        const auto & guest2 = guests_[1];
        const std::string text =
          guest2.name + ", I would like to introduce " + guest1.name +
          ". Their favorite drink is " + guest1.drink +
          ", and their interest is " + guest1.interest + ".";

        speakAsync(text, [this]() {
          transitionTo(
            ReceptionistState::INTRODUCE_G2_TO_G1,
            "Introduced Guest 1 to Guest 2");
        });
        break;
      }

      case ReceptionistState::INTRODUCE_G2_TO_G1: {
        const auto & guest1 = guests_[0];
        const auto & guest2 = guests_[1];
        const std::string text =
          guest1.name + ", this is " + guest2.name +
          ". Their favorite drink is " + guest2.drink +
          ", and their interest is " + guest2.interest + ".";

        speakAsync(text, [this]() {
          transitionTo(
            ReceptionistState::EVALUATE_SIMILARITY,
            "Introduced Guest 2 to Guest 1");
        });
        break;
      }

      case ReceptionistState::EVALUATE_SIMILARITY: {
        // [BONUS] Evaluasi Kesamaan Minat (+50 pts)
        const std::vector<PersonProfile> all_people = {host_, guests_[0], guests_[1]};
        std::string similarity_speech;

        for (size_t i = 0; i < all_people.size(); ++i) {
          for (size_t j = i + 1; j < all_people.size(); ++j) {
            if (normalize(all_people[i].interest) == normalize(all_people[j].interest)) {
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

        if (!similarity_speech.empty()) {
          speakAsync(similarity_speech, [this]() {
            transitionTo(ReceptionistState::FINISH_TASK, "Interest similarity announced");
          });
        } else {
          transitionTo(ReceptionistState::FINISH_TASK, "No identical interest found");
        }
        break;
      }

      case ReceptionistState::FINISH_TASK:
        speakAsync("Enjoy the party everyone! My task here is completed.", [this]() {
          RCLCPP_INFO(get_logger(), "=== RECEPTIONIST TASK FINISHED SUCCESSFULLY ===");
          transitionTo(ReceptionistState::COMPLETED, "Receptionist task finished");
        });
        break;

      case ReceptionistState::COMPLETED:
      case ReceptionistState::ABORTED:
        break;
    }
  }
};

// ==============================================================================
// 4. ENTRY POINT UTAMA
// ==============================================================================
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ReceptionistTaskNode>();

  rclcpp::executors::MultiThreadedExecutor executor(
    rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}

