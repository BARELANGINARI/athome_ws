// ==============================================================================
// Barelang Inari - RoboCup@Home 2026
// Package  : receptionist
// Node     : receptionist_manager (ReceptionistTaskNode)
// Deskripsi: State Machine (FSM) utama untuk tugas Receptionist (Static Head).
//            Subsystems: ASR/TTS (athome_hri), Vision (ZED Static RGB-D),
//            Nav2 (receptionist_node / navigate_to_pose), Arm (athome_arm).
// edit     : 5 november 2026 pukul 21.00 (Fixed Navigation Sync, Vision Settle, & Thread Safety)
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
#include <vector>

#include "rclcpp/rclcpp.hpp"
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
  CONFIRM_GUEST_INFO,
  WAIT_CONFIRM_GUEST_INFO,
  NAV_TO_BEVERAGE,
  ASK_DRINK,
  WAIT_DRINK,
  CHECK_BEVERAGE,
  NAV_TO_LIVING_ROOM,
  INTRODUCE_G1_TO_G2,
  INTRODUCE_G2_TO_G1,
  EVALUATE_SIMILARITY,
  SEARCH_CHAIR,
  OFFER_SEAT,
  WAIT_GUEST_SIT,
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
    case ReceptionistState::INTRODUCE_G1_TO_G2:      return "INTRODUCE_G1_TO_G2";
    case ReceptionistState::INTRODUCE_G2_TO_G1:      return "INTRODUCE_G2_TO_G1";
    case ReceptionistState::EVALUATE_SIMILARITY:     return "EVALUATE_SIMILARITY";
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
// 2. STRUKTUR DATA HOST, GUEST, DAN DETEKSI VISION
// ==============================================================================
struct PersonProfile {
  std::string id;
  std::string name;
  std::string drink;
  std::string interest;
  std::string gender{"unknown"};
  std::string shirt_color{"casual attire"};
  std::string hair_color{"dark hair"};
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
    declare_parameter<std::string>("host.name", "Alex");
    declare_parameter<std::string>("host.drink", "Coke");
    declare_parameter<std::string>("host.interest", "Robotics");
    declare_parameter<std::string>("host.gender", "male");
    declare_parameter<std::string>("host.shirt_color", "black");
    declare_parameter<std::string>("host.hair_color", "black");
    declare_parameter<int>("camera_width", 672);

    host_.id = "host";
    host_.name = get_parameter("host.name").as_string();
    host_.drink = get_parameter("host.drink").as_string();
    host_.interest = get_parameter("host.interest").as_string();
    host_.gender = get_parameter("host.gender").as_string();
    host_.shirt_color = get_parameter("host.shirt_color").as_string();
    host_.hair_color = get_parameter("host.hair_color").as_string();
    host_.attributes = {
      host_.gender,
      host_.shirt_color + " shirt",
      host_.hair_color + " hair"
    };
    camera_width_ = get_parameter("camera_width").as_int();

    callback_group_ = create_callback_group(
      rclcpp::CallbackGroupType::Reentrant);

    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.callback_group = callback_group_;

    // --------------------------------------------------------------------------
    // Publisher & Subscriber HRI (TTS, STT, & Task Status)
    // --------------------------------------------------------------------------
    tts_pub_ = create_publisher<std_msgs::msg::String>("/hri/tts_say", 10);
    status_pub_ = create_publisher<std_msgs::msg::String>("/receptionist/status", 10);
    person_name_pub_ = create_publisher<std_msgs::msg::String>("/zed/set_person_name", 10);

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
      "/zed/person_label", rclcpp::SensorDataQoS(),
      std::bind(&ReceptionistTaskNode::onPersonLabel, this, std::placeholders::_1),
      sub_opts);

    drink_sub_ = create_subscription<std_msgs::msg::String>(
      "/zed/drink_detections", rclcpp::SensorDataQoS(),
      std::bind(&ReceptionistTaskNode::onDrinkDetections, this, std::placeholders::_1),
      sub_opts);

    // Fallback ke /zed/bag_detections (tidak menghapus latest_drinks_ jika kosong)
    object_sub_ = create_subscription<std_msgs::msg::String>(
      "/zed/bag_detections", rclcpp::SensorDataQoS(),
      std::bind(&ReceptionistTaskNode::onBagDetectionsFallback, this, std::placeholders::_1),
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

    search_chair_client_ = create_client<Trigger>(
      "/search_chair", rclcpp::ServicesQoS(), callback_group_);

    arm_ambil_tas_client_ = create_client<Trigger>(
      "/arm/ambil_tas", rclcpp::ServicesQoS(), callback_group_);
    arm_ambil_tas_simple_client_ = create_client<Trigger>(
      "/arm/ambil_tas_simple", rclcpp::ServicesQoS(), callback_group_);

    arm_point_position_client_ = create_client<Trigger>(
      "/arm/point_position", rclcpp::ServicesQoS(), callback_group_);

    // Timer siklus FSM (10 Hz / 100 ms)
    fsm_timer_ = create_wall_timer(
      100ms, std::bind(&ReceptionistTaskNode::fsmTick, this));

    task_start_time_ = now();
    state_enter_time_ = now();
    last_wait_prompt_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);

    guests_.clear();
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
  static constexpr double GLOBAL_TIMEOUT_SEC        = 600.0;
  static constexpr double DIALOG_REPROMPT_SEC       = 12.0;
  static constexpr double WAIT_GUEST_PROMPT_SEC     = 8.0;
  static constexpr double VISION_INITIAL_SETTLE_SEC = 1.2;
  static constexpr double MIN_SIT_WAIT_SEC          = 4.0;
  static constexpr double PERSON_FRESHNESS_SEC      = 2.0;
  static constexpr double TTS_MIN_TIMEOUT_SEC       = 4.0;
  static constexpr double TTS_SEC_PER_WORD          = 0.5;
  static constexpr double TTS_TIMEOUT_MARGIN_SEC    = 2.0;
  static constexpr double NAV_RETRY_COOLDOWN_SEC    = 2.0;

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
  int camera_width_{672};

  // Data Persepsi Vision
  std::mutex vision_mutex_;
  std::vector<DetectedObject> latest_drinks_;
  rclcpp::Time last_drink_msg_time_{0, 0, RCL_ROS_TIME};
  std::vector<DetectedObject> table_drinks_snapshot_;
  bool drinks_detected_on_arrival_{false};
  std::vector<std::string> latest_attributes_{
    "male", "casual attire", "dark hair"
  };
  std::string empty_seat_position_{"center"};
  bool has_explicit_empty_seat_{false};
  int nearest_person_bbox_x_{0};
  int nearest_person_bbox_w_{0};
  int nearest_person_bbox_y_{0};

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
  double tts_timeout_sec_{TTS_MIN_TIMEOUT_SEC};
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

  TriggerClient::SharedPtr arm_ambil_tas_client_;
  TriggerClient::SharedPtr arm_ambil_tas_simple_client_;
  TriggerClient::SharedPtr arm_point_position_client_;

  // ============================================================================
  // SUB-ROUTINES / HELPER FUNCTIONS
  // ============================================================================
  bool isBusy() const
  {
    return speaking_.load() || service_busy_.load() || in_callback_.load();
  }

  // FIX: Vision dimatikan selama SEMUA state navigasi agar tidak menangkap
  // objek/orang di tengah perjalanan.
  bool isVisionActive() const
  {
    const ReceptionistState current = state_.load();
    return current != ReceptionistState::INIT &&
           current != ReceptionistState::NAV_TO_DOOR &&
           current != ReceptionistState::NAV_TO_BEVERAGE &&
           current != ReceptionistState::NAV_TO_LIVING_ROOM &&
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
    RCLCPP_INFO(get_logger(), "\033[1;32m[TASK STATUS]\033[0m %s", status_text.c_str());
  }

  void resetGuestProfileFields(PersonProfile & profile, int idx)
  {
    profile = PersonProfile{};
    profile.id = "guest_" + std::to_string(idx);
    profile.gender = "unknown";
    profile.shirt_color = "casual attire";
    profile.hair_color = "dark hair";
  }

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
      latest_drinks_.clear();
      has_explicit_empty_seat_ = false;
      nearest_person_bbox_w_ = 0;
    }
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
        latest_drinks_.clear();
      }
      if (next_state == ReceptionistState::OFFER_SEAT) {
        has_explicit_empty_seat_ = false;
      }
    }

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
    } else {
      client = nav_living_room_client_;
      srv_name = "/receptionist/go_to_living_room";
    }

    RCLCPP_INFO(get_logger(), "\033[1;34m[NAV2]\033[0m Navigating to '%s' via %s...",
      waypoint_name.c_str(), srv_name.c_str());

    const rclcpp::Time nav_start = now();

    callTrigger(
      client, srv_name,
      [this, waypoint_name, srv_name, nav_start, on_finished](bool success, const std::string & msg) {
        const double elapsed = (now() - nav_start).seconds();
        if (!success) {
          RCLCPP_ERROR(
            get_logger(),
            "\033[1;31m[NAV2]\033[0m Navigation to '%s' FAILED (%s). Will retry after cooldown.",
            waypoint_name.c_str(), msg.c_str());
          last_nav_fail_time_ = now();
        } else {
          if (elapsed < 0.5) {
            RCLCPP_WARN(
              get_logger(),
              "\033[1;33m[NAV2 WARNING]\033[0m Service %s returned TRUE in %.2fs (< 0.5s)! "
              "Pastikan receptionist_node menunggu NavigateToPose selesai sebelum me-return response.",
              srv_name.c_str(), elapsed);
          }
          RCLCPP_INFO(
            get_logger(),
            "\033[1;34m[NAV2]\033[0m Reached '%s' in %.1fs: %s",
            waypoint_name.c_str(), elapsed, msg.c_str());
        }
        if (on_finished) {
          on_finished(success);
        }
      },
      120.0);
  }

  // Menunjuk kursi kosong menggunakan service /arm/ambil_tas (fallback ke /arm/ambil_tas_simple)
  void pointToSeatWithArm(std::function<void(bool)> on_finished)
  {
    TriggerClient::SharedPtr client = arm_point_position_client_;
    std::string srv_name = "/arm/point_position";

    RCLCPP_INFO(get_logger(), "\033[1;36m[ARM]\033[0m Actuating arm to point to chair via %s...", srv_name.c_str());

    if ((!client || !client->service_is_ready()) &&
        arm_ambil_tas_simple_client_ && arm_ambil_tas_simple_client_->service_is_ready())
    {
      client = arm_ambil_tas_simple_client_;
      srv_name = "/arm/ambil_tas_simple";
    }

    // RCLCPP_INFO(
    //   get_logger(),
    //   "\033[1;36m[ARM]\033[0m Actuating servo to indicate empty chair via %s...",
    //   srv_name.c_str());

    callTrigger(
      client, srv_name,
      [this, srv_name, on_finished](bool success, const std::string & msg) {
        if (success) {
          RCLCPP_INFO(
            get_logger(),
            "\033[1;36m[ARM]\033[0m Servo actuation (%s) succeeded: %s",
            srv_name.c_str(), msg.c_str());
        } else {
          RCLCPP_WARN(
            get_logger(),
            "\033[1;33m[ARM]\033[0m Servo actuation (%s) finished with warning: %s",
            srv_name.c_str(), msg.c_str());
        }
        if (on_finished) {
          on_finished(success);
        }
      },
      15.0);
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

  // Menyimpan snapshot minuman di meja saat tiba di beverage area
  size_t snapshotBeveragesOnTable()
  {
    std::lock_guard<std::mutex> lock(vision_mutex_);
    table_drinks_snapshot_ = latest_drinks_;
    drinks_detected_on_arrival_ = !table_drinks_snapshot_.empty();
    return table_drinks_snapshot_.size();
  }

  // Menentukan sisi (left/center/right) dari minuman favorit berdasarkan deteksi vision
  std::string locateBeverageSide(const std::string & favorite_drink, bool & found_match)
  {
    std::lock_guard<std::mutex> lock(vision_mutex_);
    found_match = false;
    const std::string target = normalize(favorite_drink);

    const auto & drinks_list = !table_drinks_snapshot_.empty()
      ? table_drinks_snapshot_
      : latest_drinks_;

    for (const auto & drink : drinks_list) {
      const std::string d_name = normalize(drink.name);
      if (!d_name.empty() &&
          (d_name == target || d_name.find(target) != std::string::npos ||
           target.find(d_name) != std::string::npos))
      {
        found_match = true;
        const std::string side = calculateHorizontalPosition(drink.bbox_x, drink.bbox_w);
        RCLCPP_INFO(
          get_logger(),
          "\033[1;32m[BEVERAGE]\033[0m Cocok item '%s' di sisi %s (3D: x=%.2f m, y=%.2f m)",
          drink.name.c_str(), side.c_str(), drink.x, drink.y);
        return side;
      }
    }

    if (!drinks_list.empty()) {
      found_match = true;
      const auto & first = drinks_list.front();
      const std::string side = calculateHorizontalPosition(first.bbox_x, first.bbox_w);
      RCLCPP_INFO(
        get_logger(),
        "\033[1;33m[BEVERAGE]\033[0m Fallback item terdekat '%s' di sisi %s (3D: x=%.2f m, y=%.2f m)",
        first.name.c_str(), side.c_str(), first.x, first.y);
      return side;
    }

    return "center";
  }

  // Menentukan lokasi kursi kosong berdasarkan deteksi vision (/zed/empty_seat)
  std::string detectEmptyChairSide()
  {
    std::lock_guard<std::mutex> lock(vision_mutex_);
    if (has_explicit_empty_seat_ && !empty_seat_position_.empty()) {
      return empty_seat_position_;
    }

    // Jika ada orang terdeteksi di satu sisi frame, arahkan ke sisi kursi yang kosong
    if (nearest_person_bbox_w_ > 0) {
      const std::string occupied_side = calculateHorizontalPosition(
        nearest_person_bbox_x_, nearest_person_bbox_w_);
      if (occupied_side == "left") {
        return "right";
      } else if (occupied_side == "right") {
        return "left";
      } else {
        // Jika orang duduk di tengah, arahkan ke kanan agar tidak menunjuk orang tersebut
        return "right";
      }
    }

    return "center";
  }

  // Mengekstrak atribut visual dari ZED (/zed/person_attributes) secara thread-safe
  void updateCurrentGuestVisualAttributes(int person_id)
  {
    std::vector<std::string> raw_attrs;
    {
      std::lock_guard<std::mutex> v_lock(vision_mutex_);
      raw_attrs = latest_attributes_;
    }

    RCLCPP_INFO(
      get_logger(),
      "\033[1;32m[VISION]\033[0m Extracting visual attributes for person_id=%d",
      person_id);

    std::lock_guard<std::mutex> g_lock(guest_mutex_);
    std::vector<std::string> attrs;
    for (const auto & a : raw_attrs) {
      if (std::find(attrs.begin(), attrs.end(), a) == attrs.end()) {
        attrs.push_back(a);
      }

      std::string lower_a = a;
      std::transform(lower_a.begin(), lower_a.end(), lower_a.begin(), ::tolower);
      if (lower_a == "male" || lower_a == "female") {
        current_guest_.gender = lower_a;
      } else if (lower_a.find("shirt") != std::string::npos ||
                 lower_a.find("clothes") != std::string::npos ||
                 lower_a.find("attire") != std::string::npos)
      {
        current_guest_.shirt_color = a;
      } else if (lower_a.find("hair") != std::string::npos) {
        current_guest_.hair_color = a;
      }
    }
    if (attrs.empty()) {
      attrs.push_back("casual attire");
    }
    current_guest_.attributes = attrs;
  }

  static std::string formatPersonDescription(const PersonProfile & person)
  {
    std::string gender = person.gender.empty() ? "person" : person.gender;
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
           (now() - last_person_seen_time_).seconds() < PERSON_FRESHNESS_SEC;
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

  // Pencocokan frase utuh (word-boundary) agar tidak salah tangkap substring
  static bool containsWholePhrase(const std::string & norm_haystack, const std::string & phrase)
  {
    const std::string padded_h = " " + norm_haystack + " ";
    const std::string padded_p = " " + phrase + " ";
    return padded_h.find(padded_p) != std::string::npos;
  }

  // Ekstraksi slot dari kalimat STT ("person_name", "interest", "drink")
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
        "adel", "angel", "axel", "charlie", "jane", "john", "jules", "morgan",
        "paris", "robin", "simone", "alex", "michael", "sarah", "david", "emma"
      };
      for (const auto & name : kKnownNames) {
        if (containsWholePhrase(norm, name)) {
          return capitalizeWords(name);
        }
      }
      return stripPrefixes({
        "my name is ", "i am ", "im ", "call me ", "this is ", "name is "
      });
    }

    if (slot == "drink") {
      // Frase spesifik (2 kata) diperiksa lebih dulu sebelum kata tunggal ("tea", "juice")
      static const std::vector<std::string> kKnownDrinks = {
        "orange juice", "apple juice", "iced tea", "green tea",
        "coke", "cola", "pepsi", "sprite", "fanta",
        "milk", "water", "coffee", "tea", "juice", "soda",
        "botol", "bottle", "cracker", "cracker box", "snack",
        "pringles", "cheezit", "cheez-it", "chips"
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
        "football", "basketball", "coding", "programming", "dancing", "painting"
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
      "no", "nope", "nah", "not correct", "wrong", "false", "incorrect", "negative",
      "not really", "it is not", "it isnt", "bukan", "tidak", "enggak", "ga"
    };
    static const std::vector<std::string> kYesPhrases = {
      "yes", "yeah", "yep", "correct", "right", "sure", "true", "exactly", "of course",
      "that is right", "it is", "ya", "benar", "betul"
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
      "\033[1;33m[STT HEARD]\033[0m \"%s\" | State: \033[1;36m%s\033[0m",
      message->data.c_str(),
      stateToString(state_.load()).c_str());
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

  // Callback atribut visual tamu (/zed/person_attributes)
  void onPersonAttributes(const std_msgs::msg::String::SharedPtr message)
  {
    if (!isVisionActive() || !message || message->data.empty()) return;
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
    }
  }

  // Callback deteksi kursi kosong (/zed/empty_seat: "left", "center", "right")
  void onEmptySeat(const std_msgs::msg::String::SharedPtr message)
  {
    if (!isVisionActive() || !message || message->data.empty()) return;
    std::lock_guard<std::mutex> lock(vision_mutex_);
    const std::string raw = normalize(message->data);
    if (raw == "none" || raw.empty()) {
      has_explicit_empty_seat_ = false;
      return;
    }
    const auto pipe = raw.find('|');
    empty_seat_position_ = (pipe != std::string::npos) ? raw.substr(0, pipe) : raw;
    has_explicit_empty_seat_ = true;
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "\033[1;32m[EMPTY_SEAT]\033[0m Kursi kosong terdeteksi di sisi: %s",
      empty_seat_position_.c_str());
  }

  // ============================================================================
  // HELPER SERVICE CLIENT NON-BLOCKING (TANPA DETACHED THREAD / MEMORY SAFE)
  // ============================================================================
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
  // MAIN FINITE STATE MACHINE (FSM TICK)
  // ============================================================================
  void fsmTick()
  {
    const ReceptionistState current_state = state_.load();

    // 1. Global timeout watchdog
    if ((now() - task_start_time_).seconds() > GLOBAL_TIMEOUT_SEC &&
        current_state != ReceptionistState::COMPLETED &&
        current_state != ReceptionistState::ABORTED)
    {
      RCLCPP_ERROR(get_logger(), "Global receptionist task timeout reached.");
      state_ = ReceptionistState::ABORTED;
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
        RCLCPP_INFO(get_logger(), "=== STARTING RECEPTIONIST TASK ===");
        guests_.clear();
        initCurrentGuest(1);
        transitionTo(ReceptionistState::NAV_TO_DOOR, "Start cycle for Guest 1: navigating to door");
        break;

      // ------------------------------------------------------------------------
      // STATE 1: NAVIGASI KE PINTU & DETEKSI TAMU
      // ------------------------------------------------------------------------
      case ReceptionistState::NAV_TO_DOOR:
        if (last_nav_fail_time_.nanoseconds() > 0 &&
            (now() - last_nav_fail_time_).seconds() < NAV_RETRY_COOLDOWN_SEC)
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
        if ((now() - state_enter_time_).seconds() < VISION_INITIAL_SETTLE_SEC) {
          return;
        }
        if (isPersonInFront()) {
          speakAsync(
            "Hello! Welcome to the party. I am Barelang Inari from Batam State Polytechnic; I am your robot receptionist.",
            [this]() {
              transitionTo(ReceptionistState::ASK_NAME, "Guest detected and greeted at entrance");
            });
        } else if (last_wait_prompt_time_.nanoseconds() == 0 ||
                   (now() - last_wait_prompt_time_).seconds() >= WAIT_GUEST_PROMPT_SEC)
        {
          last_wait_prompt_time_ = now();
          speakAsync("I am waiting for a guest; please stand in front of me.");
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
            {
              std::lock_guard<std::mutex> lock(guest_mutex_);
              current_guest_.name = parsed_name;
            }
            if (person_name_pub_) {
              std_msgs::msg::String m;
              m.data = parsed_name;
              person_name_pub_->publish(m);
            }
            RCLCPP_INFO(
              get_logger(),
              "\033[1;32m[DIALOG NAME]\033[0m Guest %d Name -> \033[1;32m\"%s\"\033[0m (from \"%s\")",
              guest_idx_, parsed_name.c_str(), raw.c_str());
            updateCurrentGuestVisualAttributes(nearest_person_id_.load());
            transitionTo(ReceptionistState::ASK_INTEREST, "Guest name captured and stored");
          } else {
            RCLCPP_WARN(
              get_logger(),
              "\033[1;33m[DIALOG NAME]\033[0m Unrecognized input \"%s\", asking again.",
              raw.c_str());
            state_enter_time_ = now();
            speakAsync("I am sorry, I did not catch your name. May I please have your name?");
          }
        } else if ((now() - state_enter_time_).seconds() > DIALOG_REPROMPT_SEC) {
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
          ". What is your main interest or hobby?",
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
            speakAsync("I am sorry, I did not catch that. What is your main interest or hobby?");
          }
        } else if ((now() - state_enter_time_).seconds() > DIALOG_REPROMPT_SEC) {
          state_enter_time_ = now();
          speakAsync("Could you please tell me your main interest or hobby?");
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
          g_interest + ". Is that correct? Please say yes if it is, or no if it is not.";

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
            {
              std::lock_guard<std::mutex> lock(guest_mutex_);
              resetGuestProfileFields(current_guest_, guest_idx_);
            }
            if (person_name_pub_) {
              std_msgs::msg::String m;
              m.data = "";
              person_name_pub_->publish(m);
            }
            speakAsync("I am sorry. Let us start over. May I please have your name?", [this]() {
              transitionTo(ReceptionistState::WAIT_NAME, "Restarted name and interest capture");
            });
          } else {
            RCLCPP_WARN(
              get_logger(),
              "\033[1;33m[DIALOG CONFIRM]\033[0m Heard: \"%s\" | Decision: \033[1;33mUNRECOGNIZED\033[0m (reprompting yes/no)",
              raw.c_str());
            state_enter_time_ = now();
            speakAsync("Please say yes if it is, or no if it is not.");
          }
        } else if ((now() - state_enter_time_).seconds() > DIALOG_REPROMPT_SEC) {
          RCLCPP_INFO(
            get_logger(),
            "\033[1;33m[DIALOG CONFIRM]\033[0m Confirmation timeout (%.1fs without speech), reprompting.",
            DIALOG_REPROMPT_SEC);
          state_enter_time_ = now();
          speakAsync("Please say yes if it is, or no if it is not.");
        }
        break;

      // ------------------------------------------------------------------------
      // STATE 2: BEVERAGE AREA & DRINK CHECK
      // ------------------------------------------------------------------------
      case ReceptionistState::NAV_TO_BEVERAGE:
        if (last_nav_fail_time_.nanoseconds() > 0 &&
            (now() - last_nav_fail_time_).seconds() < NAV_RETRY_COOLDOWN_SEC)
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
        // FIX: Tunggu kamera ZED settle setelah robot berhenti di meja minuman
        if ((now() - state_enter_time_).seconds() < VISION_INITIAL_SETTLE_SEC) {
          return;
        }

        const size_t drink_count = snapshotBeveragesOnTable();
        std::string prompt;
        if (!drinks_detected_on_arrival_ || drink_count == 0) {
          prompt = "There are no drinks on this table; please tell me, what is your favorite drink?";
        } else {
          prompt = "There are " + std::to_string(drink_count) +
                   " drinks on the table. What is your favorite drink?";
        }

        speakAsync(prompt, [this]() {
          transitionTo(ReceptionistState::WAIT_DRINK, "Asked for favorite drink after table check");
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
            updateCurrentGuestVisualAttributes(nearest_person_id_.load());
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

            transitionTo(ReceptionistState::CHECK_BEVERAGE, "Favorite drink captured and stored");
          } else {
            RCLCPP_WARN(
              get_logger(),
              "\033[1;33m[DIALOG DRINK]\033[0m Unrecognized input \"%s\", asking again.",
              raw.c_str());
            state_enter_time_ = now();
            speakAsync("I am sorry, I did not catch that. What is your favorite drink?");
          }
        } else if ((now() - state_enter_time_).seconds() > DIALOG_REPROMPT_SEC) {
          state_enter_time_ = now();
          speakAsync("What is your favorite drink?");
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
            (now() - last_nav_fail_time_).seconds() < NAV_RETRY_COOLDOWN_SEC)
        {
          return;
        }

        std::string escort_msg =
          "We are moving to the living room; please follow me slowly from behind.";

        {
          std::lock_guard<std::mutex> lock(guest_mutex_);
          if (guest_idx_ == 1) {
            escort_msg +=
              " The host, " + host_.name + ", is in the living room. " +
              formatPersonDescription(host_) + ". Their favorite drink is " +
              host_.drink + ", and their interest is " + host_.interest + ".";
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
          if (guest_idx_ == 1) {
            transitionTo(ReceptionistState::SEARCH_CHAIR, "Arrived at living_room with Guest 1; searching for empty chair");
          } else {
            transitionTo(
              ReceptionistState::INTRODUCE_G1_TO_G2, "Arrived at living_room with Guest 2, starting introductions"
            );
          }
        });
        break;

      }

      // ------------------------------------------------------------------------
      // STATE 4: INTRODUCTIONS & SHARED INTERESTS (SAAT BERSAMA TAMU 2)
      // ------------------------------------------------------------------------
      case ReceptionistState::INTRODUCE_G1_TO_G2: {
        PersonProfile guest1, guest2;
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
          guest1 = guests_[0];
          guest2 = guests_[1];
        }

        // Pastikan kamera sempat settle atau mendeteksi orang di ruang tamu
        if (!isPersonInFront() && (now() - state_enter_time_).seconds() < 2.0) {
          return;
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
            transitionTo(ReceptionistState::SEARCH_CHAIR, "Shared interest announced; searching for empty chair");
          });
        } else {
          transitionTo(ReceptionistState::SEARCH_CHAIR, "No identical interest found; searching for empty chair");
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
        if ((now() - state_enter_time_).seconds() < VISION_INITIAL_SETTLE_SEC) {
          return;
        }

        const std::string seat_pos = detectEmptyChairSide();
        PersonProfile cg;
        {
          std::lock_guard<std::mutex> lock(guest_mutex_);
          cg = current_guest_;
        }

        // menentukan bagian 1 kalimat (berhenti tepat pada kalimat 'on the ')
        std::string part1_msg;
        if (guest_idx_ == 1) {
          part1_msg =
            host_.name + ", this is our guest, " + cg.name + ". " +
            formatPersonDescription(cg) + ". Their favorite drink is " +
            cg.drink + ", and their interest is " + cg.interest + ". " +
            cg.name + ", please take a seat in the empty chair on the ";
        } else {
          part1_msg = cg.name + ", please take a seat in the empty chair on the ";
        }

        speakAsync(part1_msg, [this, seat_pos]() {
          // tepat setelah kalimat diatas selesai, gerakkan lengan menunjuk
          pointToSeatWithArm([this, seat_pos](bool /*success*/) {
            // setelah lengan menunjuk, lanjutkan menyebutkan sisi kursi
            std::string part2_msg = seat_pos + " side that I am indicating.";
            speakAsync(part2_msg, [this]() {
              transitionTo(
                ReceptionistState::WAIT_GUEST_SIT,
                "Waiting for guest to occupy the indicated chair");
            });
          });
        });
        break;
      }

      case ReceptionistState::WAIT_GUEST_SIT: {
        // FIX: Beri waktu minimal (MIN_SIT_WAIT_SEC = 4.0s) agar tamu berjalan ke kursi,
        // sehingga Host yang sudah duduk tidak langsung memicu state ini selesai dalam 100ms.
        const double wait_elapsed = (now() - state_enter_time_).seconds();
        if (wait_elapsed < MIN_SIT_WAIT_SEC) {
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

          if (guest_idx_ == 1) {
            initCurrentGuest(2);
            transitionTo(
              ReceptionistState::NAV_TO_DOOR,
              "Guest 1 seated confirmed; returning to door for Guest 2");
          } else {
            transitionTo(
              ReceptionistState::FINISH_TASK,
              "Guest 2 seated confirmed; completing task");
          }
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
            RCLCPP_INFO(get_logger(), "=== RECEPTIONIST TASK FINISHED SUCCESSFULLY ===");
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
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ReceptionistTaskNode>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}