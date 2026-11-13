// ==============================================================================
// Barelang Inari - RoboCup@Home
// Package  : help_me_carry
// Node     : help_me_carry_manager (HelpMeCarryOrchestrator)
// Deskripsi: State Machine (FSM) utama untuk tugas Help Me Carry.
//            Mengatur interaksi HRI (TTS/STT), deteksi operator (ZED camera),
//            following person via Nav2, deteksi gesture tas, manipulasi lengan,
//            dan autonomous homing.
// edit : 2 nov 2026 22.02
// ==============================================================================

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"

using namespace std::chrono_literals;

// ==============================================================================
// 1. DEFINISI STATE FINITE STATE MACHINE (FSM)
// ==============================================================================
enum class TaskState {
  INIT,                     // Inisialisasi node & menunggu TTS engine siap
  WAIT_WAKEWORD,            // Menunggu perintah sapaan/wakeword ("hello robot")
  CHECK_OPERATOR_PRESENCE,  // Memastikan operator berdiri tepat di depan kamera
  WAIT_START_COMMAND,       // Menunggu perintah suara "follow me"
  FOLLOW_OPERATOR,          // Mengikuti operator berjalan ke lokasi tujuan
  RECOVER_OPERATOR,         // Mode pemulihan jika operator hilang dari pandangan
  ARRIVED,                  // Robot tiba di tujuan; periksa operator & tas atau mundur
  WAIT_BAG_POINTING,        // Menunggu operator menunjuk tas yang ingin dibawa
  GRASP_BAG,                // Memanggil service lengan robot untuk mengambil tas
  RETURN_TO_START,          // Autonomous homing kembali ke koordinat awal
  PUT_DOWN_BAG,             // Memanggil service lengan robot untuk meletakkan tas
  COMPLETED,                // Tugas selesai seluruhnya
  ABORTED                   // Tugas dibatalkan karena timeout / error kritis
};

static std::string stateToString(TaskState state)
{
  switch (state) {
    case TaskState::INIT:                    return "INIT";
    case TaskState::WAIT_WAKEWORD:           return "WAIT_WAKEWORD";
    case TaskState::CHECK_OPERATOR_PRESENCE: return "CHECK_OPERATOR_PRESENCE";
    case TaskState::WAIT_START_COMMAND:      return "WAIT_START_COMMAND";
    case TaskState::FOLLOW_OPERATOR:         return "FOLLOW_OPERATOR";
    case TaskState::RECOVER_OPERATOR:        return "RECOVER_OPERATOR";
    case TaskState::ARRIVED:                 return "ARRIVED";
    case TaskState::WAIT_BAG_POINTING:       return "WAIT_BAG_POINTING";
    case TaskState::GRASP_BAG:               return "GRASP_BAG";
    case TaskState::RETURN_TO_START:         return "RETURN_TO_START";
    case TaskState::PUT_DOWN_BAG:            return "PUT_DOWN_BAG";
    case TaskState::COMPLETED:               return "COMPLETED";
    case TaskState::ABORTED:                 return "ABORTED";
    default:                                 return "UNKNOWN";
  }
}

// ==============================================================================
// 2. KELAS UTAMA: HelpMeCarryOrchestrator
// ==============================================================================
class HelpMeCarryOrchestrator : public rclcpp::Node {
public:
  using Trigger = std_srvs::srv::Trigger;
  using TriggerClient = rclcpp::Client<Trigger>;
  using TriggerResult = std::function<void(bool, const std::string &)>;

  HelpMeCarryOrchestrator()
  : Node("help_me_carry_orchestrator")
  {
    // Callback group eksklusif untuk mencegah race condition antar callback
    callback_group_ = create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions options;
    options.callback_group = callback_group_;

    // --------------------------------------------------------------------------
    // Publisher & Subscriber HRI / Vision
    // --------------------------------------------------------------------------
    tts_pub_ = create_publisher<std_msgs::msg::String>("/hri/tts_say", 10);

    tts_done_sub_ = create_subscription<std_msgs::msg::String>(
      "/hri/tts_done", 10,
      std::bind(&HelpMeCarryOrchestrator::onTtsDone, this, std::placeholders::_1),
      options);

    stt_sub_ = create_subscription<std_msgs::msg::String>(
      "/hri/stt_transcript", 10,
      std::bind(&HelpMeCarryOrchestrator::onTranscript, this, std::placeholders::_1),
      options);

    emergency_sub_ = create_subscription<std_msgs::msg::String>(
      "/stt/keyword_interrupt", 10,
      std::bind(&HelpMeCarryOrchestrator::onTranscript, this, std::placeholders::_1),
      options);

    person_sub_ = create_subscription<std_msgs::msg::String>(
      "/zed/person_label", rclcpp::SensorDataQoS(),
      std::bind(&HelpMeCarryOrchestrator::onPersonLabel, this, std::placeholders::_1),
      options);

    gesture_sub_ = create_subscription<std_msgs::msg::String>(
      "/zed/gesture", 10,
      std::bind(&HelpMeCarryOrchestrator::onGesture, this, std::placeholders::_1),
      options);

    follow_status_sub_ = create_subscription<std_msgs::msg::String>(
      "/follow/status", 10,
      std::bind(&HelpMeCarryOrchestrator::onFollowStatus, this, std::placeholders::_1),
      options);z

    bag_sub_ = create_subscription<std_msgs::msg::String>(
      "/zed/bag_detections", rclcpp::SensorDataQoS(),
      std::bind(&HelpMeCarryOrchestrator::onBagDetections, this, std::placeholders::_1),
      options);

    // --------------------------------------------------------------------------
    // Service Clients (Following, Vision Alignment, Arm, & Navigation)
    // --------------------------------------------------------------------------
    follow_start_client_ = create_client<Trigger>(
      "/follow/person", rclcpp::ServicesQoS(), callback_group_);
    follow_stop_client_ = create_client<Trigger>(
      "/follow/stop", rclcpp::ServicesQoS(), callback_group_);
    step_back_client_ = create_client<Trigger>(
      "/task/step_back", rclcpp::ServicesQoS(), callback_group_);
    align_bag_client_ = create_client<Trigger>(
      "/task/align_bag", rclcpp::ServicesQoS(), callback_group_);
    grab_bag_client_ = create_client<Trigger>(
      "/arm/ambil_tas_simple", rclcpp::ServicesQoS(), callback_group_);
    put_down_bag_client_ = create_client<Trigger>(
      "/arm/letakkan_tas", rclcpp::ServicesQoS(), callback_group_);
    homing_client_ = create_client<Trigger>(
      "/homing", rclcpp::ServicesQoS(), callback_group_);

    // Timer siklus FSM (10 Hz / 100 ms)
    fsm_timer_ = create_wall_timer(
      100ms, std::bind(&HelpMeCarryOrchestrator::fsmTick, this), callback_group_);

    task_start_time_ = now();
    state_enter_time_ = now();

    RCLCPP_INFO(get_logger(),
      "\n============================================================\n"
      "  \033[1;32m[SYSTEM]\033[0m Help Me Carry Orchestrator Started\n"
      "  \033[1;36m[STATE]\033[0m  INIT\n"
      "  \033[1;33m[MIC]\033[0m    LISTENING\n"
      "============================================================");
  }

private:
  // ----------------------------------------------------------------------------
  // Konstanta Timeout & Grace Period
  // STATUS FSM
  // ----------------------------------------------------------------------------
  static constexpr double GLOBAL_TIMEOUT_SEC       = 300.0;  // Batas waktu total tugas (5 menit)
  static constexpr double OPERATOR_TIMEOUT_SEC     = 5.0;    // Toleransi oklusi rintangan sebelum masuk recovery
  static constexpr double FOLLOW_GRACE_PERIOD_SEC  = 2.0;    // Masa tenggang deteksi hilang saat awal follow
  static constexpr double ARRIVED_GRACE_PERIOD_SEC = 5.0;    // Jeda abaikan perintah arrived saat awal follow
  static constexpr double TTS_MIN_TIMEOUT_SEC      = 4.0;    // Batas minimal watchdog TTS
  static constexpr double TTS_SEC_PER_WORD         = 0.5;    // Estimasi durasi ucapan per kata
  static constexpr double TTS_TIMEOUT_MARGIN_SEC   = 2.0;    // Margin sintesis + latensi playback
  static constexpr double FOLLOW_PROMPT_INTERVAL_SEC = 3.0;
  // ----------------------------------------------------------------------------
  // Status Variabel Atomic & Waktu
  // ----------------------------------------------------------------------------
  std::atomic<TaskState> state_{TaskState::INIT};
  std::atomic<bool> busy_{false};
  std::atomic<bool> speaking_{false};
  std::atomic<bool> wakeword_requested_{false};
  std::atomic<bool> follow_requested_{false};
  std::atomic<bool> arrived_requested_{false};
  std::atomic<bool> emergency_stop_requested_{false};
  std::atomic<bool> operator_seen_{false};
  std::atomic<bool> bag_seen_{false};
  std::atomic<bool> tracker_lost_{false};

  rclcpp::Time task_start_time_;
  rclcpp::Time state_enter_time_;
  rclcpp::Time last_operator_seen_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_bag_seen_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_operator_warn_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time tts_cooldown_until_{0, 0, RCL_ROS_TIME};
  rclcpp::Time tts_started_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_command_time_{0, 0, RCL_ROS_TIME};

  std::mutex data_mutex_;
  std::mutex tts_mutex_;              // Proteksi state TTS (ditulis dari thread callback service)
  double tts_timeout_sec_{TTS_MIN_TIMEOUT_SEC};
  std::string last_command_;
  std::function<void()> pending_tts_done_;
  bool bag_prompt_sent_{false};
  bool recovery_prompt_sent_{false};
  std::string detected_gesture_{"NONE"};
  bool gesture_confirmed_{false};
  int homing_retry_count_{0};

  int follow_prompt_count_{0};
  rclcpp::Time follow_prompt_next_time_{0, 0, RCL_ROS_TIME};

  // ----------------------------------------------------------------------------
  // Node Interfaces (Pub, Sub, Clients)
  // ----------------------------------------------------------------------------
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp::TimerBase::SharedPtr fsm_timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr tts_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr tts_done_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr stt_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr emergency_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr person_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr gesture_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr follow_status_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr bag_sub_;

  TriggerClient::SharedPtr follow_start_client_;
  TriggerClient::SharedPtr follow_stop_client_;
  TriggerClient::SharedPtr step_back_client_;
  TriggerClient::SharedPtr align_bag_client_;
  TriggerClient::SharedPtr grab_bag_client_;
  TriggerClient::SharedPtr put_down_bag_client_;
  TriggerClient::SharedPtr homing_client_;

  // ----------------------------------------------------------------------------
  // Helper: Deteksi Operator & Tas di Depan Robot
  // ----------------------------------------------------------------------------
  bool isOperatorInFront() const
  {
    return operator_seen_.load() && (now() - last_operator_seen_time_).seconds() < 2.5;
  }

  bool isBagInFront() const
  {
    return bag_seen_.load() && (now() - last_bag_seen_time_).seconds() < 2.5;
  }

  // ----------------------------------------------------------------------------
  // Helper: Transisi State FSM
  // ----------------------------------------------------------------------------
  void transitionTo(TaskState next_state, const std::string & reason)
  {
    RCLCPP_INFO(
      get_logger(),
      "\033[1;36m[FSM STATE]\033[0m %s \033[1;33m->\033[0m \033[1;32m%s\033[0m | Reason: %s",
      stateToString(state_.load()).c_str(),
      stateToString(next_state).c_str(),
      reason.c_str());

    state_enter_time_ = now();
    bag_prompt_sent_ = false;
    recovery_prompt_sent_ = false;

    // Reset flag penting saat berpindah ke state tertentu
    if (next_state == TaskState::FOLLOW_OPERATOR) {
      arrived_requested_ = false;
      tracker_lost_ = false;
      follow_prompt_count_ = 0;
      follow_prompt_next_time_ = now();
    }
    if (next_state == TaskState::CHECK_OPERATOR_PRESENCE) {
      last_operator_warn_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    }
    if (next_state == TaskState::WAIT_BAG_POINTING) {
      detected_gesture_ = "NONE";
      gesture_confirmed_ = false;
    }
    if (next_state == TaskState::RETURN_TO_START && state_.load() != TaskState::RETURN_TO_START) {
      homing_retry_count_ = 0;
    }

    state_ = next_state;
  }

  // ----------------------------------------------------------------------------
  // Helper: Text-to-Speech (TTS) Asinkron
  // ----------------------------------------------------------------------------
  // Estimasi batas waktu TTS berdasarkan jumlah kata (~0.5 s/kata + margin 3 s, minimal 6 s)
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
    return std::max(TTS_MIN_TIMEOUT_SEC, static_cast<double>(words) * TTS_SEC_PER_WORD + TTS_TIMEOUT_MARGIN_SEC);
  }

  // Mengucapkan teks; FSM terkunci (busy_) sampai /hri/tts_done diterima.
  // Aksi lanjutan WAJIB diletakkan di on_done agar dieksekusi setelah TTS selesai.
  void speakAsync(const std::string & text, std::function<void()> on_done = nullptr)
  {
    {
      std::lock_guard<std::mutex> lock(tts_mutex_);
      // Kunci FSM sebelum publish agar tidak ada celah race dengan fsmTick
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
    RCLCPP_INFO(get_logger(), "\033[1;35m[TTS SAY]\033[0m \"%s\" | Mic: \033[1;31m[MUTED]\033[0m", text.c_str());
  }

  void onTtsDone(const std_msgs::msg::String::SharedPtr = nullptr)
  {
    std::function<void()> callback;
    {
      std::lock_guard<std::mutex> lock(tts_mutex_);
      if (!speaking_.load()) {
        return;  // Abaikan tts_done ganda / yang tidak diminta orchestrator
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

  // Cek apakah mic aktif mendengarkan (tidak sedang bicara & jeda cooldown terlewati)
  bool canListen() const
  {
    return !speaking_.load() && now() >= tts_cooldown_until_;
  }

  // Normalisasi string: ubah ke lowercase, hilangkan tanda baca, dan rapikan spasi
  static std::string normalize(std::string text)
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

  // Helper memotong kalimat menjadi token kata
  static std::vector<std::string> splitWords(const std::string & text)
  {
    std::vector<std::string> words;
    std::istringstream iss(text);
    std::string w;
    while (iss >> w) {
      words.push_back(w);
    }
    return words;
  }

  // ============================================================================
  // 3. NLU (NATURAL LANGUAGE UNDERSTANDING) & INTENT MATCHING
  // ============================================================================
  // Menghitung kemiripan teks menggunakan Normalized Levenshtein Edit Distance [0.0 - 1.0]
  static double calculateSimilarity(const std::string & s1, const std::string & s2)
  {
    if (s1.empty() && s2.empty()) return 1.0;
    if (s1.empty() || s2.empty()) return 0.0;
    if (s1 == s2) return 1.0;

    const size_t m = s1.size();
    const size_t n = s2.size();
    std::vector<std::vector<size_t>> dp(m + 1, std::vector<size_t>(n + 1, 0));

    for (size_t i = 0; i <= m; ++i) dp[i][0] = i;
    for (size_t j = 0; j <= n; ++j) dp[0][j] = j;

    for (size_t i = 1; i <= m; ++i) {
      for (size_t j = 1; j <= n; ++j) {
        const size_t cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
        dp[i][j] = std::min({
          dp[i - 1][j] + 1,         // Deletion
          dp[i][j - 1] + 1,         // Insertion
          dp[i - 1][j - 1] + cost   // Substitution
        });
      }
    }

    const size_t dist = dp[m][n];
    const size_t max_len = std::max(m, n);
    return 1.0 - (static_cast<double>(dist) / static_cast<double>(max_len));
  }

  // Menghitung skor kecocokan string input terhadap daftar kandidat frasa (berbasis token kata)
  static double scoreIntent(const std::string & text, const std::vector<std::string> & candidates)
  {
    if (text.empty()) return 0.0;
    double best_score = 0.0;

    const auto text_words = splitWords(text);

    for (const auto & phrase : candidates) {
      if (text == phrase) {
        return 1.0;
      }

      const auto phrase_words = splitWords(phrase);
      const size_t p_len = phrase_words.size();
      const size_t t_len = text_words.size();

      if (p_len == 0) continue;

      if (t_len >= p_len) {
        for (size_t i = 0; i + p_len <= t_len; ++i) {
          std::string sub_phrase;
          for (size_t j = 0; j < p_len; ++j) {
            if (j > 0) sub_phrase += " ";
            sub_phrase += text_words[i + j];
          }
          if (sub_phrase == phrase) {
            best_score = std::max(best_score, 0.98);
          }
          best_score = std::max(best_score, calculateSimilarity(sub_phrase, phrase));
        }
      } else {
        best_score = std::max(best_score, calculateSimilarity(text, phrase));
      }
    }
    return best_score;
  }

  // Evaluasi kecocokan intent berdasarkan batas ambang (threshold)
  bool matchIntent(const std::string & text, const std::string & target_intent,
                   const std::vector<std::string> & candidates, double threshold = 0.75)
  {
    double score = scoreIntent(text, candidates);
    if (score >= threshold) {
      RCLCPP_INFO(
        get_logger(),
        "  \033[1;32m[NLU MATCH]\033[0m Intent: \033[1m%s\033[0m | Score: \033[1;32m%.2f\033[0m (thresh: %.2f)",
        target_intent.c_str(), score, threshold);
      return true;
    } else {
      RCLCPP_INFO(
        get_logger(),
        "  \033[2m[NLU CHECK]\033[0m Target: %s | Score: %.2f (< %.2f)",
        target_intent.c_str(), score, threshold);
      return false;
    }
  }

  // Daftar kamus frasa suara (Corpus) lengkap dengan variasi fonetik
  static const std::vector<std::string> & getWakewordCorpus()
  {
    static const std::vector<std::string> corpus = {
      "hello robot", "halo robot", "hey robot", "hello barelang", "hi robot",
      "robot", "hallo", "hai", "hello", "helo", "helo robot", "hai robot", "hey barelang", "halo barelang"
    };
    return corpus;
  }

  static const std::vector<std::string> & getFollowCorpus()
  {
    static const std::vector<std::string> corpus = {
      "follow me", "please follow me", "come with me", "lets go follow me",
      "start following me", "guide me follow me", "following me",
      "fallow me", "follow mi", "folow me", "come on follow me", "ayo follow me"
    };
    return corpus;
  }

  static const std::vector<std::string> & getArrivedCorpus()
  {
    static const std::vector<std::string> corpus = {
      "arrived at the location", "we have arrived", "this is the location", "this is the location",
      "reached the car", "arrived at the car", "stop here we arrived",
      "i have arrived", "we are here", "here is the car", "arrived",
    };
    return corpus;
  }

  static const std::vector<std::string> & getStopCorpus()
  {
    static const std::vector<std::string> corpus = {
      "stop", "stopping", "stop now", "stop please", "stop following",
      "emergency stop", "berhenti", "berhenti dulu"
    };
    return corpus;
  }

  // ============================================================================
  // 4. CALLBACK SUBSCRIPTION ROS 2
  // ============================================================================
  // Callback hasil transkripsi suara STT
  void onTranscript(const std_msgs::msg::String::SharedPtr message)
  {
    if (!canListen()) {
      RCLCPP_DEBUG(get_logger(), "[STT IGNORED] Mic MUTED: \"%s\"", message->data.c_str());
      return;
    }

    const std::string command = normalize(message->data);
    if (command.empty()) {
      return;
    }

    // Debounce perintah berulang dalam rentang 2 detik
    const auto current_time = now();
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (command == last_command_ && (current_time - last_command_time_).seconds() < 2.0) {
        return;
      }
      last_command_ = command;
      last_command_time_ = current_time;
    }

    const TaskState current_state = state_.load();
    RCLCPP_INFO(
      get_logger(),
      "\033[1;33m[STT HEARD]\033[0m \"%s\" | Mic: \033[1;32m[LISTENING]\033[0m | State: \033[1;36m%s\033[0m",
      message->data.c_str(),
      stateToString(current_state).c_str());

    // 1. Sapaan di state WAIT_WAKEWORD
    if (current_state == TaskState::WAIT_WAKEWORD &&
        matchIntent(command, "WAKEWORD", getWakewordCorpus()))
    {
      if (!operator_seen_.load()) {
        RCLCPP_WARN(get_logger(), "Wakeword terdeteksi, tapi operator belum terlihat di kamera.");
      }
      wakeword_requested_ = true;
      return;
    }

    // 2. Perintah Follow di state WAIT_START_COMMAND
    if (current_state == TaskState::WAIT_START_COMMAND &&
        matchIntent(command, "FOLLOW_ME", getFollowCorpus()))
    {
      follow_requested_ = true;
      return;
    }

    // 3. Perintah Arrived & Emergency Stop di state FOLLOW_OPERATOR
    if (current_state == TaskState::FOLLOW_OPERATOR) {
      if (matchIntent(command, "ARRIVED", getArrivedCorpus())) {
        // Abaikan perintah arrived jika masih dalam masa jeda awal follow
        if ((now() - state_enter_time_).seconds() < ARRIVED_GRACE_PERIOD_SEC) {
          RCLCPP_WARN(
            get_logger(),
            "[NLU IGNORED] Perintah ARRIVED diabaikan: masa tunggu follow aktif (< %.1f s).",
            ARRIVED_GRACE_PERIOD_SEC);
          return;
        }
        arrived_requested_ = true;
        return;
      }
      if (matchIntent(command, "STOP", getStopCorpus())) {
        emergency_stop_requested_ = true;
        return;
      }
    }
  }

  // Callback label deteksi orang dari kamera ZED
  void onPersonLabel(const std_msgs::msg::String::SharedPtr message)
  {
    if (message->data.find("NO_PERSON") != std::string::npos) {
      operator_seen_ = false;
    } else if (message->data.find("operator:") != std::string::npos ||
               (state_.load() != TaskState::FOLLOW_OPERATOR &&
                message->data.find("person:") != std::string::npos) ||
               (!tracker_lost_.load() && message->data.find("person:") != std::string::npos))
    {
      operator_seen_ = true;
      last_operator_seen_time_ = now();
    }
  }

  // Callback sinkronisasi status pelacakan dari help_me_carry_node
  void onFollowStatus(const std_msgs::msg::String::SharedPtr message)
  {
    if (message->data == "TRACKING" || message->data == "PREDICTING") {
      tracker_lost_ = false;
      operator_seen_ = true;
      last_operator_seen_time_ = now();
    } else if (message->data == "LOST") {
      tracker_lost_ = true;
      operator_seen_ = false;
    }
  }

  // Callback deteksi gesture penunjukan tas dari kamera ZED
  void onGesture(const std_msgs::msg::String::SharedPtr message)
  {
    if (state_.load() != TaskState::WAIT_BAG_POINTING || gesture_confirmed_) {
      return;
    }
    const std::string gest = message->data;
    if (gest.find("LEFT") != std::string::npos || gest.find("KIRI") != std::string::npos) {
      detected_gesture_ = "LEFT";
    } else if (gest.find("RIGHT") != std::string::npos || gest.find("KANAN") != std::string::npos) {
      detected_gesture_ = "RIGHT";
    }
  }

  // Callback deteksi tas dari kamera ZED YOLO (/zed/bag_detections)
  void onBagDetections(const std_msgs::msg::String::SharedPtr message)
  {
    if (!message || message->data.empty()) {
      return;
    }
    // Format YOLO ZED: "bag|x,y,z;bag|x,y,z;..."
    if (message->data.find(',') != std::string::npos) {
      bag_seen_ = true;
      last_bag_seen_time_ = now();
    }
  }

  // ============================================================================
  // 5. HELPER SERVICE CLIENT
  // ============================================================================
  // Memanggil service ROS 2 bertipe Trigger secara non-blocking di thread terpisah
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
          get_logger(), "Service %s belum siap setelah timeout %ld detik.",
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
          } catch (const std::exception & exception) {
            result(false, exception.what());
          }
        });
    }).detach();
  }

  // Menghentikan tracking/following person
  void stopFollowing(std::function<void(bool)> on_done)
  {
    callTrigger(
      follow_stop_client_, "/follow/stop",
      [on_done](bool success, const std::string &) { on_done(success); });
  }

  // ============================================================================
  // 6. MAIN FINITE STATE MACHINE (FSM TICK)
  // ============================================================================
  void fsmTick()
  {
    const TaskState current_state = state_.load();

    // Watchdog: Cek batas waktu global seluruh task
    if ((now() - task_start_time_).seconds() > GLOBAL_TIMEOUT_SEC &&
      current_state != TaskState::COMPLETED && current_state != TaskState::ABORTED)
    {
      RCLCPP_ERROR(get_logger(), "Global task timeout reached.");
      state_ = TaskState::ABORTED;
      stopFollowing([](bool) {});
      return;
    }

    // Watchdog: Deteksi operator hilang saat following (berdasarkan timeout / status tracker LOST)
    if (current_state == TaskState::FOLLOW_OPERATOR &&
      (now() - state_enter_time_).seconds() > FOLLOW_GRACE_PERIOD_SEC &&
      (tracker_lost_.load() || (now() - last_operator_seen_time_).seconds() > OPERATOR_TIMEOUT_SEC))
    {
      operator_seen_ = false;
      transitionTo(TaskState::RECOVER_OPERATOR, "Operator lost (timeout > 5s or tracker lost)");
    }

    // Emergency Stop jika diucapkan perintah berhenti darurat
    if (emergency_stop_requested_.exchange(false)) {
      busy_ = true;
      stopFollowing([this](bool) {
        // busy_ tetap terkunci; transisi dilakukan setelah TTS selesai
        speakAsync("Okay, stopping.", [this]() {
          transitionTo(TaskState::WAIT_START_COMMAND, "Emergency stop");
        });
      });
      return;
    }

    // Safety watchdog: Lepas kuncian busy jika tts_done tidak diterima dalam estimasi durasi kalimat
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
        "TTS timeout (> %.1f s, no /hri/tts_done received), resetting busy state.", tts_timeout);
      onTtsDone(nullptr);
    }

    if (busy_.load()) {
      return;
    }

    // --------------------------------------------------------------------------
    // Evaluasi State Aktif
    // --------------------------------------------------------------------------
    switch (state_.load()) {
      // ------------------------------------------------------------------------
      // State 0: INIT - Menunggu TTS Engine siap
      // ------------------------------------------------------------------------
      case TaskState::INIT:
        if (tts_pub_->get_subscription_count() == 0) {
          return;
        }
        transitionTo(TaskState::WAIT_WAKEWORD, "TTS ready, waiting for wakeword");
        break;

      // ------------------------------------------------------------------------
      // State 1: WAIT_WAKEWORD - Menunggu sapaan awal
      // ------------------------------------------------------------------------
      case TaskState::WAIT_WAKEWORD:
        if (wakeword_requested_.exchange(false)) {
          speakAsync(
            "Hello, I'm Barelang Inari from Batam State Polytechnic, I want to help you to carry the luggage.",
            [this]() {
              transitionTo(TaskState::CHECK_OPERATOR_PRESENCE, "Greeting completed, checking operator");
            });
        }
        break;

      // ------------------------------------------------------------------------
      // State 2: CHECK_OPERATOR_PRESENCE - Memastikan operator terlihat di depan
      // ------------------------------------------------------------------------
      case TaskState::CHECK_OPERATOR_PRESENCE:
        if (isOperatorInFront()) {
          RCLCPP_INFO(get_logger(), "\033[1;32m[ZED VISION]\033[0m Operator terdeteksi di depan.");
          speakAsync(
            "If you are ready, please guide me to the place, and say follow me.",
            [this]() {
              transitionTo(TaskState::WAIT_START_COMMAND, "Operator in front confirmed");
            });
        } else {
          if ((now() - last_operator_warn_time_).seconds() > 4.0) {
            last_operator_warn_time_ = now();
            RCLCPP_WARN(get_logger(), "\033[1;33m[ZED VISION]\033[0m Operator belum terlihat di depan.");
            speakAsync("i cant see you, please stand in front of me");
          }
        }
        break;

      // ------------------------------------------------------------------------
      // State 3: WAIT_START_COMMAND - Menunggu instruksi "follow me"
      // ------------------------------------------------------------------------
      case TaskState::WAIT_START_COMMAND:
        if (follow_requested_.exchange(false)) {
          // Ucapkan konfirmasi DULU, baru robot mulai bergerak setelah TTS selesai
          speakAsync(
            "Okay, I am following you. Please walk slowly so that I can follow you from behind.",
            [this]() {
              busy_ = true;
              callTrigger(
                follow_start_client_, "/follow/person",
                [this](bool success, const std::string & message) {
                  if (!success) {
                    RCLCPP_ERROR(get_logger(), "Follow gagal: %s", message.c_str());
                    speakAsync("Person following is not ready.");
                    return;
                  }
                  last_operator_seen_time_ = now();
                  transitionTo(TaskState::FOLLOW_OPERATOR, "Follow started");
                  busy_ = false;
                });
            });
        }
        break;

      // ------------------------------------------------------------------------
      // State 4: FOLLOW_OPERATOR - Mengikuti operator (Grace period 5 detik aktif) = grace periode-> kondisi dimana robot tidak menerima inputan darimanapun
      // ------------------------------------------------------------------------
      case TaskState::FOLLOW_OPERATOR:
        if (follow_prompt_count_ < 3 && !speaking_.load() && now() >= follow_prompt_next_time_) {
          ++follow_prompt_count_;
          speakAsync(
            "Please say this is the location if you already in the location.",
            [this]() {
              if (follow_prompt_count_ < 3 && state_.load() == TaskState::FOLLOW_OPERATOR)
              {
                follow_prompt_next_time_ = now() + rclcpp::Duration::from_seconds(
                  FOLLOW_PROMPT_INTERVAL_SEC
                );
              }
            }
          );
          break;
        }
        if ((now() - state_enter_time_).seconds() > ARRIVED_GRACE_PERIOD_SEC &&
            arrived_requested_.exchange(false))
        {
          busy_ = true;
          stopFollowing([this](bool success) {
            if (!success) {
              transitionTo(TaskState::ABORTED, "Could not stop following");
              busy_ = false;
              return;
            }
            transitionTo(TaskState::ARRIVED, "Arrival command");
            busy_ = false;
          });
        }
        break;

      // ------------------------------------------------------------------------
      // State 5: ARRIVED - Cek operator & tas; panggil /task/step_back jika belum terdeteksi
      // ------------------------------------------------------------------------
      case TaskState::ARRIVED: {
        const bool operator_detected = isOperatorInFront();
        const bool bag_detected = isBagInFront();

        if (operator_detected && bag_detected) {
          RCLCPP_INFO(
            get_logger(),
            "\033[1;32m[ARRIVED]\033[0m Operator dan tas sudah terdeteksi di depan robot. Langsung menuju penunjukan tas.");
          speakAsync("We have arrived. Please point to the bag you want me to carry.", [this]() {
            transitionTo(TaskState::WAIT_BAG_POINTING, "Operator & bag visible");
          });
        } else {
          RCLCPP_INFO(
            get_logger(),
            "\033[1;33m[ARRIVED]\033[0m Operator (%s) atau tas (%s) belum terdeteksi lengkap. Memanggil service /task/step_back...",
            operator_detected ? "terdeteksi" : "tidak",
            bag_detected ? "terdeteksi" : "tidak");

          // Robot baru mundur setelah kalimat selesai diucapkan
          speakAsync("We have arrived. Stepping back to find you and the bag.", [this]() {
            busy_ = true;
            callTrigger(
              step_back_client_, "/task/step_back",
              [this](bool success, const std::string & message) {
                RCLCPP_INFO(
                  get_logger(),
                  "[ARRIVED] Selesai /task/step_back: %s (success=%s)",
                  message.c_str(), success ? "true" : "false");
                speakAsync("Please point to the bag you want me to carry.", [this]() {
                  transitionTo(TaskState::WAIT_BAG_POINTING, "Step back finished");
                });
              },
              20s);
          });
        }
        break;
      }

      // ------------------------------------------------------------------------
      // State 5: RECOVER_OPERATOR - Meminta operator kembali ke depan robot
      // ------------------------------------------------------------------------
      case TaskState::RECOVER_OPERATOR:
        if (!recovery_prompt_sent_) {
          recovery_prompt_sent_ = true;
          busy_ = true;
          stopFollowing([this](bool) {
            // busy_ tetap terkunci sampai prompt selesai diucapkan
            speakAsync("I lost sight of you. Please stand in front of me.");
          });
          break;
        } else if (isOperatorInFront()) {
          // Ucapkan konfirmasi DULU, baru lanjut follow setelah TTS selesai
          speakAsync("I see you. Resuming follow.", [this]() {
            busy_ = true;
            callTrigger(
              follow_start_client_, "/follow/person",
              [this](bool success, const std::string & message) {
                if (success) {
                  last_operator_seen_time_ = now();
                  transitionTo(TaskState::FOLLOW_OPERATOR, "Operator reacquired");
                } else {
                  RCLCPP_WARN(get_logger(), "Recovery follow gagal: %s", message.c_str());
                }
                busy_ = false;
              });
          });
        } else if ((now() - state_enter_time_).seconds() > 15.0) {
          transitionTo(TaskState::ABORTED, "Operator recovery timeout");
        }
        break;

      // ------------------------------------------------------------------------
      // State 6: WAIT_BAG_POINTING - Menunggu gesture penunjukan tas
      // ------------------------------------------------------------------------
      case TaskState::WAIT_BAG_POINTING:
        if (!bag_prompt_sent_) {
          bag_prompt_sent_ = true;
          detected_gesture_ = "NONE";
          gesture_confirmed_ = false;
          speakAsync("Which bag should I take?");
          break;
        }

        if (!gesture_confirmed_) {
          if (detected_gesture_ != "NONE") {
            gesture_confirmed_ = true;
            const std::string side = (detected_gesture_ == "LEFT") ? "right" : "left";
            RCLCPP_INFO(
              get_logger(),
              "\033[1;32m[GESTURE]\033[0m Terdeteksi %s -> Menunjuk tas di sisi %s operator",
              detected_gesture_.c_str(), side.c_str());

            speakAsync(
              "I already know which bag you are pointing to. You are pointing to the bag on your " + side + " side.",
              [this]() {
                busy_ = true;
                callTrigger(
                  align_bag_client_, "/task/align_bag",
                  [this](bool success, const std::string & message) {
                    if (success) {
                      transitionTo(TaskState::GRASP_BAG, "Bag aligned");
                    } else {
                      RCLCPP_WARN(get_logger(), "Align bag gagal: %s", message.c_str());
                      bag_prompt_sent_ = false;
                      gesture_confirmed_ = false;
                      detected_gesture_ = "NONE";
                    }
                    busy_ = false;  // Lepas kunci FSM setelah state baru siap
                  });
              });
          } else if ((now() - state_enter_time_).seconds() > 30.0) {
            state_enter_time_ = now();
            RCLCPP_WARN(get_logger(), "Timeout menunggu gesture pointing operator.");
            speakAsync("Please point to the bag you want me to carry.");
          }
        }
        break;

      // ------------------------------------------------------------------------
      // State 7: GRASP_BAG - Pengambilan tas via Lengan Robot (+ Human Fallback)
      // ------------------------------------------------------------------------
      case TaskState::GRASP_BAG:
        busy_ = true;
        callTrigger(
          grab_bag_client_, "/arm/ambil_tas",
          [this](bool success, const std::string & message) {
            // busy_ TIDAK dilepas di sini: speakAsync menjaga kunci sampai TTS selesai,
            // lalu transisi state (dan aksi berikutnya) dieksekusi di callback on_done.
            if (success) {
              speakAsync("I have the bag. I am returning to the starting point.", [this]() {
                transitionTo(TaskState::RETURN_TO_START, "Bag grasped");
              });
            } else {
              RCLCPP_WARN(
                get_logger(),
                "Grasp otonom gagal (%s). Mengaktifkan fallback bantuan operator...",
                message.c_str());
              speakAsync(
                "I could not grasp the bag automatically. Please hand the bag to me, then I will return to the starting point.",
                [this]() {
                  transitionTo(TaskState::RETURN_TO_START, "Bag handed over via human fallback");
                });
            }
          });
        break;

      // ------------------------------------------------------------------------
      // State 8: RETURN_TO_START - Navigasi otonom kembali ke titik awal (Homing)
      // ------------------------------------------------------------------------
      case TaskState::RETURN_TO_START:
        busy_ = true;
        callTrigger(
          homing_client_, "/homing",
          [this](bool success, const std::string & message) {
            if (success) {
              speakAsync("I have returned to the starting point. I will put down the bag now.", [this]() {
                transitionTo(TaskState::PUT_DOWN_BAG, "Homing completed, placing bag down");
              });
            } else if (homing_retry_count_ < 2) {
              homing_retry_count_++;
              RCLCPP_WARN(
                get_logger(),
                "Homing gagal (%s). Mencoba ulang (%d/2)...",
                message.c_str(), homing_retry_count_);
              // Retry homing baru dijalankan setelah kalimat selesai (busy_ dilepas oleh onTtsDone)
              speakAsync("Path is blocked, retrying navigation to the starting point.");
            } else {
              RCLCPP_ERROR(get_logger(), "Homing gagal setelah retry: %s", message.c_str());
              transitionTo(TaskState::ABORTED, "Homing failed after retries");
              busy_ = false;
            }
          },
          125s);
        break;

      // ------------------------------------------------------------------------
      // State 9: PUT_DOWN_BAG - Meletakkan tas di titik awal
      // ------------------------------------------------------------------------
      case TaskState::PUT_DOWN_BAG:
        busy_ = true;
        callTrigger(
          put_down_bag_client_, "/arm/letakkan_tas",
          [this](bool success, const std::string & message) {
            if (success) {
              speakAsync("I have placed the bag down. Task completed.", [this]() {
                transitionTo(TaskState::COMPLETED, "Bag placed down successfully");
              });
            } else {
              RCLCPP_WARN(get_logger(), "Letakkan tas tidak berhasil: %s. Melanjutkan ke completed.", message.c_str());
              speakAsync("Task completed.", [this]() {
                transitionTo(TaskState::COMPLETED, "Task completed (fallback)");
              });
            }
          });
        break;

      case TaskState::COMPLETED:
      case TaskState::ABORTED:
        break;
    }
  }
};

// ==============================================================================
// 7. ENTRY POINT UTAMA
// ==============================================================================
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HelpMeCarryOrchestrator>();

  // MultiThreadedExecutor dengan 2 thread untuk callback non-blocking
  rclcpp::executors::MultiThreadedExecutor executor(
    rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
