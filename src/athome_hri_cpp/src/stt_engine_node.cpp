#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include "athome_hri_cpp/audio_capture.hpp"
#include "athome_hri_cpp/audio_enhancer.hpp"
#include "athome_hri_cpp/neural_vad.hpp"
#include "athome_hri_cpp/whisper_asr.hpp"

namespace athome_hri_cpp
{

namespace
{
std::string resolveModelPath(const std::string & path)
{
  if (path.empty()) {
    return "";
  }
  const std::filesystem::path p(path);
  if (p.is_absolute() || std::filesystem::exists(p)) {
    return path;
  }
  try {
    const std::string share_dir = ament_index_cpp::get_package_share_directory("athome_hri_cpp");
    const auto cand = std::filesystem::path(share_dir) / p;
    if (std::filesystem::exists(cand)) {
      return cand.string();
    }
  } catch (...) {
  }
  return path;
}

std::string makeTaskHeader(const std::string & task_mode)
{
  std::string lower = task_mode;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  if (lower == "receptionist" || lower == "rec") {
    return "\033[1;36m[RECEPTIONIST]\033[0m";
  }
  if (lower == "help_me_carry" || lower == "hmc") {
    return "\033[1;32m[HELP_ME_CARRY]\033[0m";
  }
  if (lower == "restaurant" || lower == "resto" || lower == "waiter") {
    return "\033[1;34m[RESTAURANT]\033[0m";
  }
  return "\033[1;33m[HRI_CPP]\033[0m";
}
}  // namespace

class SttEngineCppNode : public rclcpp::Node
{
public:
  SttEngineCppNode()
  : Node("stt_engine")
  {
    // 1. Read ALL tunable parameters strictly from config/params.yaml (no hardcoded C++ defaults)
    const int mic_idx = static_cast<int>(requireYamlParameter<int64_t>("mic_device_index"));
    enable_local_mic_.store(requireYamlParameter<bool>("enable_local_mic"));
    enable_live_meter_.store(requireYamlParameter<bool>("enable_live_meter"));
    live_meter_interval_sec_.store(requireYamlParameter<double>("live_meter_interval_sec"));
    const int sample_rate = static_cast<int>(requireYamlParameter<int64_t>("stt_sample_rate"));

    vad_config_.block_duration_sec = requireYamlParameter<double>("vad_block_duration_sec");
    vad_config_.speech_band_low_hz = requireYamlParameter<double>("vad_speech_band_low_hz");
    vad_config_.speech_band_high_hz = requireYamlParameter<double>("vad_speech_band_high_hz");
    vad_config_.threshold_dbfs = requireYamlParameter<double>("threshold_dbfs");
    vad_config_.threshold_flatness = requireYamlParameter<double>("threshold_flatness");
    vad_config_.adaptive_vad_enabled = requireYamlParameter<bool>("adaptive_vad_enabled");
    vad_config_.adaptive_snr_margin_db = requireYamlParameter<double>("adaptive_snr_margin_db");
    vad_config_.adaptive_max_threshold_dbfs =
      requireYamlParameter<double>("adaptive_max_threshold_dbfs");
    vad_config_.adaptive_noise_alpha_up = requireYamlParameter<double>("adaptive_noise_alpha_up");
    vad_config_.adaptive_noise_alpha_down = requireYamlParameter<double>("adaptive_noise_alpha_down");
    vad_config_.pre_roll_frames = static_cast<int>(requireYamlParameter<int64_t>("pre_roll_frames"));
    vad_config_.hangover_frames = static_cast<int>(requireYamlParameter<int64_t>("hangover_frames"));
    vad_config_.min_speech_frames = static_cast<int>(requireYamlParameter<int64_t>("min_speech_frames"));
    vad_config_.max_speech_frames = static_cast<int>(requireYamlParameter<int64_t>("max_speech_frames"));
    vad_config_.sample_rate = sample_rate;

    const std::string denoiser_backend = requireYamlParameter<std::string>("stt_denoiser_backend");
    const std::string df_model_path = resolveModelPath(
      requireYamlParameter<std::string>("stt_deepfilter_model_path"));
    const double highpass_hz = requireYamlParameter<double>("stt_highpass_hz");

    SpectralDenoiserConfig spectral_cfg;
    spectral_cfg.alpha = static_cast<float>(requireYamlParameter<double>("stt_spectral_alpha"));
    spectral_cfg.spectral_floor =
      static_cast<float>(requireYamlParameter<double>("stt_spectral_floor"));
    spectral_cfg.noise_percentile = requireYamlParameter<double>("stt_spectral_noise_percentile");

    NeuralVadConfig nvad_cfg;
    nvad_cfg.model_path = resolveModelPath(
      requireYamlParameter<std::string>("stt_silero_vad_model_path"));
    nvad_cfg.sample_rate = sample_rate;
    nvad_cfg.threshold = static_cast<float>(requireYamlParameter<double>("stt_vad_threshold"));
    nvad_cfg.min_speech_ms = static_cast<int>(requireYamlParameter<int64_t>("stt_vad_min_speech_ms"));
    nvad_cfg.min_silence_ms = static_cast<int>(requireYamlParameter<int64_t>("stt_vad_silence_ms"));
    nvad_cfg.speech_pad_ms = static_cast<int>(requireYamlParameter<int64_t>("stt_vad_speech_pad_ms"));

    WhisperAsrConfig asr_cfg;
    asr_cfg.model_path = resolveModelPath(requireYamlParameter<std::string>("whisper_model_path"));
    asr_cfg.use_gpu = requireYamlParameter<bool>("whisper_use_gpu");
    asr_cfg.n_threads = static_cast<int>(requireYamlParameter<int64_t>("whisper_n_threads"));
    asr_cfg.language = requireYamlParameter<std::string>("stt_language");
    asr_cfg.beam_size = static_cast<int>(requireYamlParameter<int64_t>("stt_beam_size"));
    asr_cfg.temperature = static_cast<float>(requireYamlParameter<double>("stt_temperature"));
    asr_cfg.no_speech_threshold =
      static_cast<float>(requireYamlParameter<double>("stt_no_speech_threshold"));
    asr_cfg.log_prob_threshold =
      static_cast<float>(requireYamlParameter<double>("stt_log_prob_threshold"));
    asr_cfg.entropy_threshold =
      static_cast<float>(requireYamlParameter<double>("stt_entropy_threshold"));
    asr_cfg.initial_prompt = requireYamlParameter<std::string>("stt_initial_prompt");
    asr_cfg.dedup_max_ngram = static_cast<int>(requireYamlParameter<int64_t>("stt_dedup_max_ngram"));
    asr_cfg.min_transcript_chars =
      static_cast<int>(requireYamlParameter<int64_t>("stt_min_transcript_chars"));
    asr_cfg.hallucination_blacklist =
      requireYamlParameter<std::vector<std::string>>("stt_hallucination_blacklist");

    max_utterance_queue_.store(
      std::max<size_t>(1, static_cast<size_t>(requireYamlParameter<int64_t>("stt_max_utterance_queue"))));
    tts_speaking_timeout_sec_ = requireYamlParameter<double>("tts_speaking_timeout_sec");
    tts_post_mute_sec_.store(std::max(0.0, requireYamlParameter<double>("tts_post_mute_sec")));

    const char * env_task = std::getenv("ATHOME_TASK");
    declare_parameter<std::string>("task", env_task ? env_task : "all");
    task_header_ = makeTaskHeader(get_parameter("task").as_string());

    // 2. Initialize DSP + Denoising Frontend (CPU)
    enhancer_ = std::make_unique<AudioEnhancer>(
      denoiser_backend, sample_rate, highpass_hz, df_model_path, spectral_cfg);
    RCLCPP_INFO(
      get_logger(), "%s\n  Denoiser: %s (HPF %.0f Hz, CPU)",
      task_header_.c_str(), enhancer_->activeBackend().c_str(), highpass_hz);

    // 3. Initialize Silero Neural VAD (ONNX Runtime CPU)
    neural_vad_ = std::make_unique<SileroNeuralVad>(nvad_cfg);
    RCLCPP_INFO(
      get_logger(), "%s\n  Neural VAD: %s",
      task_header_.c_str(),
      neural_vad_->isReady() ? "Silero ONNX (CPU, 1 thread)" : "Disabled (Adaptive DSP VAD active)");

    // 4. Initialize whisper.cpp ASR
    asr_ = std::make_unique<WhisperAsr>(asr_cfg);
    if (asr_->isReady()) {
      RCLCPP_INFO(
        get_logger(), "%s\n  Whisper.cpp Ready (%s, threads=%d)",
        task_header_.c_str(),
        asr_->usingGpu() ? "CUDA" : "CPU",
        asr_cfg.n_threads);
    } else {
      RCLCPP_WARN(
        get_logger(), "%s\n  Whisper.cpp NOT ready (check whisper_model_path / scripts/setup_deps.sh)",
        task_header_.c_str());
    }

    // 5. ROS 2 Publishers & Subscribers (Drop-in topics)
    transcript_pub_ = create_publisher<std_msgs::msg::String>("/hri/stt_transcript", 10);
    tts_say_sub_ = create_subscription<std_msgs::msg::String>(
      "/hri/tts_say", 10,
      std::bind(&SttEngineCppNode::onTtsSay, this, std::placeholders::_1));
    tts_done_sub_ = create_subscription<std_msgs::msg::String>(
      "/hri/tts_done", 10,
      std::bind(&SttEngineCppNode::onTtsDone, this, std::placeholders::_1));

    // 6. Start ASR worker thread
    asr_running_.store(true);
    asr_thread_ = std::thread(&SttEngineCppNode::asrWorkerLoop, this);

    // 7. Start PortAudio microphone capture
    capture_ = std::make_unique<AudioCapture>(
      mic_idx,
      vad_config_,
      [this](std::vector<float> && audio) { enqueueUtterance(std::move(audio)); },
      [this]() { return isMicPaused(); },
      [this](const VadFrameMetrics & m) { onVadMetrics(m); });

    if (capture_->start()) {
      RCLCPP_INFO(
        get_logger(), "%s\n  Mic Ready: %s (HW %d Hz -> %d Hz)",
        task_header_.c_str(),
        capture_->activeDeviceName().c_str(),
        capture_->activeSampleRate(),
        sample_rate);
    } else {
      RCLCPP_ERROR(get_logger(), "%s\n  Failed to start PortAudio mic capture!", task_header_.c_str());
    }

    param_cb_handle_ = add_on_set_parameters_callback(
      std::bind(&SttEngineCppNode::onParamsChanged, this, std::placeholders::_1));
  }

  ~SttEngineCppNode() override
  {
    if (capture_) {
      capture_->stop();
    }
    asr_running_.store(false);
    asr_cv_.notify_all();
    if (asr_thread_.joinable()) {
      asr_thread_.join();
    }
  }

private:
  template<typename T>
  T requireYamlParameter(const std::string & name)
  {
    try {
      return declare_parameter<T>(name);
    } catch (const std::exception & e) {
      RCLCPP_FATAL(
        get_logger(),
        "Parameter '%s' wajib diatur di file YAML (athome_hri_cpp/config/params.yaml)! Error: %s",
        name.c_str(), e.what());
      throw;
    }
  }

  bool isMicPaused()
  {
    const double now_sec = std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
    if (tts_speaking_.load() && now_sec >= tts_speaking_until_sec_.load()) {
      RCLCPP_WARN(get_logger(), "Watchdog /hri/tts_done timeout; re-enabling microphone.");
      tts_speaking_.store(false);
    }
    return (!enable_local_mic_.load()) ||
           tts_speaking_.load() ||
           (now_sec < tts_post_mute_until_sec_.load());
  }

  void onTtsSay(const std_msgs::msg::String::SharedPtr msg)
  {
    if (msg && !msg->data.empty()) {
      const double now_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
      tts_epoch_.fetch_add(1);
      tts_speaking_until_sec_.store(now_sec + tts_speaking_timeout_sec_);
      tts_post_mute_until_sec_.store(0.0);
      tts_speaking_.store(true);
      std::lock_guard<std::mutex> lock(asr_queue_mutex_);
      while (!asr_queue_.empty()) {
        asr_queue_.pop();
      }
    }
  }

  void onTtsDone(const std_msgs::msg::String::SharedPtr /*msg*/)
  {
    const double now_sec = std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
    tts_speaking_.store(false);
    tts_speaking_until_sec_.store(0.0);
    tts_post_mute_until_sec_.store(now_sec + tts_post_mute_sec_.load());
  }

  void enqueueUtterance(std::vector<float> && audio)
  {
    if (isMicPaused()) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(asr_queue_mutex_);
      const size_t max_q = max_utterance_queue_.load();
      while (asr_queue_.size() >= max_q) {
        asr_queue_.pop();
      }
      asr_queue_.push(UtteranceItem{std::move(audio), tts_epoch_.load()});
    }
    asr_cv_.notify_one();
  }

  void asrWorkerLoop()
  {
    while (asr_running_.load()) {
      UtteranceItem item;
      {
        std::unique_lock<std::mutex> lock(asr_queue_mutex_);
        asr_cv_.wait(lock, [this]() { return !asr_queue_.empty() || !asr_running_.load(); });
        if (!asr_running_.load() && asr_queue_.empty()) {
          break;
        }
        item = std::move(asr_queue_.front());
        asr_queue_.pop();
      }

      if (item.epoch != tts_epoch_.load() || isMicPaused()) {
        continue;
      }

      // Stage 1: CPU Denoising Frontend (120 Hz HPF + DeepFilterNet2 / RNNoise / Spectral)
      std::vector<float> clean_audio = enhancer_->enhance(item.audio);

      // Stage 2: Strict CPU Silero Neural VAD Gate
      if (neural_vad_->isReady()) {
        clean_audio = neural_vad_->filterSpeech(clean_audio);
        if (clean_audio.empty()) {
          RCLCPP_DEBUG(get_logger(), "[ASR] Rejected by Silero Neural VAD (non-speech/noise).");
          continue;
        }
      }

      if (item.epoch != tts_epoch_.load() || isMicPaused()) {
        continue;
      }

      // Stage 3 & 4: whisper.cpp Anti-Loop Decoding + N-Gram Deduplication + Blacklist
      const AsrResult res = asr_->transcribe(clean_audio);
      if (res.accepted && !res.text.empty()) {
        if (item.epoch != tts_epoch_.load() || isMicPaused()) {
          RCLCPP_INFO(
            get_logger(), "%s\n  STT Dropped (TTS echo/stale): \"%s\"",
            task_header_.c_str(), res.text.c_str());
          continue;
        }
        std_msgs::msg::String out;
        out.data = res.text;
        transcript_pub_->publish(out);
        RCLCPP_INFO(
          get_logger(), "%s\n  STT: \"%s\" (RTF: %.3f | Latency: %.2fs | Audio: %.2fs)",
          task_header_.c_str(), res.text.c_str(), res.rtf, res.latency_sec, res.audio_duration_sec);
      } else if (!res.rejected_raw.empty()) {
        RCLCPP_INFO(
          get_logger(), "%s\n  STT Rejected (noise/hallucination): \"%s\"",
          task_header_.c_str(), res.rejected_raw.c_str());
      }
    }
  }

  void onVadMetrics(const VadFrameMetrics & m)
  {
    if (!enable_live_meter_.load()) {
      return;
    }
    const auto now_tp = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now_tp - last_meter_log_).count() < live_meter_interval_sec_.load()) {
      return;
    }
    last_meter_log_ = now_tp;

    const int bars = std::clamp(static_cast<int>((m.dbfs_speech + 60.0) / 10.0), 0, 6);
    std::string bar_str(static_cast<size_t>(bars), '#');
    bar_str.append(static_cast<size_t>(6 - bars), '-');

    const char * status = m.is_paused
      ? "\033[2m[MUTED]\033[0m"
      : (m.is_voice ? "\033[1;32m[VOICE]\033[0m"
                    : (m.is_speaking ? "\033[1;33m[HOLD]\033[0m" : "\033[2m[SILENCE]\033[0m"));

    RCLCPP_INFO(
      get_logger(), "%s\n  VAD: [%s] %5.1f dBFS (thr %.1f | floor %.1f) %s",
      task_header_.c_str(), bar_str.c_str(),
      m.dbfs_speech, m.effective_threshold_dbfs, m.noise_floor_dbfs, status);
  }

  rcl_interfaces::msg::SetParametersResult onParamsChanged(
    const std::vector<rclcpp::Parameter> & params)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    bool vad_dirty = false;
    bool enhancer_dirty = false;
    bool nvad_dirty = false;
    bool asr_dirty = false;

    std::string backend = get_parameter("stt_denoiser_backend").as_string();
    std::string df_model = resolveModelPath(get_parameter("stt_deepfilter_model_path").as_string());
    double highpass_hz = get_parameter("stt_highpass_hz").as_double();
    int sample_rate = static_cast<int>(get_parameter("stt_sample_rate").as_int());

    SpectralDenoiserConfig spectral_cfg;
    spectral_cfg.alpha = static_cast<float>(get_parameter("stt_spectral_alpha").as_double());
    spectral_cfg.spectral_floor = static_cast<float>(get_parameter("stt_spectral_floor").as_double());
    spectral_cfg.noise_percentile = get_parameter("stt_spectral_noise_percentile").as_double();

    float nv_thresh = static_cast<float>(get_parameter("stt_vad_threshold").as_double());
    int nv_min_sp = static_cast<int>(get_parameter("stt_vad_min_speech_ms").as_int());
    int nv_min_sil = static_cast<int>(get_parameter("stt_vad_silence_ms").as_int());
    int nv_pad = static_cast<int>(get_parameter("stt_vad_speech_pad_ms").as_int());

    std::string lang = get_parameter("stt_language").as_string();
    int beam = static_cast<int>(get_parameter("stt_beam_size").as_int());
    float temp = static_cast<float>(get_parameter("stt_temperature").as_double());
    float no_sp = static_cast<float>(get_parameter("stt_no_speech_threshold").as_double());
    float log_p = static_cast<float>(get_parameter("stt_log_prob_threshold").as_double());
    float ent = static_cast<float>(get_parameter("stt_entropy_threshold").as_double());
    std::string prompt = get_parameter("stt_initial_prompt").as_string();
    int dedup_n = static_cast<int>(get_parameter("stt_dedup_max_ngram").as_int());
    int min_chars = static_cast<int>(get_parameter("stt_min_transcript_chars").as_int());
    auto bl = get_parameter("stt_hallucination_blacklist").as_string_array();

    for (const auto & p : params) {
      const auto & n = p.get_name();
      if (n == "enable_local_mic") {
        enable_local_mic_.store(p.as_bool());
      } else if (n == "enable_live_meter") {
        enable_live_meter_.store(p.as_bool());
      } else if (n == "live_meter_interval_sec") {
        live_meter_interval_sec_.store(std::max(0.1, p.as_double()));
      } else if (n == "stt_max_utterance_queue") {
        max_utterance_queue_.store(std::max<size_t>(1, static_cast<size_t>(p.as_int())));
      } else if (n == "threshold_dbfs") {
        vad_config_.threshold_dbfs = p.as_double(); vad_dirty = true;
      } else if (n == "threshold_flatness") {
        vad_config_.threshold_flatness = p.as_double(); vad_dirty = true;
      } else if (n == "adaptive_vad_enabled") {
        vad_config_.adaptive_vad_enabled = p.as_bool(); vad_dirty = true;
      } else if (n == "adaptive_snr_margin_db") {
        vad_config_.adaptive_snr_margin_db = p.as_double(); vad_dirty = true;
      } else if (n == "adaptive_max_threshold_dbfs") {
        vad_config_.adaptive_max_threshold_dbfs = p.as_double(); vad_dirty = true;
      } else if (n == "adaptive_noise_alpha_up") {
        vad_config_.adaptive_noise_alpha_up = p.as_double(); vad_dirty = true;
      } else if (n == "adaptive_noise_alpha_down") {
        vad_config_.adaptive_noise_alpha_down = p.as_double(); vad_dirty = true;
      } else if (n == "hangover_frames") {
        vad_config_.hangover_frames = static_cast<int>(p.as_int()); vad_dirty = true;
      } else if (n == "min_speech_frames") {
        vad_config_.min_speech_frames = static_cast<int>(p.as_int()); vad_dirty = true;
      } else if (n == "max_speech_frames") {
        vad_config_.max_speech_frames = static_cast<int>(p.as_int()); vad_dirty = true;
      } else if (n == "stt_denoiser_backend") {
        backend = p.as_string(); enhancer_dirty = true;
      } else if (n == "stt_highpass_hz") {
        highpass_hz = p.as_double(); enhancer_dirty = true;
      } else if (n == "stt_spectral_alpha") {
        spectral_cfg.alpha = static_cast<float>(p.as_double()); enhancer_dirty = true;
      } else if (n == "stt_spectral_floor") {
        spectral_cfg.spectral_floor = static_cast<float>(p.as_double()); enhancer_dirty = true;
      } else if (n == "stt_spectral_noise_percentile") {
        spectral_cfg.noise_percentile = p.as_double(); enhancer_dirty = true;
      } else if (n == "stt_vad_threshold") {
        nv_thresh = static_cast<float>(p.as_double()); nvad_dirty = true;
      } else if (n == "stt_vad_min_speech_ms") {
        nv_min_sp = static_cast<int>(p.as_int()); nvad_dirty = true;
      } else if (n == "stt_vad_silence_ms") {
        nv_min_sil = static_cast<int>(p.as_int()); nvad_dirty = true;
      } else if (n == "stt_vad_speech_pad_ms") {
        nv_pad = static_cast<int>(p.as_int()); nvad_dirty = true;
      } else if (n == "stt_language") {
        lang = p.as_string(); asr_dirty = true;
      } else if (n == "stt_beam_size") {
        beam = static_cast<int>(p.as_int()); asr_dirty = true;
      } else if (n == "stt_temperature") {
        temp = static_cast<float>(p.as_double()); asr_dirty = true;
      } else if (n == "stt_no_speech_threshold") {
        no_sp = static_cast<float>(p.as_double()); asr_dirty = true;
      } else if (n == "stt_log_prob_threshold") {
        log_p = static_cast<float>(p.as_double()); asr_dirty = true;
      } else if (n == "stt_entropy_threshold") {
        ent = static_cast<float>(p.as_double()); asr_dirty = true;
      } else if (n == "stt_initial_prompt") {
        prompt = p.as_string(); asr_dirty = true;
      } else if (n == "stt_dedup_max_ngram") {
        dedup_n = static_cast<int>(p.as_int()); asr_dirty = true;
      } else if (n == "stt_min_transcript_chars") {
        min_chars = static_cast<int>(p.as_int()); asr_dirty = true;
      } else if (n == "stt_hallucination_blacklist") {
        bl = p.as_string_array(); asr_dirty = true;
      } else if (n == "tts_post_mute_sec") {
        tts_post_mute_sec_.store(std::max(0.0, p.as_double()));
      }
    }

    if (vad_dirty && capture_) {
      capture_->updateVadConfig(vad_config_);
    }
    if (enhancer_dirty && enhancer_) {
      enhancer_->updateConfig(backend, highpass_hz, sample_rate, df_model, spectral_cfg);
    }
    if (nvad_dirty && neural_vad_) {
      neural_vad_->updateThresholds(nv_thresh, nv_min_sp, nv_min_sil, nv_pad);
    }
    if (asr_dirty && asr_) {
      asr_->updateRuntimeConfig(lang, beam, temp, no_sp, log_p, ent, prompt, dedup_n, min_chars, bl);
    }
    return result;
  }

  struct UtteranceItem
  {
    std::vector<float> audio;
    uint64_t epoch{0};
  };

  std::string task_header_;
  std::atomic<bool> enable_local_mic_{true};
  std::atomic<bool> enable_live_meter_{true};
  std::atomic<double> live_meter_interval_sec_{1.0};
  std::atomic<size_t> max_utterance_queue_{4};
  std::atomic<bool> tts_speaking_{false};
  std::atomic<double> tts_speaking_until_sec_{0.0};
  std::atomic<double> tts_post_mute_sec_{0.6};
  std::atomic<double> tts_post_mute_until_sec_{0.0};
  std::atomic<uint64_t> tts_epoch_{0};
  double tts_speaking_timeout_sec_{20.0};
  std::chrono::steady_clock::time_point last_meter_log_{};

  VadConfig vad_config_;
  std::unique_ptr<AudioEnhancer> enhancer_;
  std::unique_ptr<SileroNeuralVad> neural_vad_;
  std::unique_ptr<WhisperAsr> asr_;
  std::unique_ptr<AudioCapture> capture_;

  std::atomic<bool> asr_running_{false};
  std::thread asr_thread_;
  std::mutex asr_queue_mutex_;
  std::condition_variable asr_cv_;
  std::queue<UtteranceItem> asr_queue_;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr transcript_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr tts_say_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr tts_done_sub_;
  OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;
};

}  // namespace athome_hri_cpp

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<athome_hri_cpp::SttEngineCppNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
