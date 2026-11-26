#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "athome_hri_cpp/audio_enhancer.hpp"

namespace athome_hri_cpp
{

struct VadConfig
{
  double threshold_dbfs{-30.0};
  double threshold_flatness{0.40};
  bool adaptive_vad_enabled{true};
  double adaptive_snr_margin_db{10.0};
  double adaptive_max_threshold_dbfs{-12.0};
  double adaptive_noise_alpha_up{0.05};
  double adaptive_noise_alpha_down{0.15};
  double speech_band_low_hz{300.0};
  double speech_band_high_hz{3400.0};
  int pre_roll_frames{10};
  int hangover_frames{16};
  int min_speech_frames{9};
  int max_speech_frames{250};
  int sample_rate{16000};
  double block_duration_sec{0.03};  // 30 ms (480 samples @ 16 kHz)
};

struct VadFrameMetrics
{
  double dbfs{-100.0};
  double dbfs_speech{-100.0};
  double effective_threshold_dbfs{-30.0};
  double noise_floor_dbfs{-50.0};
  double flatness{1.0};
  bool is_voice{false};
  bool is_speaking{false};
  bool is_paused{false};
};

/**
 * @brief Real-time PortAudio microphone capture with Adaptive Noise-Floor VAD.
 *
 * The PortAudio callback only copies raw samples into a thread-safe queue (no DSP/FFT/malloc
 * in real-time audio callback). A dedicated worker thread performs resampling, 300-3400 Hz
 * speech-band energy + spectral flatness computation, adaptive noise-floor tracking, and
 * utterance segmentation with pre-roll, hangover, min_speech, and max_speech bounds.
 */
class AudioCapture
{
public:
  struct VadDropInfo
  {
    int voice_frames{0};
    int min_required_frames{0};
    double duration_sec{0.0};
    double peak_dbfs{-100.0};
  };

  using UtteranceCallback = std::function<void(std::vector<float> &&)>;
  using PauseCheckCallback = std::function<bool()>;
  using MetricsCallback = std::function<void(const VadFrameMetrics &)>;
  using VadDropCallback = std::function<void(const VadDropInfo &)>;

  AudioCapture(
    int device_index,
    const VadConfig & config,
    UtteranceCallback on_utterance,
    PauseCheckCallback is_paused_fn,
    MetricsCallback on_metrics = nullptr,
    VadDropCallback on_vad_drop = nullptr);
  ~AudioCapture();

  AudioCapture(const AudioCapture &) = delete;
  AudioCapture & operator=(const AudioCapture &) = delete;

  bool start();
  void stop();
  void updateVadConfig(const VadConfig & config);
  std::string activeDeviceName() const { return active_device_name_; }
  int activeSampleRate() const { return hw_sample_rate_; }
  void pushRawSamples(const float * samples, size_t count);

private:
  int resolveInputDevice(int configured_index, std::string & out_name) const;
  void workerLoop();
  void processFrame(const std::vector<float> & frame);
  void computeFeatures(
    const std::vector<float> & frame,
    double & out_dbfs,
    double & out_dbfs_speech,
    double & out_flatness);
  void emitUtterance(bool force_emit);

  int configured_device_index_{-1};
  std::string active_device_name_{"default"};
  int hw_sample_rate_{16000};
  int target_sample_rate_{16000};
  size_t target_block_size_{480};

  mutable std::mutex config_mutex_;
  VadConfig config_;

  UtteranceCallback on_utterance_;
  PauseCheckCallback is_paused_fn_;
  MetricsCallback on_metrics_;
  VadDropCallback on_vad_drop_;

  void * pa_stream_{nullptr};
  bool pa_initialized_{false};

  std::atomic<bool> running_{false};
  std::thread worker_thread_;

  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::queue<std::vector<float>> raw_queue_;
  static constexpr size_t kMaxRawQueueDepth = 200;

  // Speech-band 300-3400 Hz cascade (2x HPF + 2x LPF)
  BiquadFilter bp_hp1_, bp_hp2_, bp_lp1_, bp_lp2_;
  std::vector<float> hann_window_;

  // VAD state machine variables
  bool is_speaking_{false};
  int silence_counter_{0};
  int active_voice_frames_{0};
  double segment_peak_dbfs_{-100.0};
  double noise_floor_dbfs_{-50.0};
  std::deque<std::vector<float>> pre_roll_buffer_;
  std::vector<std::vector<float>> speech_buffer_;
};

}  // namespace athome_hri_cpp
