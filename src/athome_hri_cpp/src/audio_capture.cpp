#include "athome_hri_cpp/audio_capture.hpp"

#include <portaudio.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <iostream>

namespace athome_hri_cpp
{

namespace
{
constexpr double kButterworthQ1 = 0.541196100146197;
constexpr double kButterworthQ2 = 1.3065629648763766;
constexpr double kPi = 3.14159265358979323846;

int paStreamCallback(
  const void * input_buffer,
  void * output_buffer,
  unsigned long frames_per_buffer,
  const PaStreamCallbackTimeInfo * /*time_info*/,
  PaStreamCallbackFlags /*status_flags*/,
  void * user_data)
{
  (void)output_buffer;
  auto * self = static_cast<AudioCapture *>(user_data);
  if (self != nullptr && input_buffer != nullptr && frames_per_buffer > 0) {
    self->pushRawSamples(static_cast<const float *>(input_buffer), frames_per_buffer);
  }
  return paContinue;
}
}  // namespace

AudioCapture::AudioCapture(
  int device_index,
  const VadConfig & config,
  UtteranceCallback on_utterance,
  PauseCheckCallback is_paused_fn,
  MetricsCallback on_metrics)
: configured_device_index_(device_index),
  target_sample_rate_(std::max(8000, config.sample_rate)),
  config_(config),
  on_utterance_(std::move(on_utterance)),
  is_paused_fn_(std::move(is_paused_fn)),
  on_metrics_(std::move(on_metrics))
{
  target_block_size_ = std::max<size_t>(
    160, static_cast<size_t>(std::llround(target_sample_rate_ * config_.block_duration_sec)));

  const double sr = static_cast<double>(target_sample_rate_);
  const double low_hz = std::clamp(config_.speech_band_low_hz, 50.0, sr * 0.4);
  const double high_hz = std::clamp(config_.speech_band_high_hz, low_hz + 100.0, sr * 0.48);
  bp_hp1_.setCoeffs(dsp::designHighpassBiquad(low_hz, sr, kButterworthQ1));
  bp_hp2_.setCoeffs(dsp::designHighpassBiquad(low_hz, sr, kButterworthQ2));
  bp_lp1_.setCoeffs(dsp::designLowpassBiquad(high_hz, sr, kButterworthQ1));
  bp_lp2_.setCoeffs(dsp::designLowpassBiquad(high_hz, sr, kButterworthQ2));

  hann_window_.resize(target_block_size_);
  for (size_t i = 0; i < target_block_size_; ++i) {
    hann_window_[i] = 0.5f * (
      1.0f - std::cos(static_cast<float>(2.0 * kPi * i / (target_block_size_ - 1))));
  }
}

AudioCapture::~AudioCapture()
{
  stop();
}

int AudioCapture::resolveInputDevice(int configured_index, std::string & out_name) const
{
  const int count = Pa_GetDeviceCount();
  if (count <= 0) {
    out_name = "none";
    return paNoDevice;
  }

  if (configured_index >= 0 && configured_index < count) {
    const PaDeviceInfo * info = Pa_GetDeviceInfo(configured_index);
    if (info != nullptr && info->maxInputChannels > 0) {
      out_name = info->name ? info->name : "configured";
      return configured_index;
    }
  }

  for (int i = 0; i < count; ++i) {
    const PaDeviceInfo * info = Pa_GetDeviceInfo(i);
    if (info == nullptr || info->maxInputChannels <= 0 || !info->name) {
      continue;
    }
    std::string lower(info->name);
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower.find("usb") != std::string::npos) {
      out_name = info->name;
      return i;
    }
  }

  const PaDeviceIndex def = Pa_GetDefaultInputDevice();
  if (def != paNoDevice) {
    const PaDeviceInfo * info = Pa_GetDeviceInfo(def);
    out_name = (info && info->name) ? info->name : "default";
    return def;
  }

  out_name = "none";
  return paNoDevice;
}

bool AudioCapture::start()
{
  if (running_.load()) {
    return true;
  }

  const PaError init_err = Pa_Initialize();
  if (init_err != paNoError) {
    std::cerr << "[AudioCapture] Pa_Initialize failed: " << Pa_GetErrorText(init_err) << "\n";
    return false;
  }
  pa_initialized_ = true;

  const int dev_idx = resolveInputDevice(configured_device_index_, active_device_name_);
  if (dev_idx == paNoDevice) {
    std::cerr << "[AudioCapture] No input audio device available.\n";
    Pa_Terminate();
    pa_initialized_ = false;
    return false;
  }

  const PaDeviceInfo * dev_info = Pa_GetDeviceInfo(dev_idx);
  PaStreamParameters input_params{};
  input_params.device = dev_idx;
  input_params.channelCount = 1;
  input_params.sampleFormat = paFloat32;
  input_params.suggestedLatency = dev_info ? dev_info->defaultLowInputLatency : 0.03;
  input_params.hostApiSpecificStreamInfo = nullptr;

  std::vector<int> candidate_rates = {target_sample_rate_};
  if (dev_info && dev_info->defaultSampleRate > 0) {
    const int def_sr = static_cast<int>(std::llround(dev_info->defaultSampleRate));
    if (def_sr != target_sample_rate_) {
      candidate_rates.push_back(def_sr);
    }
  }
  for (int r : {48000, 44100}) {
    if (std::find(candidate_rates.begin(), candidate_rates.end(), r) == candidate_rates.end()) {
      candidate_rates.push_back(r);
    }
  }

  PaStream * stream = nullptr;
  for (int sr : candidate_rates) {
    const unsigned long hw_frames = static_cast<unsigned long>(
      std::max<long>(128, std::llround(sr * config_.block_duration_sec)));
    const PaError err = Pa_OpenStream(
      &stream, &input_params, nullptr, static_cast<double>(sr),
      hw_frames, paClipOff, paStreamCallback, this);
    if (err == paNoError && stream != nullptr) {
      hw_sample_rate_ = sr;
      break;
    }
    stream = nullptr;
  }

  if (stream == nullptr) {
    std::cerr << "[AudioCapture] Failed to open PortAudio input stream on device "
              << active_device_name_ << "\n";
    Pa_Terminate();
    pa_initialized_ = false;
    return false;
  }

  pa_stream_ = stream;
  running_.store(true);
  worker_thread_ = std::thread(&AudioCapture::workerLoop, this);

  const PaError start_err = Pa_StartStream(stream);
  if (start_err != paNoError) {
    std::cerr << "[AudioCapture] Pa_StartStream failed: " << Pa_GetErrorText(start_err) << "\n";
    stop();
    return false;
  }
  return true;
}

void AudioCapture::stop()
{
  running_.store(false);
  queue_cv_.notify_all();

  if (pa_stream_ != nullptr) {
    auto * stream = static_cast<PaStream *>(pa_stream_);
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    pa_stream_ = nullptr;
  }
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
  if (pa_initialized_) {
    Pa_Terminate();
    pa_initialized_ = false;
  }
}

void AudioCapture::updateVadConfig(const VadConfig & config)
{
  std::lock_guard<std::mutex> lock(config_mutex_);
  config_ = config;
}

void AudioCapture::pushRawSamples(const float * samples, size_t count)
{
  if (!running_.load()) {
    return;
  }
  std::vector<float> buf(samples, samples + count);
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (raw_queue_.size() >= kMaxRawQueueDepth) {
      raw_queue_.pop();
    }
    raw_queue_.push(std::move(buf));
  }
  queue_cv_.notify_one();
}

void AudioCapture::workerLoop()
{
  std::vector<float> residual;
  residual.reserve(target_block_size_ * 2);

  while (running_.load()) {
    std::vector<float> chunk;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [this]() { return !raw_queue_.empty() || !running_.load(); });
      if (!running_.load() && raw_queue_.empty()) {
        break;
      }
      chunk = std::move(raw_queue_.front());
      raw_queue_.pop();
    }

    std::vector<float> resampled = (hw_sample_rate_ == target_sample_rate_)
      ? std::move(chunk)
      : dsp::resampleLinear(chunk.data(), chunk.size(), hw_sample_rate_, target_sample_rate_);

    residual.insert(residual.end(), resampled.begin(), resampled.end());
    while (residual.size() >= target_block_size_) {
      std::vector<float> frame(
        residual.begin(),
        residual.begin() + static_cast<std::ptrdiff_t>(target_block_size_));
      residual.erase(
        residual.begin(),
        residual.begin() + static_cast<std::ptrdiff_t>(target_block_size_));
      processFrame(frame);
    }
  }
}

void AudioCapture::computeFeatures(
  const std::vector<float> & frame,
  double & out_dbfs,
  double & out_dbfs_speech,
  double & out_flatness)
{
  constexpr double kEps = 1e-10;
  double sum_sq = 0.0;
  for (float s : frame) {
    sum_sq += static_cast<double>(s) * static_cast<double>(s);
  }
  const double rms = std::sqrt(sum_sq / static_cast<double>(frame.size()) + kEps);
  out_dbfs = 20.0 * std::log10(rms + kEps);

  // Speech-band 300-3400 Hz energy
  std::vector<float> speech = frame;
  bp_hp1_.processInPlace(speech);
  bp_hp2_.processInPlace(speech);
  bp_lp1_.processInPlace(speech);
  bp_lp2_.processInPlace(speech);

  double sum_sq_sp = 0.0;
  for (float s : speech) {
    sum_sq_sp += static_cast<double>(s) * static_cast<double>(s);
  }
  const double rms_sp = std::sqrt(sum_sq_sp / static_cast<double>(speech.size()) + kEps);
  out_dbfs_speech = 20.0 * std::log10(rms_sp + kEps);

  // Spectral flatness via Radix-2 FFT (power-of-two >= frame.size())
  size_t n_fft = 1;
  while (n_fft < frame.size()) {
    n_fft <<= 1;
  }
  std::vector<std::complex<float>> fft_buf(n_fft, {0.0f, 0.0f});
  for (size_t i = 0; i < frame.size(); ++i) {
    fft_buf[i] = {frame[i] * hann_window_[i], 0.0f};
  }
  dsp::fftRadix2(fft_buf, false);

  const size_t n_bins = n_fft / 2 + 1;
  double log_sum = 0.0;
  double lin_sum = 0.0;
  for (size_t k = 0; k < n_bins; ++k) {
    const double mag = static_cast<double>(std::abs(fft_buf[k])) + kEps;
    log_sum += std::log(mag);
    lin_sum += mag;
  }
  const double geom_mean = std::exp(log_sum / static_cast<double>(n_bins));
  const double arith_mean = lin_sum / static_cast<double>(n_bins);
  out_flatness = geom_mean / std::max(arith_mean, kEps);
}

void AudioCapture::emitUtterance(bool force_emit)
{
  const VadConfig cfg = [this]() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_;
  }();

  const size_t min_total = static_cast<size_t>(
    std::max(1, cfg.min_speech_frames) + std::max(0, cfg.pre_roll_frames));
  if ((force_emit && !speech_buffer_.empty()) || speech_buffer_.size() >= min_total) {
    size_t total_samples = 0;
    for (const auto & f : speech_buffer_) {
      total_samples += f.size();
    }
    std::vector<float> full_audio;
    full_audio.reserve(total_samples);
    for (const auto & f : speech_buffer_) {
      full_audio.insert(full_audio.end(), f.begin(), f.end());
    }
    if (on_utterance_ && !full_audio.empty()) {
      on_utterance_(std::move(full_audio));
    }
  }
  is_speaking_ = false;
  silence_counter_ = 0;
  speech_buffer_.clear();
}

void AudioCapture::processFrame(const std::vector<float> & frame)
{
  double dbfs = -100.0;
  double dbfs_speech = -100.0;
  double flatness = 1.0;
  computeFeatures(frame, dbfs, dbfs_speech, flatness);

  const VadConfig cfg = [this]() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_;
  }();

  const bool paused = (is_paused_fn_ && is_paused_fn_());
  if (paused) {
    pre_roll_buffer_.clear();
    speech_buffer_.clear();
    is_speaking_ = false;
    silence_counter_ = 0;
    if (on_metrics_) {
      VadFrameMetrics m;
      m.dbfs = dbfs;
      m.dbfs_speech = dbfs_speech;
      m.effective_threshold_dbfs = cfg.threshold_dbfs;
      m.noise_floor_dbfs = noise_floor_dbfs_;
      m.flatness = flatness;
      m.is_paused = true;
      on_metrics_(m);
    }
    return;
  }

  double effective_thresh = cfg.threshold_dbfs;
  if (cfg.adaptive_vad_enabled) {
    effective_thresh = std::max(
      cfg.threshold_dbfs,
      std::min(cfg.adaptive_max_threshold_dbfs, noise_floor_dbfs_ + cfg.adaptive_snr_margin_db));
  }

  const bool frame_is_voice =
    (dbfs_speech > effective_thresh) && (flatness < cfg.threshold_flatness);

  // Slow adaptive noise-floor tracking during non-speech
  if (cfg.adaptive_vad_enabled && dbfs_speech > -95.0) {
    if (!is_speaking_ && !frame_is_voice) {
      const double alpha = (dbfs_speech > noise_floor_dbfs_)
        ? std::clamp(cfg.adaptive_noise_alpha_up, 0.001, 0.5)
        : std::clamp(cfg.adaptive_noise_alpha_down, 0.001, 0.5);
      noise_floor_dbfs_ = (1.0 - alpha) * noise_floor_dbfs_ + alpha * dbfs_speech;
    } else if (dbfs_speech < noise_floor_dbfs_) {
      noise_floor_dbfs_ = 0.90 * noise_floor_dbfs_ + 0.10 * dbfs_speech;
    }
  }

  if (frame_is_voice) {
    if (!is_speaking_) {
      is_speaking_ = true;
      for (const auto & pre : pre_roll_buffer_) {
        speech_buffer_.push_back(pre);
      }
    }
    speech_buffer_.push_back(frame);
    silence_counter_ = 0;

    if (static_cast<int>(speech_buffer_.size()) >= std::max(30, cfg.max_speech_frames)) {
      emitUtterance(true);
    }
  } else {
    if (is_speaking_) {
      speech_buffer_.push_back(frame);
      ++silence_counter_;
      if (silence_counter_ >= std::max(1, cfg.hangover_frames)) {
        emitUtterance(false);
      }
    } else {
      pre_roll_buffer_.push_back(frame);
      while (static_cast<int>(pre_roll_buffer_.size()) > std::max(0, cfg.pre_roll_frames)) {
        pre_roll_buffer_.pop_front();
      }
    }
  }

  if (on_metrics_) {
    VadFrameMetrics m;
    m.dbfs = dbfs;
    m.dbfs_speech = dbfs_speech;
    m.effective_threshold_dbfs = effective_thresh;
    m.noise_floor_dbfs = noise_floor_dbfs_;
    m.flatness = flatness;
    m.is_voice = frame_is_voice;
    m.is_speaking = is_speaking_;
    m.is_paused = false;
    on_metrics_(m);
  }
}

}  // namespace athome_hri_cpp
