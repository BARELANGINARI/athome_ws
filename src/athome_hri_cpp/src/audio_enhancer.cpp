#include "athome_hri_cpp/audio_enhancer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#ifdef HAVE_DEEPFILTERNET
#include <deep_filter.h>
#endif

#ifdef HAVE_RNNOISE
#include <rnnoise.h>
#endif

namespace athome_hri_cpp
{

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kButterworthQ1 = 0.541196100146197;
constexpr double kButterworthQ2 = 1.3065629648763766;
constexpr int kNativeDenoiserSampleRate = 48000;
}  // namespace

float BiquadFilter::processSample(float x)
{
  const double xd = static_cast<double>(x);
  const double y = c_.b0 * xd + z1_;
  z1_ = c_.b1 * xd - c_.a1 * y + z2_;
  z2_ = c_.b2 * xd - c_.a2 * y;
  return static_cast<float>(y);
}

void BiquadFilter::processInPlace(std::vector<float> & data)
{
  for (float & s : data) {
    s = processSample(s);
  }
}

namespace dsp
{

void fftRadix2(std::vector<std::complex<float>> & buf, bool inverse)
{
  const size_t n = buf.size();
  if (n <= 1) {
    return;
  }

  for (size_t i = 1, j = 0; i < n; ++i) {
    size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      std::swap(buf[i], buf[j]);
    }
  }

  const double sign = inverse ? 1.0 : -1.0;
  for (size_t len = 2; len <= n; len <<= 1) {
    const double ang = sign * 2.0 * kPi / static_cast<double>(len);
    const std::complex<float> wlen(
      static_cast<float>(std::cos(ang)),
      static_cast<float>(std::sin(ang)));
    for (size_t i = 0; i < n; i += len) {
      std::complex<float> w(1.0f, 0.0f);
      const size_t half = len >> 1;
      for (size_t j = 0; j < half; ++j) {
        const std::complex<float> u = buf[i + j];
        const std::complex<float> v = buf[i + j + half] * w;
        buf[i + j] = u + v;
        buf[i + j + half] = u - v;
        w *= wlen;
      }
    }
  }

  if (inverse) {
    const float inv_n = 1.0f / static_cast<float>(n);
    for (auto & c : buf) {
      c *= inv_n;
    }
  }
}

BiquadCoeffs designHighpassBiquad(double cutoff_hz, double sample_rate, double q)
{
  const double w0 = 2.0 * kPi * cutoff_hz / sample_rate;
  const double cos_w0 = std::cos(w0);
  const double alpha = std::sin(w0) / (2.0 * q);
  const double a0 = 1.0 + alpha;

  BiquadCoeffs c;
  c.b0 = ((1.0 + cos_w0) * 0.5) / a0;
  c.b1 = (-(1.0 + cos_w0)) / a0;
  c.b2 = ((1.0 + cos_w0) * 0.5) / a0;
  c.a1 = (-2.0 * cos_w0) / a0;
  c.a2 = (1.0 - alpha) / a0;
  return c;
}

BiquadCoeffs designLowpassBiquad(double cutoff_hz, double sample_rate, double q)
{
  const double w0 = 2.0 * kPi * cutoff_hz / sample_rate;
  const double cos_w0 = std::cos(w0);
  const double alpha = std::sin(w0) / (2.0 * q);
  const double a0 = 1.0 + alpha;

  BiquadCoeffs c;
  c.b0 = ((1.0 - cos_w0) * 0.5) / a0;
  c.b1 = (1.0 - cos_w0) / a0;
  c.b2 = ((1.0 - cos_w0) * 0.5) / a0;
  c.a1 = (-2.0 * cos_w0) / a0;
  c.a2 = (1.0 - alpha) / a0;
  return c;
}

std::vector<float> resampleLinear(const float * input, size_t count, int src_rate, int dst_rate)
{
  if (count == 0 || !input) {
    return {};
  }
  if (src_rate == dst_rate || src_rate <= 0 || dst_rate <= 0) {
    return std::vector<float>(input, input + count);
  }
  const double ratio = static_cast<double>(dst_rate) / static_cast<double>(src_rate);
  const size_t out_len = std::max<size_t>(
    1, static_cast<size_t>(std::llround(static_cast<double>(count) * ratio)));
  std::vector<float> out(out_len, 0.0f);
  if (count == 1) {
    std::fill(out.begin(), out.end(), input[0]);
    return out;
  }
  const double step = static_cast<double>(count - 1) / static_cast<double>(std::max<size_t>(1, out_len - 1));
  for (size_t i = 0; i < out_len; ++i) {
    const double pos = static_cast<double>(i) * step;
    const size_t idx0 = static_cast<size_t>(pos);
    const size_t idx1 = std::min(idx0 + 1, count - 1);
    const float frac = static_cast<float>(pos - static_cast<double>(idx0));
    out[i] = input[idx0] * (1.0f - frac) + input[idx1] * frac;
  }
  return out;
}

}  // namespace dsp

AudioEnhancer::AudioEnhancer(
  const std::string & backend,
  int sample_rate,
  double highpass_hz,
  const std::string & deepfilter_model_path,
  const SpectralDenoiserConfig & spectral_cfg)
: requested_backend_(backend),
  df_model_path_(deepfilter_model_path),
  sample_rate_(sample_rate),
  highpass_hz_(highpass_hz),
  spectral_cfg_(spectral_cfg)
{
  initHighpass();
  initBackend();
}

AudioEnhancer::~AudioEnhancer()
{
  releaseNativeBackends();
}

void AudioEnhancer::releaseNativeBackends()
{
#ifdef HAVE_DEEPFILTERNET
  if (df_state_ != nullptr) {
    df_free(static_cast<DFState *>(df_state_));
    df_state_ = nullptr;
  }
#endif
#ifdef HAVE_RNNOISE
  if (rnnoise_state_ != nullptr) {
    rnnoise_destroy(static_cast<DenoiseState *>(rnnoise_state_));
    rnnoise_state_ = nullptr;
  }
#endif
}

void AudioEnhancer::initHighpass()
{
  if (highpass_hz_ <= 0.0 || sample_rate_ <= 0) {
    hp_enabled_ = false;
    return;
  }
  const double nyq = 0.5 * static_cast<double>(sample_rate_);
  const double cutoff = std::clamp(highpass_hz_, 20.0, nyq * 0.9);
  hp_stage1_ = dsp::designHighpassBiquad(cutoff, static_cast<double>(sample_rate_), kButterworthQ1);
  hp_stage2_ = dsp::designHighpassBiquad(cutoff, static_cast<double>(sample_rate_), kButterworthQ2);
  hp_enabled_ = true;
}

void AudioEnhancer::initBackend()
{
  releaseNativeBackends();

  std::string req = requested_backend_;
  std::transform(req.begin(), req.end(), req.begin(), ::tolower);

  if (req == "none") {
    active_backend_ = "none";
    return;
  }

  std::vector<std::string> order;
  if (req == "auto" || req.empty()) {
    order = {"deepfilternet2", "rnnoise", "spectral"};
  } else {
    order = {req, "spectral"};
  }

  for (const auto & cand : order) {
    if (cand == "deepfilternet2" || cand == "deepfilternet" || cand == "df2") {
#ifdef HAVE_DEEPFILTERNET
      if (!df_model_path_.empty()) {
        df_state_ = df_create(df_model_path_.c_str(), 100.0f, nullptr);
        if (df_state_ != nullptr) {
          active_backend_ = "deepfilternet2";
          return;
        }
      }
#endif
    } else if (cand == "rnnoise") {
#ifdef HAVE_RNNOISE
      rnnoise_state_ = rnnoise_create(nullptr);
      if (rnnoise_state_ != nullptr) {
        active_backend_ = "rnnoise";
        return;
      }
#endif
    } else if (cand == "spectral") {
      active_backend_ = "spectral";
      return;
    }
  }

  active_backend_ = "none";
}

void AudioEnhancer::updateConfig(
  const std::string & backend,
  double highpass_hz,
  int sample_rate,
  const std::string & deepfilter_model_path,
  const SpectralDenoiserConfig & spectral_cfg)
{
  spectral_cfg_ = spectral_cfg;
  bool need_hp = false;
  if (sample_rate > 0 && sample_rate != sample_rate_) {
    sample_rate_ = sample_rate;
    need_hp = true;
  }
  if (std::abs(highpass_hz - highpass_hz_) > 1e-6) {
    highpass_hz_ = highpass_hz;
    need_hp = true;
  }
  if (need_hp) {
    initHighpass();
  }
  if (backend != requested_backend_ || deepfilter_model_path != df_model_path_) {
    requested_backend_ = backend;
    df_model_path_ = deepfilter_model_path;
    initBackend();
  }
}

void AudioEnhancer::applyZeroPhaseHighpass(std::vector<float> & sig) const
{
  if (!hp_enabled_ || sig.size() < 32) {
    return;
  }
  BiquadFilter f1(hp_stage1_), f2(hp_stage2_);
  f1.processInPlace(sig);
  f2.processInPlace(sig);

  std::reverse(sig.begin(), sig.end());
  f1.reset();
  f2.reset();
  f1.processInPlace(sig);
  f2.processInPlace(sig);
  std::reverse(sig.begin(), sig.end());
}

std::vector<float> AudioEnhancer::enhanceDeepFilter(const std::vector<float> & sig) const
{
#ifdef HAVE_DEEPFILTERNET
  if (df_state_ == nullptr || sig.empty()) {
    return sig;
  }
  auto * state = static_cast<DFState *>(df_state_);
  const size_t frame_len = static_cast<size_t>(df_get_frame_length(state));
  if (frame_len == 0) {
    return sig;
  }
  std::vector<float> audio_48k = dsp::resampleLinear(
    sig.data(), sig.size(), sample_rate_, kNativeDenoiserSampleRate);
  const size_t orig_48k_len = audio_48k.size();
  const size_t rem = orig_48k_len % frame_len;
  if (rem != 0) {
    audio_48k.resize(orig_48k_len + (frame_len - rem), 0.0f);
  }
  std::vector<float> out_48k(audio_48k.size(), 0.0f);
  for (size_t offset = 0; offset < audio_48k.size(); offset += frame_len) {
    df_process_frame(state, audio_48k.data() + offset, out_48k.data() + offset);
  }
  out_48k.resize(orig_48k_len);
  return dsp::resampleLinear(out_48k.data(), out_48k.size(), kNativeDenoiserSampleRate, sample_rate_);
#else
  return sig;
#endif
}

std::vector<float> AudioEnhancer::enhanceRnnoise(const std::vector<float> & sig) const
{
#ifdef HAVE_RNNOISE
  if (rnnoise_state_ == nullptr || sig.empty()) {
    return sig;
  }
  auto * state = static_cast<DenoiseState *>(rnnoise_state_);
  constexpr size_t kFrameLen = 480;
  std::vector<float> audio_48k = dsp::resampleLinear(
    sig.data(), sig.size(), sample_rate_, kNativeDenoiserSampleRate);
  const size_t orig_48k_len = audio_48k.size();
  const size_t rem = orig_48k_len % kFrameLen;
  if (rem != 0) {
    audio_48k.resize(orig_48k_len + (kFrameLen - rem), 0.0f);
  }
  std::vector<float> frame_in(kFrameLen), frame_out(kFrameLen);
  std::vector<float> out_48k(audio_48k.size(), 0.0f);
  for (size_t offset = 0; offset < audio_48k.size(); offset += kFrameLen) {
    for (size_t i = 0; i < kFrameLen; ++i) {
      frame_in[i] = audio_48k[offset + i] * 32767.0f;
    }
    rnnoise_process_frame(state, frame_out.data(), frame_in.data());
    for (size_t i = 0; i < kFrameLen; ++i) {
      out_48k[offset + i] = frame_out[i] / 32768.0f;
    }
  }
  out_48k.resize(orig_48k_len);
  return dsp::resampleLinear(out_48k.data(), out_48k.size(), kNativeDenoiserSampleRate, sample_rate_);
#else
  return sig;
#endif
}

std::vector<float> AudioEnhancer::enhanceSpectral(const std::vector<float> & sig) const
{
  const size_t n_fft = (sample_rate_ >= 16000) ? 512 : 256;
  const size_t hop = n_fft / 2;
  if (sig.size() < n_fft) {
    return sig;
  }

  std::vector<float> win(n_fft);
  for (size_t i = 0; i < n_fft; ++i) {
    win[i] = 0.5f * (1.0f - std::cos(static_cast<float>(2.0 * kPi * i / (n_fft - 1))));
  }

  const size_t n_frames = 1 + (sig.size() - n_fft) / hop;
  const size_t n_bins = n_fft / 2 + 1;

  std::vector<std::vector<std::complex<float>>> stft_frames(
    n_frames, std::vector<std::complex<float>>(n_fft));
  std::vector<double> frame_energies(n_frames, 0.0);

  for (size_t f = 0; f < n_frames; ++f) {
    const size_t start = f * hop;
    for (size_t i = 0; i < n_fft; ++i) {
      stft_frames[f][i] = std::complex<float>(sig[start + i] * win[i], 0.0f);
    }
    dsp::fftRadix2(stft_frames[f], false);

    double energy = 0.0;
    for (size_t k = 0; k < n_bins; ++k) {
      energy += static_cast<double>(std::norm(stft_frames[f][k]));
    }
    frame_energies[f] = energy / static_cast<double>(n_bins);
  }

  // Estimate stationary + quasi-stationary noise PSD from the quietest frames
  std::vector<size_t> order(n_frames);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return frame_energies[a] < frame_energies[b];
  });
  const double pct = std::clamp(spectral_cfg_.noise_percentile, 0.01, 0.90);
  const size_t n_noise = std::max<size_t>(1, static_cast<size_t>(std::ceil(n_frames * pct)));
  std::vector<float> noise_psd(n_bins, 1e-10f);
  for (size_t i = 0; i < n_noise; ++i) {
    const size_t idx = order[i];
    for (size_t k = 0; k < n_bins; ++k) {
      noise_psd[k] += std::norm(stft_frames[idx][k]);
    }
  }
  for (size_t k = 0; k < n_bins; ++k) {
    noise_psd[k] = noise_psd[k] / static_cast<float>(n_noise) + 1e-10f;
  }

  const float alpha = std::max(0.0f, spectral_cfg_.alpha);
  const float spectral_floor = std::clamp(spectral_cfg_.spectral_floor, 0.0f, 1.0f);
  std::vector<std::vector<float>> gains(n_frames, std::vector<float>(n_bins, 1.0f));
  for (size_t f = 0; f < n_frames; ++f) {
    for (size_t k = 0; k < n_bins; ++k) {
      const float p = std::norm(stft_frames[f][k]);
      const float snr_post = std::max(p / noise_psd[k], 1e-6f);
      gains[f][k] = std::max(1.0f - alpha / snr_post, spectral_floor);
    }
  }

  // Mild 3-frame temporal smoothing to suppress musical noise
  if (n_frames >= 3) {
    auto smoothed = gains;
    for (size_t f = 1; f + 1 < n_frames; ++f) {
      for (size_t k = 0; k < n_bins; ++k) {
        smoothed[f][k] = 0.25f * gains[f - 1][k] + 0.5f * gains[f][k] + 0.25f * gains[f + 1][k];
      }
    }
    gains = std::move(smoothed);
  }

  // Overlap-add ISTFT
  std::vector<float> out(sig.size(), 0.0f);
  std::vector<float> norm_acc(sig.size(), 0.0f);
  for (size_t f = 0; f < n_frames; ++f) {
    auto & spec = stft_frames[f];
    spec[0] *= gains[f][0];
    if (n_fft % 2 == 0) {
      spec[n_fft / 2] *= gains[f][n_fft / 2];
    }
    for (size_t k = 1; k < n_fft / 2; ++k) {
      const float g = gains[f][k];
      spec[k] *= g;
      spec[n_fft - k] *= g;
    }
    dsp::fftRadix2(spec, true);

    const size_t start = f * hop;
    for (size_t i = 0; i < n_fft; ++i) {
      out[start + i] += spec[i].real() * win[i];
      norm_acc[start + i] += win[i] * win[i];
    }
  }

  for (size_t i = 0; i < sig.size(); ++i) {
    if (norm_acc[i] > 1e-8f) {
      out[i] /= norm_acc[i];
    } else {
      out[i] = sig[i];
    }
  }
  return out;
}

std::vector<float> AudioEnhancer::enhance(const std::vector<float> & audio) const
{
  if (audio.empty()) {
    return {};
  }
  std::vector<float> sig = audio;
  const double mean = std::accumulate(sig.begin(), sig.end(), 0.0) / static_cast<double>(sig.size());
  for (float & s : sig) {
    s -= static_cast<float>(mean);
  }

  applyZeroPhaseHighpass(sig);

  if (active_backend_ == "deepfilternet2") {
    sig = enhanceDeepFilter(sig);
  } else if (active_backend_ == "rnnoise") {
    sig = enhanceRnnoise(sig);
  } else if (active_backend_ == "spectral") {
    sig = enhanceSpectral(sig);
  }

  for (float & s : sig) {
    s = std::clamp(s, -1.0f, 1.0f);
  }
  return sig;
}

}  // namespace athome_hri_cpp
