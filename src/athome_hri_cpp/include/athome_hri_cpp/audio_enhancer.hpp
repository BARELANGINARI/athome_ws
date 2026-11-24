#pragma once

#include <complex>
#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

namespace athome_hri_cpp
{

struct BiquadCoeffs
{
  double b0{1.0}, b1{0.0}, b2{0.0};
  double a1{0.0}, a2{0.0};
};

class BiquadFilter
{
public:
  explicit BiquadFilter(const BiquadCoeffs & c = {}) : c_(c) {}
  void setCoeffs(const BiquadCoeffs & c) { c_ = c; reset(); }
  void reset() { z1_ = 0.0; z2_ = 0.0; }
  float processSample(float x);
  void processInPlace(std::vector<float> & data);

private:
  BiquadCoeffs c_;
  double z1_{0.0};
  double z2_{0.0};
};

// Shared pure-C++ DSP utilities (Radix-2 FFT & Butterworth filter design)
namespace dsp
{
void fftRadix2(std::vector<std::complex<float>> & buf, bool inverse = false);
BiquadCoeffs designHighpassBiquad(double cutoff_hz, double sample_rate, double q);
BiquadCoeffs designLowpassBiquad(double cutoff_hz, double sample_rate, double q);
std::vector<float> resampleLinear(const float * input, size_t count, int src_rate, int dst_rate);
}  // namespace dsp

struct SpectralDenoiserConfig
{
  float alpha{1.4f};
  float spectral_floor{0.08f};
  double noise_percentile{0.15};
};

/**
 * @brief CPU-only single-channel speech enhancement frontend (zero GPU VRAM).
 *
 * Stages:
 *   1. DC offset removal + Zero-phase 4th-order Butterworth High-Pass Filter (default 120 Hz)
 *      to attenuate directional-mic proximity bass boost and robot motor/fan ego-noise.
 *   2. Denoising backend ("auto" -> deepfilternet2 -> rnnoise -> spectral):
 *      - deepfilternet2: optional native C API (libdf) if compiled with HAVE_DEEPFILTERNET
 *      - rnnoise:        optional native C API (librnnoise) if compiled with HAVE_RNNOISE
 *      - spectral:       built-in pure C++ STFT/ISTFT Wiener spectral subtraction
 *      - none:           high-pass filter only
 */
class AudioEnhancer
{
public:
  AudioEnhancer(
    const std::string & backend = "auto",
    int sample_rate = 16000,
    double highpass_hz = 120.0,
    const std::string & deepfilter_model_path = "",
    const SpectralDenoiserConfig & spectral_cfg = {});
  ~AudioEnhancer();

  AudioEnhancer(const AudioEnhancer &) = delete;
  AudioEnhancer & operator=(const AudioEnhancer &) = delete;

  void updateConfig(
    const std::string & backend,
    double highpass_hz,
    int sample_rate,
    const std::string & deepfilter_model_path = "",
    const SpectralDenoiserConfig & spectral_cfg = {});

  std::vector<float> enhance(const std::vector<float> & audio) const;
  std::string activeBackend() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_backend_;
  }

private:
  void initHighpass();
  void initBackend();
  void releaseNativeBackends();

  void applyZeroPhaseHighpass(std::vector<float> & sig) const;
  std::vector<float> enhanceSpectral(const std::vector<float> & sig) const;
  std::vector<float> enhanceDeepFilter(const std::vector<float> & sig) const;
  std::vector<float> enhanceRnnoise(const std::vector<float> & sig) const;

  std::string requested_backend_;
  std::string active_backend_{"none"};
  std::string df_model_path_;
  int sample_rate_{16000};
  double highpass_hz_{120.0};
  SpectralDenoiserConfig spectral_cfg_;

  BiquadCoeffs hp_stage1_;
  BiquadCoeffs hp_stage2_;
  bool hp_enabled_{false};

  mutable std::mutex mutex_;
  void * df_state_{nullptr};
  void * rnnoise_state_{nullptr};
};

}  // namespace athome_hri_cpp
