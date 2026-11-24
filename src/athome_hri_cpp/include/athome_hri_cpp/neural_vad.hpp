#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace athome_hri_cpp
{

struct NeuralVadConfig
{
  std::string model_path;
  int sample_rate{16000};
  float threshold{0.65f};
  int min_speech_ms{250};
  int min_silence_ms{350};
  int speech_pad_ms{150};
};

/**
 * @brief Strict CPU-only Silero Neural VAD gate via ONNX Runtime C++ (1 CPU thread, 0 VRAM).
 *
 * Supports both Silero VAD v5 (input + state + sr) and v4 (input + h + c + sr) ONNX graphs.
 * If ONNX Runtime is not linked or the model file is absent, `isReady()` returns false and
 * `filterSpeech()` acts as a transparent pass-through.
 */
class SileroNeuralVad
{
public:
  explicit SileroNeuralVad(const NeuralVadConfig & config);
  ~SileroNeuralVad();

  SileroNeuralVad(const SileroNeuralVad &) = delete;
  SileroNeuralVad & operator=(const SileroNeuralVad &) = delete;

  bool isReady() const;
  void updateThresholds(float threshold, int min_speech_ms, int min_silence_ms, int speech_pad_ms);

  /**
   * @brief Filter audio to speech-only segments.
   * @return Concatenated speech samples, or empty vector if pure noise / non-speech.
   */
  std::vector<float> filterSpeech(const std::vector<float> & audio);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  NeuralVadConfig config_;
};

}  // namespace athome_hri_cpp
