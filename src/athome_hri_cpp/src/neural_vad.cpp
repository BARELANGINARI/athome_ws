#include "athome_hri_cpp/neural_vad.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace athome_hri_cpp
{

struct SileroNeuralVad::Impl
{
#ifdef HAVE_ONNXRUNTIME
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "athome_silero_vad"};
  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo mem_info{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  bool is_v5{true};
  bool ready{false};
#else
  bool ready{false};
#endif
};

SileroNeuralVad::SileroNeuralVad(const NeuralVadConfig & config)
: impl_(std::make_unique<Impl>()), config_(config)
{
#ifdef HAVE_ONNXRUNTIME
  if (config_.model_path.empty()) {
    return;
  }
  std::ifstream ifs(config_.model_path, std::ios::binary);
  if (!ifs.good()) {
    std::cerr << "[SileroNeuralVad] Model not found at: " << config_.model_path << "\n";
    return;
  }
  try {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetInterOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    impl_->session = std::make_unique<Ort::Session>(impl_->env, config_.model_path.c_str(), opts);

    // Auto-detect Silero v5 (input, state, sr) vs v4 (input, h, c, sr)
    Ort::AllocatorWithDefaultOptions alloc;
    const size_t num_inputs = impl_->session->GetInputCount();
    bool has_state = false;
    for (size_t i = 0; i < num_inputs; ++i) {
      auto name_ptr = impl_->session->GetInputNameAllocated(i, alloc);
      if (std::string(name_ptr.get()) == "state") {
        has_state = true;
      }
    }
    impl_->is_v5 = has_state;
    impl_->ready = true;
  } catch (const std::exception & e) {
    std::cerr << "[SileroNeuralVad] Init failed: " << e.what() << "\n";
    impl_->ready = false;
  }
#endif
}

SileroNeuralVad::~SileroNeuralVad() = default;

bool SileroNeuralVad::isReady() const
{
  return impl_->ready;
}

void SileroNeuralVad::updateThresholds(
  float threshold, int min_speech_ms, int min_silence_ms, int speech_pad_ms)
{
  config_.threshold = threshold;
  config_.min_speech_ms = min_speech_ms;
  config_.min_silence_ms = min_silence_ms;
  config_.speech_pad_ms = speech_pad_ms;
}

std::vector<float> SileroNeuralVad::filterSpeech(const std::vector<float> & audio)
{
#ifdef HAVE_ONNXRUNTIME
  if (!impl_->ready || audio.empty()) {
    return audio;
  }

  constexpr size_t kWindowSize = 512;  // 32 ms @ 16 kHz
  constexpr size_t kContextSize = 64;  // Silero v5 context
  const int sr = config_.sample_rate;
  const int64_t sr_val = static_cast<int64_t>(sr);

  std::vector<float> state_v5(2 * 1 * 128, 0.0f);
  std::vector<float> h_v4(2 * 1 * 64, 0.0f);
  std::vector<float> c_v4(2 * 1 * 64, 0.0f);
  std::vector<float> context(kContextSize, 0.0f);

  std::vector<float> probs;
  const size_t num_chunks = (audio.size() + kWindowSize - 1) / kWindowSize;
  probs.reserve(num_chunks);

  try {
    for (size_t offset = 0; offset < audio.size(); offset += kWindowSize) {
      std::vector<float> chunk(kWindowSize, 0.0f);
      const size_t copy_len = std::min(kWindowSize, audio.size() - offset);
      std::copy_n(audio.data() + offset, copy_len, chunk.data());

      float speech_prob = 0.0f;
      if (impl_->is_v5) {
        std::vector<float> input_buf(kContextSize + kWindowSize, 0.0f);
        std::copy(context.begin(), context.end(), input_buf.begin());
        std::copy(chunk.begin(), chunk.end(), input_buf.begin() + kContextSize);
        std::copy(chunk.end() - kContextSize, chunk.end(), context.begin());

        std::array<int64_t, 2> in_shape{1, static_cast<int64_t>(kContextSize + kWindowSize)};
        std::array<int64_t, 3> st_shape{2, 1, 128};
        std::array<int64_t, 1> sr_shape{1};

        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(
          impl_->mem_info, input_buf.data(), input_buf.size(), in_shape.data(), in_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(
          impl_->mem_info, state_v5.data(), state_v5.size(), st_shape.data(), st_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
          impl_->mem_info, const_cast<int64_t *>(&sr_val), 1, sr_shape.data(), sr_shape.size()));

        const char * in_names[] = {"input", "state", "sr"};
        const char * out_names[] = {"output", "stateN"};
        auto outputs = impl_->session->Run(
          Ort::RunOptions{nullptr}, in_names, inputs.data(), 3, out_names, 2);

        speech_prob = outputs[0].GetTensorData<float>()[0];
        const float * new_state = outputs[1].GetTensorData<float>();
        std::copy_n(new_state, state_v5.size(), state_v5.data());
      } else {
        std::array<int64_t, 2> in_shape{1, static_cast<int64_t>(kWindowSize)};
        std::array<int64_t, 3> hc_shape{2, 1, 64};
        std::array<int64_t, 1> sr_shape{1};

        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(
          impl_->mem_info, chunk.data(), chunk.size(), in_shape.data(), in_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
          impl_->mem_info, const_cast<int64_t *>(&sr_val), 1, sr_shape.data(), sr_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(
          impl_->mem_info, h_v4.data(), h_v4.size(), hc_shape.data(), hc_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(
          impl_->mem_info, c_v4.data(), c_v4.size(), hc_shape.data(), hc_shape.size()));

        const char * in_names[] = {"input", "sr", "h", "c"};
        const char * out_names[] = {"output", "hn", "cn"};
        auto outputs = impl_->session->Run(
          Ort::RunOptions{nullptr}, in_names, inputs.data(), 4, out_names, 3);

        speech_prob = outputs[0].GetTensorData<float>()[0];
        std::copy_n(outputs[1].GetTensorData<float>(), h_v4.size(), h_v4.data());
        std::copy_n(outputs[2].GetTensorData<float>(), c_v4.size(), c_v4.data());
      }
      probs.push_back(speech_prob);
    }
  } catch (const std::exception & e) {
    std::cerr << "[SileroNeuralVad] Inference error (passing audio through): " << e.what() << "\n";
    return audio;
  }

  // Hysteresis segment extraction matching Silero get_speech_timestamps
  const float neg_thresh = std::max(0.01f, config_.threshold - 0.15f);
  const size_t min_speech_samples = static_cast<size_t>((sr * config_.min_speech_ms) / 1000);
  const size_t min_silence_samples = static_cast<size_t>((sr * config_.min_silence_ms) / 1000);
  const size_t speech_pad_samples = static_cast<size_t>((sr * config_.speech_pad_ms) / 1000);

  struct Segment { size_t start; size_t end; };
  std::vector<Segment> segments;

  bool triggered = false;
  size_t temp_end = 0;
  size_t cur_start = 0;

  for (size_t i = 0; i < probs.size(); ++i) {
    const float p = probs[i];
    const size_t sample_pos = i * kWindowSize;

    if (p >= config_.threshold && temp_end != 0) {
      temp_end = 0;
    }
    if (p >= config_.threshold && !triggered) {
      triggered = true;
      cur_start = sample_pos;
      continue;
    }
    if (triggered && p < neg_thresh) {
      if (temp_end == 0) {
        temp_end = sample_pos;
      }
      if (sample_pos - temp_end >= min_silence_samples) {
        if (temp_end - cur_start >= min_speech_samples) {
          segments.push_back({cur_start, temp_end});
        }
        cur_start = 0;
        temp_end = 0;
        triggered = false;
      }
    }
  }

  if (triggered && (audio.size() - cur_start >= min_speech_samples)) {
    segments.push_back({cur_start, audio.size()});
  }

  if (segments.empty()) {
    return {};
  }

  // Apply padding and merge overlapping segments
  for (auto & seg : segments) {
    seg.start = (seg.start > speech_pad_samples) ? (seg.start - speech_pad_samples) : 0;
    seg.end = std::min(audio.size(), seg.end + speech_pad_samples);
  }

  std::vector<float> out;
  size_t last_end = 0;
  for (const auto & seg : segments) {
    const size_t s = std::max(seg.start, last_end);
    if (s < seg.end) {
      out.insert(out.end(), audio.begin() + static_cast<std::ptrdiff_t>(s),
        audio.begin() + static_cast<std::ptrdiff_t>(seg.end));
      last_end = seg.end;
    }
  }
  return out;
#else
  return audio;
#endif
}

}  // namespace athome_hri_cpp
