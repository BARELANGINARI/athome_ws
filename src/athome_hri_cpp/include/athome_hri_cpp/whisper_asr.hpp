#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace athome_hri_cpp
{

struct WhisperAsrConfig
{
  std::string model_path;
  bool use_gpu{true};
  bool flash_attn{true};
  int n_threads{2};
  std::string language{"en"};
  int beam_size{2};
  float temperature{0.0f};
  float no_speech_threshold{0.5f};
  float log_prob_threshold{-0.8f};
  float entropy_threshold{2.2f};
  std::string initial_prompt;
  int dedup_max_ngram{6};
  int min_transcript_chars{3};
  std::vector<std::string> hallucination_blacklist;
};

struct AsrResult
{
  std::string text;
  std::string raw_text;
  std::string rejected_raw;
  std::string reject_reason;
  double latency_sec{0.0};
  double audio_duration_sec{0.0};
  double rtf{0.0};
  double encode_ms{0.0};
  double decode_ms{0.0};
  double avg_logprob{0.0};
  double no_speech_prob{0.0};
  bool has_scores{false};
  bool accepted{false};
};

/**
 * @brief Thread-safe whisper.cpp C++ ASR engine with anti-loop decoding and n-gram deduplication.
 */
class WhisperAsr
{
public:
  explicit WhisperAsr(const WhisperAsrConfig & config);
  ~WhisperAsr();

  WhisperAsr(const WhisperAsr &) = delete;
  WhisperAsr & operator=(const WhisperAsr &) = delete;

  bool isReady() const;
  bool usingGpu() const;
  bool cudaCompiled() const;
  bool flashAttnActive() const;
  std::string backendSummary() const;
  std::string systemInfo() const;

  void updateRuntimeConfig(
    const std::string & language,
    int beam_size,
    float temperature,
    float no_speech_threshold,
    float log_prob_threshold,
    float entropy_threshold,
    const std::string & initial_prompt,
    int dedup_max_ngram,
    int min_transcript_chars,
    const std::vector<std::string> & blacklist);

  AsrResult transcribe(const std::vector<float> & audio_16k);

  /**
   * @brief Collapse consecutive identical word/phrase n-gram repetitions (case/punctuation agnostic).
   */
  static std::string deduplicateNgrams(const std::string & text, int max_ngram = 6);

private:
  void setBlacklistLocked(const std::vector<std::string> & items);
  static std::string trim(const std::string & s);
  static std::string toLowerCleaned(const std::string & s);

  struct Impl;
  std::unique_ptr<Impl> impl_;
  mutable std::mutex mutex_;
  WhisperAsrConfig config_;
  std::unordered_set<std::string> blacklist_;
};

}  // namespace athome_hri_cpp
