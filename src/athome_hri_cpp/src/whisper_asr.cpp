#include "athome_hri_cpp/whisper_asr.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

#ifdef HAVE_WHISPER_CPP
#include <whisper.h>
#endif

namespace athome_hri_cpp
{

struct WhisperAsr::Impl
{
#ifdef HAVE_WHISPER_CPP
  whisper_context * ctx{nullptr};
#endif
  bool ready{false};
  bool using_gpu{false};
};

WhisperAsr::WhisperAsr(const WhisperAsrConfig & config)
: impl_(std::make_unique<Impl>()), config_(config)
{
  setBlacklistLocked(config_.hallucination_blacklist);

#ifdef HAVE_WHISPER_CPP
  if (config_.model_path.empty()) {
    std::cerr << "[WhisperAsr] Empty model_path.\n";
    return;
  }
  std::ifstream ifs(config_.model_path, std::ios::binary);
  if (!ifs.good()) {
    std::cerr << "[WhisperAsr] Model file not found: " << config_.model_path << "\n";
    return;
  }

  whisper_context_params cparams = whisper_context_default_params();
  cparams.use_gpu = config_.use_gpu;
  impl_->ctx = whisper_init_from_file_with_params(config_.model_path.c_str(), cparams);
  if (impl_->ctx != nullptr) {
    impl_->ready = true;
    impl_->using_gpu = config_.use_gpu;
    return;
  }

  if (config_.use_gpu) {
    std::cerr << "[WhisperAsr] GPU init failed, retrying on CPU...\n";
    cparams.use_gpu = false;
    impl_->ctx = whisper_init_from_file_with_params(config_.model_path.c_str(), cparams);
    if (impl_->ctx != nullptr) {
      impl_->ready = true;
      impl_->using_gpu = false;
    }
  }
#else
  std::cerr << "[WhisperAsr] Built without whisper.cpp (HAVE_WHISPER_CPP not defined).\n";
#endif
}

WhisperAsr::~WhisperAsr()
{
#ifdef HAVE_WHISPER_CPP
  if (impl_->ctx != nullptr) {
    whisper_free(impl_->ctx);
    impl_->ctx = nullptr;
  }
#endif
}

bool WhisperAsr::isReady() const
{
  return impl_->ready;
}

bool WhisperAsr::usingGpu() const
{
  return impl_->using_gpu;
}

std::string WhisperAsr::trim(const std::string & s)
{
  const auto first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = s.find_last_not_of(" \t\r\n");
  return s.substr(first, last - first + 1);
}

std::string WhisperAsr::toLowerCleaned(const std::string & s)
{
  std::string out = trim(s);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  while (!out.empty() && (out.back() == '.' || out.back() == ',' || out.back() == '!' || out.back() == '?')) {
    out.pop_back();
  }
  return trim(out);
}

void WhisperAsr::setBlacklistLocked(const std::vector<std::string> & items)
{
  blacklist_.clear();
  for (const auto & item : items) {
    const std::string c = toLowerCleaned(item);
    if (!c.empty()) {
      blacklist_.insert(c);
    }
  }
}

void WhisperAsr::updateRuntimeConfig(
  const std::string & language,
  int beam_size,
  float temperature,
  float no_speech_threshold,
  float log_prob_threshold,
  float entropy_threshold,
  const std::string & initial_prompt,
  int dedup_max_ngram,
  int min_transcript_chars,
  const std::vector<std::string> & blacklist)
{
  std::lock_guard<std::mutex> lock(mutex_);
  config_.language = language;
  config_.beam_size = std::max(1, beam_size);
  config_.temperature = temperature;
  config_.no_speech_threshold = no_speech_threshold;
  config_.log_prob_threshold = log_prob_threshold;
  config_.entropy_threshold = entropy_threshold;
  config_.initial_prompt = initial_prompt;
  config_.dedup_max_ngram = std::max(1, dedup_max_ngram);
  config_.min_transcript_chars = std::max(1, min_transcript_chars);
  setBlacklistLocked(blacklist);
}

std::string WhisperAsr::deduplicateNgrams(const std::string & text, int max_ngram)
{
  std::istringstream iss(text);
  std::vector<std::string> tokens;
  std::string tok;
  while (iss >> tok) {
    tokens.push_back(tok);
  }
  if (tokens.size() < 2) {
    return trim(text);
  }

  auto norm_tok = [](const std::string & t) {
    std::string out;
    for (unsigned char c : t) {
      if (std::isalnum(c) || c == '_') {
        out.push_back(static_cast<char>(std::tolower(c)));
      }
    }
    return out;
  };

  bool changed = true;
  while (changed) {
    changed = false;
    const int n_max = std::min(std::max(1, max_ngram), static_cast<int>(tokens.size() / 2));
    for (int n = n_max; n >= 1; --n) {
      std::vector<std::string> out;
      size_t i = 0;
      const size_t un = static_cast<size_t>(n);
      while (i < tokens.size()) {
        if (i + 2 * un <= tokens.size()) {
          bool any_nonempty = false;
          bool match = true;
          for (size_t k = 0; k < un; ++k) {
            const std::string lk = norm_tok(tokens[i + k]);
            const std::string rk = norm_tok(tokens[i + un + k]);
            if (!lk.empty()) {
              any_nonempty = true;
            }
            if (lk != rk) {
              match = false;
              break;
            }
          }
          if (any_nonempty && match) {
            for (size_t k = 0; k < un; ++k) {
              out.push_back(tokens[i + k]);
            }
            std::vector<std::string> base(un);
            for (size_t k = 0; k < un; ++k) {
              base[k] = norm_tok(tokens[i + k]);
            }
            i += 2 * un;
            while (i + un <= tokens.size()) {
              bool next_match = true;
              for (size_t k = 0; k < un; ++k) {
                if (norm_tok(tokens[i + k]) != base[k]) {
                  next_match = false;
                  break;
                }
              }
              if (next_match) {
                i += un;
              } else {
                break;
              }
            }
            changed = true;
            continue;
          }
        }
        out.push_back(tokens[i]);
        ++i;
      }
      if (changed) {
        tokens = std::move(out);
        break;
      }
    }
  }

  std::ostringstream oss;
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i > 0) {
      oss << ' ';
    }
    oss << tokens[i];
  }
  return trim(oss.str());
}

AsrResult WhisperAsr::transcribe(const std::vector<float> & audio_16k)
{
  AsrResult res;
  if (audio_16k.empty()) {
    res.reject_reason = "empty_audio";
    return res;
  }
  res.audio_duration_sec = static_cast<double>(audio_16k.size()) / 16000.0;

#ifdef HAVE_WHISPER_CPP
  if (!impl_->ready || impl_->ctx == nullptr) {
    res.reject_reason = "not_ready";
    return res;
  }

  const auto t0 = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);

  whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
  wparams.n_threads = std::max(1, config_.n_threads);
  wparams.beam_search.beam_size = std::max(1, config_.beam_size);
  wparams.language = config_.language.c_str();
  wparams.translate = false;
  wparams.no_context = true;
  wparams.single_segment = false;
  wparams.print_special = false;
  wparams.print_progress = false;
  wparams.print_realtime = false;
  wparams.print_timestamps = false;
  wparams.temperature = config_.temperature;
  wparams.temperature_inc = 0.0f;
  wparams.no_speech_thold = config_.no_speech_threshold;
  wparams.logprob_thold = config_.log_prob_threshold;
  wparams.entropy_thold = config_.entropy_threshold;
  wparams.suppress_blank = true;
  wparams.suppress_nst = true;
  wparams.initial_prompt = config_.initial_prompt.empty() ? nullptr : config_.initial_prompt.c_str();

  if (whisper_full(impl_->ctx, wparams, audio_16k.data(), static_cast<int>(audio_16k.size())) != 0) {
    res.reject_reason = "whisper_error";
    return res;
  }

  std::ostringstream joined;
  std::ostringstream all_raw;
  double max_no_speech = 0.0;
  double sum_seg_logprob = 0.0;
  int seg_score_count = 0;
  bool had_no_speech_reject = false;
  bool had_logprob_reject = false;

  const int n_segments = whisper_full_n_segments(impl_->ctx);
  const whisper_token eot_tok = whisper_token_eot(impl_->ctx);
  for (int i = 0; i < n_segments; ++i) {
    const float no_speech = whisper_full_get_segment_no_speech_prob(impl_->ctx, i);
    max_no_speech = std::max(max_no_speech, static_cast<double>(no_speech));

    const int n_tokens = whisper_full_n_tokens(impl_->ctx, i);
    double sum_logprob = 0.0;
    int text_tokens = 0;
    for (int j = 0; j < n_tokens; ++j) {
      const whisper_token_data td = whisper_full_get_token_data(impl_->ctx, i, j);
      if (td.id >= eot_tok) {
        continue;
      }
      sum_logprob += static_cast<double>(td.plog);
      ++text_tokens;
    }
    const double avg_logprob =
      (text_tokens > 0) ? (sum_logprob / static_cast<double>(text_tokens)) : 0.0;
    if (text_tokens > 0) {
      sum_seg_logprob += avg_logprob;
      ++seg_score_count;
    }

    const char * seg_text = whisper_full_get_segment_text(impl_->ctx, i);
    const std::string t = (seg_text != nullptr) ? trim(seg_text) : "";
    if (!t.empty()) {
      if (all_raw.tellp() > 0) {
        all_raw << ' ';
      }
      all_raw << t;
    }

    if (no_speech >= config_.no_speech_threshold) {
      had_no_speech_reject = true;
      continue;
    }
    if (text_tokens > 0 && avg_logprob < static_cast<double>(config_.log_prob_threshold)) {
      had_logprob_reject = true;
      continue;
    }

    if (!t.empty()) {
      if (joined.tellp() > 0) {
        joined << ' ';
      }
      joined << t;
    }
  }

  const auto t1 = std::chrono::steady_clock::now();
  res.latency_sec = std::chrono::duration<double>(t1 - t0).count();
  res.rtf = (res.audio_duration_sec > 0.0) ? (res.latency_sec / res.audio_duration_sec) : 0.0;
  res.raw_text = trim(all_raw.str());
  res.no_speech_prob = max_no_speech;
  if (seg_score_count > 0) {
    res.avg_logprob = sum_seg_logprob / static_cast<double>(seg_score_count);
    res.has_scores = true;
  }

  std::string raw_text = trim(joined.str());
  if (raw_text.empty()) {
    res.rejected_raw = res.raw_text;
    res.reject_reason = had_no_speech_reject
      ? "no_speech_prob"
      : (had_logprob_reject ? "low_logprob" : "empty_whisper");
    return res;
  }

  std::string deduped = deduplicateNgrams(raw_text, config_.dedup_max_ngram);
  const std::string cleaned = toLowerCleaned(deduped);

  if (deduped.empty()) {
    res.rejected_raw = res.raw_text;
    res.reject_reason = "empty_after_dedup";
  } else if (static_cast<int>(cleaned.size()) < std::max(1, config_.min_transcript_chars)) {
    res.rejected_raw = std::move(deduped);
    res.reject_reason = "min_chars";
  } else if (blacklist_.find(cleaned) != blacklist_.end()) {
    res.rejected_raw = std::move(deduped);
    res.reject_reason = "blacklist";
  } else {
    res.text = std::move(deduped);
    res.accepted = true;
    res.reject_reason = "ok";
  }
#else
  res.reject_reason = "not_ready";
#endif
  return res;
}

}  // namespace athome_hri_cpp
