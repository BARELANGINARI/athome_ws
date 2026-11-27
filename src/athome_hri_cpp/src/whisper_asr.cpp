#include "athome_hri_cpp/whisper_asr.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

#ifdef HAVE_WHISPER_CPP
#include <ggml-backend.h>
#include <whisper.h>
#endif

namespace athome_hri_cpp
{

namespace
{
#ifdef HAVE_WHISPER_CPP
struct WhisperLogCapture
{
  bool capturing{false};
  std::string buffer;
};

std::mutex g_whisper_log_mutex;
WhisperLogCapture g_whisper_log_capture;

void whisperLogCallback(enum ggml_log_level level, const char * text, void * /*user_data*/)
{
  if (text == nullptr) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(g_whisper_log_mutex);
    if (g_whisper_log_capture.capturing) {
      g_whisper_log_capture.buffer.append(text);
    }
  }
  if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
    std::cerr << "[whisper.cpp] " << text;
  }
}

std::string toLowerCopy(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

bool containsAny(const std::string & haystack, std::initializer_list<const char *> needles)
{
  for (const char * needle : needles) {
    if (haystack.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}
#endif
}  // namespace

struct WhisperAsr::Impl
{
#ifdef HAVE_WHISPER_CPP
  whisper_context * ctx{nullptr};
#endif
  bool ready{false};
  bool using_gpu{false};
  bool cuda_compiled{false};
  bool flash_attn_active{false};
  std::string system_info;
  std::string backend_summary{"uninitialized"};
};

WhisperAsr::WhisperAsr(const WhisperAsrConfig & config)
: impl_(std::make_unique<Impl>()), config_(config)
{
  setBlacklistLocked(config_.hallucination_blacklist);

#ifdef HAVE_WHISPER_CPP
  if (config_.model_path.empty()) {
    std::cerr << "[WhisperAsr] Empty model_path.\n";
    impl_->backend_summary = "not ready (empty model_path)";
    return;
  }
  std::ifstream ifs(config_.model_path, std::ios::binary);
  if (!ifs.good()) {
    std::cerr << "[WhisperAsr] Model file not found: " << config_.model_path << "\n";
    impl_->backend_summary = "not ready (model file not found)";
    return;
  }

  static std::once_flag backend_init_flag;
  std::call_once(backend_init_flag, []() {
    ggml_backend_load_all();
  });

  const char * sys_info_cstr = whisper_print_system_info();
  impl_->system_info = (sys_info_cstr != nullptr) ? sys_info_cstr : "";

  const ggml_backend_reg_t cuda_reg = ggml_backend_reg_by_name("CUDA");
  ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
#ifdef HAVE_WHISPER_CUDA
  const bool cmake_cuda = true;
#else
  const bool cmake_cuda = false;
#endif
  impl_->cuda_compiled =
    cmake_cuda || (cuda_reg != nullptr) || (impl_->system_info.find("CUDA") != std::string::npos);

  const bool gpu_backend_present = (cuda_reg != nullptr) && (gpu_dev != nullptr);
  std::string gpu_label = "none";
  size_t vram_free_before = 0;
  size_t vram_total = 0;
  if (gpu_dev != nullptr) {
    const char * dname = ggml_backend_dev_name(gpu_dev);
    const char * ddesc = ggml_backend_dev_description(gpu_dev);
    gpu_label = std::string(dname ? dname : "GPU") + " (" + std::string(ddesc ? ddesc : "unknown") + ")";
    ggml_backend_dev_memory(gpu_dev, &vram_free_before, &vram_total);
  }

  std::string fallback_reason;
  whisper_context_params cparams = whisper_context_default_params();
  if (config_.use_gpu && !gpu_backend_present) {
    if (!impl_->cuda_compiled || cuda_reg == nullptr) {
      fallback_reason = "whisper.cpp compiled without GGML_CUDA=1 (no CUDA backend registered)";
    } else {
      fallback_reason = "no CUDA GPU device detected by ggml";
    }
    std::cerr << "[WhisperAsr] WARNING: whisper_use_gpu=true requested, but "
              << fallback_reason << ". Running on CPU (threads=" << config_.n_threads << ").\n";
    cparams.use_gpu = false;
    cparams.flash_attn = false;
  } else {
    cparams.use_gpu = config_.use_gpu;
    cparams.flash_attn = config_.use_gpu && config_.flash_attn;
  }

  auto initWithCapturedLogs = [&](const whisper_context_params & params, std::string & out_logs) {
    {
      std::lock_guard<std::mutex> lock(g_whisper_log_mutex);
      g_whisper_log_capture.buffer.clear();
      g_whisper_log_capture.capturing = true;
    }
    whisper_log_set(whisperLogCallback, nullptr);
    whisper_context * ctx = whisper_init_from_file_with_params(config_.model_path.c_str(), params);
    {
      std::lock_guard<std::mutex> lock(g_whisper_log_mutex);
      g_whisper_log_capture.capturing = false;
      out_logs = g_whisper_log_capture.buffer;
    }
    return ctx;
  };

  std::string init_logs;
  impl_->ctx = initWithCapturedLogs(cparams, init_logs);

  if (impl_->ctx == nullptr && cparams.use_gpu && cparams.flash_attn) {
    std::cerr << "[WhisperAsr] GPU init with flash_attn=true failed, retrying with flash_attn=false...\n";
    cparams.flash_attn = false;
    impl_->ctx = initWithCapturedLogs(cparams, init_logs);
  }

  if (impl_->ctx == nullptr && cparams.use_gpu) {
    std::cerr << "[WhisperAsr] GPU init failed, retrying on CPU...\n";
    fallback_reason = "whisper_init_from_file_with_params() failed with use_gpu=true";
    cparams.use_gpu = false;
    cparams.flash_attn = false;
    impl_->ctx = initWithCapturedLogs(cparams, init_logs);
  }

  if (impl_->ctx == nullptr) {
    impl_->backend_summary = "init failed";
    return;
  }

  impl_->ready = true;
  if (cparams.use_gpu && gpu_dev != nullptr) {
    size_t vram_free_after = 0;
    ggml_backend_dev_memory(gpu_dev, &vram_free_after, &vram_total);
    const double free_before_mb = static_cast<double>(vram_free_before) / 1048576.0;
    const double free_after_mb = static_cast<double>(vram_free_after) / 1048576.0;
    const double total_mb = static_cast<double>(vram_total) / 1048576.0;
    const double delta_mb = (vram_free_before > vram_free_after)
      ? static_cast<double>(vram_free_before - vram_free_after) / 1048576.0
      : 0.0;

    const std::string lower_logs = toLowerCopy(init_logs);
    const bool saw_gpu_failure = containsAny(lower_logs, {
      "no gpu found",
      "failed to initialize",
      "failed to allocate",
      "falling back to cpu",
      "using cpu backend",
    });
    const bool saw_cuda_backend =
      (lower_logs.find("whisper_backend_init_gpu: using ") != std::string::npos) &&
      (lower_logs.find("whisper_backend_init_gpu: using cpu") == std::string::npos);

    const bool gpu_verified = !saw_gpu_failure && (saw_cuda_backend || delta_mb >= 50.0);
    if (gpu_verified) {
      impl_->using_gpu = true;
      impl_->flash_attn_active = cparams.flash_attn;
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(0)
          << gpu_label << ", flash_attn=" << (impl_->flash_attn_active ? "ON" : "OFF")
          << ", VRAM free " << free_after_mb << "/" << total_mb << " MB"
          << " (model ~" << delta_mb << " MB)";
      impl_->backend_summary = oss.str();
    } else {
      impl_->using_gpu = false;
      impl_->flash_attn_active = false;
      std::ostringstream rsn;
      rsn << std::fixed << std::setprecision(0)
          << "whisper.cpp silently fell back to CPU (VRAM free "
          << free_before_mb << "->" << free_after_mb << "/" << total_mb << " MB)";
      fallback_reason = rsn.str();
      std::cerr << "[WhisperAsr] WARNING: " << fallback_reason << "\n";
    }
  } else {
    impl_->using_gpu = false;
    impl_->flash_attn_active = false;
  }

  if (!impl_->using_gpu) {
    std::ostringstream oss;
    oss << "CPU (threads=" << std::max(1, config_.n_threads)
        << ", cuda_compiled=" << (impl_->cuda_compiled ? "true" : "false");
    if (!fallback_reason.empty()) {
      oss << "; " << fallback_reason;
    }
    oss << ")";
    impl_->backend_summary = oss.str();
  }
#else
  std::cerr << "[WhisperAsr] Built without whisper.cpp (HAVE_WHISPER_CPP not defined).\n";
  impl_->backend_summary = "unavailable (HAVE_WHISPER_CPP not defined)";
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

bool WhisperAsr::cudaCompiled() const
{
  return impl_->cuda_compiled;
}

bool WhisperAsr::flashAttnActive() const
{
  return impl_->flash_attn_active;
}

std::string WhisperAsr::backendSummary() const
{
  return impl_->backend_summary;
}

std::string WhisperAsr::systemInfo() const
{
  return impl_->system_info;
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

  whisper_reset_timings(impl_->ctx);
  if (whisper_full(impl_->ctx, wparams, audio_16k.data(), static_cast<int>(audio_16k.size())) != 0) {
    res.reject_reason = "whisper_error";
    return res;
  }
  if (const whisper_timings * tm = whisper_get_timings(impl_->ctx)) {
    res.encode_ms = static_cast<double>(tm->encode_ms);
    res.decode_ms = static_cast<double>(tm->decode_ms);
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
