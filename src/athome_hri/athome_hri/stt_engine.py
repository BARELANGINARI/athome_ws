import logging
import os
import re
import tempfile
import threading
import time
from typing import Callable, Iterable, Optional, Union

import numpy as np

from athome_hri.audio_enhancer import AudioEnhancer
from athome_hri.config import load_default_hri_params

logger = logging.getLogger("athome_hri.stt")

# PyAV & Faster-Whisper compatibility patch
try:
    import av

    _orig_av_open = av.open

    def _patched_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)

    av.open = _patched_av_open
except Exception:
    pass

try:
    from faster_whisper import WhisperModel

    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

try:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    HAS_NEURAL_VAD = True
except ImportError:
    HAS_NEURAL_VAD = False

AsrResultCallback = Callable[[str, str], None]


class SpeechToText:
    """
    Modul khusus transkripsi audio (numpy float32 array atau .wav/.webm bytes)
    ke teks via Faster-Whisper dengan thread-safety dan proteksi halusinasi.
    Seluruh parameter bawaan dimuat dari config/params.yaml.
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        language: Optional[str] = None,
        on_asr_result: Optional[AsrResultCallback] = None,
        fallback_model_size: Optional[str] = None,
        sample_rate: Optional[int] = None,
        beam_size: Optional[int] = None,
        no_speech_threshold: Optional[float] = None,
        log_prob_threshold: Optional[float] = None,
        compression_ratio_threshold: Optional[float] = None,
        vad_threshold: Optional[float] = None,
        vad_min_speech_ms: Optional[int] = None,
        vad_silence_ms: Optional[int] = None,
        vad_speech_pad_ms: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        temperature: Optional[float] = None,
        denoiser_backend: Optional[str] = None,
        highpass_hz: Optional[float] = None,
        dedup_max_ngram: Optional[int] = None,
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
        hallucination_blacklist: Optional[Iterable[str]] = None,
    ) -> None:
        defaults = load_default_hri_params("stt_engine")
        self.model_size = str(defaults["whisper_model"] if model_size is None else model_size)
        self.fallback_model_size = str(
            defaults["stt_fallback_model"] if fallback_model_size is None else fallback_model_size
        )
        self.language = str(defaults["stt_language"] if language is None else language)
        self.sample_rate = int(defaults["stt_sample_rate"] if sample_rate is None else sample_rate)
        self.on_asr_result = on_asr_result
        self.beam_size = int(defaults["stt_beam_size"] if beam_size is None else beam_size)
        self.no_speech_threshold = float(
            defaults["stt_no_speech_threshold"]
            if no_speech_threshold is None
            else no_speech_threshold
        )
        self.log_prob_threshold = float(
            defaults["stt_log_prob_threshold"]
            if log_prob_threshold is None
            else log_prob_threshold
        )
        self.compression_ratio_threshold = float(
            defaults["stt_compression_ratio_threshold"]
            if compression_ratio_threshold is None
            else compression_ratio_threshold
        )
        self.vad_threshold = float(
            defaults["stt_vad_threshold"] if vad_threshold is None else vad_threshold
        )
        self.vad_min_speech_ms = int(
            defaults["stt_vad_min_speech_ms"]
            if vad_min_speech_ms is None
            else vad_min_speech_ms
        )
        self.vad_silence_ms = int(
            defaults["stt_vad_silence_ms"] if vad_silence_ms is None else vad_silence_ms
        )
        self.vad_speech_pad_ms = int(
            defaults["stt_vad_speech_pad_ms"]
            if vad_speech_pad_ms is None
            else vad_speech_pad_ms
        )
        self.repetition_penalty = float(
            defaults["stt_repetition_penalty"]
            if repetition_penalty is None
            else repetition_penalty
        )
        self.no_repeat_ngram_size = int(
            defaults["stt_no_repeat_ngram_size"]
            if no_repeat_ngram_size is None
            else no_repeat_ngram_size
        )
        self.temperature = float(
            defaults["stt_temperature"] if temperature is None else temperature
        )
        self.denoiser_backend = str(
            defaults["stt_denoiser_backend"]
            if denoiser_backend is None
            else denoiser_backend
        )
        self.highpass_hz = float(
            defaults["stt_highpass_hz"] if highpass_hz is None else highpass_hz
        )
        self.dedup_max_ngram = max(
            1,
            int(
                defaults["stt_dedup_max_ngram"]
                if dedup_max_ngram is None
                else dedup_max_ngram
            ),
        )
        self.initial_prompt = str(
            defaults["stt_initial_prompt"] if initial_prompt is None else initial_prompt
        )
        self.hotwords = str(defaults["stt_hotwords"] if hotwords is None else hotwords)
        self.set_hallucination_blacklist(
            hallucination_blacklist
            if hallucination_blacklist is not None
            else defaults["stt_hallucination_blacklist"]
        )
        self.enhancer = AudioEnhancer(
            backend=self.denoiser_backend,
            sample_rate=self.sample_rate,
            highpass_hz=self.highpass_hz,
        )
        self.whisper: Optional["WhisperModel"] = None
        self._lock = threading.Lock()

        if not HAS_WHISPER:
            logger.warning("faster_whisper is not installed. STT disabled.")
            return

        device, compute_type = self._detect_compute_backend()
        self._load_model(device, compute_type)

    @staticmethod
    def _detect_compute_backend() -> tuple[str, str]:
        """Detect CUDA GPU acceleration availability for CTranslate2."""
        try:
            import ctranslate2

            cuda_count = ctranslate2.get_cuda_device_count()
            if cuda_count > 0:
                logger.info(
                    "NVIDIA GPU detected (%d device). Using CUDA + int8_float16.",
                    cuda_count,
                )
                return "cuda", "int8_float16"
            logger.info("No GPU detected. Using CPU (int8).")
        except Exception:
            logger.info("CTranslate2 fallback to CPU (int8).")
        return "cpu", "int8"

    def set_hallucination_blacklist(self, items: Iterable[str]) -> None:
        self.hallucination_blacklist = {
            str(item).strip().lower() for item in items if str(item).strip()
        }

    def update_enhancer(
        self,
        *,
        backend: Optional[str] = None,
        highpass_hz: Optional[float] = None,
        sample_rate: Optional[int] = None,
    ) -> None:
        if backend is not None:
            self.denoiser_backend = str(backend)
        if highpass_hz is not None:
            self.highpass_hz = float(highpass_hz)
        if sample_rate is not None:
            self.sample_rate = int(sample_rate)
        self.enhancer.update_config(
            backend=backend,
            highpass_hz=highpass_hz,
            sample_rate=sample_rate,
        )

    @staticmethod
    def deduplicate_ngrams(text: str, max_ngram: int = 6) -> str:
        """Remove consecutive identical word/phrase n-gram repetitions."""
        tokens = text.split()
        if len(tokens) < 2:
            return text.strip()

        def _norm(tok: str) -> str:
            return re.sub(r"[^\w]+", "", tok.lower())

        changed = True
        while changed:
            changed = False
            n_max = min(max_ngram, len(tokens) // 2)
            for n in range(n_max, 0, -1):
                out = []
                i = 0
                while i < len(tokens):
                    if i + 2 * n <= len(tokens):
                        left = [_norm(t) for t in tokens[i : i + n]]
                        right = [_norm(t) for t in tokens[i + n : i + 2 * n]]
                        if any(left) and left == right:
                            out.extend(tokens[i : i + n])
                            i += 2 * n
                            while i + n <= len(tokens):
                                nxt = [_norm(t) for t in tokens[i : i + n]]
                                if nxt == left:
                                    i += n
                                else:
                                    break
                            changed = True
                            continue
                    out.append(tokens[i])
                    i += 1
                if changed:
                    tokens = out
                    break
        return " ".join(tokens).strip()

    def _apply_neural_vad(self, audio: np.ndarray) -> np.ndarray:
        """Run strict CPU Silero Neural VAD gate before Whisper decoder."""
        if not HAS_NEURAL_VAD or audio.size == 0:
            return audio
        try:
            vad_opts = VadOptions(
                threshold=self.vad_threshold,
                min_speech_duration_ms=self.vad_min_speech_ms,
                min_silence_duration_ms=self.vad_silence_ms,
                speech_pad_ms=self.vad_speech_pad_ms,
            )
            speech_chunks = get_speech_timestamps(
                audio, vad_opts, sampling_rate=self.sample_rate
            )
            if not speech_chunks:
                return np.empty(0, dtype=np.float32)
            return np.concatenate(
                [audio[c["start"] : c["end"]] for c in speech_chunks]
            ).astype(np.float32)
        except Exception as exc:
            logger.debug("Neural VAD fallback ke internal Whisper VAD: %s", exc)
            return audio

    def _load_model(self, device: str, compute_type: str) -> None:
        """Load Faster-Whisper model with fallback to CPU or fallback_model_size on failure."""
        candidates = [(self.model_size, device, compute_type)]
        if device == "cuda":
            candidates.append((self.model_size, "cuda", "float16"))
            candidates.append((self.model_size, "cpu", "int8"))
        if self.fallback_model_size and self.model_size != self.fallback_model_size:
            candidates.append(
                (
                    self.fallback_model_size,
                    device,
                    "float16" if device == "cuda" else "int8",
                )
            )
            if device == "cuda":
                candidates.append((self.fallback_model_size, "cpu", "int8"))

        for model_name, dev, comp in candidates:
            try:
                logger.info(
                    "Loading Faster-Whisper model '%s' (%s, %s)...",
                    model_name,
                    dev,
                    comp,
                )
                self.whisper = WhisperModel(
                    model_name,
                    device=dev,
                    compute_type=comp,
                    cpu_threads=4 if dev == "cpu" else 1,
                )
                self.model_size = model_name
                logger.info("Faster-Whisper ready ('%s' on %s).", model_name, dev)
                return
            except Exception as exc:
                logger.warning(
                    "Failed to load Faster-Whisper '%s' (%s, %s): %s",
                    model_name,
                    dev,
                    comp,
                    exc,
                )

        logger.error("All Faster-Whisper model loading attempts failed.")

    def transcribe(self, audio_input: Union[np.ndarray, bytes]) -> str:
        """
        Menerima array float32 numpy (16 kHz) dari mikrofon lokal atau raw bytes (.wav/.webm)
        dari WebSocket, lalu mengembalikan teks transkripsi yang telah difilter secara thread-safe.
        """
        if self.whisper is None or audio_input is None:
            return ""
        if isinstance(audio_input, np.ndarray) and audio_input.size == 0:
            return ""
        if isinstance(audio_input, (bytes, bytearray)) and len(audio_input) == 0:
            return ""

        temp_path: Optional[str] = None
        t0 = time.perf_counter()
        dur = 0.0

        try:
            neural_vad_applied = False
            if isinstance(audio_input, np.ndarray):
                # 1. Denoising Frontend (CPU-only: DeepFilterNet2 / RNNoise / Spectral + HPF)
                enhanced_audio = self.enhancer.enhance(
                    audio_input.astype(np.float32), self.sample_rate
                )
                # 2. Neural VAD ketat (Silero ONNX CPU) sebelum decoder Whisper
                if HAS_NEURAL_VAD:
                    gated_audio = self._apply_neural_vad(enhanced_audio)
                    if gated_audio.size == 0:
                        logger.debug("[ASR] Rejected by Neural VAD (non-speech/noise).")
                        return ""
                    audio_source = gated_audio
                    neural_vad_applied = True
                else:
                    audio_source = enhanced_audio
                dur = len(audio_source) / float(self.sample_rate)
            else:
                suffix = ".wav" if audio_input[:4] == b"RIFF" else ".webm"
                with tempfile.NamedTemporaryFile(
                    prefix="hri_stt_", suffix=suffix, delete=False
                ) as tmp_file:
                    tmp_file.write(audio_input)
                    temp_path = tmp_file.name
                audio_source = temp_path

            transcribe_kwargs = {
                "beam_size": self.beam_size,
                "language": self.language,
                "initial_prompt": self.initial_prompt or None,
                "condition_on_previous_text": False,
                "temperature": self.temperature,
                "no_speech_threshold": self.no_speech_threshold,
                "log_prob_threshold": self.log_prob_threshold,
                "compression_ratio_threshold": self.compression_ratio_threshold,
                "vad_filter": not neural_vad_applied,
                "vad_parameters": {
                    "threshold": self.vad_threshold,
                    "min_speech_duration_ms": self.vad_min_speech_ms,
                    "min_silence_duration_ms": self.vad_silence_ms,
                    "speech_pad_ms": self.vad_speech_pad_ms,
                },
            }
            try:
                import inspect
                sig_params = inspect.signature(self.whisper.transcribe).parameters
                if self.hotwords and "hotwords" in sig_params:
                    transcribe_kwargs["hotwords"] = self.hotwords
                if "repetition_penalty" in sig_params:
                    transcribe_kwargs["repetition_penalty"] = self.repetition_penalty
                if "no_repeat_ngram_size" in sig_params:
                    transcribe_kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size
            except Exception:
                pass

            with self._lock:
                segments, _ = self.whisper.transcribe(
                    audio_source,
                    **transcribe_kwargs,
                )

                # Filter ketat: no_speech rendah, logprob memadai, dan bebas loop (compression_ratio)
                valid_text = [
                    seg.text.strip()
                    for seg in segments
                    if (
                        seg.no_speech_prob < self.no_speech_threshold
                        and seg.avg_logprob >= self.log_prob_threshold
                        and seg.compression_ratio < self.compression_ratio_threshold
                    )
                ]

            text = " ".join(valid_text).strip()
            # 4. Post-processing deduplikasi n-gram (buang pengulangan kata/frasa identik)
            if text:
                text = self.deduplicate_ngrams(text, self.dedup_max_ngram)

            proc_time = time.perf_counter() - t0
            rtf = (proc_time / dur) if dur > 0 else 0.0

            cleaned_check = text.lower().rstrip(".!,?")
            if (
                text
                and cleaned_check not in self.hallucination_blacklist
                and len(cleaned_check) > 2
            ):
                metrics_str = (
                    f"RTF: {rtf:.3f} | Latency: {proc_time:.2f}s | Audio: {dur:.2f}s"
                    if dur > 0
                    else f"Latency: {proc_time:.2f}s"
                )
                logger.info('[ASR] "%s" (%s)', text, metrics_str)
                if self.on_asr_result:
                    self.on_asr_result(f'"{text}"', metrics_str)
                return text

            if text:
                logger.info(
                    "[ASR] Rejected - noise/hallucination Raw: %s (%.2fs)",
                    text,
                    proc_time,
                )
                if self.on_asr_result:
                    self.on_asr_result(
                        f"[Rejected: {text}]",
                        f"Noise/Hallucination ({proc_time:.2f}s)",
                    )
            return ""
        except Exception as exc:
            logger.error("Transcription failed: %s", exc)
            return ""
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
