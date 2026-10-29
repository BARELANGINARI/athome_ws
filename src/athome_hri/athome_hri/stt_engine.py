"""Speech-to-Text (STT) engine module powered by Faster-Whisper."""

import logging
import os
import tempfile
import time
from typing import Callable, Optional, Set, Union

import numpy as np

logger = logging.getLogger("athome_hri.stt")

# Patch kompatibilitas PyAV & Faster-Whisper
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

AsrResultCallback = Callable[[str, str], None]


class SpeechToText:
    """
    Modul khusus transkripsi audio (numpy float32 array atau .wav/.webm bytes)
    ke teks via Faster-Whisper (identik dengan utilitas/record_stream_visualize.py).
    """

    DEFAULT_MODEL_SIZE: str = "deepdml/faster-whisper-large-v3-turbo-ct2"
    FALLBACK_MODEL_SIZE: str = "base.en"
    SAMPLE_RATE: int = 16000
    DEFAULT_BEAM_SIZE: int = 2
    DEFAULT_NO_SPEECH_THRESHOLD: float = 0.5
    DEFAULT_LOG_PROB_THRESHOLD: float = -0.8
    DEFAULT_COMPRESSION_RATIO_THRESHOLD: float = 1.8
    DEFAULT_VAD_SILENCE_MS: int = 400
    DEFAULT_VAD_SPEECH_PAD_MS: int = 200

    HALLUCINATION_BLACKLIST: Set[str] = {
        "terima kasih",
        "terimakasih",
        "thank you",
        "thanks",
    }

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        language: str = "en",
        on_asr_result: Optional[AsrResultCallback] = None,
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.on_asr_result = on_asr_result
        self.whisper: Optional["WhisperModel"] = None

        if not HAS_WHISPER:
            logger.warning("faster_whisper tidak terinstal. STT tidak aktif.")
            return

        device, compute_type = self._detect_compute_backend()
        self._load_model(device, compute_type)

    @staticmethod
    def _detect_compute_backend() -> tuple[str, str]:
        """Mendeteksi ketersediaan akselerasi GPU CUDA untuk CTranslate2."""
        try:
            import ctranslate2

            cuda_count = ctranslate2.get_cuda_device_count()
            if cuda_count > 0:
                logger.info(
                    "GPU NVIDIA terdeteksi (%d device). Menggunakan CUDA + int8_float16.",
                    cuda_count,
                )
                return "cuda", "int8_float16"
            logger.info("GPU tidak terdeteksi. Menggunakan CPU (int8).")
        except Exception:
            logger.info("CTranslate2 fallback ke CPU (int8).")
        return "cpu", "int8"

    def _load_model(self, device: str, compute_type: str) -> None:
        """Memuat model Faster-Whisper dengan fallback otomatis ke CPU atau base.en bila gagal."""
        candidates = [(self.model_size, device, compute_type)]
        if device == "cuda":
            candidates.append((self.model_size, "cuda", "float16"))
            candidates.append((self.model_size, "cpu", "int8"))
        if self.model_size != self.FALLBACK_MODEL_SIZE:
            candidates.append(
                (
                    self.FALLBACK_MODEL_SIZE,
                    device,
                    "float16" if device == "cuda" else "int8",
                )
            )
            if device == "cuda":
                candidates.append((self.FALLBACK_MODEL_SIZE, "cpu", "int8"))

        for model_name, dev, comp in candidates:
            try:
                logger.info(
                    "Memuat model Faster-Whisper '%s' (%s, %s)...",
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
                logger.info("Faster-Whisper siap ('%s' pada %s).", model_name, dev)
                return
            except Exception as exc:
                logger.warning(
                    "Gagal memuat Faster-Whisper '%s' (%s, %s): %s",
                    model_name,
                    dev,
                    comp,
                    exc,
                )

        logger.error("Semua percobaan pemuatan model Faster-Whisper gagal.")

    def transcribe(self, audio_input: Union[np.ndarray, bytes]) -> str:
        """
        Menerima array float32 numpy (16 kHz) dari mikrofon lokal atau raw bytes (.wav/.webm)
        dari WebSocket, lalu mengembalikan teks transkripsi yang telah difilter.
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
            if isinstance(audio_input, np.ndarray):
                audio_source: Union[np.ndarray, str] = audio_input.astype(np.float32)
                dur = len(audio_source) / float(self.SAMPLE_RATE)
            else:
                suffix = ".wav" if audio_input[:4] == b"RIFF" else ".webm"
                with tempfile.NamedTemporaryFile(
                    prefix="hri_stt_", suffix=suffix, delete=False
                ) as tmp_file:
                    tmp_file.write(audio_input)
                    temp_path = tmp_file.name
                audio_source = temp_path

            segments, _ = self.whisper.transcribe(
                audio_source,
                beam_size=self.DEFAULT_BEAM_SIZE,
                language=self.language,
                condition_on_previous_text=True,
                no_speech_threshold=self.DEFAULT_NO_SPEECH_THRESHOLD,
                log_prob_threshold=self.DEFAULT_LOG_PROB_THRESHOLD,
                compression_ratio_threshold=self.DEFAULT_COMPRESSION_RATIO_THRESHOLD,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": self.DEFAULT_VAD_SILENCE_MS,
                    "speech_pad_ms": self.DEFAULT_VAD_SPEECH_PAD_MS,
                },
            )

            valid_text = [
                seg.text.strip()
                for seg in segments
                if seg.no_speech_prob < self.DEFAULT_NO_SPEECH_THRESHOLD
            ]
            text = " ".join(valid_text).strip()

            proc_time = time.perf_counter() - t0
            rtf = (proc_time / dur) if dur > 0 else 0.0

            cleaned_check = text.lower().rstrip(".!,?")
            if (
                text
                and cleaned_check not in self.HALLUCINATION_BLACKLIST
                and len(text) > 2
            ):
                metrics_str = (
                    f"RTF: {rtf:.3f} | Latensi: {proc_time:.2f}s | Audio: {dur:.2f}s"
                    if dur > 0
                    else f"Latensi: {proc_time:.2f}s"
                )
                logger.info('[ASR] "%s" (%s)', text, metrics_str)
                if self.on_asr_result:
                    self.on_asr_result(f'"{text}"', metrics_str)
                return text

            if text:
                logger.info(
                    "[ASR] Ditolak - noise/halusinasi Raw: %s (%.2fs)",
                    text,
                    proc_time,
                )
                if self.on_asr_result:
                    self.on_asr_result(
                        f"[Ditolak: {text}]",
                        f"Noise/Halusinasi ({proc_time:.2f}s)",
                    )
            return ""
        except Exception as exc:
            logger.error("Transkripsi gagal: %s", exc)
            return ""
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
