"""Speech-to-Text (STT) engine module powered by Faster-Whisper."""

import logging
import os
import tempfile
from typing import Optional

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


class SpeechToText:
    """Modul khusus transkripsi audio raw (.wav/.webm) ke teks via Faster-Whisper."""

    DEFAULT_BEAM_SIZE: int = 5
    DEFAULT_VAD_SILENCE_MS: int = 400

    def __init__(self, model_size: str = "base.en", language: str = "en") -> None:
        self.model_size = model_size
        self.language = language
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
                    "GPU NVIDIA terdeteksi (%d device). Menggunakan CUDA + FP16.",
                    cuda_count,
                )
                return "cuda", "float16"
            logger.info("GPU tidak terdeteksi. Menggunakan CPU (int8).")
        except Exception:
            logger.info("CTranslate2 fallback ke CPU (int8).")
        return "cpu", "int8"

    def _load_model(self, device: str, compute_type: str) -> None:
        """Memuat model Faster-Whisper dengan fallback otomatis ke CPU bila gagal."""
        try:
            logger.info(
                "Memuat model Faster-Whisper '%s' (%s, %s)...",
                self.model_size,
                device,
                compute_type,
            )
            self.whisper = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=4 if device == "cpu" else 1,
            )
            logger.info("Faster-Whisper siap.")
        except Exception as exc:
            logger.error("Gagal memuat Faster-Whisper (%s): %s", device, exc)
            if device == "cuda":
                try:
                    logger.info("Mencoba fallback ke CPU (int8)...")
                    self.whisper = WhisperModel(
                        self.model_size,
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=4,
                    )
                    logger.info("Faster-Whisper CPU siap.")
                except Exception as fallback_exc:
                    logger.error("Fallback CPU juga gagal: %s", fallback_exc)

    def transcribe(self, audio_bytes: bytes) -> str:
        """Menerima raw audio bytes (.wav atau .webm) dan mengembalikan teks transkripsi."""
        if not self.whisper or not audio_bytes:
            return ""

        suffix = ".wav" if audio_bytes[:4] == b"RIFF" else ".webm"
        temp_path: Optional[str] = None

        try:
            with tempfile.NamedTemporaryFile(
                prefix="hri_stt_", suffix=suffix, delete=False
            ) as tmp_file:
                tmp_file.write(audio_bytes)
                temp_path = tmp_file.name

            segments, _ = self.whisper.transcribe(
                temp_path,
                beam_size=self.DEFAULT_BEAM_SIZE,
                language=self.language,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": self.DEFAULT_VAD_SILENCE_MS},
            )
            return " ".join(seg.text for seg in segments).strip()
        except Exception as exc:
            logger.error("Transkripsi gagal: %s", exc)
            return ""
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
