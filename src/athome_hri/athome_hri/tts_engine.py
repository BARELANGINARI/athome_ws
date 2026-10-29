"""Text-to-Speech (TTS) engine module powered by Piper TTS."""

import base64
import io
import logging
import os
from pathlib import Path
from typing import Optional
import wave

logger = logging.getLogger("athome_hri.tts")

try:
    from piper.voice import PiperVoice

    HAS_PIPER = True
except ImportError:
    HAS_PIPER = False


class TextToSpeech:
    """Modul khusus sintesis suara teks ke audio WAV (Base64) via Piper TTS."""

    def __init__(self, model_filename: str = "en_US-lessac-medium.onnx") -> None:
        self.piper: Optional["PiperVoice"] = None
        if not HAS_PIPER:
            logger.warning("piper-tts tidak terinstal. TTS tidak aktif.")
            return

        model_path = self._find_model(model_filename)
        if not model_path:
            logger.warning(
                "Model '%s' tidak ditemukan di lokasi pencarian.", model_filename
            )
            return

        try:
            logger.info("Memuat model Piper TTS dari: %s", model_path)
            self.piper = PiperVoice.load(model_path)
            logger.info("Piper TTS siap.")
        except Exception as exc:
            logger.error("Gagal memuat model Piper: %s", exc)

    @staticmethod
    def _find_model(filename: str) -> Optional[str]:
        """Mencari path file model ONNX melalui ament_index atau direktori relatif paket."""
        candidates: list[Path] = []

        try:
            from ament_index_python.packages import get_package_share_directory

            share_dir = Path(get_package_share_directory("athome_hri"))
            candidates.append(share_dir / "model" / filename)
        except Exception:
            pass

        module_dir = Path(__file__).resolve().parent
        candidates.extend(
            [
                module_dir.parent / "model" / filename,
                module_dir / "model" / filename,
                Path.cwd() / "src" / "athome_hri" / "model" / filename,
            ]
        )

        for path in candidates:
            if path.is_file():
                return str(path)
        return None

    def synthesize_wav_base64(self, text: str) -> str:
        """Menyintesis teks menjadi audio WAV mono 16-bit dan mengembalikan string Base64."""
        if not self.piper or not text or not text.strip():
            return ""

        try:
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.piper.config.sample_rate)
                for chunk in self.piper.synthesize(text):
                    wav_file.writeframes(chunk.audio_int16_bytes)

            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as exc:
            logger.error("Sintesis audio gagal untuk '%s': %s", text, exc)
            return ""
