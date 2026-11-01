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
    """Text-to-speech synthesis module to 16-bit mono WAV (Base64) via Piper TTS."""

    def __init__(self, model_filename: str = "en_US-lessac-medium.onnx") -> None:
        self.piper: Optional["PiperVoice"] = None
        if not HAS_PIPER:
            logger.warning("piper-tts is not installed. TTS disabled.")
            return

        model_path = self._find_model(model_filename)
        if not model_path:
            logger.warning(
                "Model '%s' not found in search paths.", model_filename
            )
            return

        try:
            logger.info("Loading Piper TTS model from: %s", model_path)
            self.piper = PiperVoice.load(model_path)
            logger.info("Piper TTS ready.")
        except Exception as exc:
            logger.error("Failed to load Piper model: %s", exc)

    @staticmethod
    def _find_model(filename: str) -> Optional[str]:
        """Search ONNX model file path via ament_index or relative package directory."""
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
        """Synthesize text into mono 16-bit WAV audio and return Base64 string."""
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
                # Tambahkan trailing silence (300 ms) agar suku kata terakhir tidak terpotong hardware buffer
                trailing_silence = b"\x00" * int(self.piper.config.sample_rate * 2 * 0.3)
                wav_file.writeframes(trailing_silence)

            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as exc:
            logger.error("Audio synthesis failed for '%s': %s", text, exc)
            return ""
