import base64
import io
import os
import wave

try:
    from piper.voice import PiperVoice
    HAS_PIPER = True
except ImportError:
    HAS_PIPER = False


class TextToSpeech:
    """Modul khusus sintesis suara teks ke audio bytes via Piper TTS."""

    def __init__(self, model_filename: str = "en_US-lessac-medium.onnx"):
        self.piper = None
        if not HAS_PIPER:
            print("[TTS WARN] piper-tts tidak terinstal. TTS tidak aktif.")
            return

        model_path = self._find_model(model_filename)
        if model_path:
            try:
                print(f"[TTS] Memuat model Piper TTS dari: {model_path}")
                self.piper = PiperVoice.load(model_path)
                print("[TTS] Piper TTS siap.")
            except Exception as e:
                print(f"[TTS Error] Gagal memuat model Piper: {e}")
        else:
            print(f"[TTS WARN] Model '{model_filename}' tidak ditemukan di lokasi pencarian.")

    def _find_model(self, filename: str) -> str:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "model", filename),
            os.path.join(os.path.dirname(__file__), "model", filename),
            os.path.join(os.getcwd(), "src", "athome_hri", "model", filename),
            f"/home/yesaya/Documents/cluster_3/draft_program/athome-ws/src/athome_hri/model/{filename}",
        ]
        try:
            from ament_index_python.packages import get_package_share_directory
            candidates.append(os.path.join(get_package_share_directory("athome_hri"), "model", filename))
        except Exception:
            pass

        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def synthesize_wav_base64(self, text: str) -> str:
        """Menyintesis teks menjadi audio WAV dan mengembalikan data dalam string Base64."""
        if not self.piper or not text:
            return ""

        try:
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.piper.config.sample_rate)
                for chunk in self.piper.synthesize(text):
                    wav_file.writeframes(chunk.audio_int16_bytes)

            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode("utf-8")
        except Exception as e:
            print(f"[TTS Error] Sintesis audio gagal untuk '{text}': {e}")
            return ""

