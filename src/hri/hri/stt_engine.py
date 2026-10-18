"""
Speech-to-Text (STT) Engine using Faster-Whisper.
Dioptimalkan untuk GPU NVIDIA (CUDA FP16) dengan fallback CPU (int8).
"""

import os

# Patch kompatibilitas PyAV & Faster-Whisper
try:
    import av
    orig_av_open = av.open

    def patched_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return orig_av_open(*args, **kwargs)

    av.open = patched_av_open
except Exception:
    pass

try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False


class SpeechToText:
    """Modul khusus transkripsi audio raw/webm ke teks via Faster-Whisper."""

    def __init__(self, model_size: str = "base.en"):
        self.whisper = None
        if not HAS_WHISPER:
            print("[STT WARN] faster_whisper tidak terinstal. STT tidak aktif.")
            return

        device = "cpu"
        compute_type = "int8"
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                device = "cuda"
                compute_type = "float16"
                print(f"[STT] GPU NVIDIA terdeteksi ({ctranslate2.get_cuda_device_count()} device). Menggunakan CUDA + FP16.")
            else:
                print("[STT] GPU tidak terdeteksi. Menggunakan CPU.")
        except Exception:
            print("[STT] CTranslate2 fallback ke CPU.")

        try:
            print(f"[STT] Memuat model Faster-Whisper '{model_size}' ({device}, {compute_type})...")
            self.whisper = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=4 if device == "cpu" else 1,
            )
            print("[STT] Faster-Whisper siap.")
        except Exception as e:
            print(f"[STT Error] Gagal memuat Faster-Whisper ({device}): {e}")
            if device == "cuda":
                try:
                    print("[STT] Mencoba fallback ke CPU (int8)...")
                    self.whisper = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4)
                    print("[STT] Faster-Whisper CPU siap.")
                except Exception as e2:
                    print(f"[STT Error] Fallback CPU juga gagal: {e2}")

    def transcribe(self, audio_bytes: bytes) -> str:
        """Menerima raw audio bytes dan mengembalikan string teks transkripsi."""
        if not self.whisper:
            return ""

        temp_path = "/tmp/hri_input_audio.webm"
        try:
            with open(temp_path, "wb") as f:
                f.write(audio_bytes)

            segments, _ = self.whisper.transcribe(
                temp_path,
                beam_size=5,
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400),
            )
            text = " ".join([seg.text for seg in segments]).strip()
            return text
        except Exception as e:
            print(f"[STT Error] Transkripsi gagal: {e}")
            return ""
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

