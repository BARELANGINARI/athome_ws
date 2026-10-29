import os

# PyAV & Faster-Whisper compatibility patch
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
            print("[STT WARN] faster_whisper is not installed. STT disabled.")
            return

        device = "cpu"
        compute_type = "int8"
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                device = "cuda"
                compute_type = "float16"
                print(f"[STT] NVIDIA GPU detected ({ctranslate2.get_cuda_device_count()} device). Using CUDA + FP16.")
            else:
                print("[STT] No GPU detected. Using CPU.")
        except Exception:
            print("[STT] CTranslate2 fallback to CPU.")

        try:
            print(f"[STT] Loading Faster-Whisper model '{model_size}' ({device}, {compute_type})...")
            self.whisper = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=4 if device == "cpu" else 1,
            )
            print("[STT] Faster-Whisper ready.")
        except Exception as e:
            print(f"[STT Error] Failed loading Faster-Whisper ({device}): {e}")
            if device == "cuda":
                try:
                    print("[STT] Trying fallback to CPU (int8)...")
                    self.whisper = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4)
                    print("[STT] Faster-Whisper CPU ready.")
                except Exception as e2:
                    print(f"[STT Error] CPU fallback failed: {e2}")

    def transcribe(self, audio_bytes: bytes) -> str:
        """Receive raw audio bytes and return transcribed text string."""
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
            print(f"[STT Error] Transcription failed: {e}")
            return ""
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

