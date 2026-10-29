"""Audio I/O components: asynchronous TTS playback queue and local microphone VAD listener."""

import asyncio
import base64
from collections import deque
import io
import logging
import queue
import threading
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union
import wave

import numpy as np

from athome_hri.tts_engine import TextToSpeech

logger = logging.getLogger("athome_hri.audio_io")

try:
    from scipy.signal import butter, lfilter

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import sounddevice as sd

    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

try:
    import pyaudio

    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False

BroadcastCallback = Callable[[Dict[str, Any]], Awaitable[Any]]
AudioCapturedCallback = Callable[[Union[np.ndarray, bytes]], Awaitable[Any]]
QueueItem = Tuple[str, float, Optional[asyncio.Future]]


class AudioSpeechQueue:
    """Antrean pemutaran TTS asinkron dengan pencegah duplikasi (cooldown) dan estimasi durasi."""

    WORDS_DURATION_FACTOR: float = 0.42
    MIN_ESTIMATED_DURATION: float = 1.8
    POST_SPEECH_BUFFER_SEC: float = 0.3

    def __init__(
        self,
        tts_engine: Optional[TextToSpeech],
        broadcast_func: Optional[BroadcastCallback],
        default_cooldown: float = 3.0,
        min_gap: float = 0.5,
        enable_local_speaker: bool = True,
    ) -> None:
        self.tts = tts_engine
        self.broadcast_func = broadcast_func
        self.default_cooldown = default_cooldown
        self.min_gap = min_gap
        self.enable_local_speaker = enable_local_speaker
        self.queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        self.last_spoken_time: Dict[str, float] = {}
        self.global_last_spoken_time: float = 0.0
        self.is_speaking: bool = False
        self._worker_task: Optional[asyncio.Task] = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Memulai background task worker pemutaran suara."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = loop.create_task(self._worker())

    def stop(self) -> None:
        """Membatalkan worker antrean suara dan membersihkan item yang tertunda."""
        self.clear()
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()

    @classmethod
    def _estimate_duration(cls, wav_base64: str, text: str) -> float:
        if wav_base64:
            try:
                raw_bytes = base64.b64decode(wav_base64)
                with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
                    rate = wf.getframerate()
                    if rate > 0:
                        return wf.getnframes() / float(rate)
            except Exception:
                pass
        return max(
            len(text.split()) * cls.WORDS_DURATION_FACTOR,
            cls.MIN_ESTIMATED_DURATION,
        )

    @staticmethod
    def _play_wav_locally(wav_base64: str) -> bool:
        """Memutar audio WAV Base64 ke speaker lokal menggunakan PyAudio atau SoundDevice."""
        if not wav_base64:
            return False
        try:
            raw_bytes = base64.b64decode(wav_base64)
            with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
                if HAS_PYAUDIO:
                    pa = pyaudio.PyAudio()
                    try:
                        stream = pa.open(
                            format=pa.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True,
                        )
                        chunk = 1024
                        data = wf.readframes(chunk)
                        while data:
                            stream.write(data)
                            data = wf.readframes(chunk)
                        stream.stop_stream()
                        stream.close()
                        return True
                    finally:
                        pa.terminate()
                elif HAS_SOUNDDEVICE:
                    rate = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    audio_np = (
                        np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    sd.play(audio_np, samplerate=rate)
                    sd.wait()
                    return True
        except Exception as exc:
            logger.debug("Gagal memutar audio ke speaker lokal: %s", exc)
        return False

    async def _worker(self) -> None:
        while True:
            text, delay_after, done_future = await self.queue.get()
            self.is_speaking = True
            try:
                gap = time.time() - self.global_last_spoken_time
                if gap < self.min_gap:
                    await asyncio.sleep(self.min_gap - gap)

                logger.info('[ROBOT SPEAKS] "%s"', text)
                wav_base64 = (
                    await asyncio.to_thread(self.tts.synthesize_wav_base64, text)
                    if self.tts
                    else ""
                )

                if self.broadcast_func:
                    await self.broadcast_func(
                        {
                            "type": "TTS_PLAY",
                            "text": text,
                            "audio": wav_base64,
                        }
                    )

                duration = self._estimate_duration(wav_base64, text)
                t0 = time.time()
                if self.enable_local_speaker and wav_base64:
                    await asyncio.to_thread(self._play_wav_locally, wav_base64)
                elapsed = time.time() - t0
                remaining = max(0.0, duration - elapsed)
                await asyncio.sleep(remaining + delay_after + self.POST_SPEECH_BUFFER_SEC)
                self.global_last_spoken_time = time.time()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Kesalahan pada AudioSpeechQueue worker: %s", exc)
            finally:
                self.is_speaking = False
                if done_future and not done_future.done():
                    done_future.set_result(True)
                self.queue.task_done()

    def _can_enqueue(self, text: str, cooldown: Optional[float], force: bool) -> bool:
        key = text.lower()
        now = time.time()
        if not force and key in self.last_spoken_time:
            cd = self.default_cooldown if cooldown is None else cooldown
            if (now - self.last_spoken_time[key]) < cd:
                return False
        self.last_spoken_time[key] = now
        return True

    def enqueue(
        self,
        text: str,
        cooldown: Optional[float] = None,
        delay_after: float = 0.0,
        force: bool = False,
    ) -> bool:
        """Memasukkan teks ke antrean TTS secara non-blocking."""
        text_clean = text.strip()
        if not text_clean or not self._can_enqueue(text_clean, cooldown, force):
            return False
        self.queue.put_nowait((text_clean, delay_after, None))
        return True

    async def speak_and_wait(
        self,
        text: str,
        delay_after: float = 0.0,
        force: bool = True,
    ) -> bool:
        """Memasukkan teks ke antrean dan menunggu (await) hingga selesai diucapkan."""
        text_clean = text.strip()
        if not text_clean or not self._can_enqueue(text_clean, None, force):
            return False
        loop = asyncio.get_running_loop()
        done_future: asyncio.Future[bool] = loop.create_future()
        self.queue.put_nowait((text_clean, delay_after, done_future))
        await done_future
        return True

    def clear(self) -> None:
        """Mengosongkan seluruh antrean ucapan yang belum diproses."""
        while not self.queue.empty():
            try:
                _, _, fut = self.queue.get_nowait()
                if fut and not fut.done():
                    fut.set_result(False)
                self.queue.task_done()
            except Exception:
                break


class LocalMicListener:
    """
    Modul perekam mikrofon lokal dengan VAD berbasis Speech-Band dBFS (300-3400 Hz)
    dan Spectral Flatness (identik dengan utilitas/record_stream_visualize.py).
    """

    SAMPLE_RATE: int = 16000
    BLOCK_DURATION: float = 0.03  # 30 ms per frame (480 sampel @ 16 kHz)
    BANDPASS_LOWCUT: float = 300.0
    BANDPASS_HIGHCUT: float = 3400.0
    BANDPASS_ORDER: int = 4

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        on_audio_captured: AudioCapturedCallback,
        is_paused_fn: Callable[[], bool],
        device_index: Optional[int] = None,
        threshold_dbfs: float = -40.0,
        threshold_flatness: float = 0.40,
        pre_roll_frames: int = 10,
        hangover_frames: int = 25,
        min_speech_frames: int = 15,
        sample_rate: int = SAMPLE_RATE,
        block_duration: float = BLOCK_DURATION,
    ) -> None:
        self.loop = loop
        self.on_audio_captured = on_audio_captured
        self.is_paused_fn = is_paused_fn
        self.device_index = device_index
        self.threshold_dbfs = threshold_dbfs
        self.threshold_flatness = threshold_flatness
        self.pre_roll_frames = pre_roll_frames
        self.hangover_frames = hangover_frames
        self.min_speech_frames = min_speech_frames
        self.sample_rate = sample_rate
        self.block_size = int(sample_rate * block_duration)
        self.enabled: bool = False
        self._thread: Optional[threading.Thread] = None

        if HAS_SCIPY:
            nyq = 0.5 * self.sample_rate
            self._b_band, self._a_band = butter(
                self.BANDPASS_ORDER,
                [self.BANDPASS_LOWCUT / nyq, self.BANDPASS_HIGHCUT / nyq],
                btype="band",
            )
        else:
            self._b_band, self._a_band = None, None

    def _compute_features(
        self, signal: np.ndarray
    ) -> Tuple[np.ndarray, float, float, float]:
        """Menghitung Full-band dBFS, Speech-band dBFS (300-3400 Hz), dan Spectral Flatness."""
        eps = 1e-10
        rms = float(np.sqrt(np.mean(signal**2) + eps))
        dbfs = float(20.0 * np.log10(rms + eps))

        if HAS_SCIPY and self._b_band is not None and self._a_band is not None:
            speech_sig = lfilter(self._b_band, self._a_band, signal)
            rms_speech = float(np.sqrt(np.mean(speech_sig**2) + eps))
        else:
            freqs = np.fft.rfftfreq(len(signal), 1.0 / self.sample_rate)
            fft_full = np.fft.rfft(signal)
            mask = (freqs >= self.BANDPASS_LOWCUT) & (freqs <= self.BANDPASS_HIGHCUT)
            speech_sig = np.fft.irfft(fft_full * mask, n=len(signal))
            rms_speech = float(np.sqrt(np.mean(speech_sig**2) + eps))

        dbfs_speech = float(20.0 * np.log10(rms_speech + eps))

        fft_mag = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) + eps
        flatness = float(np.exp(np.mean(np.log(fft_mag))) / np.mean(fft_mag))

        return signal, dbfs, dbfs_speech, flatness

    def start(self) -> None:
        """Memulai thread pembacaan mikrofon lokal."""
        if not HAS_SOUNDDEVICE and not HAS_PYAUDIO:
            logger.warning(
                "sounddevice maupun pyaudio tidak terinstal. Input mikrofon lokal tidak dapat dijalankan."
            )
            return
        if self._thread and self._thread.is_alive():
            return
        self.enabled = True
        self._thread = threading.Thread(
            target=self._listen_loop, name="HriLocalMicListener", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Menghentikan thread pembacaan mikrofon lokal."""
        self.enabled = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)

    def _listen_loop(self) -> None:
        if HAS_SOUNDDEVICE:
            self._listen_with_sounddevice()
        elif HAS_PYAUDIO:
            self._listen_with_pyaudio()

    def _listen_with_sounddevice(self) -> None:
        audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        def audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            if status:
                logger.debug("SoundDevice status: %s", status)
            audio_queue.put(indata[:, 0].copy())

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.block_size,
                device=self.device_index,
                callback=audio_callback,
            ):
                logger.info(
                    "Mikrofon lokal aktif via SoundDevice (%d Hz, block=%d, dbfs_thresh=%.1f dBFS, flatness_max=%.2f).",
                    self.sample_rate,
                    self.block_size,
                    self.threshold_dbfs,
                    self.threshold_flatness,
                )
                self._run_vad_state_machine(audio_queue)
        except Exception as exc:
            logger.warning(
                "Gagal membuka mikrofon via SoundDevice (%s). Mencoba fallback PyAudio...",
                exc,
            )
            if HAS_PYAUDIO and self.enabled:
                self._listen_with_pyaudio()
        finally:
            logger.info("Mikrofon lokal dinonaktifkan.")

    def _listen_with_pyaudio(self) -> None:
        pa = pyaudio.PyAudio()
        stream = None
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.block_size,
            )
            logger.info(
                "Mikrofon lokal aktif via PyAudio (%d Hz, block=%d, dbfs_thresh=%.1f dBFS, flatness_max=%.2f).",
                self.sample_rate,
                self.block_size,
                self.threshold_dbfs,
                self.threshold_flatness,
            )

            audio_queue: queue.Queue[np.ndarray] = queue.Queue()
            while self.enabled:
                try:
                    pcm_bytes = stream.read(self.block_size, exception_on_overflow=False)
                except Exception:
                    time.sleep(0.01)
                    continue
                sig = (
                    np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
                audio_queue.put(sig)
                self._drain_vad_queue(audio_queue)
        except Exception as exc:
            logger.error("Gagal membuka mikrofon lokal via PyAudio: %s", exc)
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            pa.terminate()
            logger.info("Mikrofon lokal dinonaktifkan.")

    def _InitVadState(self) -> None:
        self._is_speaking = False
        self._speech_buffer: List[np.ndarray] = []
        self._pre_roll_buffer: deque[np.ndarray] = deque(maxlen=self.pre_roll_frames)
        self._silence_counter = 0

    def _run_vad_state_machine(self, audio_queue: "queue.Queue[np.ndarray]") -> None:
        self._InitVadState()
        while self.enabled:
            try:
                sig = audio_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            self._process_vad_frame(sig)

    def _drain_vad_queue(self, audio_queue: "queue.Queue[np.ndarray]") -> None:
        if not hasattr(self, "_is_speaking"):
            self._InitVadState()
        while not audio_queue.empty():
            sig = audio_queue.get_nowait()
            self._process_vad_frame(sig)

    def _process_vad_frame(self, sig: np.ndarray) -> None:
        if self.is_paused_fn and self.is_paused_fn():
            if self._is_speaking or len(self._pre_roll_buffer) > 0:
                self._pre_roll_buffer.clear()
                self._speech_buffer.clear()
                self._is_speaking = False
                self._silence_counter = 0
            return

        sig, _dbfs, dbfs_speech, flatness = self._compute_features(sig)

        # Logika Keputusan VAD (Kombinasi Energy Speech-Band & Spectral Flatness)
        frame_is_voice = (dbfs_speech > self.threshold_dbfs) and (
            flatness < self.threshold_flatness
        )

        if frame_is_voice:
            if not self._is_speaking:
                self._is_speaking = True
                # Masukkan pre-roll buffer agar awal kata tidak terpotong
                self._speech_buffer.extend(list(self._pre_roll_buffer))
            self._speech_buffer.append(sig)
            self._silence_counter = 0
        else:
            if self._is_speaking:
                self._speech_buffer.append(sig)
                self._silence_counter += 1
                # Jika sudah hening melebihi hangover time, kirim ke ASR
                if self._silence_counter >= self.hangover_frames:
                    if (
                        len(self._speech_buffer)
                        >= self.min_speech_frames + self.pre_roll_frames
                    ):
                        full_audio = np.concatenate(self._speech_buffer).astype(
                            np.float32
                        )
                        if self.on_audio_captured:
                            asyncio.run_coroutine_threadsafe(
                                self.on_audio_captured(full_audio), self.loop
                            )
                    # Reset state
                    self._is_speaking = False
                    self._speech_buffer = []
                    self._silence_counter = 0
            else:
                self._pre_roll_buffer.append(sig)
