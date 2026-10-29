"""Audio I/O components: asynchronous TTS playback queue and local microphone VAD listener."""

import asyncio
import base64
from collections import deque
import io
import logging
import math
import struct
import threading
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
import wave

from athome_hri.tts_engine import TextToSpeech

logger = logging.getLogger("athome_hri.audio_io")

try:
    import pyaudio

    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False

BroadcastCallback = Callable[[Dict[str, Any]], Awaitable[Any]]
AudioCapturedCallback = Callable[[bytes], Awaitable[Any]]
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
        """Memutar audio WAV Base64 ke speaker lokal menggunakan PyAudio."""
        if not wav_base64 or not HAS_PYAUDIO:
            return False
        try:
            raw_bytes = base64.b64decode(wav_base64)
            with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
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
    """Modul perekam mikrofon lokal dengan deteksi aktivitas suara berbasis RMS energy (VAD)."""

    FALLBACK_SAMPLE_RATES: Tuple[int, ...] = (16000, 44100, 48000)
    MIN_SPEECH_DURATION_SEC: float = 0.35
    MAX_SPEECH_DURATION_SEC: float = 8.0
    PRE_BUFFER_DURATION_SEC: float = 0.4

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        on_audio_captured: AudioCapturedCallback,
        is_paused_fn: Callable[[], bool],
        device_index: Optional[int] = None,
        energy_threshold: int = 500,
        silence_limit: float = 1.0,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
    ) -> None:
        self.loop = loop
        self.on_audio_captured = on_audio_captured
        self.is_paused_fn = is_paused_fn
        self.device_index = device_index
        self.energy_threshold = energy_threshold
        self.silence_limit = silence_limit
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.enabled: bool = False
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _calc_rms(pcm_bytes: bytes) -> float:
        count = len(pcm_bytes) // 2
        if count == 0:
            return 0.0
        shorts = struct.unpack(f"<{count}h", pcm_bytes[: count * 2])
        sum_sq = sum(s * s for s in shorts)
        return math.sqrt(sum_sq / count)

    @staticmethod
    def _frames_to_wav_bytes(frames: List[bytes], rate: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))
        return buf.getvalue()

    def start(self) -> None:
        """Memulai thread pembacaan mikrofon lokal."""
        if not HAS_PYAUDIO:
            logger.warning(
                "pyaudio tidak terinstal. Input mikrofon lokal tidak dapat dijalankan."
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

    def _open_audio_stream(self, pa: "pyaudio.PyAudio") -> Tuple[Optional[Any], int]:
        candidate_rates = (self.sample_rate,) + tuple(
            r for r in self.FALLBACK_SAMPLE_RATES if r != self.sample_rate
        )
        for rate in candidate_rates:
            try:
                stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=self.chunk_size,
                )
                return stream, rate
            except Exception:
                continue
        return None, self.sample_rate

    def _listen_loop(self) -> None:
        pa = pyaudio.PyAudio()
        stream, rate = self._open_audio_stream(pa)

        if stream is None:
            logger.error("Gagal membuka mikrofon lokal pada device.")
            pa.terminate()
            return

        logger.info(
            "Mikrofon lokal aktif (%d Hz, threshold=%d).",
            rate,
            self.energy_threshold,
        )
        pre_buffer_len = max(int(self.PRE_BUFFER_DURATION_SEC * rate / self.chunk_size), 2)
        pre_buffer: deque[bytes] = deque(maxlen=pre_buffer_len)
        recording_frames: List[bytes] = []
        is_recording = False
        silence_start: Optional[float] = None
        record_start = 0.0

        try:
            while self.enabled:
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                except Exception:
                    time.sleep(0.05)
                    continue

                if self.is_paused_fn and self.is_paused_fn():
                    pre_buffer.clear()
                    recording_frames.clear()
                    is_recording = False
                    silence_start = None
                    continue

                rms = self._calc_rms(data)
                now = time.time()

                if not is_recording:
                    pre_buffer.append(data)
                    if rms >= self.energy_threshold:
                        is_recording = True
                        record_start = now
                        silence_start = None
                        recording_frames = list(pre_buffer)
                        pre_buffer.clear()
                else:
                    recording_frames.append(data)
                    if rms < self.energy_threshold:
                        if silence_start is None:
                            silence_start = now
                        elif (now - silence_start) >= self.silence_limit:
                            is_recording = False
                    else:
                        silence_start = None

                    if (now - record_start) >= self.MAX_SPEECH_DURATION_SEC:
                        is_recording = False

                    if not is_recording:
                        duration = now - record_start
                        if duration >= self.MIN_SPEECH_DURATION_SEC and self.on_audio_captured:
                            wav_bytes = self._frames_to_wav_bytes(recording_frames, rate)
                            asyncio.run_coroutine_threadsafe(
                                self.on_audio_captured(wav_bytes), self.loop
                            )
                        recording_frames = []
                        silence_start = None
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            pa.terminate()
            logger.info("Mikrofon lokal dinonaktifkan.")

