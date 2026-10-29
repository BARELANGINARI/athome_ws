"""Audio I/O components: asynchronous TTS playback queue, local microphone VAD listener, and real-time VAD visualizer."""

import asyncio
import base64
from collections import deque
from dataclasses import dataclass
import io
import logging
import os
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

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

BroadcastCallback = Callable[[Dict[str, Any]], Awaitable[Any]]
AudioCapturedCallback = Callable[[Union[np.ndarray, bytes]], Awaitable[Any]]
QueueItem = Tuple[str, float, Optional[asyncio.Future]]


@dataclass(frozen=True)
class VadFrameMetrics:
    """Data metrik per-frame (30 ms) untuk pemrosesan VAD dan visualisasi real-time."""

    signal: np.ndarray
    fft_mag: np.ndarray
    dbfs: float
    dbfs_speech: float
    flatness: float
    is_speaking: bool
    is_paused: bool


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
        """Memutar audio WAV Base64 ke speaker lokal menggunakan SoundDevice atau PyAudio."""
        if not wav_base64:
            return False
        try:
            raw_bytes = base64.b64decode(wav_base64)
            with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
                if HAS_SOUNDDEVICE:
                    rate = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    audio_np = (
                        np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    sd.play(audio_np, samplerate=rate)
                    sd.wait()
                    return True
                elif HAS_PYAUDIO:
                    pa = pyaudio.PyAudio()
                    try:
                        chunk = 4096
                        stream = pa.open(
                            format=pa.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True,
                            frames_per_buffer=chunk,
                        )
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
                await asyncio.sleep(
                    remaining + delay_after + self.POST_SPEECH_BUFFER_SEC
                )
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

    def _can_enqueue(
        self, text: str, cooldown: Optional[float], force: bool
    ) -> bool:
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
        on_frame_metrics: Optional[Callable[[VadFrameMetrics], None]] = None,
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
        self.on_frame_metrics = on_frame_metrics
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
        self._hanning_window = np.hanning(self.block_size)

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
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        """Menghitung Full-band dBFS, Speech-band dBFS (300-3400 Hz), FFT Magnitude, dan Spectral Flatness."""
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

        window = (
            self._hanning_window
            if len(signal) == len(self._hanning_window)
            else np.hanning(len(signal))
        )
        fft_mag = np.abs(np.fft.rfft(signal * window)) + eps
        flatness = float(np.exp(np.mean(np.log(fft_mag))) / np.mean(fft_mag))

        return signal, fft_mag, dbfs, dbfs_speech, flatness

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

        def audio_callback(
            indata: np.ndarray, frames: int, time_info: Any, status: Any
        ) -> None:
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
        hw_rate = self.sample_rate
        hw_block = self.block_size
        try:
            for cand_rate in (self.sample_rate, 48000, 44100):
                try:
                    cand_block = int(cand_rate * self.BLOCK_DURATION)
                    stream = pa.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=cand_rate,
                        input=True,
                        input_device_index=self.device_index,
                        frames_per_buffer=cand_block,
                    )
                    hw_rate = cand_rate
                    hw_block = cand_block
                    break
                except Exception:
                    continue

            if stream is None:
                logger.error("Gagal membuka mikrofon lokal via PyAudio pada semua sample rate.")
                return

            logger.info(
                "Mikrofon lokal aktif via PyAudio (%d Hz -> %d Hz, block=%d, dbfs_thresh=%.1f dBFS, flatness_max=%.2f).",
                hw_rate,
                self.sample_rate,
                self.block_size,
                self.threshold_dbfs,
                self.threshold_flatness,
            )

            audio_queue: queue.Queue[np.ndarray] = queue.Queue()
            self._init_vad_state()
            while self.enabled:
                try:
                    pcm_bytes = stream.read(
                        hw_block, exception_on_overflow=False
                    )
                except Exception:
                    time.sleep(0.01)
                    continue
                sig = (
                    np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
                if hw_rate != self.sample_rate and len(sig) > 0:
                    x_old = np.linspace(0.0, 1.0, num=len(sig), endpoint=False)
                    x_new = np.linspace(
                        0.0, 1.0, num=self.block_size, endpoint=False
                    )
                    sig = np.interp(x_new, x_old, sig).astype(np.float32)
                audio_queue.put(sig)
                while not audio_queue.empty():
                    self._process_vad_frame(audio_queue.get_nowait())
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

    def _init_vad_state(self) -> None:
        self._is_speaking = False
        self._speech_buffer: List[np.ndarray] = []
        self._pre_roll_buffer: deque[np.ndarray] = deque(maxlen=self.pre_roll_frames)
        self._silence_counter = 0

    def _run_vad_state_machine(
        self, audio_queue: "queue.Queue[np.ndarray]"
    ) -> None:
        self._init_vad_state()
        while self.enabled:
            try:
                sig = audio_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            self._process_vad_frame(sig)

    def _process_vad_frame(self, sig: np.ndarray) -> None:
        sig, fft_mag, dbfs, dbfs_speech, flatness = self._compute_features(sig)
        paused = bool(self.is_paused_fn and self.is_paused_fn())

        if paused:
            if self._is_speaking or len(self._pre_roll_buffer) > 0:
                self._pre_roll_buffer.clear()
                self._speech_buffer.clear()
                self._is_speaking = False
                self._silence_counter = 0
            if self.on_frame_metrics:
                self.on_frame_metrics(
                    VadFrameMetrics(
                        signal=sig,
                        fft_mag=fft_mag,
                        dbfs=dbfs,
                        dbfs_speech=dbfs_speech,
                        flatness=flatness,
                        is_speaking=False,
                        is_paused=True,
                    )
                )
            return

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

        if self.on_frame_metrics:
            self.on_frame_metrics(
                VadFrameMetrics(
                    signal=sig,
                    fft_mag=fft_mag,
                    dbfs=dbfs,
                    dbfs_speech=dbfs_speech,
                    flatness=flatness,
                    is_speaking=self._is_speaking,
                    is_paused=False,
                )
            )


class VadVisualizer:
    """
    Visualisasi grafik real-time Matplotlib untuk VAD & ASR
    (Waveform, FFT Spectrum 300-3400 Hz, Rolling Speech-Band dBFS, dan status transkripsi).
    """

    HISTORY_LEN: int = 100
    RENDER_INTERVAL_SEC: float = 0.03
    MAX_QUEUE_SIZE: int = 60

    def __init__(self, mic_listener: LocalMicListener) -> None:
        self.mic_listener = mic_listener
        self._frame_queue: queue.Queue[VadFrameMetrics] = queue.Queue(
            maxsize=self.MAX_QUEUE_SIZE
        )
        self._latest_transcript: str = "[Menunggu suara...]"
        self._latest_asr_metrics: str = "RTF: 0.00 | Durasi: 0.0s"
        self._asr_dirty: bool = False
        self._render_task: Optional[asyncio.Task] = None
        self._active: bool = False

        self.fig: Optional[Any] = None
        self._dbfs_hist = np.full(self.HISTORY_LEN, -80.0)
        self._speech_dbfs_hist = np.full(self.HISTORY_LEN, -80.0)

    @staticmethod
    def _has_display() -> bool:
        return bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )

    def push_frame(self, metrics: VadFrameMetrics) -> None:
        """Menerima frame metrik dari thread audio secara non-blocking."""
        if not self._active:
            return
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._frame_queue.put_nowait(metrics)
        except queue.Full:
            pass

    def update_asr_status(self, transcript: str, metrics: str) -> None:
        """Memperbarui teks hasil transkripsi dan metrik RTF ASR di bagian bawah grafik."""
        self._latest_transcript = transcript
        self._latest_asr_metrics = metrics
        self._asr_dirty = True

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Menginisialisasi jendela grafik Matplotlib dan memulai task perenderan asinkron."""
        if not HAS_MATPLOTLIB:
            logger.warning(
                "matplotlib tidak terinstal. Visualisasi grafik VAD dinonaktifkan."
            )
            return
        if not self._has_display():
            logger.warning(
                "Environment DISPLAY tidak terdeteksi (headless). Visualisasi grafik VAD dinonaktifkan."
            )
            return

        try:
            self._setup_figure()
            self._active = True
            self._render_task = loop.create_task(self._render_loop())
            logger.info(
                "Visualisasi grafik VAD aktif (Gunakan panah UP/DOWN untuk dBFS, LEFT/RIGHT untuk Flatness)."
            )
        except Exception as exc:
            logger.warning("Gagal menginisialisasi jendela visualisasi VAD: %s", exc)
            self._active = False

    def stop(self) -> None:
        """Menghentikan loop perenderan dan menutup jendela Matplotlib."""
        self._active = False
        if self._render_task and not self._render_task.done():
            self._render_task.cancel()
        if HAS_MATPLOTLIB and self.fig is not None:
            try:
                plt.close(self.fig)
            except Exception:
                pass
            self.fig = None

    def _setup_figure(self) -> None:
        plt.ion()
        block_size = self.mic_listener.block_size
        sample_rate = self.mic_listener.sample_rate

        self.fig, (self.ax_wave, self.ax_spec, self.ax_dbfs) = plt.subplots(
            3, 1, figsize=(10, 8.5)
        )
        self.fig.canvas.manager.set_window_title(
            "Barelang Inari HRI - Real-Time VAD & ASR Visualizer"
        )
        plt.subplots_adjust(hspace=0.5, bottom=0.15)

        # Plot 1: Waveform & Indikator Status VAD
        x_wave = np.arange(block_size) / sample_rate * 1000.0
        (self.line_wave,) = self.ax_wave.plot(
            x_wave, np.zeros(block_size), color="#007acc"
        )
        self.ax_wave.set_ylim(-1.0, 1.0)
        self.title_wave = self.ax_wave.set_title(
            "VAD Status: SILENCE | menunggu input...",
            fontweight="bold",
            color="gray",
        )
        self.ax_wave.set_ylabel("Amplitude")

        # Plot 2: FFT Spectrum
        freqs = np.fft.rfftfreq(block_size, 1.0 / sample_rate)
        (self.line_spec,) = self.ax_spec.plot(
            freqs, np.zeros(len(freqs)), color="#d9534f"
        )
        self.ax_spec.axvspan(
            self.mic_listener.BANDPASS_LOWCUT,
            self.mic_listener.BANDPASS_HIGHCUT,
            color="green",
            alpha=0.15,
            label="Speech Band (300-3400 Hz)",
        )
        self.ax_spec.set_xlim(0, 8000)
        self.ax_spec.set_ylim(-60, 20)
        self.ax_spec.set_title("Frequency Spectrum (FFT dB)")
        self.ax_spec.set_ylabel("dB")
        self.ax_spec.legend(loc="upper right")

        # Plot 3: Rolling Energy & Threshold
        (self.line_dbfs,) = self.ax_dbfs.plot(
            self._dbfs_hist, label="Full dBFS", color="gray", alpha=0.5
        )
        (self.line_speech_dbfs,) = self.ax_dbfs.plot(
            self._speech_dbfs_hist, label="Speech-Band dBFS", color="#2ca02c"
        )
        thresh = self.mic_listener.threshold_dbfs
        flat_max = self.mic_listener.threshold_flatness
        self.line_thresh = self.ax_dbfs.axhline(
            thresh,
            color="red",
            linestyle="--",
            label=f"Thresh ({thresh:.1f} dBFS)",
        )
        self.ax_dbfs.set_ylim(-80, 0)
        self.ax_dbfs.set_title(
            f"VAD Energy | [UP/DOWN] dBFS Thresh | [LEFT/RIGHT] Flatness Max: {flat_max:.2f}"
        )
        self.ax_dbfs.set_ylabel("dBFS")
        self.ax_dbfs.legend(loc="upper left")

        # Kotak Teks Hasil Transkripsi ASR di Bagian Bawah
        self.asr_text_box = self.fig.text(
            0.5,
            0.03,
            f"ASR: {self._latest_transcript}\n{self._latest_asr_metrics}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#f0f4f8",
                edgecolor="#007acc",
            ),
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.show()

    def _on_key_press(self, event: Any) -> None:
        if event.key == "up":
            self.mic_listener.threshold_dbfs = min(
                0.0, self.mic_listener.threshold_dbfs + 1.0
            )
        elif event.key == "down":
            self.mic_listener.threshold_dbfs = max(
                -80.0, self.mic_listener.threshold_dbfs - 1.0
            )
        elif event.key == "right":
            self.mic_listener.threshold_flatness = min(
                1.0, self.mic_listener.threshold_flatness + 0.05
            )
        elif event.key == "left":
            self.mic_listener.threshold_flatness = max(
                0.05, self.mic_listener.threshold_flatness - 0.05
            )
        else:
            return

        thresh = self.mic_listener.threshold_dbfs
        flat_max = self.mic_listener.threshold_flatness
        self.line_thresh.set_ydata([thresh, thresh])
        self.line_thresh.set_label(f"Thresh ({thresh:.1f} dBFS)")
        self.ax_dbfs.set_title(
            f"VAD Energy | [UP/DOWN] dBFS Thresh | [LEFT/RIGHT] Flatness Max: {flat_max:.2f}"
        )
        self.ax_dbfs.legend(loc="upper left")

    async def _render_loop(self) -> None:
        while self._active and self.fig is not None:
            try:
                if not plt.fignum_exists(self.fig.number):
                    logger.info("Jendela visualisasi VAD ditutup oleh pengguna.")
                    self._active = False
                    break

                updated = False
                last_metrics: Optional[VadFrameMetrics] = None

                while not self._frame_queue.empty():
                    metrics = self._frame_queue.get_nowait()
                    last_metrics = metrics
                    self._dbfs_hist = np.roll(self._dbfs_hist, -1)
                    self._dbfs_hist[-1] = metrics.dbfs
                    self._speech_dbfs_hist = np.roll(self._speech_dbfs_hist, -1)
                    self._speech_dbfs_hist[-1] = metrics.dbfs_speech
                    updated = True

                if last_metrics is not None:
                    self.line_wave.set_ydata(last_metrics.signal)
                    self.line_spec.set_ydata(20.0 * np.log10(last_metrics.fft_mag))
                    self.line_dbfs.set_ydata(self._dbfs_hist)
                    self.line_speech_dbfs.set_ydata(self._speech_dbfs_hist)

                    if last_metrics.is_paused:
                        self.title_wave.set_text(
                            "VAD Status: [PAUSED - ROBOT SPEAKING / BUSY]"
                        )
                        self.title_wave.set_color("#d9534f")
                        self.line_wave.set_color("#999999")
                    elif last_metrics.is_speaking:
                        self.title_wave.set_text(
                            f"VAD Status: [SPEECH DETECTED - RECORDING] (Flatness: {last_metrics.flatness:.2f})"
                        )
                        self.title_wave.set_color("green")
                        self.line_wave.set_color("#2ca02c")
                    else:
                        self.title_wave.set_text(
                            f"VAD Status: SILENCE (Flatness: {last_metrics.flatness:.2f})"
                        )
                        self.title_wave.set_color("gray")
                        self.line_wave.set_color("#007acc")

                if self._asr_dirty:
                    self.asr_text_box.set_text(
                        f"ASR: {self._latest_transcript}\n{self._latest_asr_metrics}"
                    )
                    self._asr_dirty = False
                    updated = True

                if updated:
                    self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()

                await asyncio.sleep(self.RENDER_INTERVAL_SEC)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Kesalahan render visualisasi VAD: %s", exc)
                await asyncio.sleep(0.1)
