"""CPU-only single-channel speech enhancement frontend for athome_hri.

Pipeline:
1. Zero-phase Butterworth High-Pass Filter (default 120 Hz) to suppress
   directional-mic proximity effect and robot motor/fan low-frequency rumble.
2. Denoising backend (strictly CPU, zero GPU VRAM):
   - "deepfilternet2": DeepFilterNet2 via `df.enhance` (1 CPU thread, 48 kHz)
   - "rnnoise": RNNoise via `pyrnnoise` (CPU, 48 kHz)
   - "spectral": Built-in lightweight Wiener / spectral subtraction (`scipy`/`numpy`)
   - "none": High-pass filter only
   - "auto": Tries deepfilternet2 -> rnnoise -> spectral
"""

import logging
from math import gcd
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("athome_hri.audio_enhancer")

try:
    from scipy.signal import butter, istft, resample_poly, sosfiltfilt, stft

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class AudioEnhancer:
    """Single-channel CPU speech enhancement frontend."""

    DF_SAMPLE_RATE: int = 48000

    def __init__(
        self,
        backend: str = "auto",
        sample_rate: int = 16000,
        highpass_hz: float = 120.0,
    ) -> None:
        self.requested_backend = str(backend or "auto").strip().lower()
        self.sample_rate = int(sample_rate)
        self.highpass_hz = float(highpass_hz)
        self.active_backend: str = "none"

        self._sos_hp: Optional[np.ndarray] = None
        self._df_model: Any = None
        self._df_state: Any = None
        self._df_enhance_fn: Any = None
        self._torch: Any = None
        self._rnnoise: Any = None

        self._build_highpass_filter()
        self._init_backend(self.requested_backend)

    def _build_highpass_filter(self) -> None:
        if not HAS_SCIPY or self.highpass_hz <= 0.0:
            self._sos_hp = None
            return
        nyq = 0.5 * self.sample_rate
        cutoff = min(max(self.highpass_hz, 20.0), nyq * 0.9)
        self._sos_hp = butter(4, cutoff / nyq, btype="highpass", output="sos")

    def _init_backend(self, backend: str) -> None:
        if backend == "none":
            self.active_backend = "none"
            logger.info("AudioEnhancer active: none (HPF %.0f Hz only).", self.highpass_hz)
            return

        order = (
            ("deepfilternet2", "rnnoise", "spectral")
            if backend == "auto"
            else (backend, "spectral")
        )

        for cand in order:
            if cand in ("deepfilternet2", "deepfilternet", "df2"):
                if self._try_init_deepfilternet2():
                    self.active_backend = "deepfilternet2"
                    logger.info(
                        "AudioEnhancer active: DeepFilterNet2 (CPU, 1 thread, HPF %.0f Hz).",
                        self.highpass_hz,
                    )
                    return
            elif cand == "rnnoise":
                if self._try_init_rnnoise():
                    self.active_backend = "rnnoise"
                    logger.info(
                        "AudioEnhancer active: RNNoise (CPU, HPF %.0f Hz).",
                        self.highpass_hz,
                    )
                    return
            elif cand == "spectral":
                if HAS_SCIPY:
                    self.active_backend = "spectral"
                    logger.info(
                        "AudioEnhancer active: Spectral/Wiener DSP fallback (CPU, HPF %.0f Hz).",
                        self.highpass_hz,
                    )
                    return

        self.active_backend = "none"
        logger.warning(
            "AudioEnhancer: tidak ada backend denoiser yang tersedia; menggunakan HPF saja."
        )

    def _try_init_deepfilternet2(self) -> bool:
        try:
            import dataclasses
            import sys
            import types
            import torch

            # Compatibility shim for torchaudio >= 2.1 where torchaudio.backend was removed
            if "torchaudio.backend" not in sys.modules:
                try:
                    import torchaudio.backend.common  # noqa: F401
                except ImportError:
                    _tb = types.ModuleType("torchaudio.backend")
                    _tbc = types.ModuleType("torchaudio.backend.common")
                    _tbc.AudioMetaData = dataclasses.make_dataclass(
                        "AudioMetaData",
                        [
                            ("sample_rate", int, 0),
                            ("num_frames", int, 0),
                            ("num_channels", int, 0),
                            ("bits_per_sample", int, 0),
                            ("encoding", str, ""),
                        ],
                    )
                    _tb.common = _tbc
                    sys.modules["torchaudio.backend"] = _tb
                    sys.modules["torchaudio.backend.common"] = _tbc

            from df.enhance import enhance, init_df

            torch.set_num_threads(1)
            model, df_state, _ = init_df(
                model_base_dir="DeepFilterNet2",
                post_filter=True,
                log_level="ERROR",
            )
            model = model.to("cpu")
            model.eval()
            self._torch = torch
            self._df_model = model
            self._df_state = df_state
            self._df_enhance_fn = enhance
            return True
        except Exception as exc:
            logger.debug("DeepFilterNet2 tidak tersedia (%s).", exc)
            return False

    def _try_init_rnnoise(self) -> bool:
        try:
            from pyrnnoise import RNNoise

            self._rnnoise = RNNoise(sample_rate=self.DF_SAMPLE_RATE)
            return True
        except Exception as exc:
            logger.debug("RNNoise (pyrnnoise) tidak tersedia (%s).", exc)
            return False

    def update_config(
        self,
        *,
        backend: Optional[str] = None,
        highpass_hz: Optional[float] = None,
        sample_rate: Optional[int] = None,
    ) -> None:
        need_hp = False
        if sample_rate is not None and int(sample_rate) != self.sample_rate:
            self.sample_rate = int(sample_rate)
            need_hp = True
        if highpass_hz is not None and float(highpass_hz) != self.highpass_hz:
            self.highpass_hz = float(highpass_hz)
            need_hp = True
        if need_hp:
            self._build_highpass_filter()
        if backend is not None:
            new_backend = str(backend).strip().lower()
            if new_backend != self.requested_backend:
                self.requested_backend = new_backend
                self._init_backend(new_backend)

    @staticmethod
    def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr or audio.size == 0:
            return audio.astype(np.float32, copy=False)
        if HAS_SCIPY:
            g = gcd(src_sr, dst_sr)
            return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)
        ratio = float(dst_sr) / float(src_sr)
        out_len = max(1, int(round(len(audio) * ratio)))
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)

    def _apply_highpass(self, audio: np.ndarray) -> np.ndarray:
        if self._sos_hp is None or audio.size < 32:
            return audio
        try:
            return sosfiltfilt(self._sos_hp, audio).astype(np.float32)
        except Exception:
            return audio

    def _enhance_deepfilternet2(self, audio: np.ndarray, sr: int) -> np.ndarray:
        target_sr = int(self._df_state.sr())
        audio_48k = self._resample(audio, sr, target_sr)
        with self._torch.no_grad():
            tensor = self._torch.from_numpy(audio_48k).unsqueeze(0).to("cpu")
            enhanced_48k = (
                self._df_enhance_fn(self._df_model, self._df_state, tensor)
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        return self._resample(enhanced_48k, target_sr, sr)

    def _enhance_rnnoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio_48k = self._resample(audio, sr, self.DF_SAMPLE_RATE)
        pcm16 = np.clip(audio_48k * 32767.0, -32768.0, 32767.0).astype(np.int16)
        chunks = [
            frame
            for _vad_prob, frame in self._rnnoise.denoise_chunk(pcm16)
        ]
        if not chunks:
            return audio
        enhanced_48k = (
            np.concatenate(chunks).astype(np.float32).reshape(-1) / 32768.0
        )
        return self._resample(enhanced_48k, self.DF_SAMPLE_RATE, sr)

    def _enhance_spectral(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Lightweight causal-style Wiener / spectral subtraction on CPU."""
        if not HAS_SCIPY or audio.size < 512:
            return audio
        nperseg = 512 if sr >= 16000 else 256
        noverlap = nperseg // 2
        _, _, zxx = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(zxx)
        phase = np.angle(zxx)
        power = mag ** 2

        # Estimate noise PSD from the quietest 15% of time frames
        frame_energy = np.mean(power, axis=0)
        n_noise_frames = max(1, int(np.ceil(frame_energy.size * 0.15)))
        quiet_idx = np.argsort(frame_energy)[:n_noise_frames]
        noise_psd = np.mean(power[:, quiet_idx], axis=1, keepdims=True) + 1e-10

        # Smooth Wiener gain with gentle oversubtraction & floor to avoid artifacts
        alpha = 1.4
        spectral_floor = 0.08
        snr_post = power / noise_psd
        gain = np.maximum(1.0 - alpha / np.maximum(snr_post, 1e-6), spectral_floor)
        # Mild temporal smoothing of gain across frames
        if gain.shape[1] >= 3:
            gain[:, 1:-1] = 0.25 * gain[:, :-2] + 0.5 * gain[:, 1:-1] + 0.25 * gain[:, 2:]

        enhanced_zxx = (mag * gain) * np.exp(1j * phase)
        _, enhanced = istft(enhanced_zxx, fs=sr, nperseg=nperseg, noverlap=noverlap)
        if enhanced.size > audio.size:
            enhanced = enhanced[: audio.size]
        elif enhanced.size < audio.size:
            enhanced = np.pad(enhanced, (0, audio.size - enhanced.size))
        return enhanced.astype(np.float32)

    def enhance(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> np.ndarray:
        """Apply high-pass filter and configured CPU speech enhancement."""
        if audio is None or audio.size == 0:
            return audio
        sr = int(sample_rate or self.sample_rate)
        sig = np.asarray(audio, dtype=np.float32).reshape(-1)
        sig = sig - float(np.mean(sig))  # Remove DC offset
        sig = self._apply_highpass(sig)

        try:
            if self.active_backend == "deepfilternet2":
                sig = self._enhance_deepfilternet2(sig, sr)
            elif self.active_backend == "rnnoise":
                sig = self._enhance_rnnoise(sig, sr)
            elif self.active_backend == "spectral":
                sig = self._enhance_spectral(sig, sr)
        except Exception as exc:
            logger.warning(
                "AudioEnhancer (%s) gagal, fallback ke sinyal HPF: %s",
                self.active_backend,
                exc,
            )

        return np.clip(sig, -1.0, 1.0).astype(np.float32, copy=False)
