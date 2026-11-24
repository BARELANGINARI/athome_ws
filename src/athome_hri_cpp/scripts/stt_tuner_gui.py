#!/usr/bin/env python3
"""Interactive PyQt5 GUI for calibrating and live-tuning STT parameters per environment.

Supports both `athome_hri` (Python) and `athome_hri_cpp` (C++) YAML configs,
real-time directional-mic Speech-Band dBFS & Spectral Flatness monitoring,
2-step ambient-noise + operator-voice auto-calibration, live `ros2 param set`
tuning, and live `/hri/stt_transcript` monitoring.
"""

from collections import deque
import json
from math import gcd
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from scipy.signal import butter, lfilter, resample_poly
import yaml

try:
    import pyaudio

    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False


# ==============================================================================
# 1. Pure DSP, Calibration & YAML Helpers (Unit-Testable)
# ==============================================================================


class SpeechBandAnalyzer:
    """Computes full-band dBFS, 300-3400 Hz speech-band dBFS, and spectral flatness."""

    def __init__(self, sample_rate: int = 16000, block_size: int = 480) -> None:
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        nyq = 0.5 * self.sample_rate
        self._b, self._a = butter(4, [300.0 / nyq, 3400.0 / nyq], btype="band")
        self._hann = np.hanning(self.block_size).astype(np.float32)

    def analyze(self, frame: np.ndarray) -> Tuple[float, float, float]:
        eps = 1e-10
        sig = np.asarray(frame, dtype=np.float32).reshape(-1)
        rms = float(np.sqrt(np.mean(sig ** 2) + eps))
        dbfs = float(20.0 * np.log10(rms + eps))

        speech = lfilter(self._b, self._a, sig)
        rms_sp = float(np.sqrt(np.mean(speech ** 2) + eps))
        dbfs_speech = float(20.0 * np.log10(rms_sp + eps))

        win = self._hann if sig.size == self._hann.size else np.hanning(sig.size).astype(np.float32)
        fft_mag = np.abs(np.fft.rfft(sig * win)) + eps
        flatness = float(np.exp(np.mean(np.log(fft_mag))) / np.mean(fft_mag))
        return dbfs, dbfs_speech, flatness


def has_consecutive_repetition(text: str, max_ngram: int = 4) -> bool:
    tokens = [re.sub(r"[^\w]+", "", t.lower()) for t in text.split()]
    tokens = [t for t in tokens if t]
    if len(tokens) < 2:
        return False
    for n in range(1, min(max_ngram, len(tokens) // 2) + 1):
        for i in range(len(tokens) - 2 * n + 1):
            if tokens[i : i + n] == tokens[i + n : i + 2 * n]:
                return True
    return False


def compute_optimal_params(
    noise_stats: Dict[str, float], voice_stats: Dict[str, float]
) -> Dict[str, Any]:
    """Derive optimal VAD, Denoiser, and Whisper parameters from 5s noise + 5s voice samples."""
    noise_p50 = float(noise_stats.get("p50_dbfs", -48.0))
    noise_p90 = float(noise_stats.get("p90_dbfs", -42.0))
    noise_flat_p50 = float(noise_stats.get("p50_flatness", 0.55))

    voice_p75 = float(voice_stats.get("p75_dbfs", -20.0))
    voice_flat_p90 = float(voice_stats.get("p90_flatness", 0.25))

    snr_db = voice_p75 - noise_p50

    if noise_p50 < -45.0 and snr_db >= 18.0:
        env_class = "Quiet Lab"
    elif noise_p50 < -34.0 and snr_db >= 12.0:
        env_class = "Moderate Hall"
    elif snr_db >= 7.0:
        env_class = "Noisy Arena"
    else:
        env_class = "Extreme Noise (Low SNR)"

    # Threshold floor: safely above median noise, below operator voice p75
    raw_floor = max(noise_p50 + 6.0, min(noise_p90 + 2.0, voice_p75 - 4.0))
    threshold_dbfs = round(float(np.clip(raw_floor, -45.0, -16.0)), 1)

    # Adaptive SNR margin scales gently downward when SNR is tight so quiet speech is not clipped
    snr_margin = round(float(np.clip(snr_db * 0.45, 7.0, 14.0)), 1)
    max_thresh = round(float(np.clip(voice_p75 - 3.0, -24.0, -8.0)), 1)

    # Flatness gate sits between voice p90 and ambient noise p50
    flat_mid = 0.5 * (voice_flat_p90 + noise_flat_p50)
    threshold_flatness = round(float(np.clip(max(voice_flat_p90 + 0.08, flat_mid), 0.25, 0.55)), 2)

    if env_class == "Quiet Lab":
        hangover, min_sp = 18, 8
        nvad_th, no_sp_th, log_prob = 0.55, 0.55, -0.9
        spec_alpha, spec_floor, post_mute = 1.1, 0.12, 0.45
    elif env_class == "Moderate Hall":
        hangover, min_sp = 16, 9
        nvad_th, no_sp_th, log_prob = 0.65, 0.50, -0.8
        spec_alpha, spec_floor, post_mute = 1.4, 0.08, 0.60
    elif env_class == "Noisy Arena":
        hangover, min_sp = 13, 10
        nvad_th, no_sp_th, log_prob = 0.72, 0.45, -0.75
        spec_alpha, spec_floor, post_mute = 1.7, 0.06, 0.70
    else:
        hangover, min_sp = 12, 11
        nvad_th, no_sp_th, log_prob = 0.78, 0.40, -0.65
        spec_alpha, spec_floor, post_mute = 2.0, 0.05, 0.80

    params = {
        "threshold_dbfs": threshold_dbfs,
        "adaptive_vad_enabled": True,
        "adaptive_snr_margin_db": snr_margin,
        "adaptive_max_threshold_dbfs": max_thresh,
        "threshold_flatness": threshold_flatness,
        "hangover_frames": hangover,
        "min_speech_frames": min_sp,
        "stt_vad_threshold": nvad_th,
        "stt_no_speech_threshold": no_sp_th,
        "stt_log_prob_threshold": log_prob,
        "stt_spectral_alpha": spec_alpha,
        "stt_spectral_floor": spec_floor,
        "tts_post_mute_sec": post_mute,
    }
    diagnosis = (
        f"Environment: {env_class} | Noise p50={noise_p50:.1f} dBFS (p90={noise_p90:.1f}) | "
        f"Voice p75={voice_p75:.1f} dBFS | Est. SNR={snr_db:.1f} dB"
    )
    return {"params": params, "diagnosis": diagnosis, "environment": env_class, "snr_db": round(snr_db, 1)}


def _format_yaml_value(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return f"{val:.4g}" if abs(val) < 1e-2 else f"{val:.2f}".rstrip("0").rstrip(".") if "." in f"{val:.2f}" else str(val)
    if isinstance(val, str):
        escaped = val.replace('"', '\\"')
        return f'"{escaped}"'
    return str(val)


def update_yaml_preserving_comments(yaml_text: str, updates: Dict[str, Any]) -> Tuple[str, int]:
    """Update existing scalar keys in YAML text while preserving comments and formatting."""
    lines = yaml_text.splitlines(keepends=True)
    updated_count = 0
    out_lines: List[str] = []

    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            out_lines.append(line)
            continue

        matched_key = None
        for key in updates:
            m = re.match(rf"^(\s*)({re.escape(key)}:\s*)([^#\r\n]*?)(\s*(#.*)?)$", line.rstrip("\r\n"))
            if m:
                matched_key = key
                indent, prefix, _old_val, comment_part, _ = m.groups()
                newline = "\r\n" if line.endswith("\r\n") else ("\n" if line.endswith("\n") else "")
                new_val = _format_yaml_value(updates[key])
                pad = " " * max(1, 20 - len(new_val)) if comment_part and comment_part.strip().startswith("#") else ""
                comment_clean = comment_part.lstrip() if comment_part else ""
                if comment_clean:
                    out_lines.append(f"{indent}{prefix}{new_val}{pad}{comment_clean}{newline}")
                else:
                    out_lines.append(f"{indent}{prefix}{new_val}{newline}")
                updated_count += 1
                break

        if matched_key is None:
            out_lines.append(line)

    return "".join(out_lines), updated_count


PRESETS: Dict[str, Dict[str, Any]] = {
    "Quiet Lab": {
        "threshold_dbfs": -36.0,
        "adaptive_vad_enabled": True,
        "adaptive_snr_margin_db": 10.0,
        "adaptive_max_threshold_dbfs": -14.0,
        "threshold_flatness": 0.42,
        "hangover_frames": 18,
        "min_speech_frames": 8,
        "stt_vad_threshold": 0.55,
        "stt_no_speech_threshold": 0.55,
        "stt_log_prob_threshold": -0.9,
        "stt_spectral_alpha": 1.1,
        "stt_spectral_floor": 0.12,
        "tts_post_mute_sec": 0.45,
    },
    "Moderate Hall": {
        "threshold_dbfs": -30.0,
        "adaptive_vad_enabled": True,
        "adaptive_snr_margin_db": 10.0,
        "adaptive_max_threshold_dbfs": -12.0,
        "threshold_flatness": 0.40,
        "hangover_frames": 16,
        "min_speech_frames": 9,
        "stt_vad_threshold": 0.65,
        "stt_no_speech_threshold": 0.50,
        "stt_log_prob_threshold": -0.8,
        "stt_spectral_alpha": 1.4,
        "stt_spectral_floor": 0.08,
        "tts_post_mute_sec": 0.60,
    },
    "Noisy Competition Arena": {
        "threshold_dbfs": -24.0,
        "adaptive_vad_enabled": True,
        "adaptive_snr_margin_db": 8.5,
        "adaptive_max_threshold_dbfs": -10.0,
        "threshold_flatness": 0.34,
        "hangover_frames": 13,
        "min_speech_frames": 10,
        "stt_vad_threshold": 0.74,
        "stt_no_speech_threshold": 0.45,
        "stt_log_prob_threshold": -0.7,
        "stt_spectral_alpha": 1.8,
        "stt_spectral_floor": 0.06,
        "tts_post_mute_sec": 0.75,
    },
    "Far-Field (1.5m+)": {
        "threshold_dbfs": -38.0,
        "adaptive_vad_enabled": True,
        "adaptive_snr_margin_db": 7.5,
        "adaptive_max_threshold_dbfs": -16.0,
        "threshold_flatness": 0.44,
        "hangover_frames": 20,
        "min_speech_frames": 8,
        "stt_vad_threshold": 0.52,
        "stt_no_speech_threshold": 0.55,
        "stt_log_prob_threshold": -0.95,
        "stt_spectral_alpha": 1.5,
        "stt_spectral_floor": 0.09,
        "tts_post_mute_sec": 0.65,
    },
}


# ==============================================================================
# 2. Background Workers (PyAudio Capture, ROS 2 Transcript, ROS 2 Param Set)
# ==============================================================================


def list_pyaudio_input_devices() -> List[Tuple[int, str]]:
    if not HAS_PYAUDIO:
        return []
    devices: List[Tuple[int, str]] = []
    try:
        pa = pyaudio.PyAudio()
        try:
            for idx in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(idx)
                if info.get("maxInputChannels", 0) > 0:
                    devices.append((idx, str(info.get("name", f"Device {idx}"))))
        finally:
            pa.terminate()
    except Exception:
        pass
    return devices


class AudioMonitorThread(QtCore.QThread):
    metrics_ready = QtCore.pyqtSignal(dict)
    calibration_progress = QtCore.pyqtSignal(int)
    calibration_finished = QtCore.pyqtSignal(str, dict)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, device_index: Optional[int] = None, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.device_index = device_index
        self._running = False
        self._lock = threading.Lock()
        self._vad_cfg = {
            "threshold_dbfs": -30.0,
            "threshold_flatness": 0.40,
            "adaptive_vad_enabled": True,
            "adaptive_snr_margin_db": 10.0,
            "adaptive_max_threshold_dbfs": -12.0,
        }
        self._calib_mode: Optional[str] = None
        self._calib_until: float = 0.0
        self._calib_dbfs: List[float] = []
        self._calib_flat: List[float] = []

    def update_vad_settings(self, cfg: Dict[str, Any]) -> None:
        with self._lock:
            for k in self._vad_cfg:
                if k in cfg:
                    self._vad_cfg[k] = cfg[k]

    def start_calibration_capture(self, mode: str, duration_sec: float = 5.0) -> None:
        with self._lock:
            self._calib_mode = mode
            self._calib_until = time.monotonic() + duration_sec
            self._calib_duration = duration_sec
            self._calib_dbfs = []
            self._calib_flat = []

    def stop_monitor(self) -> None:
        self._running = False
        self.wait(2000)

    def run(self) -> None:
        if not HAS_PYAUDIO:
            self.error_occurred.emit("PyAudio tidak terinstal.")
            return

        self._running = True
        target_sr = 16000
        block_sec = 0.03
        analyzer = SpeechBandAnalyzer(target_sr, int(target_sr * block_sec))
        pa = None
        stream = None
        hw_rate = target_sr
        noise_floor = -50.0

        try:
            pa = pyaudio.PyAudio()
            rates = [16000, 48000, 44100]
            if self.device_index is not None and self.device_index >= 0:
                try:
                    def_sr = int(pa.get_device_info_by_index(self.device_index).get("defaultSampleRate", 0))
                    if def_sr > 0 and def_sr not in rates:
                        rates.insert(0, def_sr)
                except Exception:
                    pass

            for r in rates:
                try:
                    hw_block = int(r * block_sec)
                    stream = pa.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=r,
                        input=True,
                        input_device_index=None if (self.device_index is None or self.device_index < 0) else self.device_index,
                        frames_per_buffer=hw_block,
                    )
                    hw_rate = r
                    break
                except Exception:
                    stream = None

            if stream is None:
                self.error_occurred.emit("Gagal membuka stream mikrofon (mungkin sedang dipakai stt_engine_node).")
                return

            hw_block = int(hw_rate * block_sec)
            while self._running:
                raw = stream.read(hw_block, exception_on_overflow=False)
                sig = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if hw_rate != target_sr:
                    g = gcd(target_sr, hw_rate)
                    sig = resample_poly(sig, target_sr // g, hw_rate // g).astype(np.float32)

                dbfs, dbfs_sp, flat = analyzer.analyze(sig)
                with self._lock:
                    cfg = dict(self._vad_cfg)

                eff_thr = cfg["threshold_dbfs"]
                if cfg["adaptive_vad_enabled"]:
                    eff_thr = max(
                        cfg["threshold_dbfs"],
                        min(cfg["adaptive_max_threshold_dbfs"], noise_floor + cfg["adaptive_snr_margin_db"]),
                    )

                is_voice = (dbfs_sp > eff_thr) and (flat < cfg["threshold_flatness"])
                if cfg["adaptive_vad_enabled"] and dbfs_sp > -95.0 and not is_voice:
                    alpha = 0.05 if dbfs_sp > noise_floor else 0.15
                    noise_floor = (1.0 - alpha) * noise_floor + alpha * dbfs_sp

                self.metrics_ready.emit(
                    {
                        "dbfs": dbfs,
                        "dbfs_speech": dbfs_sp,
                        "flatness": flat,
                        "noise_floor": noise_floor,
                        "effective_threshold": eff_thr,
                        "threshold_dbfs": cfg["threshold_dbfs"],
                        "is_voice": is_voice,
                    }
                )

                # Handle active 5-second calibration recording
                done_payload = None
                with self._lock:
                    if self._calib_mode is not None:
                        self._calib_dbfs.append(dbfs_sp)
                        self._calib_flat.append(flat)
                        rem = max(0.0, self._calib_until - time.monotonic())
                        pct = int(min(100.0, 100.0 * (1.0 - rem / max(0.1, self._calib_duration))))
                        self.calibration_progress.emit(pct)
                        if rem <= 0.0:
                            arr_db = np.array(self._calib_dbfs, dtype=np.float32)
                            arr_fl = np.array(self._calib_flat, dtype=np.float32)
                            done_payload = (
                                self._calib_mode,
                                {
                                    "p50_dbfs": float(np.percentile(arr_db, 50)),
                                    "p75_dbfs": float(np.percentile(arr_db, 75)),
                                    "p90_dbfs": float(np.percentile(arr_db, 90)),
                                    "p50_flatness": float(np.percentile(arr_fl, 50)),
                                    "p90_flatness": float(np.percentile(arr_fl, 90)),
                                },
                            )
                            self._calib_mode = None
                if done_payload is not None:
                    self.calibration_finished.emit(done_payload[0], done_payload[1])
        except Exception as exc:
            self.error_occurred.emit(f"Audio error: {exc}")
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            if pa is not None:
                try:
                    pa.terminate()
                except Exception:
                    pass


class RosTranscriptThread(QtCore.QThread):
    transcript_received = QtCore.pyqtSignal(str, bool)
    status_changed = QtCore.pyqtSignal(str)

    def __init__(self, topic: str = "/hri/stt_transcript", parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.topic = topic
        self._stop = threading.Event()
        self._proc: Optional[subprocess.Popen] = None

    def stop_feed(self) -> None:
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self.wait(1500)

    def run(self) -> None:
        self._stop.clear()
        try:
            self._proc = subprocess.Popen(
                ["ros2", "topic", "echo", self.topic, "--field", "data"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self.status_changed.emit(f"Listening to {self.topic}...")
            if self._proc.stdout is not None:
                for line in self._proc.stdout:
                    if self._stop.is_set():
                        break
                    s = line.strip()
                    if s and s != "---":
                        txt = s.strip("'\"")
                        self.transcript_received.emit(txt, has_consecutive_repetition(txt))
        except FileNotFoundError:
            self.status_changed.emit("ros2 CLI tidak ditemukan di PATH.")
        except Exception as exc:
            self.status_changed.emit(f"ROS feed error: {exc}")


class RosParamApplyThread(QtCore.QThread):
    finished_summary = QtCore.pyqtSignal(int, int, str)

    def __init__(self, node_name: str, params: Dict[str, Any], parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.node_name = node_name
        self.params = dict(params)

    def run(self) -> None:
        applied = 0
        failed = 0
        for key, val in self.params.items():
            val_str = ("true" if val else "false") if isinstance(val, bool) else str(val)
            try:
                res = subprocess.run(
                    ["ros2", "param", "set", self.node_name, key, val_str],
                    capture_output=True,
                    text=True,
                    timeout=2.5,
                )
                if res.returncode == 0:
                    applied += 1
                else:
                    failed += 1
            except Exception:
                failed += 1
        msg = f"Applied {applied} params live to {self.node_name}" + (
            f" ({failed} skipped/offline)" if failed else ""
        )
        self.finished_summary.emit(applied, failed, msg)


# ==============================================================================
# 3. Custom Lightweight PyQt5 Visualizations
# ==============================================================================


class LevelMeterWidget(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(44)
        self._dbfs = -70.0
        self._noise = -50.0
        self._thresh = -30.0
        self._eff_thresh = -30.0
        self._is_voice = False
        self._active = False

    def set_values(
        self, dbfs: float, noise: float, thresh: float, eff_thresh: float, is_voice: bool, active: bool
    ) -> None:
        self._dbfs = dbfs
        self._noise = noise
        self._thresh = thresh
        self._eff_thresh = eff_thresh
        self._is_voice = is_voice
        self._active = active
        self.update()

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect().adjusted(4, 4, -4, -4)
        p.fillRect(rect, QtGui.QColor("#1a1d24"))
        p.setPen(QtGui.QColor("#333945"))
        p.drawRoundedRect(rect, 5, 5)

        def _x(db: float) -> int:
            ratio = max(0.0, min(1.0, (db + 70.0) / 70.0))
            return rect.left() + int(ratio * rect.width())

        if self._active:
            bar_w = max(0, _x(self._dbfs) - rect.left())
            color = QtGui.QColor("#22c55e" if self._is_voice else "#3b82f6")
            p.fillRect(rect.left() + 1, rect.top() + 1, max(0, bar_w - 2), rect.height() - 2, color)

            # Noise floor marker (amber)
            nx = _x(self._noise)
            p.setPen(QtGui.QPen(QtGui.QColor("#f59e0b"), 2))
            p.drawLine(nx, rect.top() + 2, nx, rect.bottom() - 2)

            # Static threshold (red) & Effective adaptive threshold (cyan)
            tx = _x(self._thresh)
            p.setPen(QtGui.QPen(QtGui.QColor("#ef4444"), 2))
            p.drawLine(tx, rect.top() + 2, tx, rect.bottom() - 2)

            ex = _x(self._eff_thresh)
            p.setPen(QtGui.QPen(QtGui.QColor("#06b6d4"), 2, QtCore.Qt.DashLine))
            p.drawLine(ex, rect.top() + 2, ex, rect.bottom() - 2)

        badge = "OFF" if not self._active else ("VOICE" if self._is_voice else "SILENCE")
        p.setPen(QtGui.QColor("#f8fafc"))
        p.setFont(QtGui.QFont("Monospace", 10, QtGui.QFont.Bold))
        p.drawText(
            rect.adjusted(8, 0, -8, 0),
            QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight,
            f"{self._dbfs:5.1f} dBFS  [{badge}]",
        )


class RollingPlotWidget(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(130)
        self._hist_sp: deque = deque([-65.0] * 120, maxlen=120)
        self._hist_thr: deque = deque([-30.0] * 120, maxlen=120)
        self._hist_nf: deque = deque([-50.0] * 120, maxlen=120)

    def push_point(self, dbfs_sp: float, eff_thr: float, noise_floor: float) -> None:
        self._hist_sp.append(dbfs_sp)
        self._hist_thr.append(eff_thr)
        self._hist_nf.append(noise_floor)
        self.update()

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect().adjusted(4, 4, -4, -4)
        p.fillRect(rect, QtGui.QColor("#11141a"))
        p.setPen(QtGui.QColor("#2c323d"))
        p.drawRect(rect)

        def _y(db: float) -> float:
            ratio = max(0.0, min(1.0, (db + 70.0) / 70.0))
            return rect.bottom() - ratio * rect.height()

        # Grid lines every 20 dB
        p.setPen(QtGui.QPen(QtGui.QColor("#222630"), 1, QtCore.Qt.DotLine))
        for db in (-60, -40, -20):
            yy = int(_y(db))
            p.drawLine(rect.left(), yy, rect.right(), yy)

        def _draw_series(vals: deque, color: str, width: float, style=QtCore.Qt.SolidLine) -> None:
            pts = []
            n = len(vals)
            for i, v in enumerate(vals):
                px = rect.left() + (float(i) / max(1, n - 1)) * rect.width()
                pts.append(QtCore.QPointF(px, _y(v)))
            p.setPen(QtGui.QPen(QtGui.QColor(color), width, style))
            for i in range(1, len(pts)):
                p.drawLine(pts[i - 1], pts[i])

        _draw_series(self._hist_nf, "#f59e0b", 1.5, QtCore.Qt.DotLine)
        _draw_series(self._hist_thr, "#06b6d4", 1.8, QtCore.Qt.DashLine)
        _draw_series(self._hist_sp, "#22c55e", 2.0, QtCore.Qt.SolidLine)


# ==============================================================================
# 4. Main Window
# ==============================================================================


class SttTunerWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Barelang Inari — STT Environment Tuner (Python & C++)")
        self.resize(1180, 760)

        self._ws_src = Path(__file__).resolve().parents[2]
        self._py_yaml_path = self._ws_src / "athome_hri" / "config" / "params.yaml"
        self._cpp_yaml_path = self._ws_src / "athome_hri_cpp" / "config" / "params.yaml"

        self._audio_thread: Optional[AudioMonitorThread] = None
        self._ros_thread: Optional[RosTranscriptThread] = None
        self._apply_thread: Optional[RosParamApplyThread] = None
        self._noise_stats: Optional[Dict[str, float]] = None
        self._voice_stats: Optional[Dict[str, float]] = None
        self._widgets: Dict[str, Any] = {}

        self._build_ui()
        self._apply_dark_theme()
        self._refresh_mics()
        self._load_from_yaml()

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background-color: #14171f; color: #e2e8f0; font-size: 12px; }
            QGroupBox { border: 1px solid #2d3442; border-radius: 6px; margin-top: 10px; padding-top: 10px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #38bdf8; }
            QPushButton { background-color: #252b38; border: 1px solid #3a4254; border-radius: 4px; padding: 6px 12px; }
            QPushButton:hover { background-color: #323a4b; }
            QPushButton#primaryBtn { background-color: #0284c7; border-color: #0ea5e9; font-weight: bold; }
            QPushButton#primaryBtn:hover { background-color: #0369a1; }
            QPushButton#saveBtn { background-color: #15803d; border-color: #22c55e; font-weight: bold; }
            QPushButton#saveBtn:hover { background-color: #166534; }
            QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit { background-color: #1b202b; border: 1px solid #333b4d; border-radius: 4px; padding: 3px 6px; }
            QTabWidget::pane { border: 1px solid #2d3442; border-radius: 4px; }
            QTabBar::tab { background: #1c212c; padding: 6px 14px; border: 1px solid #2d3442; }
            QTabBar::tab:selected { background: #283040; color: #38bdf8; font-weight: bold; }
            """
        )

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # ---- LEFT COLUMN: Audio Monitor, Calibration & ROS Transcript ----
        left_col = QtWidgets.QVBoxLayout()
        root.addLayout(left_col, 5)

        mic_box = QtWidgets.QGroupBox("1. Real-Time Directional Mic Monitor")
        mic_lay = QtWidgets.QVBoxLayout(mic_box)

        mic_row = QtWidgets.QHBoxLayout()
        self.mic_combo = QtWidgets.QComboBox()
        self.btn_refresh_mic = QtWidgets.QPushButton("Refresh")
        self.btn_toggle_mic = QtWidgets.QPushButton("Start Monitor")
        self.btn_toggle_mic.setObjectName("primaryBtn")
        mic_row.addWidget(self.mic_combo, 1)
        mic_row.addWidget(self.btn_refresh_mic)
        mic_row.addWidget(self.btn_toggle_mic)
        mic_lay.addLayout(mic_row)

        self.level_meter = LevelMeterWidget()
        self.rolling_plot = RollingPlotWidget()
        mic_lay.addWidget(self.level_meter)
        mic_lay.addWidget(self.rolling_plot)

        self.lbl_metrics = QtWidgets.QLabel(
            "Speech: -- dBFS | Noise Floor: -- dBFS | Eff. Thr: -- dBFS | Flatness: --"
        )
        self.lbl_metrics.setStyleSheet("color: #94a3b8; font-family: monospace;")
        mic_lay.addWidget(self.lbl_metrics)
        left_col.addWidget(mic_box)

        cal_box = QtWidgets.QGroupBox("2. Auto-Kalibrasi Ruangan (5s Noise + 5s Suara)")
        cal_lay = QtWidgets.QVBoxLayout(cal_box)
        cal_btns = QtWidgets.QHBoxLayout()
        self.btn_cal_noise = QtWidgets.QPushButton("1. Measure Noise (5s)")
        self.btn_cal_voice = QtWidgets.QPushButton("2. Measure Voice (5s)")
        self.btn_cal_apply = QtWidgets.QPushButton("Auto-Calculate Params")
        self.btn_cal_apply.setObjectName("primaryBtn")
        self.btn_cal_apply.setEnabled(False)
        cal_btns.addWidget(self.btn_cal_noise)
        cal_btns.addWidget(self.btn_cal_voice)
        cal_btns.addWidget(self.btn_cal_apply)
        cal_lay.addLayout(cal_btns)

        self.cal_progress = QtWidgets.QProgressBar()
        self.cal_progress.setValue(0)
        self.lbl_cal_status = QtWidgets.QLabel(
            "Start Monitor lalu rekam 5s kondisi ruangan hening dan 5s suara operator."
        )
        self.lbl_cal_status.setWordWrap(True)
        cal_lay.addWidget(self.cal_progress)
        cal_lay.addWidget(self.lbl_cal_status)
        left_col.addWidget(cal_box)

        ros_box = QtWidgets.QGroupBox("3. Live /hri/stt_transcript Verification")
        ros_lay = QtWidgets.QVBoxLayout(ros_box)
        ros_btns = QtWidgets.QHBoxLayout()
        self.btn_ros_feed = QtWidgets.QPushButton("Start ROS Feed")
        self.btn_ros_clear = QtWidgets.QPushButton("Clear")
        self.lbl_ros_status = QtWidgets.QLabel("Idle (stop Monitor sebelum menjalankan stt_engine_node)")
        ros_btns.addWidget(self.btn_ros_feed)
        ros_btns.addWidget(self.btn_ros_clear)
        ros_btns.addWidget(self.lbl_ros_status, 1)
        ros_lay.addLayout(ros_btns)

        self.transcript_log = QtWidgets.QTextEdit()
        self.transcript_log.setReadOnly(True)
        ros_lay.addWidget(self.transcript_log)
        left_col.addWidget(ros_box, 1)

        # ---- RIGHT COLUMN: Presets, Parameter Tabs, & YAML / Live Actions ----
        right_col = QtWidgets.QVBoxLayout()
        root.addLayout(right_col, 6)

        preset_row = QtWidgets.QHBoxLayout()
        preset_row.addWidget(QtWidgets.QLabel("Preset Lokasi:"))
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(list(PRESETS.keys()))
        self.btn_apply_preset = QtWidgets.QPushButton("Apply Preset")
        self.btn_export_json = QtWidgets.QPushButton("Export Profile")
        self.btn_import_json = QtWidgets.QPushButton("Import Profile")
        preset_row.addWidget(self.preset_combo, 1)
        preset_row.addWidget(self.btn_apply_preset)
        preset_row.addWidget(self.btn_export_json)
        preset_row.addWidget(self.btn_import_json)
        right_col.addLayout(preset_row)

        self.tabs = QtWidgets.QTabWidget()
        right_col.addWidget(self.tabs, 1)

        self._build_param_tabs()

        action_box = QtWidgets.QGroupBox("Target Config & Deployment")
        act_lay = QtWidgets.QHBoxLayout(action_box)
        self.target_yaml_combo = QtWidgets.QComboBox()
        self.target_yaml_combo.addItems(["Both (Python + C++)", "C++ YAML (athome_hri_cpp)", "Python YAML (athome_hri)"])
        self.btn_load_yaml = QtWidgets.QPushButton("Load YAML")
        self.btn_apply_live = QtWidgets.QPushButton("Apply Live (/stt_engine)")
        self.btn_apply_live.setObjectName("primaryBtn")
        self.btn_save_yaml = QtWidgets.QPushButton("Save to YAML")
        self.btn_save_yaml.setObjectName("saveBtn")

        act_lay.addWidget(self.target_yaml_combo, 1)
        act_lay.addWidget(self.btn_load_yaml)
        act_lay.addWidget(self.btn_apply_live)
        act_lay.addWidget(self.btn_save_yaml)
        right_col.addWidget(action_box)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

        # Signals
        self.btn_refresh_mic.clicked.connect(self._refresh_mics)
        self.btn_toggle_mic.clicked.connect(self._toggle_mic_monitor)
        self.btn_cal_noise.clicked.connect(lambda: self._start_calib("noise"))
        self.btn_cal_voice.clicked.connect(lambda: self._start_calib("voice"))
        self.btn_cal_apply.clicked.connect(self._apply_auto_calibration)
        self.btn_ros_feed.clicked.connect(self._toggle_ros_feed)
        self.btn_ros_clear.clicked.connect(self.transcript_log.clear)
        self.btn_apply_preset.clicked.connect(self._apply_selected_preset)
        self.btn_export_json.clicked.connect(self._export_profile)
        self.btn_import_json.clicked.connect(self._import_profile)
        self.btn_load_yaml.clicked.connect(self._load_from_yaml)
        self.btn_apply_live.clicked.connect(self._apply_live_ros)
        self.btn_save_yaml.clicked.connect(self._save_to_yaml)

    def _add_float_row(
        self, form: QtWidgets.QFormLayout, key: str, label: str, lo: float, hi: float, step: float, decimals: int = 2
    ) -> None:
        row = QtWidgets.QHBoxLayout()
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        scale = int(round(1.0 / step))
        slider.setRange(int(round(lo * scale)), int(round(hi * scale)))
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)

        slider.valueChanged.connect(lambda v: spin.setValue(v / scale))
        spin.valueChanged.connect(lambda v: slider.setValue(int(round(v * scale))))
        spin.valueChanged.connect(self._push_vad_to_monitor)

        row.addWidget(slider, 1)
        row.addWidget(spin)
        form.addRow(label, row)
        self._widgets[key] = spin

    def _add_int_row(self, form: QtWidgets.QFormLayout, key: str, label: str, lo: int, hi: int) -> None:
        row = QtWidgets.QHBoxLayout()
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(lo, hi)
        spin = QtWidgets.QSpinBox()
        spin.setRange(lo, hi)
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        row.addWidget(slider, 1)
        row.addWidget(spin)
        form.addRow(label, row)
        self._widgets[key] = spin

    def _add_bool_row(self, form: QtWidgets.QFormLayout, key: str, label: str) -> None:
        chk = QtWidgets.QCheckBox()
        chk.stateChanged.connect(self._push_vad_to_monitor)
        form.addRow(label, chk)
        self._widgets[key] = chk

    def _add_combo_row(self, form: QtWidgets.QFormLayout, key: str, label: str, items: List[str]) -> None:
        cmb = QtWidgets.QComboBox()
        cmb.addItems(items)
        form.addRow(label, cmb)
        self._widgets[key] = cmb

    def _build_param_tabs(self) -> None:
        # Tab 1: Capture & Adaptive VAD
        t1 = QtWidgets.QWidget()
        f1 = QtWidgets.QFormLayout(t1)
        self._add_float_row(f1, "threshold_dbfs", "Threshold Floor (dBFS)", -60.0, -5.0, 0.5, 1)
        self._add_float_row(f1, "threshold_flatness", "Spectral Flatness Max", 0.10, 0.90, 0.01, 2)
        self._add_bool_row(f1, "adaptive_vad_enabled", "Adaptive Noise-Floor VAD")
        self._add_float_row(f1, "adaptive_snr_margin_db", "Adaptive SNR Margin (dB)", 3.0, 25.0, 0.5, 1)
        self._add_float_row(f1, "adaptive_max_threshold_dbfs", "Adaptive Max Threshold (dBFS)", -35.0, -5.0, 0.5, 1)
        self._add_int_row(f1, "pre_roll_frames", "Pre-Roll Frames (x30ms)", 0, 30)
        self._add_int_row(f1, "hangover_frames", "Hangover Frames (x30ms)", 5, 45)
        self._add_int_row(f1, "min_speech_frames", "Min Speech Frames (x30ms)", 3, 30)
        self._add_int_row(f1, "max_speech_frames", "Max Speech Frames (x30ms)", 60, 500)
        self.tabs.addTab(t1, "Capture & VAD")

        # Tab 2: Denoiser & HPF
        t2 = QtWidgets.QWidget()
        f2 = QtWidgets.QFormLayout(t2)
        self._add_combo_row(f2, "stt_denoiser_backend", "Denoiser Backend", ["auto", "deepfilternet2", "rnnoise", "spectral", "none"])
        self._add_float_row(f2, "stt_highpass_hz", "High-Pass Filter (Hz)", 0.0, 300.0, 5.0, 0)
        self._add_float_row(f2, "stt_spectral_alpha", "Spectral Subtraction Alpha", 0.5, 3.0, 0.05, 2)
        self._add_float_row(f2, "stt_spectral_floor", "Spectral Gain Floor", 0.01, 0.50, 0.01, 2)
        self._add_float_row(f2, "stt_spectral_noise_percentile", "Noise Frame Percentile", 0.05, 0.50, 0.05, 2)
        self.tabs.addTab(t2, "Denoiser & HPF")

        # Tab 3: Neural VAD, Whisper & Echo Guard
        t3 = QtWidgets.QWidget()
        f3 = QtWidgets.QFormLayout(t3)
        self._add_float_row(f3, "stt_vad_threshold", "Silero Neural VAD Threshold", 0.20, 0.95, 0.01, 2)
        self._add_int_row(f3, "stt_vad_min_speech_ms", "Neural VAD Min Speech (ms)", 100, 1000)
        self._add_int_row(f3, "stt_vad_silence_ms", "Neural VAD Silence Split (ms)", 150, 1000)
        self._add_int_row(f3, "stt_vad_speech_pad_ms", "Neural VAD Pad (ms)", 50, 400)
        self._add_float_row(f3, "stt_no_speech_threshold", "Whisper No-Speech Threshold", 0.10, 0.95, 0.05, 2)
        self._add_float_row(f3, "stt_log_prob_threshold", "Whisper Log-Prob Min", -2.5, -0.1, 0.05, 2)
        self._add_int_row(f3, "stt_beam_size", "Whisper Beam Size", 1, 5)
        self._add_int_row(f3, "stt_dedup_max_ngram", "N-Gram Deduplication Max", 1, 10)
        self._add_float_row(f3, "tts_post_mute_sec", "Post-TTS Echo Mute (s)", 0.0, 2.0, 0.05, 2)
        self.tabs.addTab(t3, "Neural VAD & Whisper")

    def _get_all_params(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, w in self._widgets.items():
            if isinstance(w, QtWidgets.QDoubleSpinBox):
                out[key] = round(float(w.value()), 4)
            elif isinstance(w, QtWidgets.QSpinBox):
                out[key] = int(w.value())
            elif isinstance(w, QtWidgets.QCheckBox):
                out[key] = bool(w.isChecked())
            elif isinstance(w, QtWidgets.QComboBox):
                out[key] = str(w.currentText())
        return out

    def _set_params_to_ui(self, params: Dict[str, Any]) -> None:
        for key, val in params.items():
            w = self._widgets.get(key)
            if w is None:
                continue
            if isinstance(w, QtWidgets.QDoubleSpinBox):
                w.setValue(float(val))
            elif isinstance(w, QtWidgets.QSpinBox):
                w.setValue(int(val))
            elif isinstance(w, QtWidgets.QCheckBox):
                w.setChecked(bool(val))
            elif isinstance(w, QtWidgets.QComboBox):
                idx = w.findText(str(val))
                if idx >= 0:
                    w.setCurrentIndex(idx)
        self._push_vad_to_monitor()

    def _refresh_mics(self) -> None:
        self.mic_combo.clear()
        self.mic_combo.addItem("Auto / Default System Mic (-1)", -1)
        for idx, name in list_pyaudio_input_devices():
            self.mic_combo.addItem(f"[{idx}] {name}", idx)

    def _toggle_mic_monitor(self) -> None:
        if self._audio_thread is not None and self._audio_thread.isRunning():
            self._audio_thread.stop_monitor()
            self._audio_thread = None
            self.btn_toggle_mic.setText("Start Monitor")
            self.level_meter.set_values(-70.0, -50.0, -30.0, -30.0, False, False)
            self.status_bar.showMessage("Mic monitor stopped.")
            return

        dev_idx = self.mic_combo.currentData()
        self._audio_thread = AudioMonitorThread(dev_idx, self)
        self._audio_thread.metrics_ready.connect(self._on_metrics)
        self._audio_thread.calibration_progress.connect(self.cal_progress.setValue)
        self._audio_thread.calibration_finished.connect(self._on_calib_finished)
        self._audio_thread.error_occurred.connect(lambda e: self.status_bar.showMessage(e))
        self._push_vad_to_monitor()
        self._audio_thread.start()
        self.btn_toggle_mic.setText("Stop Monitor")
        self.status_bar.showMessage("Mic monitor running (16 kHz Speech-Band 300-3400 Hz).")

    def _push_vad_to_monitor(self) -> None:
        if self._audio_thread is not None and self._audio_thread.isRunning():
            self._audio_thread.update_vad_settings(self._get_all_params())

    def _on_metrics(self, m: Dict[str, Any]) -> None:
        self.level_meter.set_values(
            m["dbfs_speech"], m["noise_floor"], m["threshold_dbfs"], m["effective_threshold"], m["is_voice"], True
        )
        self.rolling_plot.push_point(m["dbfs_speech"], m["effective_threshold"], m["noise_floor"])
        self.lbl_metrics.setText(
            f"Speech: {m['dbfs_speech']:5.1f} dBFS | Noise Floor: {m['noise_floor']:5.1f} dBFS | "
            f"Eff. Thr: {m['effective_threshold']:5.1f} dBFS | Flatness: {m['flatness']:.2f}"
        )

    def _start_calib(self, mode: str) -> None:
        if self._audio_thread is None or not self._audio_thread.isRunning():
            self._toggle_mic_monitor()
        if self._audio_thread is not None and self._audio_thread.isRunning():
            self.cal_progress.setValue(0)
            msg = (
                "Merekam ambient noise selama 5 detik (harap diam)..."
                if mode == "noise"
                else "Merekam suara operator selama 5 detik (ucapkan perintah normal ke mic)..."
            )
            self.lbl_cal_status.setText(msg)
            self._audio_thread.start_calibration_capture(mode, 5.0)

    def _on_calib_finished(self, mode: str, stats: Dict[str, float]) -> None:
        if mode == "noise":
            self._noise_stats = stats
            self.lbl_cal_status.setText(
                f"Noise tersimpan: p50={stats['p50_dbfs']:.1f} dBFS, flatness={stats['p50_flatness']:.2f}. "
                "Lanjutkan ke langkah 2 (Measure Voice)."
            )
        else:
            self._voice_stats = stats
            self.lbl_cal_status.setText(
                f"Voice tersimpan: p75={stats['p75_dbfs']:.1f} dBFS, flatness={stats['p90_flatness']:.2f}."
            )
        if self._noise_stats and self._voice_stats:
            self.btn_cal_apply.setEnabled(True)

    def _apply_auto_calibration(self) -> None:
        if not (self._noise_stats and self._voice_stats):
            return
        res = compute_optimal_params(self._noise_stats, self._voice_stats)
        self._set_params_to_ui(res["params"])
        self.lbl_cal_status.setText(f"Optimal params applied! {res['diagnosis']}")
        self.status_bar.showMessage("Auto-calibration applied to UI.")

    def _toggle_ros_feed(self) -> None:
        if self._ros_thread is not None and self._ros_thread.isRunning():
            self._ros_thread.stop_feed()
            self._ros_thread = None
            self.btn_ros_feed.setText("Start ROS Feed")
            self.lbl_ros_status.setText("Stopped")
            return
        self._ros_thread = RosTranscriptThread("/hri/stt_transcript", self)
        self._ros_thread.transcript_received.connect(self._on_transcript)
        self._ros_thread.status_changed.connect(self.lbl_ros_status.setText)
        self._ros_thread.start()
        self.btn_ros_feed.setText("Stop ROS Feed")

    def _on_transcript(self, text: str, repeated: bool) -> None:
        ts = time.strftime("%H:%M:%S")
        if repeated:
            self.transcript_log.append(f"<span style='color:#f87171'>[{ts}] {text} [REPETITION]</span>")
        else:
            self.transcript_log.append(f"<span style='color:#4ade80'>[{ts}] {text}</span>")

    def _apply_selected_preset(self) -> None:
        name = self.preset_combo.currentText()
        if name in PRESETS:
            self._set_params_to_ui(PRESETS[name])
            self.status_bar.showMessage(f"Preset '{name}' loaded into UI.")

    def _export_profile(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Profile JSON", "", "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps(self._get_all_params(), indent=2), encoding="utf-8")
            self.status_bar.showMessage(f"Profile exported to {path}")

    def _import_profile(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import Profile JSON", "", "JSON (*.json)")
        if path:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self._set_params_to_ui(data)
            self.status_bar.showMessage(f"Profile imported from {path}")

    def _selected_yaml_paths(self) -> List[Path]:
        mode = self.target_yaml_combo.currentText()
        if mode.startswith("Both"):
            return [self._cpp_yaml_path, self._py_yaml_path]
        if mode.startswith("C++"):
            return [self._cpp_yaml_path]
        return [self._py_yaml_path]

    def _load_from_yaml(self) -> None:
        for cand in self._selected_yaml_paths():
            if cand.is_file():
                data = yaml.safe_load(cand.read_text(encoding="utf-8")) or {}
                for block in data.values():
                    if isinstance(block, dict) and isinstance(block.get("ros__parameters"), dict):
                        self._set_params_to_ui(block["ros__parameters"])
                        self.status_bar.showMessage(f"Loaded parameters from {cand.name}")
                        return

    def _apply_live_ros(self) -> None:
        self.btn_apply_live.setEnabled(False)
        self._apply_thread = RosParamApplyThread("/stt_engine", self._get_all_params(), self)
        self._apply_thread.finished_summary.connect(self._on_apply_live_done)
        self._apply_thread.start()

    def _on_apply_live_done(self, _ok: int, _fail: int, msg: str) -> None:
        self.btn_apply_live.setEnabled(True)
        self.status_bar.showMessage(msg)

    def _save_to_yaml(self) -> None:
        params = self._get_all_params()
        saved = []
        for path in self._selected_yaml_paths():
            if not path.is_file():
                continue
            shutil.copy2(path, path.with_suffix(".yaml.bak"))
            updated_text, count = update_yaml_preserving_comments(
                path.read_text(encoding="utf-8"), params
            )
            path.write_text(updated_text, encoding="utf-8")
            saved.append(f"{path.parent.parent.name} ({count} keys)")
        self.status_bar.showMessage("Saved to YAML: " + ", ".join(saved) if saved else "No YAML files found.")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._audio_thread is not None and self._audio_thread.isRunning():
            self._audio_thread.stop_monitor()
        if self._ros_thread is not None and self._ros_thread.isRunning():
            self._ros_thread.stop_feed()
        super().closeEvent(event)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = SttTunerWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
