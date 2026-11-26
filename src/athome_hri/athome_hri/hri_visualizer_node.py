"""Real-time PyQt5 audio & active-parameter visualizer for ROS 2 HRI launches."""

from collections import deque
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import yaml


def load_yaml_ros_params(yaml_path: str) -> Dict[str, Any]:
    """Load the first ros__parameters dictionary from a ROS 2 YAML file."""
    if not yaml_path:
        return {}
    p = Path(os.path.expanduser(yaml_path))
    if not p.is_file():
        return {}
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict):
            for block in data.values():
                if isinstance(block, dict) and isinstance(block.get("ros__parameters"), dict):
                    return dict(block["ros__parameters"])
    except Exception:
        pass
    return {}


def parse_telemetry_array(data: Any) -> Optional[Dict[str, Any]]:
    """Decode /hri/audio_telemetry Float32MultiArray payload (16-float header + PCM samples)."""
    if data is None or len(data) < 17:
        return None
    flags = int(round(float(data[14])))
    samples = np.asarray(data[16:], dtype=np.float32)
    return {
        "sample_rate": float(data[0]) if float(data[0]) > 0 else 16000.0,
        "dbfs": float(data[1]),
        "dbfs_speech": float(data[2]),
        "threshold_dbfs": float(data[3]),
        "effective_threshold_dbfs": float(data[4]),
        "noise_floor_dbfs": float(data[5]),
        "flatness": float(data[6]),
        "threshold_flatness": float(data[7]),
        "speech_band_low_hz": float(data[8]),
        "speech_band_high_hz": float(data[9]),
        "adaptive_vad_enabled": bool(float(data[10]) >= 0.5),
        "adaptive_snr_margin_db": float(data[11]),
        "adaptive_max_threshold_dbfs": float(data[12]),
        "stt_vad_threshold": float(data[13]),
        "is_voice": bool(flags & 1),
        "is_speaking": bool(flags & 2),
        "is_paused": bool(flags & 4),
        "highpass_hz": float(data[15]),
        "samples": samples,
    }


class WaveformWidget(QtWidgets.QWidget):
    """Real-time time-domain audio waveform canvas color-coded by VAD state."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(135)
        self._samples = np.zeros(480, dtype=np.float32)
        self._sample_rate = 16000.0
        self._state = "WAITING"

    def set_frame(self, samples: np.ndarray, sample_rate: float, state: str) -> None:
        self._samples = samples
        self._sample_rate = sample_rate
        self._state = state
        self.update()

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QtGui.QColor("#0f172a"))

        pad_l, pad_r, pad_t, pad_b = 48, 14, 24, 20
        pw = max(20, w - pad_l - pad_r)
        ph = max(20, h - pad_t - pad_b)
        mid_y = pad_t + ph / 2.0

        p.setPen(QtGui.QPen(QtGui.QColor("#1e293b"), 1))
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = int(pad_t + frac * ph)
            p.drawLine(pad_l, y, pad_l + pw, y)
        p.drawRect(pad_l, pad_t, pw, ph)

        color_map = {
            "VOICE": "#22c55e",
            "HOLD": "#f59e0b",
            "MUTED": "#ef4444",
            "SILENCE": "#38bdf8",
        }
        line_color = QtGui.QColor(color_map.get(self._state, "#64748b"))

        if self._samples.size > 1:
            n = self._samples.size
            step = max(1, n // max(1, pw))
            pts = []
            for idx in range(0, n, step):
                x = pad_l + (idx / float(max(1, n - 1))) * pw
                val = float(np.clip(self._samples[idx], -1.0, 1.0))
                y = mid_y - val * (ph * 0.48)
                pts.append(QtCore.QPointF(x, y))
            p.setPen(QtGui.QPen(line_color, 1.6))
            p.drawPolyline(QtGui.QPolygonF(pts))

        dur_ms = (self._samples.size / max(1.0, self._sample_rate)) * 1000.0
        p.setPen(QtGui.QColor("#e2e8f0"))
        p.setFont(QtGui.QFont("Sans", 9, QtGui.QFont.Bold))
        p.drawText(pad_l + 4, 16, f"Waveform (30ms Frame, {dur_ms:.1f} ms)")
        p.setPen(QtGui.QColor("#94a3b8"))
        p.setFont(QtGui.QFont("Sans", 8))
        p.drawText(4, pad_t + 10, "+1.0")
        p.drawText(8, int(mid_y) + 4, " 0.0")
        p.drawText(4, pad_t + ph, "-1.0")
        p.end()


class SpectrumWidget(QtWidgets.QWidget):
    """Real-time FFT magnitude spectrum (dB) with Speech Band & HPF cutoff overlays."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(145)
        self._freqs = np.fft.rfftfreq(480, 1.0 / 16000.0)
        self._spec_db = np.full_like(self._freqs, -80.0)
        self._low_hz = 300.0
        self._high_hz = 3400.0
        self._hpf_hz = 120.0

    def set_frame(
        self,
        samples: np.ndarray,
        sample_rate: float,
        low_hz: float,
        high_hz: float,
        hpf_hz: float,
    ) -> None:
        self._low_hz = low_hz
        self._high_hz = high_hz
        self._hpf_hz = hpf_hz
        if samples.size >= 16:
            win = np.hanning(samples.size)
            mag = np.abs(np.fft.rfft(samples * win)) / max(1.0, samples.size * 0.5)
            self._freqs = np.fft.rfftfreq(samples.size, 1.0 / max(1.0, sample_rate))
            self._spec_db = np.clip(20.0 * np.log10(mag + 1e-6), -90.0, 5.0)
        self.update()

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QtGui.QColor("#0f172a"))

        pad_l, pad_r, pad_t, pad_b = 48, 14, 24, 22
        pw = max(20, w - pad_l - pad_r)
        ph = max(20, h - pad_t - pad_b)
        max_hz = 8000.0
        db_min, db_max = -90.0, 0.0

        def hz_to_x(hz: float) -> float:
            return pad_l + (np.clip(hz, 0.0, max_hz) / max_hz) * pw

        def db_to_y(db: float) -> float:
            norm = (np.clip(db, db_min, db_max) - db_min) / (db_max - db_min)
            return pad_t + (1.0 - norm) * ph

        # Speech band shaded overlay
        x_low = hz_to_x(self._low_hz)
        x_high = hz_to_x(self._high_hz)
        p.fillRect(
            QtCore.QRectF(x_low, pad_t, max(1.0, x_high - x_low), ph),
            QtGui.QColor(34, 197, 94, 34),
        )

        # HPF cutoff line
        x_hpf = hz_to_x(self._hpf_hz)
        p.setPen(QtGui.QPen(QtGui.QColor("#f97316"), 1.3, QtCore.Qt.DashLine))
        p.drawLine(int(x_hpf), pad_t, int(x_hpf), pad_t + ph)

        # Grid
        p.setPen(QtGui.QPen(QtGui.QColor("#1e293b"), 1))
        for db in (-80, -60, -40, -20, 0):
            y = int(db_to_y(db))
            p.drawLine(pad_l, y, pad_l + pw, y)
        for hz in (1000, 2000, 3000, 4000, 5000, 6000, 7000):
            x = int(hz_to_x(hz))
            p.drawLine(x, pad_t, x, pad_t + ph)
        p.drawRect(pad_l, pad_t, pw, ph)

        # Spectrum curve
        if self._freqs.size > 1:
            pts = [
                QtCore.QPointF(hz_to_x(float(f)), db_to_y(float(db)))
                for f, db in zip(self._freqs, self._spec_db)
                if f <= max_hz
            ]
            if len(pts) > 1:
                p.setPen(QtGui.QPen(QtGui.QColor("#f43f5e"), 1.6))
                p.drawPolyline(QtGui.QPolygonF(pts))

        p.setPen(QtGui.QColor("#e2e8f0"))
        p.setFont(QtGui.QFont("Sans", 9, QtGui.QFont.Bold))
        p.drawText(
            pad_l + 4,
            16,
            f"FFT Spectrum (dB) | Speech Band: {self._low_hz:.0f}-{self._high_hz:.0f} Hz (green) | HPF: {self._hpf_hz:.0f} Hz (orange)",
        )
        p.setPen(QtGui.QColor("#94a3b8"))
        p.setFont(QtGui.QFont("Sans", 8))
        for db in (0, -30, -60, -90):
            p.drawText(6, int(db_to_y(db)) + 4, f"{db:>3d}")
        for hz in (0, 2000, 4000, 6000, 8000):
            p.drawText(int(hz_to_x(hz)) - 10, h - 5, f"{hz // 1000}k")
        p.end()


class EnergyHistoryWidget(QtWidgets.QWidget):
    """Rolling history of Full dBFS, Speech-Band dBFS, Noise Floor, and Thresholds."""

    def __init__(self, history_len: int = 200, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(165)
        self._len = history_len
        self._full_hist = deque([-80.0] * history_len, maxlen=history_len)
        self._speech_hist = deque([-80.0] * history_len, maxlen=history_len)
        self._floor_hist = deque([-55.0] * history_len, maxlen=history_len)
        self._eff_thr_hist = deque([-24.0] * history_len, maxlen=history_len)
        self._floor_thr = -24.0

    def push_point(
        self,
        dbfs_full: float,
        dbfs_speech: float,
        noise_floor: float,
        threshold_floor: float,
        eff_threshold: float,
    ) -> None:
        self._full_hist.append(dbfs_full)
        self._speech_hist.append(dbfs_speech)
        self._floor_hist.append(noise_floor)
        self._eff_thr_hist.append(eff_threshold)
        self._floor_thr = threshold_floor
        self.update()

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QtGui.QColor("#0f172a"))

        pad_l, pad_r, pad_t, pad_b = 48, 14, 24, 20
        pw = max(20, w - pad_l - pad_r)
        ph = max(20, h - pad_t - pad_b)
        db_min, db_max = -85.0, 0.0

        def db_to_y(val: float) -> float:
            norm = (np.clip(val, db_min, db_max) - db_min) / (db_max - db_min)
            return pad_t + (1.0 - norm) * ph

        p.setPen(QtGui.QPen(QtGui.QColor("#1e293b"), 1))
        for db in (-80, -60, -40, -20, 0):
            y = int(db_to_y(db))
            p.drawLine(pad_l, y, pad_l + pw, y)
        p.drawRect(pad_l, pad_t, pw, ph)

        def draw_series(vals: deque, color: str, width: float = 1.6, dash: bool = False) -> None:
            n = len(vals)
            if n < 2:
                return
            pts = [
                QtCore.QPointF(pad_l + (i / float(n - 1)) * pw, db_to_y(v))
                for i, v in enumerate(vals)
            ]
            pen = QtGui.QPen(QtGui.QColor(color), width)
            if dash:
                pen.setStyle(QtCore.Qt.DashLine)
            p.setPen(pen)
            p.drawPolyline(QtGui.QPolygonF(pts))

        # Floor threshold constant dashed line
        y_floor_thr = int(db_to_y(self._floor_thr))
        p.setPen(QtGui.QPen(QtGui.QColor("#a855f7"), 1.2, QtCore.Qt.DotLine))
        p.drawLine(pad_l, y_floor_thr, pad_l + pw, y_floor_thr)

        draw_series(self._full_hist, "#64748b", 1.2)
        draw_series(self._floor_hist, "#f59e0b", 1.5, dash=True)
        draw_series(self._eff_thr_hist, "#ef4444", 1.8, dash=True)
        draw_series(self._speech_hist, "#22c55e", 2.0)

        p.setPen(QtGui.QColor("#e2e8f0"))
        p.setFont(QtGui.QFont("Sans", 9, QtGui.QFont.Bold))
        p.drawText(
            pad_l + 4,
            16,
            "Energy History (dBFS) | Speech (green) | Full (gray) | Eff Thr (red) | Floor Thr (purple) | Noise (amber)",
        )
        p.setPen(QtGui.QColor("#94a3b8"))
        p.setFont(QtGui.QFont("Sans", 8))
        for db in (0, -20, -40, -60, -80):
            p.drawText(6, int(db_to_y(db)) + 4, f"{db:>3d}")
        p.end()


class HriAudioVisualizerWindow(QtWidgets.QMainWindow):
    """Live audio visualizer & active-parameter monitor for HRI launches."""

    def __init__(
        self,
        task: str = "all",
        stt_backend: str = "cpp",
        yaml_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"Barelang Inari HRI - Live Audio & Parameter Visualizer ({task} / {stt_backend})")
        self.resize(1040, 740)
        self._task = task
        self._stt_backend = stt_backend
        self._yaml_params = dict(yaml_params or {})
        self._last_msg_time = 0.0

        self._build_ui()
        self._apply_dark_theme()
        self._refresh_param_labels()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Top status bar
        top_row = QtWidgets.QHBoxLayout()
        self.lbl_identity = QtWidgets.QLabel(f"Task: {self._task.upper()}   |   Backend: {self._stt_backend.upper()}")
        self.lbl_identity.setStyleSheet("font-weight: bold; font-size: 13px; color: #38bdf8;")
        self.lbl_state_badge = QtWidgets.QLabel("WAITING STREAM")
        self.lbl_state_badge.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_state_badge.setMinimumWidth(170)
        self.lbl_state_badge.setStyleSheet(
            "padding: 4px 10px; border-radius: 4px; font-weight: bold; background: #334155; color: #e2e8f0;"
        )
        top_row.addWidget(self.lbl_identity)
        top_row.addStretch(1)
        top_row.addWidget(self.lbl_state_badge)
        root.addLayout(top_row)

        # Active parameters panel
        param_box = QtWidgets.QGroupBox("Active HRI Parameters (Live + YAML)")
        param_lay = QtWidgets.QGridLayout(param_box)
        param_lay.setVerticalSpacing(4)
        param_lay.setHorizontalSpacing(14)

        self.lbl_vad_params = QtWidgets.QLabel("-")
        self.lbl_live_metrics = QtWidgets.QLabel("-")
        self.lbl_asr_params = QtWidgets.QLabel("-")

        self.flatness_bar = QtWidgets.QProgressBar()
        self.flatness_bar.setRange(0, 100)
        self.flatness_bar.setTextVisible(True)
        self.flatness_bar.setFormat("Flatness: 0.00")

        param_lay.addWidget(QtWidgets.QLabel("<b>VAD & Denoiser:</b>"), 0, 0)
        param_lay.addWidget(self.lbl_vad_params, 0, 1, 1, 2)
        param_lay.addWidget(QtWidgets.QLabel("<b>ASR & Post-Mute:</b>"), 1, 0)
        param_lay.addWidget(self.lbl_asr_params, 1, 1, 1, 2)
        param_lay.addWidget(QtWidgets.QLabel("<b>Live Telemetry:</b>"), 2, 0)
        param_lay.addWidget(self.lbl_live_metrics, 2, 1)
        param_lay.addWidget(self.flatness_bar, 2, 2)
        param_lay.setColumnStretch(1, 3)
        param_lay.setColumnStretch(2, 1)
        root.addWidget(param_box)

        # Three audio graphs
        self.wave_widget = WaveformWidget()
        self.spec_widget = SpectrumWidget()
        self.energy_widget = EnergyHistoryWidget(history_len=200)
        root.addWidget(self.wave_widget, 2)
        root.addWidget(self.spec_widget, 2)
        root.addWidget(self.energy_widget, 3)

        # Bottom HRI dialog & event status
        dialog_box = QtWidgets.QGroupBox("Live HRI Speech & ASR Pipeline Status")
        d_lay = QtWidgets.QVBoxLayout(dialog_box)
        self.lbl_tts = QtWidgets.QLabel("ROBOT TTS : (waiting)")
        self.lbl_tts.setWordWrap(True)
        self.lbl_tts.setStyleSheet("color: #7dd3fc; font-weight: bold;")
        self.lbl_asr = QtWidgets.QLabel("USER STT  : (waiting)")
        self.lbl_asr.setWordWrap(True)
        self.lbl_asr.setStyleSheet("color: #4ade80; font-weight: bold;")
        d_lay.addWidget(self.lbl_tts)
        d_lay.addWidget(self.lbl_asr)
        root.addWidget(dialog_box)

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #111827; color: #e5e7eb; font-family: Sans; font-size: 11px; }
            QGroupBox { border: 1px solid #334155; border-radius: 6px; margin-top: 8px; padding: 8px; font-weight: bold; color: #cbd5e1; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QProgressBar { border: 1px solid #334155; border-radius: 4px; background: #0f172a; text-align: center; height: 18px; }
            QProgressBar::chunk { background-color: #38bdf8; }
        """)

    def _refresh_param_labels(self, live: Optional[Dict[str, Any]] = None) -> None:
        yp = self._yaml_params
        thr_floor = live["threshold_dbfs"] if live else yp.get("threshold_dbfs", -24.0)
        thr_flat = live["threshold_flatness"] if live else yp.get("threshold_flatness", 0.38)
        adap_en = live["adaptive_vad_enabled"] if live else yp.get("adaptive_vad_enabled", True)
        snr_m = live["adaptive_snr_margin_db"] if live else yp.get("adaptive_snr_margin_db", 7.0)
        adap_max = live["adaptive_max_threshold_dbfs"] if live else yp.get("adaptive_max_threshold_dbfs", -10.0)
        hpf = live["highpass_hz"] if live else yp.get("stt_highpass_hz", 120.0)
        nvad_thr = live["stt_vad_threshold"] if live else yp.get("stt_vad_threshold", 0.52)

        hang = yp.get("hangover_frames", "-")
        min_sp = yp.get("min_speech_frames", "-")
        pre = yp.get("pre_roll_frames", "-")
        den = yp.get("stt_denoiser_backend", "auto")
        post_mute = yp.get("tts_post_mute_sec", "-")
        no_sp = yp.get("stt_no_speech_threshold", "-")
        logp = yp.get("stt_log_prob_threshold", "-")
        model = yp.get("whisper_model_path", yp.get("whisper_model", "-"))
        model_short = Path(str(model)).name if model != "-" else "-"

        self.lbl_vad_params.setText(
            f"threshold_dbfs={float(thr_floor):.1f} | adaptive={adap_en} (margin={float(snr_m):.1f}dB, max={float(adap_max):.1f}) | "
            f"flat_max={float(thr_flat):.2f} | min/hang/pre={min_sp}/{hang}/{pre} | denoiser={den} (HPF={float(hpf):.0f}Hz)"
        )
        self.lbl_asr_params.setText(
            f"silero_thr={float(nvad_thr):.2f} | no_speech={no_sp} | log_prob={logp} | "
            f"tts_post_mute={post_mute}s | model={model_short}"
        )

    def handle_telemetry(self, tel: Dict[str, Any]) -> None:
        self._last_msg_time = time.monotonic()
        if tel["is_paused"]:
            state = "MUTED"
            badge_css = "background: #7f1d1d; color: #fecaca;"
            badge_txt = "MUTED (ROBOT TTS)"
        elif tel["is_voice"]:
            state = "VOICE"
            badge_css = "background: #14532d; color: #bbf7d0;"
            badge_txt = "VOICE DETECTED"
        elif tel["is_speaking"]:
            state = "HOLD"
            badge_css = "background: #78350f; color: #fde68a;"
            badge_txt = "HOLD (HANGOVER)"
        else:
            state = "SILENCE"
            badge_css = "background: #1e293b; color: #94a3b8;"
            badge_txt = "SILENCE"

        self.lbl_state_badge.setText(badge_txt)
        self.lbl_state_badge.setStyleSheet(f"padding: 4px 10px; border-radius: 4px; font-weight: bold; {badge_css}")

        self._refresh_param_labels(tel)
        self.lbl_live_metrics.setText(
            f"Speech={tel['dbfs_speech']:5.1f} dBFS | Full={tel['dbfs']:5.1f} dBFS | "
            f"Eff.Thr={tel['effective_threshold_dbfs']:5.1f} dBFS | Floor={tel['noise_floor_dbfs']:5.1f} dBFS"
        )

        flat = float(np.clip(tel["flatness"], 0.0, 1.0))
        self.flatness_bar.setValue(int(round(flat * 100.0)))
        self.flatness_bar.setFormat(f"Flatness: {flat:.2f} / {tel['threshold_flatness']:.2f}")
        bar_col = "#22c55e" if flat < tel["threshold_flatness"] else "#ef4444"
        self.flatness_bar.setStyleSheet(
            f"QProgressBar {{ border: 1px solid #334155; border-radius: 4px; background: #0f172a; text-align: center; height: 18px; }}"
            f"QProgressBar::chunk {{ background-color: {bar_col}; }}"
        )

        self.wave_widget.set_frame(tel["samples"], tel["sample_rate"], state)
        self.spec_widget.set_frame(
            tel["samples"], tel["sample_rate"],
            tel["speech_band_low_hz"], tel["speech_band_high_hz"], tel["highpass_hz"]
        )
        self.energy_widget.push_point(
            tel["dbfs"], tel["dbfs_speech"], tel["noise_floor_dbfs"],
            tel["threshold_dbfs"], tel["effective_threshold_dbfs"]
        )

    def handle_tts_say(self, text: str) -> None:
        self.lbl_tts.setText(f"[{time.strftime('%H:%M:%S')}] ROBOT TTS : {text}")

    def handle_stt_event(self, payload: Dict[str, Any]) -> None:
        ts = time.strftime("%H:%M:%S")
        ev = str(payload.get("event", "unknown"))
        reason = str(payload.get("reason", ""))
        raw_t = str(payload.get("raw_text", "") or "")
        final_t = str(payload.get("final_text", "") or "")
        lat = payload.get("latency_sec")
        rtf = payload.get("rtf")
        logp = payload.get("avg_logprob")

        if ev == "asr_accepted" and final_t:
            self.lbl_asr.setStyleSheet("color: #4ade80; font-weight: bold;")
            self.lbl_asr.setText(
                f"[{ts}] USER STT [ACCEPTED]: \"{final_t}\" "
                f"(lat={lat}s | rtf={rtf} | logp={logp})"
            )
        elif payload.get("status") == "rejected":
            self.lbl_asr.setStyleSheet("color: #fbbf24; font-weight: bold;")
            self.lbl_asr.setText(
                f"[{ts}] USER STT [REJECTED:{reason}]: raw=\"{raw_t}\" (logp={logp})"
            )
        else:
            self.lbl_asr.setStyleSheet("color: #f87171; font-weight: bold;")
            self.lbl_asr.setText(
                f"[{ts}] USER STT [DROPPED:{ev}/{reason}]"
                + (f" raw=\"{raw_t}\"" if raw_t else "")
            )

    def handle_stt_transcript(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.lbl_asr.setStyleSheet("color: #4ade80; font-weight: bold;")
        self.lbl_asr.setText(f"[{ts}] USER STT: \"{text}\"")


def main(args=None) -> None:
    """Entry point for ROS 2 hri_visualizer_node."""
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or os.environ.get("QT_QPA_PLATFORM")):
        print("[hri_visualizer] DISPLAY not detected (headless environment); skipping audio visualizer GUI.")
        return

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray, String

    class _SignalBridge(QtCore.QObject):
        telemetry_sig = QtCore.pyqtSignal(dict)
        tts_sig = QtCore.pyqtSignal(str)
        stt_event_sig = QtCore.pyqtSignal(dict)
        stt_transcript_sig = QtCore.pyqtSignal(str)

    rclpy.init(args=args)
    node = Node("hri_visualizer")
    task = str(node.declare_parameter("task", os.environ.get("ATHOME_TASK", "all")).value)
    stt_backend = str(node.declare_parameter("stt_backend", "cpp").value)
    cpp_params_file = str(node.declare_parameter("cpp_params_file", "").value)
    py_params_file = str(node.declare_parameter("py_params_file", "").value)
    telemetry_topic = str(node.declare_parameter("telemetry_topic", "/hri/audio_telemetry").value)
    stt_events_topic = str(node.declare_parameter("stt_events_topic", "/hri/stt_events").value)
    stt_transcript_topic = str(node.declare_parameter("stt_transcript_topic", "/hri/stt_transcript").value)
    tts_say_topic = str(node.declare_parameter("tts_say_topic", "/hri/tts_say").value)

    active_yaml = cpp_params_file if stt_backend == "cpp" else py_params_file
    yaml_params = load_yaml_ros_params(active_yaml)

    app = QtWidgets.QApplication(sys.argv)
    win = HriAudioVisualizerWindow(task=task, stt_backend=stt_backend, yaml_params=yaml_params)
    bridge = _SignalBridge()
    bridge.telemetry_sig.connect(win.handle_telemetry)
    bridge.tts_sig.connect(win.handle_tts_say)
    bridge.stt_event_sig.connect(win.handle_stt_event)
    bridge.stt_transcript_sig.connect(win.handle_stt_transcript)

    has_events = [False]

    def _on_telemetry(msg: Float32MultiArray) -> None:
        parsed = parse_telemetry_array(msg.data)
        if parsed is not None:
            bridge.telemetry_sig.emit(parsed)

    def _on_tts(msg: String) -> None:
        if msg.data.strip():
            bridge.tts_sig.emit(msg.data.strip())

    def _on_event(msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            if isinstance(payload, dict):
                has_events[0] = True
                bridge.stt_event_sig.emit(payload)
        except Exception:
            pass

    def _on_transcript(msg: String) -> None:
        if not has_events[0] and msg.data.strip():
            bridge.stt_transcript_sig.emit(msg.data.strip())

    node.create_subscription(Float32MultiArray, telemetry_topic, _on_telemetry, 10)
    node.create_subscription(String, tts_say_topic, _on_tts, 10)
    node.create_subscription(String, stt_events_topic, _on_event, 20)
    node.create_subscription(String, stt_transcript_topic, _on_transcript, 10)

    def _spin_once() -> None:
        if rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
        else:
            app.quit()

    ros_timer = QtCore.QTimer()
    ros_timer.timeout.connect(_spin_once)
    ros_timer.start(15)

    def _sigint_handler(_sig: int, _frame: Any) -> None:
        app.quit()

    signal.signal(signal.SIGINT, _sigint_handler)

    win.show()
    try:
        app.exec_()
    finally:
        ros_timer.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
