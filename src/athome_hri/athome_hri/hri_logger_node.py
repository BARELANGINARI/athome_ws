"""ROS 2 node for logging complete per-trial HRI robot prompts and STT pipeline events."""

import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, Optional

_TRIAL_PATTERN = re.compile(r"^trial_(\d+)_.*\.(txt|csv)$")


def next_trial_number(log_dir: Path) -> int:
    """Return the next 1-based trial number based on existing trial_* files in log_dir."""
    if not log_dir.exists():
        return 1
    max_num = 0
    for entry in log_dir.iterdir():
        if not entry.is_file():
            continue
        match = _TRIAL_PATTERN.match(entry.name)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return max_num + 1


def _sanitize_task_name(task: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", task.strip().lower()).strip("_")
    return cleaned or "general"


def _fmt_num(val: Any, digits: int = 3) -> str:
    if val is None or val == "":
        return ""
    try:
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return ""


class HriSessionLogger:
    """Writes per-trial HRI conversation and STT telemetry logs to .txt and .csv."""

    CSV_HEADER = [
        "timestamp",
        "turn",
        "event_type",
        "status",
        "reason",
        "task",
        "stt_backend",
        "question",
        "raw_text",
        "final_text",
        "audio_sec",
        "latency_sec",
        "rtf",
        "avg_logprob",
        "no_speech_prob",
        "response_delay_sec",
    ]

    def __init__(self, log_dir: str, task: str = "general", stt_backend: str = "unknown") -> None:
        self.log_dir = Path(os.path.expanduser(log_dir)).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.task = _sanitize_task_name(task)
        self.stt_backend = (stt_backend or "unknown").strip()
        self.trial_num = next_trial_number(self.log_dir)

        start_dt = datetime.now()
        ts_str = start_dt.strftime("%Y%m%d_%H%M%S")
        stem = f"trial_{self.trial_num:03d}_{self.task}_{ts_str}"

        self.txt_path = self.log_dir / f"{stem}.txt"
        self.csv_path = self.log_dir / f"{stem}.csv"

        self._lock = threading.Lock()
        self._turn = 0
        self._robot_count = 0
        self._user_count = 0
        self._rejected_count = 0
        self._dropped_count = 0
        self._current_question = ""
        self._question_mono: Optional[float] = None
        self._tts_done_mono: Optional[float] = None
        self._current_question_answered = False
        self._stt_events_active = False
        self._closed = False

        self._txt_file = self.txt_path.open("w", encoding="utf-8")
        self._csv_file = self.csv_path.open("w", encoding="utf-8", newline="")
        self._csv_writer = csv.writer(self._csv_file)

        self._txt_file.write(
            "========================================================================\n"
            f"Trial       : {self.trial_num:03d}\n"
            f"Task        : {self.task}\n"
            f"STT Backend : {self.stt_backend}\n"
            f"Start Time  : {start_dt.isoformat(timespec='seconds')}\n"
            f"TXT Log     : {self.txt_path}\n"
            f"CSV Log     : {self.csv_path}\n"
            "========================================================================\n"
        )
        self._txt_file.flush()

        self._csv_writer.writerow(self.CSV_HEADER)
        self._csv_file.flush()

    def _compute_delay_locked(self, mono: float) -> str:
        if not self._current_question:
            return ""
        ref_mono = self._tts_done_mono if self._tts_done_mono is not None else self._question_mono
        if ref_mono is None:
            return ""
        return f"{max(0.0, mono - ref_mono):.2f}"

    def _write_unanswered_question_locked(self, dt: datetime) -> None:
        if self._current_question and not self._current_question_answered:
            self._csv_writer.writerow([
                dt.isoformat(timespec="milliseconds"),
                "",
                "robot_prompt_unanswered",
                "unanswered",
                "",
                self.task,
                self.stt_backend,
                self._current_question,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ])
            self._csv_file.flush()

    def record_robot_prompt(
        self,
        text: str,
        now_dt: Optional[datetime] = None,
        now_mono: Optional[float] = None,
    ) -> None:
        """Log a robot TTS prompt/question from /hri/tts_say."""
        cleaned = (text or "").strip()
        if not cleaned:
            return
        dt = now_dt or datetime.now()
        mono = time.monotonic() if now_mono is None else now_mono

        with self._lock:
            if self._closed:
                return
            self._write_unanswered_question_locked(dt)
            self._current_question = cleaned
            self._question_mono = mono
            self._tts_done_mono = None
            self._current_question_answered = False
            self._robot_count += 1

            self._txt_file.write(f"[{dt.strftime('%H:%M:%S.%f')[:-3]}] [ROBOT] {cleaned}\n")
            self._txt_file.flush()

    def record_tts_done(
        self,
        now_dt: Optional[datetime] = None,
        now_mono: Optional[float] = None,
    ) -> None:
        """Mark when TTS playback finished and log the timestamp."""
        dt = now_dt or datetime.now()
        mono = time.monotonic() if now_mono is None else now_mono
        with self._lock:
            if self._closed or not self._current_question:
                return
            self._tts_done_mono = mono
            self._txt_file.write(f"[{dt.strftime('%H:%M:%S.%f')[:-3]}] [TTS_DONE]\n")
            self._txt_file.flush()

    def record_stt_event(
        self,
        payload: Dict[str, Any],
        now_dt: Optional[datetime] = None,
        now_mono: Optional[float] = None,
    ) -> None:
        """Log a structured STT pipeline event (accepted, rejected, or dropped)."""
        dt = now_dt or datetime.now()
        mono = time.monotonic() if now_mono is None else now_mono

        event_type = str(payload.get("event", "unknown")).strip() or "unknown"
        status = str(payload.get("status", "unknown")).strip() or "unknown"
        reason = str(payload.get("reason", "")).strip()
        raw_text = str(payload.get("raw_text", "") or "").strip()
        final_text = str(payload.get("final_text", "") or "").strip()

        audio_s = _fmt_num(payload.get("audio_sec"), 2)
        lat_s = _fmt_num(payload.get("latency_sec"), 2)
        rtf_s = _fmt_num(payload.get("rtf"), 3)
        logp_s = _fmt_num(payload.get("avg_logprob"), 3)
        nosp_s = _fmt_num(payload.get("no_speech_prob"), 3)

        with self._lock:
            if self._closed:
                return
            self._stt_events_active = True
            delay_str = self._compute_delay_locked(mono)
            turn_val: Any = ""

            if event_type == "asr_accepted" and final_text:
                self._turn += 1
                self._user_count += 1
                self._current_question_answered = True
                turn_val = self._turn
                tag = "[USER:ACCEPTED]"
            elif status == "rejected" or event_type == "asr_rejected":
                self._rejected_count += 1
                tag = "[STT:REJECTED]"
            else:
                self._dropped_count += 1
                tag = "[STT:DROPPED] "

            self._csv_writer.writerow([
                dt.isoformat(timespec="milliseconds"),
                turn_val,
                event_type,
                status,
                reason,
                self.task,
                self.stt_backend,
                self._current_question,
                raw_text,
                final_text,
                audio_s,
                lat_s,
                rtf_s,
                logp_s,
                nosp_s,
                delay_str,
            ])
            self._csv_file.flush()

            details = []
            if reason and reason != "ok":
                details.append(f"reason={reason}")
            if raw_text and raw_text != final_text:
                details.append(f'raw="{raw_text}"')
            if audio_s:
                details.append(f"audio={audio_s}s")
            if lat_s:
                details.append(f"lat={lat_s}s")
            if rtf_s and float(rtf_s) > 0.0:
                details.append(f"rtf={rtf_s}")
            if logp_s:
                details.append(f"logp={logp_s}")
            if nosp_s:
                details.append(f"nosp={nosp_s}")
            if payload.get("voice_frames"):
                details.append(f"frames={payload.get('voice_frames')}")
            if payload.get("peak_dbfs") is not None and event_type == "vad_dropped_short":
                details.append(f"peak={_fmt_num(payload.get('peak_dbfs'), 1)}dBFS")
            if delay_str:
                details.append(f"delay={delay_str}s")

            body_text = final_text or raw_text or f"({event_type})"
            detail_str = f" ({' | '.join(details)})" if details else ""
            self._txt_file.write(
                f"[{dt.strftime('%H:%M:%S.%f')[:-3]}] {tag} {body_text}{detail_str}\n"
            )
            self._txt_file.flush()

    def record_user_transcript(
        self,
        text: str,
        now_dt: Optional[datetime] = None,
        now_mono: Optional[float] = None,
    ) -> None:
        """Fallback logger for /hri/stt_transcript when /hri/stt_events is not active."""
        cleaned = (text or "").strip()
        if not cleaned:
            return
        dt = now_dt or datetime.now()
        mono = time.monotonic() if now_mono is None else now_mono

        with self._lock:
            if self._closed or self._stt_events_active:
                return
            self._turn += 1
            self._user_count += 1
            self._current_question_answered = True
            delay_str = self._compute_delay_locked(mono)
            delay_note = f" (delay: {delay_str}s)" if delay_str else ""

            self._csv_writer.writerow([
                dt.isoformat(timespec="milliseconds"),
                self._turn,
                "asr_accepted",
                "accepted",
                "ok",
                self.task,
                self.stt_backend,
                self._current_question,
                cleaned,
                cleaned,
                "",
                "",
                "",
                "",
                "",
                delay_str,
            ])
            self._csv_file.flush()

            self._txt_file.write(
                f"[{dt.strftime('%H:%M:%S.%f')[:-3]}] [USER]  {cleaned}{delay_note}\n"
            )
            self._txt_file.flush()

    def close(self, now_dt: Optional[datetime] = None) -> None:
        """Flush any pending prompt, append session summary, and close output files."""
        dt = now_dt or datetime.now()
        with self._lock:
            if self._closed:
                return
            self._write_unanswered_question_locked(dt)
            self._txt_file.write(
                "------------------------------------------------------------------------\n"
                f"End Time                  : {dt.isoformat(timespec='seconds')}\n"
                f"Total Robot Prompts       : {self._robot_count}\n"
                f"Total Accepted Transcripts: {self._user_count}\n"
                f"Total Rejected ASR        : {self._rejected_count}\n"
                f"Total Dropped Mic Events  : {self._dropped_count}\n"
                "========================================================================\n"
            )
            self._txt_file.flush()
            self._csv_file.flush()
            self._txt_file.close()
            self._csv_file.close()
            self._closed = True


def main(args=None) -> None:
    """Entry point for the ROS 2 hri_logger_node."""
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class HriLoggerNode(Node):
        def __init__(self) -> None:
            super().__init__("hri_logger")
            default_task = os.environ.get("ATHOME_TASK", "general")
            log_dir = str(self.declare_parameter("log_dir", "~/athome_hri_logs").value)
            task = str(self.declare_parameter("task", default_task).value)
            stt_backend = str(self.declare_parameter("stt_backend", "unknown").value)
            tts_say_topic = str(self.declare_parameter("tts_say_topic", "/hri/tts_say").value)
            tts_done_topic = str(self.declare_parameter("tts_done_topic", "/hri/tts_done").value)
            stt_topic = str(
                self.declare_parameter("stt_transcript_topic", "/hri/stt_transcript").value
            )
            stt_events_topic = str(
                self.declare_parameter("stt_events_topic", "/hri/stt_events").value
            )

            self.session_logger = HriSessionLogger(
                log_dir=log_dir,
                task=task,
                stt_backend=stt_backend,
            )

            self.create_subscription(String, tts_say_topic, self._on_tts_say, 10)
            self.create_subscription(String, tts_done_topic, self._on_tts_done, 10)
            self.create_subscription(String, stt_events_topic, self._on_stt_event, 20)
            self.create_subscription(String, stt_topic, self._on_stt_transcript, 10)

            self.get_logger().info(
                f"HRI Logger aktif (Trial {self.session_logger.trial_num:03d}, task={self.session_logger.task})\n"
                f"  TXT: {self.session_logger.txt_path}\n"
                f"  CSV: {self.session_logger.csv_path}"
            )

        def _on_tts_say(self, msg: String) -> None:
            self.session_logger.record_robot_prompt(msg.data)

        def _on_tts_done(self, _msg: String) -> None:
            self.session_logger.record_tts_done()

        def _on_stt_event(self, msg: String) -> None:
            try:
                payload = json.loads(msg.data)
                if isinstance(payload, dict):
                    self.session_logger.record_stt_event(payload)
            except Exception as exc:
                self.get_logger().warn(f"Gagal parse /hri/stt_events JSON: {exc}")

        def _on_stt_transcript(self, msg: String) -> None:
            self.session_logger.record_user_transcript(msg.data)

        def destroy_node(self) -> bool:
            self.session_logger.close()
            return super().destroy_node()

    rclpy.init(args=args)
    node = HriLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.session_logger.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
