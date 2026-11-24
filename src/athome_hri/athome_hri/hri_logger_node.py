"""ROS 2 node for logging per-trial HRI robot prompts and user STT transcripts."""

import csv
from datetime import datetime
import os
from pathlib import Path
import re
import threading
import time
from typing import Optional

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


class HriSessionLogger:
    """Writes per-trial HRI conversation logs to both human-readable .txt and structured .csv."""

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
        self._current_question = ""
        self._question_mono: Optional[float] = None
        self._tts_done_mono: Optional[float] = None
        self._current_question_answered = False
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

        self._csv_writer.writerow([
            "timestamp",
            "turn",
            "task",
            "stt_backend",
            "question",
            "user_transcript",
            "response_delay_sec",
        ])
        self._csv_file.flush()

    def _write_unanswered_question_locked(self, dt: datetime) -> None:
        if self._current_question and not self._current_question_answered:
            self._csv_writer.writerow([
                dt.isoformat(timespec="milliseconds"),
                "",
                self.task,
                self.stt_backend,
                self._current_question,
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

    def record_tts_done(self, now_mono: Optional[float] = None) -> None:
        """Mark when TTS playback finished so user response delay can be measured accurately."""
        mono = time.monotonic() if now_mono is None else now_mono
        with self._lock:
            if self._closed or not self._current_question:
                return
            self._tts_done_mono = mono

    def record_user_transcript(
        self,
        text: str,
        now_dt: Optional[datetime] = None,
        now_mono: Optional[float] = None,
    ) -> None:
        """Log a recognized user transcript from /hri/stt_transcript paired with current prompt."""
        cleaned = (text or "").strip()
        if not cleaned:
            return
        dt = now_dt or datetime.now()
        mono = time.monotonic() if now_mono is None else now_mono

        with self._lock:
            if self._closed:
                return
            self._turn += 1
            self._user_count += 1
            self._current_question_answered = True

            ref_mono = self._tts_done_mono if self._tts_done_mono is not None else self._question_mono
            if ref_mono is not None:
                delay_sec = max(0.0, mono - ref_mono)
                delay_str = f"{delay_sec:.2f}"
                delay_note = f" (delay: {delay_str}s)"
            else:
                delay_str = ""
                delay_note = ""

            self._csv_writer.writerow([
                dt.isoformat(timespec="milliseconds"),
                self._turn,
                self.task,
                self.stt_backend,
                self._current_question,
                cleaned,
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
                f"End Time               : {dt.isoformat(timespec='seconds')}\n"
                f"Total Robot Prompts    : {self._robot_count}\n"
                f"Total User Transcripts : {self._user_count}\n"
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

            self.session_logger = HriSessionLogger(
                log_dir=log_dir,
                task=task,
                stt_backend=stt_backend,
            )

            self.create_subscription(String, tts_say_topic, self._on_tts_say, 10)
            self.create_subscription(String, tts_done_topic, self._on_tts_done, 10)
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
