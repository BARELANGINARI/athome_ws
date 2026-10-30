#!/usr/bin/env python3
"""Standalone ROS 2 speech-to-text engine."""

import asyncio
import contextlib
import os
import threading
from typing import Union

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from athome_hri.audio_io import LocalMicListener, VadFrameMetrics, no_c_stderr
from athome_hri.stt_engine import SpeechToText


def find_usb_mic_index() -> Union[int, None]:
    """Mencari indeks input audio PyAudio yang mengandung 'usb'."""
    try:
        import pyaudio
        with no_c_stderr():
            p = pyaudio.PyAudio()
            dev_idx = None
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0 and "usb" in info.get("name", "").lower():
                    dev_idx = i
                    break
            p.terminate()
        return dev_idx
    except Exception:
        return None


class SttEngineNode(Node):
    """Capture local microphone audio and publish transcript text."""

    def __init__(self) -> None:
        super().__init__("stt_engine")
        self.declare_parameter("model", SpeechToText.DEFAULT_MODEL_SIZE)
        self.declare_parameter("language", "en")
        self.declare_parameter("device_index", -1)
        self.declare_parameter("threshold_dbfs", -40.0)
        self.declare_parameter("threshold_flatness", 0.40)
        self.declare_parameter("pre_roll_frames", 10)
        self.declare_parameter("hangover_frames", 25)
        self.declare_parameter("min_speech_frames", 15)

        self.transcript_pub = self.create_publisher(
            String, "/hri/stt_transcript", 10
        )
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            name="SttAsyncLoop",
            daemon=True,
        )
        self._loop_thread.start()

        self._stt = SpeechToText(
            model_size=str(self.get_parameter("model").value),
            language=str(self.get_parameter("language").value),
        )
        raw_device_index = int(self.get_parameter("device_index").value)
        if raw_device_index < 0:
            resolved_index = find_usb_mic_index()
            if resolved_index is not None:
                self.get_logger().info(
                    f"Otomatis mendeteksi USB Mic pada index {resolved_index}"
                )
            else:
                resolved_index = None
        else:
            resolved_index = raw_device_index

        self._listener = LocalMicListener(
            loop=self._loop,
            on_audio_captured=self._on_audio,
            is_paused_fn=lambda: False,
            on_frame_metrics=self._on_metrics,
            device_index=resolved_index,
            threshold_dbfs=float(self.get_parameter("threshold_dbfs").value),
            threshold_flatness=float(self.get_parameter("threshold_flatness").value),
            pre_roll_frames=int(self.get_parameter("pre_roll_frames").value),
            hangover_frames=int(self.get_parameter("hangover_frames").value),
            min_speech_frames=int(self.get_parameter("min_speech_frames").value),
        )
        self._listener.start()
        self.get_logger().info("Standalone STT engine ready.")

    def _on_metrics(self, m: VadFrameMetrics) -> None:
        thresh_dbfs = float(self.get_parameter("threshold_dbfs").value)
        thresh_flatness = float(self.get_parameter("threshold_flatness").value)
        is_voice = (m.dbfs_speech > thresh_dbfs) and (m.flatness < thresh_flatness)

        bars = int(max(0, min(15, (m.dbfs_speech + 60) / 3.33)))
        bar_str = "#" * bars + "-" * (15 - bars)

        if is_voice:
            status = "\033[1;32m[VOICE DETECTED]\033[0m"
        elif m.is_speaking:
            status = "\033[1;33m[HOLD / HANGOVER]\033[0m"
        else:
            status = "\033[2m[SILENCE / NOISE]\033[0m"

        self.get_logger().info(
            f"[VAD METER] [{bar_str}] {m.dbfs_speech:5.1f} dBFS (min: {thresh_dbfs:.1f}) | "
            f"Flatness: {m.flatness:.2f} (max: {thresh_flatness:.2f}) | {status}",
            throttle_duration_sec=0.25,
        )

    async def _transcribe(self, audio_input: Union[np.ndarray, bytes]) -> None:
        transcript = await asyncio.to_thread(self._stt.transcribe, audio_input)
        if not transcript:
            return
        message = String()
        message.data = transcript
        self.transcript_pub.publish(message)
        self.get_logger().info("[STT] %s", transcript)

    async def _on_audio(self, audio_input: Union[np.ndarray, bytes]) -> None:
        await self._transcribe(audio_input)

    def destroy_node(self):
        self._listener.stop()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SttEngineNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
