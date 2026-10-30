#!/usr/bin/env python3
"""Standalone ROS 2 text-to-speech engine."""

from concurrent.futures import ThreadPoolExecutor

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from athome_hri.audio_io import AudioSpeechQueue
from athome_hri.tts_engine import TextToSpeech


class TtsEngineNode(Node):
    """Synthesize and play text received on /hri/tts_say."""

    def __init__(self) -> None:
        super().__init__("tts_engine")
        self.declare_parameter("model", "en_US-lessac-medium.onnx")
        self.declare_parameter("enable_local_speaker", True)

        self._tts = TextToSpeech(str(self.get_parameter("model").value))
        self._play_locally = bool(
            self.get_parameter("enable_local_speaker").value
        )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Tts")
        self.tts_done_pub = self.create_publisher(String, "/hri/tts_done", 10)
        self.subscription = self.create_subscription(
            String, "/hri/tts_say", self._on_text, 10
        )
        self.get_logger().info("Standalone TTS engine ready.")

    def _on_text(self, message: String) -> None:
        text = message.data.strip()
        if text:
            self._executor.submit(self._speak, text)

    def _speak(self, text: str) -> None:
        wav_base64 = self._tts.synthesize_wav_base64(text)
        if self._play_locally and wav_base64:
            AudioSpeechQueue._play_wav_locally(wav_base64)

        done = String()
        done.data = text
        self.tts_done_pub.publish(done)
        self.get_logger().info("[TTS] %s", text)

    def destroy_node(self):
        self._executor.shutdown(wait=True)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TtsEngineNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
