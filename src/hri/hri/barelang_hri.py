"""
Barelang Inari - Human-Robot Interaction (HRI) Master Node.
Bertugas murni sebagai antarmuka suara (STT/TTS), NLU (Intent Recognition),
dan Web Dashboard tanpa interferensi kontrol pergerakan motor (cmd_vel).
"""

import asyncio
import os
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import websockets

from hri.stt_engine import SpeechToText
from hri.tts_engine import TextToSpeech
from hri.intent_recognizer import IntentRecognizer
from hri.web_server import WebServer


class BarelangHriNode(Node):
    """ROS 2 Node untuk menjembatani komunikasi suara HRI dengan ekosistem robot."""

    def __init__(self, async_loop, on_tts_command):
        super().__init__("barelang_hri_node")
        self.loop = async_loop
        self.on_tts_command = on_tts_command

        # Publishers
        self.event_pub = self.create_publisher(String, "/hri/event", 10)

        # Subscribers
        self.tts_sub = self.create_subscription(
            String, "/hri/tts_say", self._on_tts_say, 10
        )
        self.status_sub = self.create_subscription(
            String, "/luggage/status", self._on_luggage_status, 10
        )
        self.target_dir_sub = self.create_subscription(
            String, "/luggage/target_dir", self._on_target_dir, 10
        )

        self.get_logger().info("BarelangHriNode modular aktif (Audio & UI Interface).")

    def _on_tts_say(self, msg: String):
        text = msg.data.strip()
        if text and self.on_tts_command:
            asyncio.run_coroutine_threadsafe(self.on_tts_command(text), self.loop)

    def _on_luggage_status(self, msg: String):
        text = msg.data.strip()
        if hasattr(self, "web_server") and self.web_server:
            payload = {
                "type": "STATE_UPDATE",
                "state": "RUNNING",
                "substate": text,
            }
            asyncio.run_coroutine_threadsafe(self.web_server.broadcast_ui(payload), self.loop)

    def _on_target_dir(self, msg: String):
        direction = msg.data.strip()
        if hasattr(self, "web_server") and self.web_server:
            payload = {
                "type": "DIRECTION_UPDATE",
                "direction": direction,
            }
            asyncio.run_coroutine_threadsafe(self.web_server.broadcast_ui(payload), self.loop)

    def publish_event(self, event_name: str):
        msg = String()
        msg.data = event_name
        self.event_pub.publish(msg)
        self.get_logger().info(f"[HRI Event Published] {event_name}")


class HriCoordinator:
    """Orkestrator integrasi mesin suara, NLU, dan web server."""

    def __init__(self):
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.nlu = IntentRecognizer(threshold=0.68)
        self.web = WebServer(
            http_port=8000,
            ws_port=8765,
            on_audio_received=self.handle_audio,
            on_text_received=self.handle_text,
        )
        self.ros_node = None

    def set_ros_node(self, node: BarelangHriNode):
        self.ros_node = node
        self.ros_node.web_server = self.web

    async def speak(self, text: str):
        """Sintesis suara dan kirim ke web klien."""
        if not text:
            return
        print(f"[ROBOT SPEAKS] \"{text}\"")
        wav_base64 = self.tts.synthesize_wav_base64(text)

        payload = {
            "type": "TTS_PLAY",
            "text": text,
            "audio": wav_base64,
        }
        await self.web.broadcast_ui(payload)

    async def handle_audio(self, audio_bytes: bytes):
        """Menerima audio dari HP, transkripsi ke teks, lalu proses intent."""
        print(f"[STT] Memproses {len(audio_bytes)} bytes audio...")
        transcript = self.stt.transcribe(audio_bytes)
        print(f"[STT Result] '{transcript}'")

        await self.web.broadcast_ui({
            "type": "TRANSCRIPT",
            "transcript": transcript,
        })
        if transcript:
            await self.process_intent(transcript)

    async def handle_text(self, text: str):
        """Menerima teks langsung (simulasi UI) dan memproses intent."""
        print(f"[Text Input] '{text}'")
        await self.web.broadcast_ui({
            "type": "TRANSCRIPT",
            "transcript": text,
        })
        await self.process_intent(text)

    async def process_intent(self, text: str):
        """Mencocokkan intent dan mempublikasikan event ROS 2."""
        if self.nlu.match_intent(text, "WAKEWORD"):
            if self.ros_node:
                self.ros_node.publish_event("WAKEWORD_DETECTED")
        elif self.nlu.match_intent(text, "FOLLOW_ME"):
            if self.ros_node:
                self.ros_node.publish_event("FOLLOW_COMMAND")
        elif self.nlu.match_intent(text, "ARRIVED"):
            if self.ros_node:
                self.ros_node.publish_event("ARRIVED_COMMAND")


def ros_spin_thread(node):
    rclpy.spin(node)


async def main():
    rclpy.init()
    loop = asyncio.get_running_loop()

    coordinator = HriCoordinator()
    ros_node = BarelangHriNode(loop, coordinator.speak)
    coordinator.set_ros_node(ros_node)

    # Thread ROS 2
    t_ros = threading.Thread(target=ros_spin_thread, args=(ros_node,), daemon=True)
    t_ros.start()

    # Jalankan HTTP server lokal di background
    coordinator.web.start_http()

    print("========================================================================")
    print("🤖 Barelang Inari HRI Server Aktif (Modular)")
    print("🌐 HTTP Web App : http://0.0.0.0:8000")
    print("🔌 WebSocket    : ws://0.0.0.0:8765")
    print("📡 ROS 2 Topics : /hri/event (PUB), /hri/tts_say (SUB), /luggage/status (SUB)")
    print("========================================================================")

    async with websockets.serve(coordinator.web.ws_handler, "0.0.0.0", 8765, max_size=10 * 1024 * 1024):
        await asyncio.Future()


def main_entry():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[HRI] Mematikan sistem...")
        rclpy.shutdown()


if __name__ == "__main__":
    main_entry()