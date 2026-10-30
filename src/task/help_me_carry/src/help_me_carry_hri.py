#!/usr/bin/env python3
"""
Barelang Inari - Human-Robot Interaction (HRI) Master Node.

Bertugas sebagai Task Manager & antarmuka terpadu (Voice/NLU/Web/ROS 2/VAD Visualizer).
"""

import asyncio
from enum import Enum, auto
import logging
import threading
import time
from typing import Callable, Dict, Optional, Set, Tuple, Union

from geometry_msgs.msg import Twist
import numpy as np
import rclpy
from rclpy.client import Client
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
import websockets

from athome_hri.audio_io import (
    AudioSpeechQueue,
    LocalMicListener,
    VadVisualizer,
)
from athome_hri.intent_recognizer import IntentRecognizer
from athome_hri.stt_engine import SpeechToText
from athome_hri.tts_engine import TextToSpeech
from athome_hri.web_server import WebServer

logger = logging.getLogger("athome_hri.coordinator")

# =====================================================================
# 0. DEFAULT KONFIGURASI INPUT AUDIO & SERVER (Sesuai record_stream_visualize.py)
# =====================================================================
ENABLE_LOCAL_MIC: bool = True
ENABLE_WS_AUDIO: bool = True
ENABLE_VISUALIZER: bool = True
MIC_DEVICE_INDEX: Optional[int] = 4 # ganti USB MIC disini
THRESHOLD_DBFS: float = -30.0
THRESHOLD_FLATNESS: float = 0.40
PRE_ROLL_FRAMES: int = 10      # 300 ms sebelum suara terdeteksi
HANGOVER_FRAMES: int = 25      # 750 ms tunggu hening sebelum potong kalimat
MIN_SPEECH_FRAMES: int = 15    # 450 ms minimal durasi bicara
WHISPER_MODEL_SIZE: str = "deepdml/faster-whisper-large-v3-turbo-ct2"
HTTP_PORT: int = 8000
WS_PORT: int = 8765
WS_MAX_SIZE_BYTES: int = 10 * 1024 * 1024


# =====================================================================
# 1. EXPLICIT ROBOT STATE DEFINITION
# =====================================================================
class RobotState(Enum):
    """Definisi state eksplisit untuk alur tugas HRI Carry My Luggage."""

    IDLE = auto()
    READY_TO_FOLLOW = auto()
    FOLLOWING = auto()
    WAIT_GESTURE = auto()
    ALIGNING_BAG = auto()
    GRABBING_BAG = auto()
    HOMING = auto()
    DONE = auto()


BUSY_TASK_STATES: Set[RobotState] = {
    RobotState.WAIT_GESTURE,
    RobotState.ALIGNING_BAG,
    RobotState.GRABBING_BAG,
    RobotState.HOMING,
}


# =====================================================================
# 2. ROS 2 NODE (TASK MANAGER INTERFACE)
# =====================================================================
class BarelangHriNode(Node):
    """Node ROS 2 yang menjembatani komunikasi topik/service robot dengan event loop HRI."""

    TOPIC_HRI_EVENT = "/hri/event"
    TOPIC_STT_TRANSCRIPT = "/hri/stt_transcript"
    TOPIC_CMD_VEL = "cmd_vel"
    TOPIC_TTS_SAY = "/hri/tts_say"
    TOPIC_PERSON_LABEL = "/zed/person_label"
    TOPIC_GESTURE = "/zed/gesture"

    SERVICE_MAP: Dict[str, str] = {
        "follow_start": "/follow/person",
        "follow_stop": "/follow/stop",
        "align_bag": "/task/align_bag",
        "grab_bag": "/arm/ambil_tas",
        "homing": "/homing",
    }

    VALID_POINTING_GESTURES: Tuple[str, str] = ("LEFT", "RIGHT")
    OPERATOR_GRACE_PERIOD_SEC: float = 3.0

    def __init__(
        self,
        async_loop: asyncio.AbstractEventLoop,
        on_tts_command: Callable[[str], None],
        on_gesture_changed: Callable[[str], None],
    ) -> None:
        super().__init__("barelang_hri_node")
        self.loop = async_loop
        self.on_tts_command = on_tts_command
        self.on_gesture_changed = on_gesture_changed

        # Deklarasi parameter ROS 2 (Sesuai record_stream_visualize.py)
        self.declare_parameter("enable_local_mic", ENABLE_LOCAL_MIC)
        self.declare_parameter("enable_ws_audio", ENABLE_WS_AUDIO)
        self.declare_parameter("enable_visualizer", ENABLE_VISUALIZER)
        self.declare_parameter("threshold_dbfs", THRESHOLD_DBFS)
        self.declare_parameter("threshold_flatness", THRESHOLD_FLATNESS)
        self.declare_parameter("pre_roll_frames", PRE_ROLL_FRAMES)
        self.declare_parameter("hangover_frames", HANGOVER_FRAMES)
        self.declare_parameter("min_speech_frames", MIN_SPEECH_FRAMES)
        self.declare_parameter("whisper_model", WHISPER_MODEL_SIZE)
        self.declare_parameter("enable_task_manager", True)

        self.enable_local_mic: bool = bool(
            self.get_parameter("enable_local_mic").value
        )
        self.enable_ws_audio: bool = bool(self.get_parameter("enable_ws_audio").value)
        self.enable_visualizer: bool = bool(
            self.get_parameter("enable_visualizer").value
        )
        self.threshold_dbfs: float = float(
            self.get_parameter("threshold_dbfs").value
        )
        self.threshold_flatness: float = float(
            self.get_parameter("threshold_flatness").value
        )
        self.pre_roll_frames: int = int(
            self.get_parameter("pre_roll_frames").value
        )
        self.hangover_frames: int = int(
            self.get_parameter("hangover_frames").value
        )
        self.min_speech_frames: int = int(
            self.get_parameter("min_speech_frames").value
        )
        self.whisper_model: str = str(
            self.get_parameter("whisper_model").value
        )
        self.enable_task_manager: bool = bool(
            self.get_parameter("enable_task_manager").value
        )

        # Vision & Gesture State
        self._person_topic_received: bool = False
        self._last_operator_seen_time: float = 0.0
        self._gesture_lock = threading.Lock()
        self.last_gesture: str = "NONE"
        self.gesture_ready_for_task: bool = False
        self._captured_gesture: Optional[str] = None
        self.gesture_event = asyncio.Event()

        # Publishers & Subscribers
        self.event_pub = self.create_publisher(String, self.TOPIC_HRI_EVENT, 10)
        self.transcript_pub = self.create_publisher(
            String, self.TOPIC_STT_TRANSCRIPT, 10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, self.TOPIC_CMD_VEL, 10)

        self.create_subscription(String, self.TOPIC_TTS_SAY, self._on_tts_say, 10)
        self.create_subscription(
            String, self.TOPIC_PERSON_LABEL, self._on_person_label, 10
        )
        self.create_subscription(String, self.TOPIC_GESTURE, self._on_gesture, 10)

        # Service Clients
        self.srv_clients: Dict[str, Client] = {
            key: self.create_client(Trigger, srv_name)
            for key, srv_name in self.SERVICE_MAP.items()
        }
        self.get_logger().info("BarelangHriNode Task Manager aktif.")

    def stop_robot(self) -> None:
        """Mengirim perintah kecepatan nol (Twist kosong) untuk menghentikan base robot."""
        self.cmd_vel_pub.publish(Twist())

    def publish_event(self, event_name: str) -> None:
        """Memublikasikan event status HRI ke topik /hri/event."""
        msg = String()
        msg.data = event_name
        self.event_pub.publish(msg)
        self.get_logger().info(f"[HRI Event] {event_name}")

    def publish_transcript(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.transcript_pub.publish(msg)

    def _on_tts_say(self, msg: String) -> None:
        text = msg.data.strip()
        if text and self.on_tts_command:
            self.loop.call_soon_threadsafe(self.on_tts_command, text)

    @property
    def is_operator_in_front(self) -> bool:
        """
        Mengembalikan True jika operator terlihat dalam OPERATOR_GRACE_PERIOD_SEC terakhir,
        atau jika topik kamera /zed/person_label belum aktif (mode pengujian HRI standalone).
        """
        if not self._person_topic_received:
            return True
        return (
            time.monotonic() - self._last_operator_seen_time
        ) <= self.OPERATOR_GRACE_PERIOD_SEC

    def _on_person_label(self, msg: String) -> None:
        self._person_topic_received = True
        data = msg.data.strip()
        if data and data != "NO_PERSON" and (
            "operator:" in data or "person:" in data
        ):
            self._last_operator_seen_time = time.monotonic()

    def _on_gesture(self, msg: String) -> None:
        gesture = msg.data.strip().upper()
        if not gesture:
            return

        trigger_event = False
        changed = False
        with self._gesture_lock:
            if gesture != self.last_gesture:
                self.last_gesture = gesture
                changed = True
            if (
                gesture in self.VALID_POINTING_GESTURES
                and self.gesture_ready_for_task
            ):
                self.gesture_ready_for_task = False
                self._captured_gesture = gesture
                trigger_event = True

        if trigger_event:
            self.loop.call_soon_threadsafe(self.gesture_event.set)

        if changed:
            self.get_logger().info(f"[HRI] Gesture terdeteksi: {gesture}")
            if self.on_gesture_changed:
                self.loop.call_soon_threadsafe(self.on_gesture_changed, gesture)

    def reset_gesture_session(self) -> None:
        """Mereset state deteksi gesture secara thread-safe tepat sebelum menunggu gesture baru."""
        with self._gesture_lock:
            self.last_gesture = "NONE"
            self._captured_gesture = None
            self.gesture_ready_for_task = True
        self.gesture_event.clear()

    async def wait_for_new_gesture(self, timeout: float = 30.0) -> Optional[str]:
        """Menunggu secara asinkron hingga gesture tunjuk (LEFT/RIGHT) baru terdeteksi."""
        try:
            await asyncio.wait_for(self.gesture_event.wait(), timeout=timeout)
            with self._gesture_lock:
                return (
                    self._captured_gesture
                    if self._captured_gesture in self.VALID_POINTING_GESTURES
                    else None
                )
        except asyncio.TimeoutError:
            with self._gesture_lock:
                self.gesture_ready_for_task = False
            return None

    async def call_trigger(
        self, service_key: str, timeout: float = 60.0
    ) -> Tuple[bool, str]:
        """Memanggil ROS 2 Trigger service secara asinkron dengan batas waktu (timeout)."""
        client = self.srv_clients.get(service_key)
        if not client or not client.service_is_ready():
            self.get_logger().warn(f"Service '{service_key}' belum siap!")
            return False, "Service not ready"

        self.get_logger().info(f"[HRI] Memanggil service: {service_key}...")
        ros_future = client.call_async(Trigger.Request())
        async_future: asyncio.Future[Tuple[bool, str]] = self.loop.create_future()

        def _on_done(fut) -> None:
            try:
                res = fut.result()
                self.loop.call_soon_threadsafe(
                    async_future.set_result, (res.success, res.message)
                )
            except Exception as exc:
                self.loop.call_soon_threadsafe(
                    async_future.set_result, (False, str(exc))
                )

        ros_future.add_done_callback(_on_done)

        try:
            success, message = await asyncio.wait_for(async_future, timeout=timeout)
            self.get_logger().info(
                f"[HRI] Service '{service_key}' selesai (success={success}): {message}"
            )
            return success, message
        except asyncio.TimeoutError:
            self.get_logger().error(
                f"[HRI] Service '{service_key}' timeout ({timeout}s)!"
            )
            return False, "Timeout"


# =====================================================================
# 3. HRI COORDINATOR (LINEAR STATE MACHINE)
# =====================================================================
class HriCoordinator:
    """Koordinator utama state machine HRI yang menyatukan STT, NLU, TTS, Web UI, Visualizer, dan ROS 2."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.state = RobotState.IDLE
        self._stt_lock = asyncio.Lock()

        self.ros_node = BarelangHriNode(
            async_loop=loop,
            on_tts_command=lambda text: (
                self.audio_queue.enqueue(text)
                if hasattr(self, "audio_queue")
                else None
            ),
            on_gesture_changed=self._on_gesture_changed,
        )

        self.enable_local_mic: bool = self.ros_node.enable_local_mic
        self.enable_ws_audio: bool = self.ros_node.enable_ws_audio
        self.enable_visualizer: bool = self.ros_node.enable_visualizer

        self.mic_listener = LocalMicListener(
            loop=loop,
            on_audio_captured=self.handle_local_mic_audio,
            is_paused_fn=self._is_mic_paused,
            device_index=MIC_DEVICE_INDEX,
            threshold_dbfs=self.ros_node.threshold_dbfs,
            threshold_flatness=self.ros_node.threshold_flatness,
            pre_roll_frames=self.ros_node.pre_roll_frames,
            hangover_frames=self.ros_node.hangover_frames,
            min_speech_frames=self.ros_node.min_speech_frames,
        )

        self.visualizer = VadVisualizer(self.mic_listener)
        self.mic_listener.on_frame_metrics = self.visualizer.push_frame

        self.stt = SpeechToText(
            model_size=self.ros_node.whisper_model,
            on_asr_result=self.visualizer.update_asr_status,
        )
        self.tts = TextToSpeech()
        self.nlu = IntentRecognizer(threshold=0.68)
        self.web = WebServer(
            http_port=HTTP_PORT,
            ws_port=WS_PORT,
            on_audio_received=self.handle_ws_audio,
            on_text_received=self.handle_text,
        )
        self.audio_queue = AudioSpeechQueue(self.tts, self.web.broadcast_ui)
        self.audio_queue.start(loop)

        if self.enable_local_mic:
            if self.enable_visualizer:
                self.visualizer.start(loop)
            self.mic_listener.start()

    def shutdown(self) -> None:
        """Membersihkan resource audio, visualisasi, server HTTP, dan node ROS 2 saat aplikasi berhenti."""
        self.mic_listener.stop()
        self.visualizer.stop()
        self.audio_queue.stop()
        self.web.stop_http()
        self.ros_node.destroy_node()

    def _is_mic_paused(self) -> bool:
        return (
            not self.enable_local_mic
            or self.audio_queue.is_speaking
            or self.state in BUSY_TASK_STATES
        )

    def _on_gesture_changed(self, gesture: str) -> None:
        self.loop.create_task(
            self.web.broadcast_ui({"type": "GESTURE_UPDATE", "gesture": gesture})
        )

    async def set_state(self, new_state: RobotState, substate: str = "") -> None:
        """Memperbarui state robot dan menyiarkan pembaruan ke Web UI."""
        self.state = new_state
        await self.web.broadcast_ui(
            {
                "type": "STATE_UPDATE",
                "state": new_state.name,
                "substate": substate,
            }
        )

    async def handle_ws_audio(self, audio_bytes: bytes) -> None:
        if not self.enable_ws_audio:
            self.ros_node.get_logger().debug(
                "Input audio WebSocket diabaikan (enable_ws_audio=False)."
            )
            return
        await self.handle_audio(audio_bytes)

    async def handle_local_mic_audio(
        self, audio_input: Union[np.ndarray, bytes]
    ) -> None:
        if not self.enable_local_mic:
            return
        await self.handle_audio(audio_input)

    async def handle_audio(self, audio_input: Union[np.ndarray, bytes]) -> None:
        async with self._stt_lock:
            transcript = await asyncio.to_thread(self.stt.transcribe, audio_input)
        if transcript:
            logger.info('[STT TRANSCRIPT] "%s"', transcript)
        await self.web.broadcast_ui({"type": "TRANSCRIPT", "transcript": transcript})
        if transcript:
            self.ros_node.publish_transcript(transcript)
        if transcript:
            await self.process_intent(transcript)

    async def handle_text(self, text: str) -> None:
        await self.web.broadcast_ui({"type": "TRANSCRIPT", "transcript": text})
        if text.strip():
            self.ros_node.publish_transcript(text)
        await self.process_intent(text)

    async def process_intent(self, text: str) -> None:
        """Mengevaluasi teks transkripsi terhadap NLU dan memicu transisi state yang sesuai."""
        if not self.ros_node.enable_task_manager:
            return
        if self.state in BUSY_TASK_STATES:
            return

        # 1. INISIALISASI (WAKEWORD)
        if self.nlu.match_intent(text, "WAKEWORD"):
            if self.ros_node.is_operator_in_front:
                self.ros_node.publish_event("WAKEWORD_DETECTED")
                await self.set_state(
                    RobotState.READY_TO_FOLLOW, "Menunggu perintah Follow Me"
                )
                self.audio_queue.enqueue(
                    "Hello, I'm Barelang Inari, I want to help you to carry the luggage. "
                    "If you are ready, please guide me to the place, and say follow me."
                )
            else:
                self.ros_node.get_logger().warn(
                    "Wakeword terdengar, tapi operator belum terdeteksi di depan robot."
                )
            return

        # 2. MULAI MENGIKUTI (FOLLOW ME)
        if self.nlu.match_intent(text, "FOLLOW_ME"):
            self.ros_node.publish_event("FOLLOW_COMMAND")
            ok, _ = await self.ros_node.call_trigger("follow_start", timeout=5.0)
            if ok:
                await self.set_state(
                    RobotState.FOLLOWING, "Menghadapi rintangan (Nav2)"
                )
                self.audio_queue.enqueue("Following you now.")
            else:
                self.audio_queue.enqueue(
                    "Person following is not ready. Please check the system."
                )
            return

        # 3. ALUR ARRIVED (Berjalan di background task agar tidak memblokir WebSocket)
        if self.nlu.match_intent(text, "ARRIVED"):
            self.loop.create_task(self._execute_arrival_sequence())
            return

        # 4. PERINTAH BERHENTI DARURAT / MANUAL (STOP)
        if self.state == RobotState.FOLLOWING and self.nlu.match_intent(text, "STOP"):
            self.ros_node.publish_event("STOP_COMMAND")
            await self.ros_node.call_trigger("follow_stop", timeout=5.0)
            self.ros_node.stop_robot()
            await self.set_state(
                RobotState.READY_TO_FOLLOW, "Berhenti mengikuti (menunggu perintah)"
            )
            self.audio_queue.enqueue("Okay, stopping.")

    async def _execute_arrival_sequence(self) -> None:
        """Sekuens tugas linear saat sampai di lokasi pengambilan tas."""
        self.ros_node.publish_event("ARRIVED_COMMAND")
        await self.ros_node.call_trigger("follow_stop", timeout=5.0)
        self.ros_node.stop_robot()
        self.audio_queue.clear()

        # Step 1: Tanya tas mana yang harus diambil & tunggu gesture
        await self.set_state(RobotState.WAIT_GESTURE, "Menunggu gesture tunjuk tas")
        await self.audio_queue.speak_and_wait(
            "Okay, stopping. We have already arrived at the destination. "
            "Tell me, which bag should I take?"
        )

        self.ros_node.reset_gesture_session()
        gesture = await self.ros_node.wait_for_new_gesture(timeout=30.0)
        if not gesture:
            await self.audio_queue.speak_and_wait(
                "I did not detect a new left or right gesture. Please try again."
            )
            await self.set_state(RobotState.READY_TO_FOLLOW, "Gesture timeout")
            return

        # Mirroring perspektif kamera ke perspektif operator
        side = "right" if gesture == "LEFT" else "left"
        await self.audio_queue.speak_and_wait(
            f"Okay, I know which bag you're pointing to. You're pointing to the {side} bag."
        )

        # Step 2: Align Bag
        await self.set_state(RobotState.ALIGNING_BAG, f"Aligning to {side} bag")
        ok, msg = await self.ros_node.call_trigger("align_bag", timeout=45.0)
        if not ok:
            await self._abort_sequence(f"Failed to align with the bag: {msg}")
            return

        # Step 3: Arm Ambil Tas
        await self.set_state(RobotState.GRABBING_BAG, "Mengambil tas dengan lengan")
        await self.audio_queue.speak_and_wait("I am taking the bag now.")
        ok, msg = await self.ros_node.call_trigger("grab_bag", timeout=60.0)
        if not ok:
            await self._abort_sequence(f"Failed to grab the bag: {msg}")
            return

        # Step 4: Homing
        await self.set_state(RobotState.HOMING, "Kembali ke titik awal")
        await self.audio_queue.speak_and_wait(
            "I've got the bag, now I'm going back to where I started."
        )
        ok, msg = await self.ros_node.call_trigger("homing", timeout=120.0)
        if not ok:
            await self._abort_sequence(f"Failed to return to starting point: {msg}")
            return

        # Step 5: Selesai
        await self.audio_queue.speak_and_wait("I have reached the starting point.")
        await self.audio_queue.speak_and_wait("Task Completed.")
        await self.set_state(RobotState.DONE, "Task Completed")
        self.ros_node.get_logger().info("[HRI] TASK COMPLETED. Sistem standby.")

    async def _abort_sequence(self, reason: str) -> None:
        self.ros_node.get_logger().error(f"[HRI Abort] {reason}")
        await self.set_state(RobotState.IDLE, f"Error: {reason}")
        await self.audio_queue.speak_and_wait(
            "Sorry, I encountered a problem and cannot continue the task."
        )


# =====================================================================
# 4. ENTRY POINT
# =====================================================================
def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%H:%M:%S",
    )


async def main() -> None:
    _configure_logging()
    rclpy.init()
    loop = asyncio.get_running_loop()
    coordinator = HriCoordinator(loop)

    ros_spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(coordinator.ros_node,),
        name="HriRosSpin",
        daemon=True,
    )
    ros_spin_thread.start()
    coordinator.web.start_http()

    services_list = ", ".join(BarelangHriNode.SERVICE_MAP.values())
    logger.info("=" * 72)
    logger.info("Barelang Inari HRI Server Aktif (Clean Task Manager)")
    logger.info(
        "Input Audio -> Local Mic: %s | Visualizer: %s | WebSocket: %s",
        coordinator.enable_local_mic,
        coordinator.enable_visualizer,
        coordinator.enable_ws_audio,
    )
    logger.info("Services: %s", services_list)
    logger.info("=" * 72)

    try:
        async with websockets.serve(
            coordinator.web.ws_handler,
            "0.0.0.0",
            WS_PORT,
            max_size=WS_MAX_SIZE_BYTES,
        ):
            await asyncio.Future()
    finally:
        coordinator.shutdown()


def main_entry() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main_entry()
