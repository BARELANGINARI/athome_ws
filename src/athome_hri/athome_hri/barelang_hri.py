"""
Barelang Inari - Human-Robot Interaction (HRI) Master Node (Refactored).
Bertugas murni sebagai Task Manager & antarmuka (Voice/NLU/Web).
Update : 21 oktober 2026
"""

import asyncio
import base64
from enum import Enum, auto
import io
import threading
import time
from typing import Callable, Dict, Optional, Tuple
import wave

import rclpy
from rclpy.client import Client
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_srvs.srv import Trigger
import websockets

try:
    from athome_hri.stt_engine import SpeechToText
    from athome_hri.tts_engine import TextToSpeech
    from athome_hri.intent_recognizer import IntentRecognizer
    from athome_hri.web_server import WebServer
except ImportError:
    from hri.stt_engine import SpeechToText
    from hri.tts_engine import TextToSpeech
    from hri.intent_recognizer import IntentRecognizer
    from hri.web_server import WebServer


# =====================================================================
# 1. EXPLICIT STATE DEFINITION
# =====================================================================
class RobotState(Enum):
    IDLE = auto()
    READY_TO_FOLLOW = auto()
    FOLLOWING = auto()
    WAIT_GESTURE = auto()
    ALIGNING_BAG = auto()
    GRABBING_BAG = auto()
    HOMING = auto()
    DONE = auto()


# =====================================================================
# 2. AUDIO SPEECH QUEUE (MENDUKUNG ASYNC AWAIT)
# =====================================================================
class AudioSpeechQueue:
    def __init__(
        self,
        tts_engine: TextToSpeech,
        broadcast_func: Callable,
        default_cooldown: float = 3.0,
        min_gap: float = 0.5,
    ):
        self.tts = tts_engine
        self.broadcast_func = broadcast_func
        self.default_cooldown = default_cooldown
        self.min_gap = min_gap
        self.queue: asyncio.Queue = asyncio.Queue()
        self.last_spoken_time: Dict[str, float] = {}
        self.global_last_spoken_time: float = 0.0
        self._worker_task: Optional[asyncio.Task] = None

    def start(self, loop: asyncio.AbstractEventLoop):
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = loop.create_task(self._worker())

    @staticmethod
    def _estimate_duration(wav_base64: str, text: str) -> float:
        if wav_base64:
            try:
                raw_bytes = base64.b64decode(wav_base64)
                with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
                    rate = wf.getframerate()
                    if rate > 0:
                        return wf.getnframes() / float(rate)
            except Exception:
                pass
        return max(len(text.split()) * 0.42, 1.8)

    async def _worker(self):
        while True:
            text, delay_after, done_future = await self.queue.get()
            try:
                gap = time.time() - self.global_last_spoken_time
                if gap < self.min_gap:
                    await asyncio.sleep(self.min_gap - gap)

                print(f'[ROBOT SPEAKS] "{text}"')
                wav_base64 = self.tts.synthesize_wav_base64(text) if self.tts else ""

                if self.broadcast_func:
                    await self.broadcast_func({
                        "type": "TTS_PLAY",
                        "text": text,
                        "audio": wav_base64,
                    })

                duration = self._estimate_duration(wav_base64, text)
                await asyncio.sleep(duration + delay_after + 0.3)
                self.global_last_spoken_time = time.time()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[AudioQueue Error] {e}")
            finally:
                if done_future and not done_future.done():
                    done_future.set_result(True)
                self.queue.task_done()

    def _can_enqueue(self, text: str, cooldown: Optional[float], force: bool) -> bool:
        key = text.lower()
        if not force and key in self.last_spoken_time:
            cd = self.default_cooldown if cooldown is None else cooldown
            if (time.time() - self.last_spoken_time[key]) < cd:
                return False
        self.last_spoken_time[key] = time.time()
        return True

    def enqueue(
        self,
        text: str,
        cooldown: Optional[float] = None,
        delay_after: float = 0.0,
        force: bool = False,
    ) -> bool:
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
        done_future = loop.create_future()
        self.queue.put_nowait((text_clean, delay_after, done_future))
        await done_future
        return True

    def clear(self):
        while not self.queue.empty():
            try:
                _, _, fut = self.queue.get_nowait()
                if fut and not fut.done():
                    fut.set_result(False)
                self.queue.task_done()
            except Exception:
                break


# =====================================================================
# 3. ROS 2 NODE (TASK MANAGER INTERFACE)
# =====================================================================
class BarelangHriNode(Node):
    def __init__(
        self,
        async_loop: asyncio.AbstractEventLoop,
        on_tts_command: Callable[[str], None],
        on_gesture_changed: Callable[[str], None],
    ):
        super().__init__("barelang_hri_node")
        self.loop = async_loop
        self.on_tts_command = on_tts_command
        self.on_gesture_changed = on_gesture_changed

        # Vision & Gesture State
        self.is_operator_in_front = False
        self.last_gesture = "NONE"
        self.gesture_ready_for_task = True
        self.gesture_event = asyncio.Event()

        # Publishers & Subscribers
        self.event_pub = self.create_publisher(String, "/hri/event", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", 10)

        self.create_subscription(String, "/hri/tts_say", self._on_tts_say, 10)
        self.create_subscription(String, "/zed/person_label", self._on_person_label, 10)
        self.create_subscription(String, "/zed/gesture", self._on_gesture, 10)

        # Service Clients Map (Menghilangkan duplikasi client)
        self.srv_clients: Dict[str, Client] = {
            "follow_start": self.create_client(Trigger, "/follow/person"),
            "follow_stop": self.create_client(Trigger, "/follow/stop"),
            "align_bag": self.create_client(Trigger, "/task/align_bag"),
            "grab_bag": self.create_client(Trigger, "/arm/ambil_tas"),
            "homing": self.create_client(Trigger, "/homing"),
        }
        self.get_logger().info("BarelangHriNode Task Manager aktif.")

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())

    def publish_event(self, event_name: str):
        msg = String()
        msg.data = event_name
        self.event_pub.publish(msg)
        self.get_logger().info(f"[HRI Event] {event_name}")

    def _on_tts_say(self, msg: String):
        text = msg.data.strip()
        if text and self.on_tts_command:
            self.loop.call_soon_threadsafe(self.on_tts_command, text)

    def _on_person_label(self, msg: String):
        data = msg.data.strip()
        self.is_operator_in_front = bool(
            data and data != "NO_PERSON" and ("operator:" in data or "person:" in data)
        )

    def _on_gesture(self, msg: String):
        gesture = msg.data.strip().upper()
        if not gesture or gesture == self.last_gesture:
            return
        self.last_gesture = gesture

        if gesture == "NONE":
            self.gesture_ready_for_task = True
        elif gesture in ("LEFT", "RIGHT") and self.gesture_ready_for_task:
            self.gesture_ready_for_task = False
            self.loop.call_soon_threadsafe(self.gesture_event.set)

        self.get_logger().info(f"[HRI] Gesture terdeteksi: {gesture}")
        if self.on_gesture_changed:
            self.loop.call_soon_threadsafe(self.on_gesture_changed, gesture)

    def reset_gesture_session(self):
        self.gesture_ready_for_task = self.last_gesture == "NONE"
        self.last_gesture = "NONE"
        self.gesture_event.clear()

    async def wait_for_new_gesture(self, timeout: float = 30.0) -> Optional[str]:
        try:
            await asyncio.wait_for(self.gesture_event.wait(), timeout=timeout)
            return self.last_gesture if self.last_gesture in ("LEFT", "RIGHT") else None
        except asyncio.TimeoutError:
            return None

    async def call_trigger(
        self, service_key: str, timeout: float = 60.0
    ) -> Tuple[bool, str]:
        """Wrapper generik untuk memanggil Trigger service secara async."""
        client = self.srv_clients.get(service_key)
        if not client or not client.service_is_ready():
            self.get_logger().warn(f"Service '{service_key}' belum siap!")
            return False, "Service not ready"

        self.get_logger().info(f"[HRI] Memanggil service: {service_key}...")
        ros_future = client.call_async(Trigger.Request())
        async_future = self.loop.create_future()

        def _on_done(fut):
            try:
                res = fut.result()
                self.loop.call_soon_threadsafe(
                    async_future.set_result, (res.success, res.message)
                )
            except Exception as e:
                self.loop.call_soon_threadsafe(
                    async_future.set_result, (False, str(e))
                )

        ros_future.add_done_callback(_on_done)

        try:
            success, message = await asyncio.wait_for(async_future, timeout=timeout)
            self.get_logger().info(
                f"[HRI] Service '{service_key}' selesai (success={success}): {message}"
            )
            return success, message
        except asyncio.TimeoutError:
            self.get_logger().error(f"[HRI] Service '{service_key}' timeout ({timeout}s)!")
            return False, "Timeout"


# =====================================================================
# 4. HRI COORDINATOR (LINEAR STATE MACHINE)
# =====================================================================
class HriCoordinator:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.state = RobotState.IDLE

        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.nlu = IntentRecognizer(threshold=0.68)
        self.web = WebServer(
            http_port=8000,
            ws_port=8765,
            on_audio_received=self.handle_audio,
            on_text_received=self.handle_text,
        )
        self.audio_queue = AudioSpeechQueue(self.tts, self.web.broadcast_ui)
        self.audio_queue.start(loop)

        self.ros_node = BarelangHriNode(
            async_loop=loop,
            on_tts_command=lambda text: self.audio_queue.enqueue(text),
            on_gesture_changed=self._on_gesture_changed,
        )

    def _on_gesture_changed(self, gesture: str):
        asyncio.create_task(
            self.web.broadcast_ui({"type": "GESTURE_UPDATE", "gesture": gesture})
        )

    async def set_state(self, new_state: RobotState, substate: str = ""):
        self.state = new_state
        await self.web.broadcast_ui({
            "type": "STATE_UPDATE",
            "state": new_state.name,
            "substate": substate,
        })

    async def handle_audio(self, audio_bytes: bytes):
        transcript = self.stt.transcribe(audio_bytes)
        await self.web.broadcast_ui({"type": "TRANSCRIPT", "transcript": transcript})
        if transcript:
            await self.process_intent(transcript)

    async def handle_text(self, text: str):
        await self.web.broadcast_ui({"type": "TRANSCRIPT", "transcript": text})
        await self.process_intent(text)

    async def process_intent(self, text: str):
        # State Guard: Abaikan perintah suara saat robot sedang sibuk mengeksekusi task
        busy_states = {
            RobotState.WAIT_GESTURE,
            RobotState.ALIGNING_BAG,
            RobotState.GRABBING_BAG,
            RobotState.HOMING,
        }
        if self.state in busy_states:
            return

        # 1. INISIALISASI (WAKEWORD)
        if self.nlu.match_intent(text, "WAKEWORD"):
            if self.ros_node.is_operator_in_front:
                self.ros_node.publish_event("WAKEWORD_DETECTED")
                await self.set_state(RobotState.READY_TO_FOLLOW, "Menunggu perintah Follow Me")
                self.audio_queue.enqueue(
                    "Hello, I'm Barelang Inari, I want to help you to carry the luggage. "
                    "If you are ready, please guide me to the place, and say follow me."
                )
            else:
                self.ros_node.get_logger().warn(
                    "Wakeword terdengar, tapi operator belum terdeteksi di depan robot."
                )
            return

        # 2. MULAI MENGIKUTI
        if self.nlu.match_intent(text, "FOLLOW_ME"):
            self.ros_node.publish_event("FOLLOW_COMMAND")
            ok, _ = await self.ros_node.call_trigger("follow_start", timeout=5.0)
            if ok:
                await self.set_state(RobotState.FOLLOWING, "Menghadapi rintangan (Nav2)")
                self.audio_queue.enqueue("Following you now.")
            else:
                self.audio_queue.enqueue("Person following is not ready. Please check the system.")
            return

        # 3. ALUR ARRIVED (Berjalan di background task agar tidak memblokir WebSocket)
        if self.nlu.match_intent(text, "ARRIVED"):
            asyncio.create_task(self._execute_arrival_sequence())

    async def _execute_arrival_sequence(self):
        """Sekuens tugas linear tanpa nested callbacks."""
        self.ros_node.publish_event("ARRIVED_COMMAND")
        await self.ros_node.call_trigger("follow_stop", timeout=5.0)
        self.ros_node.stop_robot()
        self.audio_queue.clear()
        self.ros_node.reset_gesture_session()

        # Step 1: Tanya tas mana yang harus diambil & tunggu gesture
        await self.set_state(RobotState.WAIT_GESTURE, "Menunggu gesture tunjuk tas")
        await self.audio_queue.speak_and_wait(
            "Okay, stopping. We have already arrived at the destination. "
            "Tell me, which bag should I take?"
        )

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

    async def _abort_sequence(self, reason: str):
        self.ros_node.get_logger().error(f"[HRI Abort] {reason}")
        await self.set_state(RobotState.IDLE, f"Error: {reason}")
        await self.audio_queue.speak_and_wait(
            "Sorry, I encountered a problem and cannot continue the task."
        )


# =====================================================================
# 5. ENTRY POINT
# =====================================================================
async def main():
    rclpy.init()
    loop = asyncio.get_running_loop()

    coordinator = HriCoordinator(loop)

    t_ros = threading.Thread(
        target=rclpy.spin, args=(coordinator.ros_node,), daemon=True
    )
    t_ros.start()
    coordinator.web.start_http()

    print("========================================================================")
    print("Barelang Inari HRI Server Aktif (Clean Task Manager)")
    print("Services: /follow/person, /follow/stop, /task/align_bag, /homing, /arm/ambil_tas")
    print("========================================================================")

    async with websockets.serve(
        coordinator.web.ws_handler, "0.0.0.0", 8765, max_size=10 * 1024 * 1024
    ):
        await asyncio.Future()


def main_entry():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        rclpy.shutdown()


if __name__ == "__main__":
    main_entry()