import asyncio
import base64
import io
import os
import threading
import time
import wave
from typing import Callable, Optional

# Library ROS 2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_srvs.srv import Trigger
import websockets

try:
    from icecream import ic
except ImportError:
    def ic(*args):
        if not args:
            print("ic|")
            return
        parts = [str(a) for a in args]
        print(f"ic| {', '.join(parts)}")
        return args[0] if len(args) == 1 else args

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
# 1. AUDIO SPEECH QUEUE (ANTI-SPAM & ANTI-OVERLAPPING)
# =====================================================================
class AudioSpeechQueue:
    def __init__(
        self,
        tts_engine: TextToSpeech,
        broadcast_func,
        default_cooldown: float = 3.0,
        min_gap: float = 0.5,
    ):
        self.tts = tts_engine
        self.broadcast_func = broadcast_func
        self.default_cooldown = default_cooldown
        self.min_gap = min_gap
        self.queue: asyncio.Queue = asyncio.Queue()
        self.last_spoken_time = {}
        self.global_last_spoken_time: float = 0.0
        self._worker_task: Optional[asyncio.Task] = None

    def start(self, loop: asyncio.AbstractEventLoop):
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = loop.create_task(self._worker())

    async def _worker(self):
        while True:
            item = await self.queue.get()
            try:
                text, delay_after, state, substate, on_finish = item
                now = time.time()
                gap = now - self.global_last_spoken_time
                if gap < self.min_gap:
                    await asyncio.sleep(self.min_gap - gap)

                ic(text)
                wav_base64 = self.tts.synthesize_wav_base64(text) if self.tts else ""

                if self.broadcast_func:
                    await self.broadcast_func({
                        "type": "TTS_PLAY",
                        "text": text,
                        "audio": wav_base64,
                        "state": state,
                        "substate": substate,
                    })

                audio_duration = 0.0
                if wav_base64:
                    try:
                        raw_bytes = base64.b64decode(wav_base64)
                        with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
                            rate = wf.getframerate()
                            if rate > 0:
                                audio_duration = wf.getnframes() / float(rate)
                    except Exception:
                        pass
                if audio_duration <= 0.0:
                    audio_duration = max(len(text.split()) * 0.42, 1.8)

                await asyncio.sleep(audio_duration + delay_after + 0.3)
                self.global_last_spoken_time = time.time()

                if on_finish:
                    try:
                        if asyncio.iscoroutinefunction(on_finish):
                            await on_finish()
                        else:
                            res = on_finish()
                            if asyncio.iscoroutine(res):
                                await res
                    except Exception as e:
                        ic(e)
            except asyncio.CancelledError:
                break
            except Exception as e:
                ic(e)
            finally:
                self.queue.task_done()

    def enqueue(
        self,
        text: str,
        state: str = "",
        substate: str = "",
        cooldown: Optional[float] = None,
        delay_after: float = 0.0,
        force: bool = False,
        on_finish: Optional[Callable] = None,
    ) -> bool:
        text_clean = text.strip()
        if not text_clean:
            return False
        key = text_clean.lower()
        if not force and key in self.last_spoken_time:
            cd = self.default_cooldown if cooldown is None else cooldown
            if (time.time() - self.last_spoken_time[key]) < cd:
                return False
        self.last_spoken_time[key] = time.time()
        self.queue.put_nowait((text_clean, delay_after, state, substate, on_finish))
        return True

    def clear(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except Exception:
                break


# =====================================================================
# 2. ROS 2 NODE (TASK MANAGER INTERFACE)
# =====================================================================
class BarelangHriNode(Node):
    def __init__(self, async_loop, on_tts_command):
        super().__init__("barelang_hri_node")
        self.loop = async_loop
        self.on_tts_command = on_tts_command

        # State
        self.is_operator_in_front = False
        self.last_gesture = "NONE"

        # Publishers
        self.event_pub = self.create_publisher(String, "/hri/event", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", 10)

        # Subscribers
        self.tts_sub = self.create_subscription(String, "/hri/tts_say", self._on_tts_say, 10)
        self.person_label_sub = self.create_subscription(
            String, "/zed/person_label", self._on_person_label, 10
        )
        self.gesture_sub = self.create_subscription(
            String, "/zed/gesture", self._on_gesture, 10
        )

        # Service Clients
        self.follow_client = self.create_client(Trigger, "/follow/person")
        self.follow_stop_client = self.create_client(Trigger, "/follow/stop")
        self.align_bag_client = self.create_client(Trigger, "/task/align_bag")
        self.homing_client = self.create_client(Trigger, "/homing")

        ic("BarelangHriNode Task Manager aktif")

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())

    def _on_tts_say(self, msg: String):
        text = msg.data.strip()
        if text and self.on_tts_command:
            self.loop.call_soon_threadsafe(self.on_tts_command, text)

    def _on_person_label(self, msg: String):
        """Membaca topic deteksi operator dari vision (/zed/person_label)."""
        data = msg.data.strip()
        if data and data != "NO_PERSON" and ("operator:" in data or "person:" in data):
            self.is_operator_in_front = True
        elif data == "NO_PERSON":
            self.is_operator_in_front = False

    def _on_gesture(self, msg: String):
        """Menerima gesture dari vision (/zed/gesture) lalu broadcast ke Web UI."""
        gesture = msg.data.strip()
        if not gesture or gesture == self.last_gesture:
            return  # skip jika tidak berubah (hemat bandwidth)
        self.last_gesture = gesture
        ic(gesture)
        # Broadcast ke semua Web UI client
        asyncio.run_coroutine_threadsafe(
            self._broadcast_gesture(gesture), self.loop
        )

    async def _broadcast_gesture(self, gesture: str):
        """Kirim update gesture ke Web UI via WebSocket."""
        if not hasattr(self, "web_server") or self.web_server is None:
            return
        await self.web_server.broadcast_ui({
            "type": "GESTURE_UPDATE",
            "gesture": gesture,          # "RIGHT" | "LEFT" | "NONE"
        })

    def publish_event(self, event_name: str):
        msg = String()
        msg.data = event_name
        self.event_pub.publish(msg)
        ic(event_name)

    def start_follow(self) -> bool:
        if not self.follow_client.service_is_ready():
            return False
        self.follow_client.call_async(Trigger.Request())
        return True

    def stop_follow(self) -> bool:
        if not self.follow_stop_client.service_is_ready():
            return False
        self.follow_stop_client.call_async(Trigger.Request())
        return True

    def call_align_bag(self, on_complete: Callable):
        """Memanggil routine membaca gesture & mengambil tas."""
        if not self.align_bag_client.service_is_ready():
            ic("Service /task/align_bag belum siap")
            return False

        ic("Menjalankan Task: Align Bag")
        future = self.align_bag_client.call_async(Trigger.Request())

        def _on_done(fut):
            try:
                res = fut.result()
                ic(res.message if res else "OK")
            except Exception as e:
                ic(e)
            self.loop.call_soon_threadsafe(on_complete)

        future.add_done_callback(_on_done)
        return True

    def call_homing(self, on_complete: Callable):
        """Memanggil routine kembali ke titik start."""
        if not self.homing_client.service_is_ready():
            ic("Service /homing belum siap")
            return False

        ic("Menjalankan Task: Homing")
        future = self.homing_client.call_async(Trigger.Request())

        def _on_done(fut):
            try:
                res = fut.result()
                ic(res.message if res else "OK")
            except Exception as e:
                ic(e)
            self.loop.call_soon_threadsafe(on_complete)

        future.add_done_callback(_on_done)
        return True


# =====================================================================
# 3. HRI COORDINATOR
# =====================================================================
class HriCoordinator:
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
        self.audio_queue = AudioSpeechQueue(self.tts, self.web.broadcast_ui)
        self.ros_node: Optional[BarelangHriNode] = None

    def set_ros_node(self, node: BarelangHriNode):
        self.ros_node = node
        self.ros_node.web_server = self.web

    def speak(
        self,
        text: str,
        force: bool = False,
        cooldown: Optional[float] = None,
        on_finish: Optional[Callable] = None,
    ):
        if text:
            self.audio_queue.enqueue(
                text, force=force, cooldown=cooldown, on_finish=on_finish
            )

    async def handle_audio(self, audio_bytes: bytes):
        transcript = self.stt.transcribe(audio_bytes)
        await self.web.broadcast_ui({"type": "TRANSCRIPT", "transcript": transcript})
        if transcript:
            await self.process_intent(transcript)

    async def handle_text(self, text: str):
        await self.web.broadcast_ui({"type": "TRANSCRIPT", "transcript": text})
        await self.process_intent(text)

    async def process_intent(self, text: str):
        if not self.ros_node:
            return

        # 1. INISIALISASI (WAKEWORD)
        if self.nlu.match_intent(text, "WAKEWORD"):
            if self.ros_node.is_operator_in_front:
                self.ros_node.publish_event("WAKEWORD_DETECTED")
                self.speak(
                    "Hello, I'm Barelang Inari, I want to help you to carry the luggage. "
                    "If you are ready, please guide me to the place, and say follow me."
                )
            else:
                ic("Wakeword detected, operator not in front")
            return

        # 2. MULAI MENGIKUTI
        if self.nlu.match_intent(text, "FOLLOW_ME"):
            started = self.ros_node.start_follow()
            self.ros_node.publish_event("FOLLOW_COMMAND")
            if started:
                await self.web.broadcast_ui({
                    "type": "STATE_UPDATE",
                    "state": "FOLLOWING",
                    "substate": "Menghadapi rintangan (Nav2)",
                })
                self.speak("Following you now.")
            else:
                self.speak("Person following is not ready. Please check the system.")
            return

        # 3. ALUR ARRIVED -> ALIGN BAG -> HOMING -> SELESAI
        is_arrived = self.nlu.match_intent(text, "ARRIVED")
        if is_arrived:
            self.ros_node.publish_event("ARRIVED_COMMAND")
            self.ros_node.stop_follow()
            self.ros_node.stop_robot()
            self.audio_queue.clear()

            await self.web.broadcast_ui({
                "type": "STATE_UPDATE",
                "state": "WAIT_GESTURE",
                "substate": "Service /task/align_bag siap",
            })

            # -- Chain 3: Homing Selesai --
            def on_homing_finished():
                def on_task_complete_spoken():
                    ic("TASK COMPLETED. Sistem standby")
                    asyncio.run_coroutine_threadsafe(
                        self.web_server.broadcast_ui({
                            "type": "STATE_UPDATE",
                            "state": "DONE",
                            "substate": "Task Completed",
                        }),
                        self.ros_node.loop,
                    )

                # Speak part 1, then part 2
                self.speak(
                    "I have reached the starting point.",
                    force=True,
                    on_finish=lambda: self.speak(
                        "Task Completed.",
                        force=True,
                        on_finish=on_task_complete_spoken,
                    ),
                )

            # -- Chain 2: Align Bag Selesai --
            def on_align_bag_finished():
                def trigger_homing():
                    self.ros_node.call_homing(on_complete=on_homing_finished)

                self.speak(
                    "I've got the bag; now I'm going back to where I started.",
                    force=True,
                    on_finish=trigger_homing,
                )

            # -- Chain 1: Ucapan Arrived Selesai --
            def on_arrived_spoken():
                # Langsung serahkan ke service align bag untuk mencari gesture & tas
                self.ros_node.call_align_bag(on_complete=on_align_bag_finished)

            # Trigger awal dari rantai kejadian
            stop_text = (
                "Okay, stopping. We have already arrived at the destination. "
                "Tell me, which bag should I take?"
            )
            self.speak(stop_text, force=True, on_finish=on_arrived_spoken)
            return


def ros_spin_thread(node):
    rclpy.spin(node)


async def main():
    rclpy.init()
    loop = asyncio.get_running_loop()

    coordinator = HriCoordinator()
    coordinator.audio_queue.start(loop)

    ros_node = BarelangHriNode(loop, coordinator.speak)
    coordinator.set_ros_node(ros_node)

    t_ros = threading.Thread(target=ros_spin_thread, args=(ros_node,), daemon=True)
    t_ros.start()
    coordinator.web.start_http()

    ic("Barelang Inari HRI Server Aktif")

    async with websockets.serve(coordinator.web.ws_handler, "0.0.0.0", 8765, max_size=10 * 1024 * 1024):
        await asyncio.Future()


def main_entry():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        rclpy.shutdown()


if __name__ == "__main__":
    main_entry()