"""
Barelang Inari - Human-Robot Interaction (HRI) Master Node
Task: Carry My Luggage (RoboCup @Home)
Terintegrasi penuh dengan Vision (ZED), Obstacle Avoidance (/scan), Motor (cmd_vel), dan Nav2.
Dioptimalkan untuk Ubuntu 24.04, CPU Intel i7, GPU NVIDIA GeForce RTX 2060 Mobile.
"""

import asyncio
import base64
from difflib import SequenceMatcher
import http.server
import io
import json
import math
import os
import socketserver
import threading
import time
import wave

# --- 0. PATCH KOMPATIBILITAS PyAV v19 & FASTER-WHISPER ---
try:
  import av

  orig_av_open = av.open

  def patched_av_open(*args, **kwargs):
    kwargs.pop("metadata_errors", None)
    return orig_av_open(*args, **kwargs)

  av.open = patched_av_open
except Exception:
  pass

from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
import websockets

# Optional imports for STT & TTS
try:
  from faster_whisper import WhisperModel

  HAS_WHISPER = True
except ImportError:
  HAS_WHISPER = False
  print("[WARN] faster_whisper tidak ditemukan. STT akan menggunakan fallback.")

try:
  from piper.voice import PiperVoice

  HAS_PIPER = True
except ImportError:
  HAS_PIPER = False
  print("[WARN] piper-tts tidak ditemukan. TTS audio byte tidak aktif.")


# =====================================================================
# 1. CORE ENGINES: FASTER-WHISPER (GPU/CPU) & PIPER TTS
# =====================================================================
class SpeechEngines:

  def __init__(self):
    self.whisper = None
    self.piper = None

    # Inisialisasi Faster-Whisper dengan Auto-Detect GPU (RTX 2060 Mobile)
    if HAS_WHISPER:
      device = "cpu"
      compute_type = "int8"
      try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
          device = "cuda"
          compute_type = "float16"
          print(
              f"[STT] GPU NVIDIA terdeteksi ({ctranslate2.get_cuda_device_count()} device). Menggunakan CUDA + FP16."
          )
        else:
          print("[STT] GPU tidak terdeteksi oleh CTranslate2. Menggunakan CPU.")
      except Exception:
        print("[STT] Fallback CTranslate2 ke CPU.")

      try:
        print(f"[STT] Memuat model Faster-Whisper ({device}, {compute_type})...")
        self.whisper = WhisperModel(
            "base.en",
            device=device,
            compute_type=compute_type,
            cpu_threads=4 if device == "cpu" else 1,
        )
        print("[STT] Faster-Whisper siap.")
      except Exception as e:
        print(f"[STT Error] Gagal memuat Faster-Whisper ({device}): {e}")
        if device == "cuda":
          print("[STT] Mencoba fallback ke CPU (int8)...")
          try:
            self.whisper = WhisperModel("base.en", device="cpu", compute_type="int8", cpu_threads=4)
            print("[STT] Faster-Whisper CPU siap.")
          except Exception as e2:
            print(f"[STT Error] Fallback CPU juga gagal: {e2}")

    # Inisialisasi Piper TTS (ONNX Model)
    if HAS_PIPER:
      piper_paths = [
          "/opt/models/hri/en_US-lessac-medium.onnx",
          "/home/yesaya/Documents/cluster_3/hri_barelang_inari/hri_ws/models/en_US-lessac-medium.onnx",
          "/home/yesaya/Documents/cluster_3/hri_athome/hri_ws/models/en_US-lessac-medium.onnx",
      ]
      model_path = None
      for p in piper_paths:
        if os.path.exists(p):
          model_path = p
          break

      if model_path:
        try:
          print(f"[TTS] Memuat model Piper TTS dari: {model_path}")
          self.piper = PiperVoice.load(model_path)
          print("[TTS] Piper TTS siap.")
        except Exception as e:
          print(f"[TTS Error] Gagal memuat model Piper: {e}")
      else:
        print("[TTS] Warning: File model .onnx Piper tidak ditemukan.")

  def transcribe(self, audio_bytes: bytes) -> str:
    """Menerima raw audio bytes dari smartphone dan mentranskripsi via Faster-Whisper."""
    if not self.whisper:
      return ""

    temp_path = "/tmp/hri_input_audio.webm"
    try:
      with open(temp_path, "wb") as f:
        f.write(audio_bytes)

      segments, _ = self.whisper.transcribe(
          temp_path,
          beam_size=5,
          language="en",
          vad_filter=True,
          vad_parameters=dict(min_silence_duration_ms=400),
      )
      text = " ".join([seg.text for seg in segments]).strip()
      return text
    except Exception as e:
      print(f"[STT Error] Transkripsi gagal: {e}")
      return ""

  def synthesize_wav_base64(self, text: str) -> str:
    """Menyintesis suara robot via Piper dan mengembalikan string WAV Base64."""
    if not self.piper:
      return ""

    try:
      buffer = io.BytesIO()
      with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(self.piper.config.sample_rate)
        for chunk in self.piper.synthesize(text):
          wav_file.writeframes(chunk.audio_int16_bytes)

      buffer.seek(0)
      return base64.b64encode(buffer.read()).decode("utf-8")
    except Exception as e:
      print(f"[TTS Error] Sintesis gagal: {e}")
      return ""


# =====================================================================
# 2. ALGORITMA PEMAHAMAN INPUT (FUZZY SEMANTIC MATCHING)
# =====================================================================
class IntentRecognizer:

  def __init__(self, threshold=0.68):
    self.threshold = threshold
    self.intent_corpus = {
        "WAKEWORD": [
            "hello robot",
            "halo robot",
            "hey robot",
            "hello barelang",
            "hi robot",
            "robot",
        ],
        "FOLLOW_ME": [
            "follow me",
            "please follow me",
            "come with me",
            "lets go follow me",
            "start following me",
            "guide me follow me",
            "following me",
        ],
        "ARRIVED": [
            "arrived at the location",
            "we have arrived",
            "this is the location",
            "reached the car",
            "arrived at the car",
            "stop here we arrived",
            "i have arrived",
            "we are here",
            "arrived",
        ],
    }

  def match_intent(self, transcript: str, target_intent: str) -> bool:
    if not transcript or not transcript.strip():
      return False

    text_clean = transcript.lower().strip()
    candidates = self.intent_corpus.get(target_intent, [])
    best_score = 0.0

    for phrase in candidates:
      score = SequenceMatcher(None, text_clean, phrase).ratio()
      if phrase in text_clean:
        score = max(score, 0.95)
      best_score = max(best_score, score)

    print(
        f"[NLU] Input: '{transcript}' | Target: {target_intent} | Similarity: {best_score:.2f}"
    )
    return best_score >= self.threshold


# =====================================================================
# 3. MODULAR PERSON TRACKER & OBSTACLE AVOIDANCE
# =====================================================================
class ModularPersonTracker:
  """Abstraksi modular pelacak operator & penghindar rintangan."""

  def __init__(self, target_dist=1.0, max_speed=0.25):
    self.target_dist = target_dist
    self.max_speed = max_speed
    self.kp_linear = 0.35
    self.kp_angular = 1.2
    self.min_front_obstacle_m = 0.60  # Batas aman rintangan depan

  def compute_velocity(self, person_3d: tuple, front_min_scan: float) -> Twist:
    """Menghitung Twist kecepatan berdasarkan koordinat 3D orang dan jarak LiDAR."""
    cmd = Twist()
    if not person_3d:
      return cmd

    x_depth, y_lat, _ = person_3d  # x = jarak depan (m), y = deviasi lateral (m)

    # 1. Sudut arah menghadap orang (bearing)
    bearing_rad = math.atan2(y_lat, x_depth)
    cmd.angular.z = max(-0.8, min(0.8, self.kp_angular * bearing_rad))

    # 2. Kecepatan maju menuju jarak target
    dist_error = x_depth - self.target_dist
    if dist_error > 0.05:
      raw_linear = self.kp_linear * dist_error
      cmd.linear.x = max(0.0, min(self.max_speed, raw_linear))
    else:
      cmd.linear.x = 0.0

    # 3. Obstacle Avoidance: potong kecepatan maju jika ada rintangan dekat di depan
    if front_min_scan < self.min_front_obstacle_m:
      cmd.linear.x = 0.0

    return cmd


# =====================================================================
# 4. ROS 2 NODE INTERFACE
# =====================================================================
class BarelangHriNode(Node):

  def __init__(self, fsm, loop):
    super().__init__("barelang_hri_node")
    self.fsm = fsm
    self.loop = loop
    self.tracker = ModularPersonTracker(target_dist=1.0, max_speed=0.25)

    # Data sensor & status
    self.last_person_3d = None
    self.last_person_seen_time = time.time()
    self.is_operator_lost = False
    self.front_min_scan = float("inf")

    # Publishers
    self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
    self.hri_status_pub = self.create_publisher(String, "/luggage/hri_status", 10)

    # Subscriptions ke modul eksternal (ZED, LiDAR, LuggageTask)
    self.person_sub = self.create_subscription(
        String, "/zed/person_keypoints", self.on_person_keypoints, 10
    )
    self.bag_sub = self.create_subscription(
        String, "/zed/bag_detections", self.on_bag_detections, 10
    )
    self.scan_sub = self.create_subscription(
        LaserScan, "/scan", self.on_scan, rclpy.qos.qos_profile_sensor_data
    )
    self.gesture_status_sub = self.create_subscription(
        String, "/gesture_drive/status", self.on_gesture_status, 10
    )
    self.gesture_bag_sub = self.create_subscription(
        String, "/gesture_drive/target_bag", self.on_gesture_target_bag, 10
    )

    # Nav2 Action Client untuk kembali ke titik Start
    self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

    # Timer kontrol pergerakan & pemantauan operator (10 Hz)
    self.create_timer(0.1, self.motion_control_loop)
    self.get_logger().info("BarelangHriNode aktif dan terintegrasi dengan Vision & Nav2.")

  def on_person_keypoints(self, msg: String):
    raw = msg.data.strip()
    if raw == "NO_PERSON" or not raw:
      self.last_person_3d = None
      return

    # Parse koordinat 3D orang terdekat ("person:x,y,w,h|x_3d,y_3d,z_3d|...")
    try:
      first_person = raw.split(";")[0]
      parts = first_person.split("|")
      if len(parts) >= 2:
        coords = [float(v) for v in parts[1].split(",")]
        if len(coords) == 3 and not any(math.isnan(c) for c in coords):
          self.last_person_3d = (coords[0], coords[1], coords[2])
          self.last_person_seen_time = time.time()

          if self.is_operator_lost and self.fsm.state == "RECOVERY":
            self.is_operator_lost = False
            self.get_logger().info("Operator kembali terdeteksi di kamera ZED.")
            asyncio.run_coroutine_threadsafe(
                self.fsm.handle_vision_event("FOUND"), self.loop
            )
    except Exception:
      pass

  def on_bag_detections(self, msg: String):
    self.fsm.latest_bags_data = msg.data

  def on_scan(self, msg: LaserScan):
    # Hitung jarak terdekat di sektor depan ±30 derajat
    if not msg.ranges:
      return
    n = len(msg.ranges)
    sector = int(n * (30.0 / 360.0))
    front_ranges = []
    # Sektor kanan depan & kiri depan
    for r in msg.ranges[:sector] + msg.ranges[-sector:]:
      if math.isfinite(r) and r >= msg.range_min:
        front_ranges.append(r)
    self.front_min_scan = min(front_ranges) if front_ranges else float("inf")

  def on_gesture_status(self, msg: String):
    self.get_logger().info(f"[Motion Stack Status] {msg.data}")

  def on_gesture_target_bag(self, msg: String):
    self.fsm.pointed_direction = msg.data
    self.get_logger().info(f"[Locked Bag Info] {msg.data}")

  def motion_control_loop(self):
    """Loop kontrol motor saat robot berada dalam state FOLLOWING."""
    if self.fsm.state != "FOLLOWING":
      return

    # Cek apakah operator hilang lebih dari 2.0 detik
    if time.time() - self.last_person_seen_time > 2.0:
      if not self.is_operator_lost:
        self.is_operator_lost = True
        self.stop_robot()
        self.get_logger().warn("Operator hilang dari pandangan ZED! Memicu RECOVERY.")
        asyncio.run_coroutine_threadsafe(
            self.fsm.handle_vision_event("LOST"), self.loop
        )
      return

    # Jika operator terlihat, hitung dan kirim kecepatan
    if self.last_person_3d:
      cmd = self.tracker.compute_velocity(self.last_person_3d, self.front_min_scan)
      self.cmd_pub.publish(cmd)

  def stop_robot(self):
    self.cmd_pub.publish(Twist())

  def send_nav_goal_to_start(self):
    """Mengirim perintah Nav2 kembali ke Start Location (0.0, 0.0)."""
    self.get_logger().info("Mengirim action goal Nav2 menuju Start Point...")
    if not self.nav_client.wait_for_server(timeout_sec=3.0):
      self.get_logger().warn("Nav2 Action Server 'navigate_to_pose' tidak merespons.")
      return

    goal = NavigateToPose.Goal()
    goal.pose = PoseStamped()
    goal.pose.header.frame_id = "map"
    goal.pose.header.stamp = self.get_clock().now().to_msg()
    goal.pose.pose.position.x = 0.0
    goal.pose.pose.position.y = 0.0
    goal.pose.pose.orientation.w = 1.0

    self.nav_client.send_goal_async(goal)


# =====================================================================
# 5. FINITE STATE MACHINE HRI (BARELANG INARI)
# =====================================================================
class BarelangInariFSM:

  def __init__(self):
    self.state = "WAIT_WAKEWORD"
    self.nlu = IntentRecognizer(threshold=0.68)
    self.engines = SpeechEngines()
    self.ros_node = None
    self.active_ws = set()

    self.latest_bags_data = ""
    self.pointed_direction = "kanan"
    self.queue_detected = False
    self.recovery_task = None

  def set_ros_node(self, ros_node: BarelangHriNode):
    self.ros_node = ros_node

  async def broadcast_ui(self, payload: dict):
    """Mengirim pesan JSON ke seluruh client smartphone yang terhubung."""
    if not self.active_ws:
      return
    msg = json.dumps(payload)
    await asyncio.gather(
        *[ws.send(msg) for ws in self.active_ws if ws.open],
        return_exceptions=True,
    )

  async def speak_sequence(self, tts_items: list, substate: str = "-"):
    """Menyintesis suara dengan Piper dan mengirim ke smartphone client."""
    for item in tts_items:
      text = item.get("text", "")
      delay = item.get("delay_after", 0.0)

      print(f"[ROBOT SPEAKS] \"{text}\"")
      wav_base64 = self.engines.synthesize_wav_base64(text)

      payload = {
          "type": "TTS_PLAY",
          "state": self.state,
          "substate": substate,
          "text": text,
          "audio": wav_base64,
          "delay_after": delay,
      }
      await self.broadcast_ui(payload)

      # Perkiraan waktu pemutaran audio di HP + jeda
      est_duration = max(len(text.split()) * 0.36, 1.2) + delay
      await asyncio.sleep(est_duration)

  async def handle_audio_input(self, audio_bytes: bytes):
    """Menerima rekaman audio smartphone, transkripsi Whisper, lalu NLU."""
    print(f"[STT] Menerima {len(audio_bytes)} bytes audio dari HP...")
    transcript = self.engines.transcribe(audio_bytes)
    print(f"[STT Result] '{transcript}'")

    await self.broadcast_ui({
        "type": "TRANSCRIPT",
        "state": self.state,
        "transcript": transcript,
    })

    if not transcript:
      if self.state == "WAIT_FOLLOW":
        await self.speak_sequence([{
            "text": (
                "If you are ready, please guide me to the place, and say follow me."
            ),
            "delay_after": 0.0,
        }])
      return

    await self.process_transcript(transcript)

  async def process_transcript(self, text: str):
    # STATE 0: Menunggu Wake Word "hello robot"
    if self.state == "WAIT_WAKEWORD":
      if self.nlu.match_intent(text, "WAKEWORD"):
        self.state = "WAIT_FOLLOW"
        await self.speak_sequence([
            {
                "text": (
                    "Hello, I'm Barelang Inari, I want to help you to carry the luggage."
                ),
                "delay_after": 0.4,
            },
            {
                "text": (
                    "If you are ready, please guide me to the place, and say follow me."
                ),
                "delay_after": 0.0,
            },
        ])
      else:
        print("[HRI] Wakeword belum diucapkan. Robot tetap diam.")

    # STATE 1: Menunggu perintah "follow me"
    elif self.state == "WAIT_FOLLOW":
      if self.nlu.match_intent(text, "FOLLOW_ME"):
        self.state = "FOLLOWING"
        if self.ros_node:
          self.ros_node.last_person_seen_time = time.time()
          self.ros_node.is_operator_lost = False

        await self.speak_sequence(
            [{"text": "Following you now.", "delay_after": 0.0}],
            substate="TrackOperator & AvoidObstacles",
        )
      else:
        await self.speak_sequence([
            {
                "text": (
                    "If you are ready, please guide me to the place, and say follow me."
                ),
                "delay_after": 0.0,
            },
        ])

    # STATE 2: Sedang mengikuti operator
    elif self.state == "FOLLOWING":
      if self.nlu.match_intent(text, "ARRIVED"):
        if self.ros_node:
          self.ros_node.stop_robot()
        await self.run_bag_pickup_and_return()

  async def handle_vision_event(self, event_val: str):
    """Menangani event dari kamera ZED (LOST / FOUND)."""
    if self.state == "FOLLOWING" and event_val == "LOST":
      self.state = "RECOVERY"
      if self.ros_node:
        self.ros_node.stop_robot()

      # Batalkan recovery task sebelumnya jika masih berjalan
      if self.recovery_task and not self.recovery_task.done():
        self.recovery_task.cancel()
      self.recovery_task = asyncio.create_task(self._run_recovery_loop())

    elif self.state == "RECOVERY" and event_val == "FOUND":
      self.state = "FOLLOWING"
      if self.recovery_task and not self.recovery_task.done():
        self.recovery_task.cancel()

      await self.speak_sequence(
          [{"text": "I see you. Resuming following.", "delay_after": 0.0}],
          substate="TrackOperator & AvoidObstacles",
      )

  async def _run_recovery_loop(self):
    """Mengulang kalimat lost tiap 2 detik maksimal 5 kali."""
    for attempt in range(1, 6):
      if self.state != "RECOVERY":
        break

      await self.speak_sequence(
          [{
              "text": (
                  "Please stop, and stand in my front camera, because I lost you."
              ),
              "delay_after": 0.0,
          }],
          substate=f"LostOperator (Call {attempt}/5)",
      )

      if self.state != "RECOVERY":
        break

      await asyncio.sleep(2.0)

    if self.state == "RECOVERY":
      print("[HRI] Operator belum ditemukan setelah 5 kali panggilan.")

  async def run_bag_pickup_and_return(self):
    """STATE 3 & 4: Dummy Bag Pickup & Return to Start."""
    # --- State 3: Pengambilan Tas ---
    self.state = "BAG_PICKUP"
    sub_steps = [
        ("DetectPointing", "Mendeteksi arah tunjuk operator..."),
        ("IdentifyTargetBag", "Mengidentifikasi tas target..."),
        ("GraspBag", "Mengeksekusi pengambilan tas..."),
    ]
    for sub, desc in sub_steps:
      print(f"[BAG_PICKUP] {desc}")
      await self.broadcast_ui({
          "type": "STATE_UPDATE",
          "state": self.state,
          "substate": sub,
      })
      await asyncio.sleep(2.2)

    # Kalimat konfirmasi sebelum kembali ke Start
    self.state = "RETURN_START"
    await self.speak_sequence(
        [{
            "text": (
                "Thank you, I have the bag now, and I will back to starting point."
            ),
            "delay_after": 0.0,
        }],
        substate="Init Return",
    )

    # Kirim action Nav2 kembali ke Start Position
    if self.ros_node:
      self.ros_node.send_nav_goal_to_start()

    # --- State 4: Kembali ke Start (Delivery) ---
    return_steps = [
        ("NavigateToStart", "Global Planner menuju koordinat Start..."),
        ("CheckQueue", "Memeriksa deteksi antrean 3-4 orang..."),
        (
            "SocialQueueing" if self.queue_detected else "DirectNavigation",
            "Navigasi menuju titik drop...",
        ),
        ("PlaceBag", "Meletakkan tas di lantai..."),
    ]
    for step, desc in return_steps:
      print(f"[RETURN_START] {desc}")
      await self.broadcast_ui({
          "type": "STATE_UPDATE",
          "state": self.state,
          "substate": step,
      })
      await asyncio.sleep(2.5)

    # Selesai
    self.state = "COMPLETED"
    await self.speak_sequence(
        [{"text": "I've already finished the task.", "delay_after": 0.0}],
        substate="Done",
    )


# =====================================================================
# 6. WEBSOCKET & HTTP SERVER RUNNER
# =====================================================================
fsm = BarelangInariFSM()


async def ws_handler(websocket):
  fsm.active_ws.add(websocket)
  print(f"[SERVER] HP Terhubung dari {websocket.remote_address}")

  # Kirim status awal
  await websocket.send(
      json.dumps({
          "type": "STATE_UPDATE",
          "state": fsm.state,
          "substate": "-",
          "text": "-",
      })
  )

  try:
    async for message in websocket:
      if isinstance(message, bytes):
        await fsm.handle_audio_input(message)
      else:
        data = json.loads(message)
        msg_type = data.get("type", "")

        if msg_type == "AUDIO_BASE64":
          audio_bytes = base64.b64decode(data.get("audio", ""))
          await fsm.handle_audio_input(audio_bytes)
        elif msg_type == "VISION_EVENT":
          await fsm.handle_vision_event(data.get("value", ""))
        elif msg_type == "SIM_TEXT":
          await fsm.process_transcript(data.get("text", ""))
  except websockets.exceptions.ConnectionClosed:
    pass
  finally:
    fsm.active_ws.discard(websocket)
    print("[SERVER] HP Terputus.")


def start_http_server(web_dir, port=8000):
  """Menjalankan HTTP server lokal sederhana di port 8000 untuk menyajikan index.html."""

  class Handler(http.server.SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
      super().__init__(*args, directory=web_dir, **kwargs)

    def log_message(self, format, *args):
      pass  # Redam log akses rutin

  try:
    with socketserver.TCPServer(("0.0.0.0", port), Handler) as httpd:
      print(f"[HTTP] Web UI tersedia di http://0.0.0.0:{port}")
      httpd.serve_forever()
  except Exception as e:
    print(f"[HTTP Info] Port {port} tidak dapat dibuka (mungkin sudah jalan): {e}")


def ros_spin_thread(node):
  rclpy.spin(node)


async def main():
  rclpy.init()
  loop = asyncio.get_running_loop()

  # Inisialisasi ROS 2 Node
  ros_node = BarelangHriNode(fsm, loop)
  fsm.set_ros_node(ros_node)

  # Jalankan ROS 2 spin di thread terpisah
  t_ros = threading.Thread(target=ros_spin_thread, args=(ros_node,), daemon=True)
  t_ros.start()

  # Jalankan HTTP server lokal di background agar index.html langsung bisa dibuka
  current_dir = os.path.dirname(os.path.abspath(__file__))
  t_http = threading.Thread(
      target=start_http_server, args=(current_dir, 8000), daemon=True
  )
  t_http.start()

  print("========================================================================")
  print("🤖 Barelang Inari HRI Server Aktif")
  print("🌐 HTTP Web App : http://<IP_ROBOT>:8000")
  print("🔌 WebSocket    : ws://0.0.0.0:8765")
  print("🎙️ Faster-Whisper (base.en) | 🗣️ Piper TTS (en_US-lessac)")
  print("========================================================================")

  async with websockets.serve(ws_handler, "0.0.0.0", 8765, max_size=10 * 1024 * 1024):
    await asyncio.Future()


def main_entry():
  try:
    asyncio.run(main())
  except KeyboardInterrupt:
    print("\n[HRI] Mematikan sistem...")
    rclpy.shutdown()


if __name__ == "__main__":
  main_entry()