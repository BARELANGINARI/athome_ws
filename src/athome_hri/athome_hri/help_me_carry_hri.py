import asyncio
from enum import Enum, auto
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union

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
    VadFrameMetrics,
    VadVisualizer,
    format_vad_meter,
    resolve_mic_device_index,
    resolve_speaker_device_index,
)
from athome_hri.config import (
    apply_param_updates,
    configure_hri_logging,
    declare_hri_parameters,
)
from athome_hri.intent_recognizer import IntentRecognizer
from athome_hri.stt_engine import SpeechToText
from athome_hri.tts_engine import TextToSpeech
from athome_hri.web_server import WebServer

logger = logging.getLogger("athome_hri.coordinator")

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
    TOPIC_CMD_VEL = "cmd_vel"
    TOPIC_TTS_SAY = "/hri/tts_say"
    TOPIC_PERSON_LABEL = "/vision/person_label"
    TOPIC_GESTURE = "/vision/gesture"

    SERVICE_MAP: Dict[str, str] = {
        "follow_start": "/follow/person",
        "follow_stop": "/follow/stop",
        "align_bag": "/task/align_bag",
        "grab_bag": "/arm/ambil_tas",
        "homing": "/homing",
    }

    VALID_POINTING_GESTURES: Tuple[str, str] = ("LEFT", "RIGHT")
    OPERATOR_GRACE_PERIOD_SEC: float = 3.0

    _PARAM_NAMES = (
        "mic_device_index",
        "speaker_device_index",
        "enable_local_mic",
        "enable_ws_audio",
        "enable_visualizer",
        "enable_live_meter",
        "enable_local_speaker",
        "threshold_dbfs",
        "threshold_flatness",
        "adaptive_vad_enabled",
        "adaptive_snr_margin_db",
        "pre_roll_frames",
        "hangover_frames",
        "min_speech_frames",
        "max_speech_frames",
        "whisper_model",
        "stt_fallback_model",
        "stt_language",
        "stt_sample_rate",
        "stt_beam_size",
        "stt_no_speech_threshold",
        "stt_log_prob_threshold",
        "stt_compression_ratio_threshold",
        "stt_vad_threshold",
        "stt_vad_min_speech_ms",
        "stt_vad_silence_ms",
        "stt_vad_speech_pad_ms",
        "stt_repetition_penalty",
        "stt_no_repeat_ngram_size",
        "stt_temperature",
        "stt_denoiser_backend",
        "stt_highpass_hz",
        "stt_dedup_max_ngram",
        "stt_initial_prompt",
        "stt_hotwords",
        "stt_hallucination_blacklist",
        "tts_model",
        "nlu_threshold",
        "http_port",
        "ws_port",
    )
    _LEGACY_ALIASES = {
        "device_index": "mic_device_index",
        "language": "stt_language",
    }

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
        self._runtime_updaters: Dict[str, Callable[[Any], None]] = {}

        params = declare_hri_parameters(
            self,
            self._PARAM_NAMES,
            legacy_aliases=self._LEGACY_ALIASES,
        )

        self.enable_local_mic: bool = bool(params["enable_local_mic"])
        self.enable_ws_audio: bool = bool(params["enable_ws_audio"])
        self.enable_visualizer: bool = bool(params["enable_visualizer"])
        self.enable_live_meter: bool = bool(params["enable_live_meter"])
        self.enable_local_speaker: bool = bool(params["enable_local_speaker"])

        resolved_mic, auto_detected = resolve_mic_device_index(
            params["mic_device_index"]
        )
        self.mic_device_index: Optional[int] = resolved_mic
        if auto_detected:
            self.get_logger().info(
                f"Otomatis mendeteksi USB Mic pada index {self.mic_device_index}"
            )
        elif self.mic_device_index is None:
            self.get_logger().warn(
                "USB Mic tidak terdeteksi; menggunakan perangkat input audio default sistem."
            )

        self.speaker_device_index: Optional[int] = resolve_speaker_device_index(
            params["speaker_device_index"]
        )
        self.threshold_dbfs: float = float(params["threshold_dbfs"])
        self.threshold_flatness: float = float(params["threshold_flatness"])
        self.adaptive_vad_enabled: bool = bool(params["adaptive_vad_enabled"])
        self.adaptive_snr_margin_db: float = float(params["adaptive_snr_margin_db"])
        self.pre_roll_frames: int = int(params["pre_roll_frames"])
        self.hangover_frames: int = int(params["hangover_frames"])
        self.min_speech_frames: int = int(params["min_speech_frames"])
        self.max_speech_frames: int = int(params["max_speech_frames"])
        self.whisper_model: str = str(params["whisper_model"])
        self.stt_fallback_model: str = str(params["stt_fallback_model"])
        self.stt_language: str = str(params["stt_language"])
        self.stt_sample_rate: int = int(params["stt_sample_rate"])
        self.stt_beam_size: int = int(params["stt_beam_size"])
        self.stt_no_speech_threshold: float = float(params["stt_no_speech_threshold"])
        self.stt_log_prob_threshold: float = float(params["stt_log_prob_threshold"])
        self.stt_compression_ratio_threshold: float = float(params["stt_compression_ratio_threshold"])
        self.stt_vad_threshold: float = float(params["stt_vad_threshold"])
        self.stt_vad_min_speech_ms: int = int(params["stt_vad_min_speech_ms"])
        self.stt_vad_silence_ms: int = int(params["stt_vad_silence_ms"])
        self.stt_vad_speech_pad_ms: int = int(params["stt_vad_speech_pad_ms"])
        self.stt_repetition_penalty: float = float(params["stt_repetition_penalty"])
        self.stt_no_repeat_ngram_size: int = int(params["stt_no_repeat_ngram_size"])
        self.stt_temperature: float = float(params["stt_temperature"])
        self.stt_denoiser_backend: str = str(params["stt_denoiser_backend"])
        self.stt_highpass_hz: float = float(params["stt_highpass_hz"])
        self.stt_dedup_max_ngram: int = int(params["stt_dedup_max_ngram"])
        self.stt_initial_prompt: str = str(params["stt_initial_prompt"])
        self.stt_hotwords: str = str(params["stt_hotwords"])
        self.stt_hallucination_blacklist = list(params["stt_hallucination_blacklist"])
        self.tts_model: str = str(params["tts_model"])
        self.nlu_threshold: float = float(params["nlu_threshold"])
        self.http_port: int = int(params["http_port"])
        self.ws_port: int = int(params["ws_port"])

        self.add_on_set_parameters_callback(self._on_params_changed)

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
        self.get_logger().info("\033[1;36m[HRI]\033[0m Ready")

    def register_runtime_updaters(
        self, updaters: Dict[str, Callable[[Any], None]]
    ) -> None:
        """Daftarkan callback pembaruan runtime untuk komponen audio/HRI."""
        self._runtime_updaters = dict(updaters)

    def _on_params_changed(self, params):
        attr_casts: Dict[str, Callable[[Any], Any]] = {
            "enable_local_mic": bool,
            "enable_ws_audio": bool,
            "enable_live_meter": bool,
            "enable_local_speaker": bool,
            "speaker_device_index": lambda v: resolve_speaker_device_index(int(v)),
            "threshold_dbfs": float,
            "threshold_flatness": float,
            "hangover_frames": int,
            "min_speech_frames": int,
            "stt_language": str,
            "nlu_threshold": float,
        }
        updaters: Dict[str, Callable[[Any], None]] = {}
        for attr, cast in attr_casts.items():

            def _make_handler(a: str = attr, c: Callable[[Any], Any] = cast):
                def _apply(raw_val: Any) -> None:
                    val = c(raw_val)
                    setattr(self, a, val)
                    hook = self._runtime_updaters.get(a)
                    if hook is not None:
                        hook(val)

                return _apply

            updaters[attr] = _make_handler()

        return apply_param_updates(
            params,
            updaters,
            node_logger=self.get_logger(),
            restart_required=(
                "mic_device_index",
                "enable_visualizer",
                "pre_roll_frames",
                "whisper_model",
                "tts_model",
                "http_port",
                "ws_port",
            ),
        )

    def stop_robot(self) -> None:
        """Mengirim perintah kecepatan nol (Twist kosong) untuk menghentikan base robot."""
        self.cmd_vel_pub.publish(Twist())

    def publish_event(self, event_name: str) -> None:
        """Memublikasikan event status HRI ke topik /hri/event."""
        msg = String()
        msg.data = event_name
        self.event_pub.publish(msg)
        self.get_logger().info(f"\033[1;36m[HRI]\033[0m Event: {event_name}")

    def _on_tts_say(self, msg: String) -> None:
        text = msg.data.strip()
        if text and self.on_tts_command:
            self.loop.call_soon_threadsafe(self.on_tts_command, text)

    @property
    def is_operator_in_front(self) -> bool:
        """
        Mengembalikan True jika operator terlihat dalam OPERATOR_GRACE_PERIOD_SEC terakhir,
        atau jika topik kamera /vision/person_label belum aktif (mode pengujian HRI standalone).
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
            self.get_logger().info(f"\033[1;33m[GESTURE]\033[0m {gesture}")
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
            self.get_logger().warn(f"\033[1;33m[HRI]\033[0m Service '{service_key}' not ready!")
            return False, "Service not ready"

        self.get_logger().info(f"\033[1;36m[HRI]\033[0m Calling: {service_key}")
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
            device_index=self.ros_node.mic_device_index,
            threshold_dbfs=self.ros_node.threshold_dbfs,
            threshold_flatness=self.ros_node.threshold_flatness,
            pre_roll_frames=self.ros_node.pre_roll_frames,
            hangover_frames=self.ros_node.hangover_frames,
            min_speech_frames=self.ros_node.min_speech_frames,
            max_speech_frames=self.ros_node.max_speech_frames,
            adaptive_vad_enabled=self.ros_node.adaptive_vad_enabled,
            adaptive_snr_margin_db=self.ros_node.adaptive_snr_margin_db,
        )

        self.visualizer = VadVisualizer(self.mic_listener)
        self.mic_listener.on_frame_metrics = self._on_frame_metrics

        self.stt = SpeechToText(
            model_size=self.ros_node.whisper_model,
            language=self.ros_node.stt_language,
            on_asr_result=self.visualizer.update_asr_status,
            fallback_model_size=self.ros_node.stt_fallback_model,
            sample_rate=self.ros_node.stt_sample_rate,
            beam_size=self.ros_node.stt_beam_size,
            no_speech_threshold=self.ros_node.stt_no_speech_threshold,
            log_prob_threshold=self.ros_node.stt_log_prob_threshold,
            compression_ratio_threshold=self.ros_node.stt_compression_ratio_threshold,
            vad_threshold=self.ros_node.stt_vad_threshold,
            vad_min_speech_ms=self.ros_node.stt_vad_min_speech_ms,
            vad_silence_ms=self.ros_node.stt_vad_silence_ms,
            vad_speech_pad_ms=self.ros_node.stt_vad_speech_pad_ms,
            repetition_penalty=self.ros_node.stt_repetition_penalty,
            no_repeat_ngram_size=self.ros_node.stt_no_repeat_ngram_size,
            temperature=self.ros_node.stt_temperature,
            denoiser_backend=self.ros_node.stt_denoiser_backend,
            highpass_hz=self.ros_node.stt_highpass_hz,
            dedup_max_ngram=self.ros_node.stt_dedup_max_ngram,
            initial_prompt=self.ros_node.stt_initial_prompt,
            hotwords=self.ros_node.stt_hotwords,
            hallucination_blacklist=self.ros_node.stt_hallucination_blacklist,
        )
        self.tts = TextToSpeech(model_filename=self.ros_node.tts_model)
        self.nlu = IntentRecognizer(threshold=self.ros_node.nlu_threshold)
        self.web = WebServer(
            http_port=self.ros_node.http_port,
            ws_port=self.ros_node.ws_port,
            on_audio_received=self.handle_ws_audio,
            on_text_received=self.handle_text,
        )
        self.audio_queue = AudioSpeechQueue(
            self.tts,
            self.web.broadcast_ui,
            enable_local_speaker=self.ros_node.enable_local_speaker,
            speaker_device_index=self.ros_node.speaker_device_index,
        )
        self.audio_queue.start(loop)

        self.ros_node.register_runtime_updaters(
            {
                "enable_local_mic": lambda v: setattr(self, "enable_local_mic", v),
                "enable_ws_audio": lambda v: setattr(self, "enable_ws_audio", v),
                "enable_local_speaker": lambda v: setattr(
                    self.audio_queue, "enable_local_speaker", v
                ),
                "speaker_device_index": lambda v: setattr(
                    self.audio_queue, "speaker_device_index", v
                ),
                "threshold_dbfs": lambda v: self.mic_listener.update_vad_params(
                    threshold_dbfs=v
                ),
                "threshold_flatness": lambda v: self.mic_listener.update_vad_params(
                    threshold_flatness=v
                ),
                "hangover_frames": lambda v: self.mic_listener.update_vad_params(
                    hangover_frames=v
                ),
                "min_speech_frames": lambda v: self.mic_listener.update_vad_params(
                    min_speech_frames=v
                ),
                "stt_language": lambda v: setattr(self.stt, "language", str(v)),
                "stt_beam_size": lambda v: setattr(self.stt, "beam_size", int(v)),
                "stt_no_speech_threshold": lambda v: setattr(self.stt, "no_speech_threshold", float(v)),
                "stt_log_prob_threshold": lambda v: setattr(self.stt, "log_prob_threshold", float(v)),
                "stt_compression_ratio_threshold": lambda v: setattr(self.stt, "compression_ratio_threshold", float(v)),
                "stt_vad_silence_ms": lambda v: setattr(self.stt, "vad_silence_ms", int(v)),
                "stt_vad_speech_pad_ms": lambda v: setattr(self.stt, "vad_speech_pad_ms", int(v)),
                "stt_initial_prompt": lambda v: setattr(self.stt, "initial_prompt", str(v)),
                "stt_hotwords": lambda v: setattr(self.stt, "hotwords", str(v)),
                "stt_hallucination_blacklist": lambda v: self.stt.set_hallucination_blacklist(v),
                "nlu_threshold": lambda v: setattr(self.nlu, "threshold", v),
            }
        )

        if self.enable_local_mic:
            if self.enable_visualizer:
                self.visualizer.start(loop)
            self.mic_listener.start()

    def _on_frame_metrics(self, metrics: VadFrameMetrics) -> None:
        if self.enable_visualizer:
            self.visualizer.push_frame(metrics)
        if self.ros_node.enable_live_meter:
            self.ros_node.get_logger().info(
                format_vad_meter(
                    metrics,
                    self.mic_listener.threshold_dbfs,
                    self.mic_listener.threshold_flatness,
                    task_header="\033[1;32m[HELP_ME_CARRY]\033[0m",
                ),
                throttle_duration_sec=1.0,
            )

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
            await self.process_intent(transcript)

    async def handle_text(self, text: str) -> None:
        await self.web.broadcast_ui({"type": "TRANSCRIPT", "transcript": text})
        await self.process_intent(text)

    async def process_intent(self, text: str) -> None:
        """Mengevaluasi teks transkripsi terhadap NLU dan memicu transisi state yang sesuai."""
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
        self.ros_node.get_logger().info("\033[1;32m[HRI]\033[0m Task completed, standby")

    async def _abort_sequence(self, reason: str) -> None:
        self.ros_node.get_logger().error(f"\033[1;31m[HRI]\033[0m Aborted: {reason}")
        await self.set_state(RobotState.IDLE, f"Error: {reason}")
        await self.audio_queue.speak_and_wait(
            "Sorry, I encountered a problem and cannot continue the task."
        )


# =====================================================================
# 4. ENTRY POINT
# =====================================================================
async def main() -> None:
    configure_hri_logging()
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
            coordinator.ros_node.ws_port,
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
