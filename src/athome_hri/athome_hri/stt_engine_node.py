import asyncio
import threading
import time
from typing import Union

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from athome_hri.audio_io import (
    LocalMicListener,
    VadFrameMetrics,
    format_vad_meter,
    resolve_mic_device_index,
)
from athome_hri.config import (
    apply_param_updates,
    configure_hri_logging,
    declare_hri_parameters,
)
from athome_hri.stt_engine import SpeechToText


class SttEngineNode(Node):
    """Capture local microphone audio and publish transcript text."""

    TTS_SPEAKING_TIMEOUT_SEC: float = 20.0
    _PARAM_NAMES = (
        "mic_device_index",
        "enable_local_mic",
        "enable_live_meter",
        "threshold_dbfs",
        "threshold_flatness",
        "pre_roll_frames",
        "hangover_frames",
        "min_speech_frames",
        "whisper_model",
        "stt_language",
    )
    _LEGACY_ALIASES = {
        "device_index": "mic_device_index",
        "model": "whisper_model",
        "language": "stt_language",
    }

    def __init__(self) -> None:
        super().__init__("stt_engine")
        params = declare_hri_parameters(
            self,
            self._PARAM_NAMES,
            legacy_aliases=self._LEGACY_ALIASES,
        )

        self._enable_local_mic: bool = bool(params["enable_local_mic"])
        self._enable_live_meter: bool = bool(params["enable_live_meter"])

        self.transcript_pub = self.create_publisher(
            String, "/hri/stt_transcript", 10
        )

        self._tts_speaking = False
        self._tts_speaking_until = 0.0
        self.tts_say_sub = self.create_subscription(
            String, "/hri/tts_say", self._on_tts_say, 10
        )
        self.tts_done_sub = self.create_subscription(
            String, "/hri/tts_done", self._on_tts_done, 10
        )

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            name="SttAsyncLoop",
            daemon=True,
        )
        self._loop_thread.start()

        self.get_logger().info(
            f"Memuat model Whisper '{params['whisper_model']}' (language={params['stt_language']})..."
        )
        self._stt = SpeechToText(
            model_size=str(params["whisper_model"]),
            language=str(params["stt_language"]),
        )
        resolved_index, auto_detected = resolve_mic_device_index(
            params["mic_device_index"]
        )
        if auto_detected:
            self.get_logger().info(
                f"Otomatis mendeteksi USB Mic pada index {resolved_index}"
            )
        elif resolved_index is None:
            self.get_logger().warn(
                "USB Mic tidak terdeteksi; menggunakan perangkat input audio default sistem."
            )

        self._listener = LocalMicListener(
            loop=self._loop,
            on_audio_captured=self._on_audio,
            is_paused_fn=self._is_mic_paused,
            on_frame_metrics=self._on_metrics,
            device_index=resolved_index,
            threshold_dbfs=float(params["threshold_dbfs"]),
            threshold_flatness=float(params["threshold_flatness"]),
            pre_roll_frames=int(params["pre_roll_frames"]),
            hangover_frames=int(params["hangover_frames"]),
            min_speech_frames=int(params["min_speech_frames"]),
        )
        self.add_on_set_parameters_callback(self._on_params_changed)
        self._listener.start()
        self.get_logger().info("Standalone STT engine ready.")

    def _on_params_changed(self, params):
        updaters = {
            "enable_local_mic": lambda v: setattr(self, "_enable_local_mic", bool(v)),
            "enable_live_meter": lambda v: setattr(self, "_enable_live_meter", bool(v)),
            "threshold_dbfs": lambda v: self._listener.update_vad_params(
                threshold_dbfs=float(v)
            ),
            "threshold_flatness": lambda v: self._listener.update_vad_params(
                threshold_flatness=float(v)
            ),
            "hangover_frames": lambda v: self._listener.update_vad_params(
                hangover_frames=int(v)
            ),
            "min_speech_frames": lambda v: self._listener.update_vad_params(
                min_speech_frames=int(v)
            ),
            "stt_language": lambda v: setattr(self._stt, "language", str(v)),
        }
        return apply_param_updates(
            params,
            updaters,
            node_logger=self.get_logger(),
            restart_required=("mic_device_index", "whisper_model", "pre_roll_frames"),
        )

    def _is_mic_paused(self) -> bool:
        if self._tts_speaking and time.monotonic() >= self._tts_speaking_until:
            self.get_logger().warn(
                "Watchdog /hri/tts_done timeout; membuka kembali input mikrofon."
            )
            self._tts_speaking = False
        return (not self._enable_local_mic) or self._tts_speaking

    def _on_tts_say(self, msg: String) -> None:
        if msg.data.strip():
            self._tts_speaking = True
            self._tts_speaking_until = time.monotonic() + self.TTS_SPEAKING_TIMEOUT_SEC

    def _on_tts_done(self, _msg: String) -> None:
        self._tts_speaking = False
        self._tts_speaking_until = 0.0

    def _on_metrics(self, m: VadFrameMetrics) -> None:
        if not self._enable_live_meter:
            return

        self.get_logger().info(
            format_vad_meter(
                m,
                self._listener.threshold_dbfs,
                self._listener.threshold_flatness,
            ),
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
    configure_hri_logging()
    rclpy.init(args=args)
    node = SttEngineNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
