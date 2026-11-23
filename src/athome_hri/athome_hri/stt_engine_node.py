import asyncio
import os
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
    VadVisualizer,
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
        "enable_visualizer",
        "enable_live_meter",
        "threshold_dbfs",
        "threshold_flatness",
        "pre_roll_frames",
        "hangover_frames",
        "min_speech_frames",
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
        "stt_initial_prompt",
        "stt_hotwords",
        "stt_hallucination_blacklist",
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
        self._enable_visualizer: bool = bool(params["enable_visualizer"])
        self._enable_live_meter: bool = bool(params["enable_live_meter"])

        self.declare_parameter("task", os.environ.get("ATHOME_TASK", "all"))
        self._task_mode = str(self.get_parameter("task").value).lower()

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
            f"\033[1;33m[STT]\033[0m Loading model '{params['whisper_model']}' ({params['stt_language']})..."
        )
        self._stt = SpeechToText(
            model_size=str(params["whisper_model"]),
            language=str(params["stt_language"]),
            fallback_model_size=str(params["stt_fallback_model"]),
            sample_rate=int(params["stt_sample_rate"]),
            beam_size=int(params["stt_beam_size"]),
            no_speech_threshold=float(params["stt_no_speech_threshold"]),
            log_prob_threshold=float(params["stt_log_prob_threshold"]),
            compression_ratio_threshold=float(params["stt_compression_ratio_threshold"]),
            vad_threshold=float(params["stt_vad_threshold"]),
            vad_min_speech_ms=int(params["stt_vad_min_speech_ms"]),
            vad_silence_ms=int(params["stt_vad_silence_ms"]),
            vad_speech_pad_ms=int(params["stt_vad_speech_pad_ms"]),
            repetition_penalty=float(params["stt_repetition_penalty"]),
            no_repeat_ngram_size=int(params["stt_no_repeat_ngram_size"]),
            temperature=float(params["stt_temperature"]),
            initial_prompt=str(params["stt_initial_prompt"]),
            hotwords=str(params["stt_hotwords"]),
            hallucination_blacklist=params["stt_hallucination_blacklist"],
        )
        resolved_index, auto_detected = resolve_mic_device_index(
            params["mic_device_index"]
        )
        if auto_detected:
            self.get_logger().info(
                f"\033[1;33m[STT]\033[0m Mic device index {resolved_index}"
            )
        elif resolved_index is None:
            self.get_logger().warn(
                "\033[1;33m[STT]\033[0m Default system mic in use"
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
        self._visualizer = VadVisualizer(self._listener)
        self._stt.on_asr_result = self._visualizer.update_asr_status
        self._vis_timer = self.create_timer(
            VadVisualizer.RENDER_INTERVAL_SEC,
            self._on_visualizer_tick,
        )

        self.add_on_set_parameters_callback(self._on_params_changed)
        if self._enable_local_mic and self._enable_visualizer:
            self._visualizer.start()
        self._listener.start()
        self.get_logger().info(f"{self.task_header}\n  STT: Ready")

    @property
    def task_header(self) -> str:
        if self._task_mode in ("receptionist", "rec"):
            return "\033[1;36m[RECEPTIONIST]\033[0m"
        elif self._task_mode in ("help_me_carry", "hmc"):
            return "\033[1;32m[HELP_ME_CARRY]\033[0m"
        elif self._task_mode in ("restaurant", "resto", "waiter"):
            return "\033[1;34m[RESTAURANT]\033[0m"
        return "\033[1;33m[HRI]\033[0m"

    def _set_visualizer_enabled(self, enabled: bool) -> None:
        self._enable_visualizer = bool(enabled)
        if self._enable_visualizer and self._enable_local_mic:
            self._visualizer.start()
        else:
            self._visualizer.stop()

    def _on_visualizer_tick(self) -> None:
        if self._enable_visualizer:
            self._visualizer.render_once()

    def _on_params_changed(self, params):
        updaters = {
            "enable_local_mic": lambda v: setattr(self, "_enable_local_mic", bool(v)),
            "enable_visualizer": self._set_visualizer_enabled,
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
            "stt_sample_rate": lambda v: setattr(self._stt, "sample_rate", int(v)),
            "stt_beam_size": lambda v: setattr(self._stt, "beam_size", int(v)),
            "stt_no_speech_threshold": lambda v: setattr(self._stt, "no_speech_threshold", float(v)),
            "stt_log_prob_threshold": lambda v: setattr(self._stt, "log_prob_threshold", float(v)),
            "stt_compression_ratio_threshold": lambda v: setattr(self._stt, "compression_ratio_threshold", float(v)),
            "stt_vad_threshold": lambda v: setattr(self._stt, "vad_threshold", float(v)),
            "stt_vad_min_speech_ms": lambda v: setattr(self._stt, "vad_min_speech_ms", int(v)),
            "stt_vad_silence_ms": lambda v: setattr(self._stt, "vad_silence_ms", int(v)),
            "stt_vad_speech_pad_ms": lambda v: setattr(self._stt, "vad_speech_pad_ms", int(v)),
            "stt_repetition_penalty": lambda v: setattr(self._stt, "repetition_penalty", float(v)),
            "stt_no_repeat_ngram_size": lambda v: setattr(self._stt, "no_repeat_ngram_size", int(v)),
            "stt_temperature": lambda v: setattr(self._stt, "temperature", float(v)),
            "stt_initial_prompt": lambda v: setattr(self._stt, "initial_prompt", str(v)),
            "stt_hotwords": lambda v: setattr(self._stt, "hotwords", str(v)),
            "stt_hallucination_blacklist": lambda v: self._stt.set_hallucination_blacklist(v),
        }
        return apply_param_updates(
            params,
            updaters,
            node_logger=self.get_logger(),
            restart_required=("mic_device_index", "whisper_model", "stt_fallback_model", "pre_roll_frames"),
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
        if self._enable_visualizer:
            self._visualizer.push_frame(m)
        if not self._enable_live_meter:
            return

        self.get_logger().info(
            format_vad_meter(
                m,
                self._listener.threshold_dbfs,
                self._listener.threshold_flatness,
                task_header=self.task_header,
            ),
            throttle_duration_sec=1.0,
        )

    async def _transcribe(self, audio_input: Union[np.ndarray, bytes]) -> None:
        transcript = await asyncio.to_thread(self._stt.transcribe, audio_input)
        if not transcript:
            return
        message = String()
        message.data = transcript
        self.transcript_pub.publish(message)
        self.get_logger().info(f"{self.task_header}\n  STT: \"{transcript}\"")

    async def _on_audio(self, audio_input: Union[np.ndarray, bytes]) -> None:
        await self._transcribe(audio_input)

    def destroy_node(self):
        self._visualizer.stop()
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
