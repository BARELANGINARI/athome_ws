from concurrent.futures import ThreadPoolExecutor

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from athome_hri.audio_io import AudioSpeechQueue, resolve_speaker_device_index
from athome_hri.config import (
    apply_param_updates,
    configure_hri_logging,
    declare_hri_parameters,
)
from athome_hri.tts_engine import TextToSpeech


class TtsEngineNode(Node):
    """Synthesize and play text received on /hri/tts_say."""

    _PARAM_NAMES = (
        "tts_model",
        "tts_use_cuda",
        "enable_local_speaker",
        "speaker_device_index",
    )
    _LEGACY_ALIASES = {"model": "tts_model"}

    def __init__(self) -> None:
        super().__init__("tts_engine")
        params = declare_hri_parameters(
            self,
            self._PARAM_NAMES,
            legacy_aliases=self._LEGACY_ALIASES,
        )

        self._tts_model = str(params["tts_model"])
        self._tts_use_cuda = bool(params["tts_use_cuda"])
        self._tts = TextToSpeech(self._tts_model, use_cuda=self._tts_use_cuda)
        self._play_locally = bool(params["enable_local_speaker"])
        self._speaker_device_index = resolve_speaker_device_index(
            params["speaker_device_index"]
        )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Tts")
        self.tts_done_pub = self.create_publisher(String, "/hri/tts_done", 10)
        self.subscription = self.create_subscription(
            String, "/hri/tts_say", self._on_text, 10
        )
        self.add_on_set_parameters_callback(self._on_params_changed)
        self.get_logger().info("Standalone TTS engine ready.")

    def _reload_tts(self) -> None:
        self._tts = TextToSpeech(self._tts_model, use_cuda=self._tts_use_cuda)

    def _on_params_changed(self, params):
        def _update_model(v):
            self._tts_model = str(v)
            self._reload_tts()

        def _update_cuda(v):
            self._tts_use_cuda = bool(v)
            self._reload_tts()

        updaters = {
            "enable_local_speaker": lambda v: setattr(
                self, "_play_locally", bool(v)
            ),
            "speaker_device_index": lambda v: setattr(
                self, "_speaker_device_index", resolve_speaker_device_index(int(v))
            ),
            "tts_model": _update_model,
            "tts_use_cuda": _update_cuda,
        }
        return apply_param_updates(params, updaters, node_logger=self.get_logger())

    def _on_text(self, message: String) -> None:
        text = message.data.strip()
        if text:
            self._executor.submit(self._speak, text)

    def _speak(self, text: str) -> None:
        wav_base64 = self._tts.synthesize_wav_base64(text)
        if self._play_locally and wav_base64:
            AudioSpeechQueue._play_wav_locally(
                wav_base64, device_index=self._speaker_device_index
            )

        done = String()
        done.data = text
        self.tts_done_pub.publish(done)
        self.get_logger().info("[TTS] %s", text)

    def destroy_node(self):
        self._executor.shutdown(wait=True)
        return super().destroy_node()


def main(args=None) -> None:
    configure_hri_logging()
    rclpy.init(args=args)
    node = TtsEngineNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
