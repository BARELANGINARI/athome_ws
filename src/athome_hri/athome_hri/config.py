"""Single-source-of-truth parameter defaults and ROS 2 YAML loader for athome_hri."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

logger = logging.getLogger("athome_hri.config")

DEFAULT_HRI_PARAMS: Dict[str, Any] = {
    "mic_device_index": -1,
    "speaker_device_index": -1,
    "enable_local_mic": True,
    "enable_visualizer": True,
    "enable_live_meter": True,
    "enable_local_speaker": True,
    "threshold_dbfs": -30.0,
    "threshold_flatness": 0.40,
    "adaptive_vad_enabled": True,
    "adaptive_snr_margin_db": 10.0,
    "pre_roll_frames": 10,
    "hangover_frames": 16,
    "min_speech_frames": 9,
    "max_speech_frames": 250,
    "whisper_model": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "stt_fallback_model": "base.en",
    "stt_language": "en",
    "stt_sample_rate": 16000,
    "stt_beam_size": 2,
    "stt_no_speech_threshold": 0.5,
    "stt_log_prob_threshold": -0.8,
    "stt_compression_ratio_threshold": 1.8,
    "stt_vad_threshold": 0.65,
    "stt_vad_min_speech_ms": 250,
    "stt_vad_silence_ms": 350,
    "stt_vad_speech_pad_ms": 150,
    "stt_repetition_penalty": 1.15,
    "stt_no_repeat_ngram_size": 3,
    "stt_temperature": 0.0,
    "stt_denoiser_backend": "auto",
    "stt_highpass_hz": 120.0,
    "stt_dedup_max_ngram": 6,
    "stt_initial_prompt": "",
    "stt_hotwords": "hello robot halo robot follow me arrived we are here stop berhenti",
    "stt_hallucination_blacklist": [
        "terima kasih",
        "terimakasih",
        "thank you",
        "thanks",
        "thank you very much",
        "you",
    ],
    "tts_model": "en_US-lessac-medium.onnx",
    "tts_use_cuda": True,
    "tts_post_mute_sec": 0.6,
}


def find_params_yaml_path() -> Optional[Path]:
    """Locate the active athome_hri params.yaml file if present."""
    candidates = []
    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(
            Path(get_package_share_directory("athome_hri")) / "config" / "params.yaml"
        )
    except Exception:
        pass

    module_dir = Path(__file__).resolve().parent
    candidates.extend(
        [
            module_dir.parent / "config" / "params.yaml",
            Path.cwd() / "src" / "athome_hri" / "config" / "params.yaml",
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def coerce_param_value(raw_value: Any, template: Any) -> Any:
    """Cast raw YAML/CLI value to match the expected parameter type."""
    if raw_value is None:
        return template
    if isinstance(template, bool):
        if isinstance(raw_value, str):
            return raw_value.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw_value)
    if isinstance(template, int):
        return int(raw_value)
    if isinstance(template, float):
        return float(raw_value)
    if isinstance(template, str):
        return str(raw_value)
    if isinstance(template, list):
        if isinstance(raw_value, str):
            return [item.strip() for item in raw_value.split(",") if item.strip()]
        if isinstance(raw_value, (list, tuple, set)):
            return [str(item).strip() for item in raw_value if str(item).strip()]
    return raw_value


def _extract_ros_params(yaml_data: Dict[str, Any], key: str) -> Dict[str, Any]:
    block = yaml_data.get(key)
    if isinstance(block, dict):
        params = block.get("ros__parameters")
        if isinstance(params, dict):
            return params
    return {}


def _apply_yaml_section(
    target: Dict[str, Any],
    section: Dict[str, Any],
    node_name: Optional[str],
) -> None:
    """Merge canonical keys and legacy YAML aliases from one ros__parameters dict."""
    model_target = "tts_model" if node_name == "tts_engine" else "whisper_model"
    legacy_map = {
        "device_index": "mic_device_index",
        "language": "stt_language",
        "model": model_target,
    }
    for legacy_key, canonical_key in legacy_map.items():
        if canonical_key not in section and legacy_key in section:
            target[canonical_key] = coerce_param_value(
                section[legacy_key], DEFAULT_HRI_PARAMS[canonical_key]
            )
    for key, raw_val in section.items():
        if key in DEFAULT_HRI_PARAMS:
            target[key] = coerce_param_value(raw_val, DEFAULT_HRI_PARAMS[key])


def load_default_hri_params(
    node_name: Optional[str] = None,
    params_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load default HRI parameters from params.yaml merged onto code defaults."""
    resolved = dict(DEFAULT_HRI_PARAMS)
    yaml_file = params_path if params_path is not None else find_params_yaml_path()
    if yaml_file is None or not yaml_file.is_file():
        return resolved

    try:
        import yaml

        with yaml_file.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except Exception as exc:
        logger.warning("Gagal membaca default YAML %s: %s", yaml_file, exc)
        return resolved

    if not isinstance(data, dict):
        return resolved

    _apply_yaml_section(resolved, _extract_ros_params(data, "/**"), node_name)
    if node_name:
        clean_name = node_name.lstrip("/")
        _apply_yaml_section(resolved, _extract_ros_params(data, clean_name), clean_name)
        _apply_yaml_section(
            resolved, _extract_ros_params(data, f"/{clean_name}"), clean_name
        )

    return resolved


def declare_hri_parameters(
    node: Any,
    param_names: Iterable[str],
    legacy_aliases: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Declare unified HRI parameters on a ROS 2 node using YAML-backed defaults."""
    defaults = load_default_hri_params(node.get_name())
    overrides = getattr(node, "_parameter_overrides", {}) or {}

    alias_lookup: Dict[str, list[str]] = {}
    if legacy_aliases:
        for old_name, new_name in legacy_aliases.items():
            alias_lookup.setdefault(new_name, []).append(old_name)

    values: Dict[str, Any] = {}
    for name in param_names:
        default_val = defaults[name]
        if name not in overrides:
            for old_name in alias_lookup.get(name, ()):
                if old_name in overrides:
                    default_val = coerce_param_value(
                        overrides[old_name].value, DEFAULT_HRI_PARAMS[name]
                    )
                    break

        if not node.has_parameter(name):
            node.declare_parameter(name, default_val)
        values[name] = coerce_param_value(
            node.get_parameter(name).value, DEFAULT_HRI_PARAMS[name]
        )

    return values


def apply_param_updates(
    params: Iterable[Any],
    updaters: Dict[str, Callable[[Any], None]],
    node_logger: Any = None,
    restart_required: Iterable[str] = (),
) -> Any:
    """Apply runtime ROS 2 parameter changes and return SetParametersResult."""
    from rcl_interfaces.msg import SetParametersResult

    restart_set = set(restart_required)
    for param in params:
        handler = updaters.get(param.name)
        if handler is not None:
            try:
                handler(param.value)
            except Exception as exc:
                return SetParametersResult(
                    successful=False,
                    reason=f"Failed updating '{param.name}': {exc}",
                )
        elif param.name in restart_set and node_logger is not None:
            node_logger.info(
                f"Parameter '{param.name}' diubah ke {param.value!r}; "
                "restart node untuk menerapkan ulang perangkat/model/port."
            )
    return SetParametersResult(successful=True)


def configure_hri_logging(level: int = logging.INFO) -> None:
    """Configure root Python logging so athome_hri submodules emit console logs."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%H:%M:%S",
    )

