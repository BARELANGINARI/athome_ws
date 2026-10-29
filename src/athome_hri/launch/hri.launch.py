from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch Barelang HRI speech and interaction node with VAD and ASR parameters."""
    return LaunchDescription([
        DeclareLaunchArgument(
            "enable_local_mic",
            default_value="true",
            description="Enable local microphone input via VAD (Bandpass + Spectral Flatness)",
        ),
        DeclareLaunchArgument(
            "enable_ws_audio",
            default_value="true",
            description="Enable audio input from Web UI WebSocket",
        ),
        DeclareLaunchArgument(
            "enable_visualizer",
            default_value="true",
            description="Enable real-time VAD & ASR visualization window (Matplotlib)",
        ),
        DeclareLaunchArgument(
            "threshold_dbfs",
            default_value="-40.0",
            description="Speech-Band (300-3400 Hz) energy threshold in dBFS",
        ),
        DeclareLaunchArgument(
            "threshold_flatness",
            default_value="0.40",
            description="Maximum Spectral Flatness threshold for voiced speech detection",
        ),
        DeclareLaunchArgument(
            "pre_roll_frames",
            default_value="10",
            description="Pre-roll frames count (30 ms/frame, 10 = 300 ms)",
        ),
        DeclareLaunchArgument(
            "hangover_frames",
            default_value="25",
            description="Silence hangover frames before segment cut (25 = 750 ms)",
        ),
        DeclareLaunchArgument(
            "min_speech_frames",
            default_value="15",
            description="Minimum speech frames to trigger ASR (15 = 450 ms)",
        ),
        DeclareLaunchArgument(
            "whisper_model",
            default_value="deepdml/faster-whisper-large-v3-turbo-ct2",
            description="Faster-Whisper model for STT transcription",
        ),
        Node(
            package="athome_hri",
            executable="help_me_carry_hri",
            name="barelang_hri_node",
            output="screen",
            emulate_tty=True,
            parameters=[
                {
                    "enable_local_mic": LaunchConfiguration("enable_local_mic"),
                    "enable_ws_audio": LaunchConfiguration("enable_ws_audio"),
                    "enable_visualizer": LaunchConfiguration("enable_visualizer"),
                    "threshold_dbfs": LaunchConfiguration("threshold_dbfs"),
                    "threshold_flatness": LaunchConfiguration("threshold_flatness"),
                    "pre_roll_frames": LaunchConfiguration("pre_roll_frames"),
                    "hangover_frames": LaunchConfiguration("hangover_frames"),
                    "min_speech_frames": LaunchConfiguration("min_speech_frames"),
                    "whisper_model": LaunchConfiguration("whisper_model"),
                }
            ],
        ),
    ])
