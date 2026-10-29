from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "enable_local_mic",
            default_value="true",
            description="Aktifkan input mikrofon lokal via VAD (Bandpass + Spectral Flatness)",
        ),
        DeclareLaunchArgument(
            "enable_ws_audio",
            default_value="true",
            description="Aktifkan input audio dari Web UI WebSocket",
        ),
        DeclareLaunchArgument(
            "enable_visualizer",
            default_value="true",
            description="Aktifkan jendela grafik visualisasi VAD & ASR real-time (Matplotlib)",
        ),
        DeclareLaunchArgument(
            "threshold_dbfs",
            default_value="-40.0",
            description="Ambang batas energi Speech-Band (300-3400 Hz) dalam dBFS",
        ),
        DeclareLaunchArgument(
            "threshold_flatness",
            default_value="0.40",
            description="Batas maksimum Spectral Flatness untuk deteksi suara vokal",
        ),
        DeclareLaunchArgument(
            "pre_roll_frames",
            default_value="10",
            description="Jumlah frame pre-roll (30 ms/frame, 10 = 300 ms)",
        ),
        DeclareLaunchArgument(
            "hangover_frames",
            default_value="25",
            description="Jumlah frame hangover hening sebelum kalimat dipotong (25 = 750 ms)",
        ),
        DeclareLaunchArgument(
            "min_speech_frames",
            default_value="15",
            description="Minimal frame bicara agar diproses ASR (15 = 450 ms)",
        ),
        DeclareLaunchArgument(
            "whisper_model",
            default_value="deepdml/faster-whisper-large-v3-turbo-ct2",
            description="Model Faster-Whisper untuk transkripsi STT",
        ),
        Node(
            package="help_me_carry",
            executable="help_me_carry_hri.py",
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
