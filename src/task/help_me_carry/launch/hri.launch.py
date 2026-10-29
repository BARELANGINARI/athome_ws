from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "enable_local_mic",
            default_value="true",
            description="Aktifkan input mikrofon lokal via PyAudio VAD",
        ),
        DeclareLaunchArgument(
            "enable_ws_audio",
            default_value="true",
            description="Aktifkan input audio dari Web UI WebSocket",
        ),
        DeclareLaunchArgument(
            "mic_energy_threshold",
            default_value="500",
            description="Ambang batas energi RMS untuk deteksi suara mikrofon lokal",
        ),
        DeclareLaunchArgument(
            "mic_silence_limit",
            default_value="1.0",
            description="Durasi hening (detik) untuk mengakhiri perekaman mikrofon lokal",
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
                    "mic_energy_threshold": LaunchConfiguration("mic_energy_threshold"),
                    "mic_silence_limit": LaunchConfiguration("mic_silence_limit"),
                }
            ],
        ),
    ])
