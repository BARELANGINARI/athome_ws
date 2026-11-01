from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "http_port",
            default_value="8080",
            description="HTTP port to serve the face web interface",
        ),
        DeclareLaunchArgument(
            "ws_port",
            default_value="8766",
            description="WebSocket bridge port for ROS 2 expression streaming",
        ),
        DeclareLaunchArgument(
            "browser",
            default_value="false",
            description="Automatically launch Chromium browser on HDMI LCD",
        ),
        DeclareLaunchArgument(
            "kiosk",
            default_value="true",
            description="Run browser in borderless full-screen kiosk mode",
        ),
        Node(
            package="athome_face",
            executable="face_node",
            name="robot_face_node",
            output="screen",
            emulate_tty=True,
            parameters=[
                {
                    "http_port": LaunchConfiguration("http_port"),
                    "ws_port": LaunchConfiguration("ws_port"),
                    "launch_browser": LaunchConfiguration("browser"),
                    "kiosk_mode": LaunchConfiguration("kiosk"),
                }
            ],
        ),
    ])

