from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=PathJoinSubstitution([
            FindPackageShare("athome_hri"),
            "config",
            "params.yaml",
        ]),
        description="Path to ROS 2 parameters YAML file for HRI nodes",
    )

    return LaunchDescription([
        params_file_arg,
        Node(
            package="athome_hri",
            executable="stt_engine_node",
            name="stt_engine",
            output="screen",
            emulate_tty=True,
            parameters=[LaunchConfiguration("params_file")],
        ),
        Node(
            package="athome_hri",
            executable="tts_engine_node",
            name="tts_engine",
            output="screen",
            emulate_tty=True,
        ),
        Node(
            package="help_me_carry",
            executable="help_me_carry_manager",
            name="help_me_carry_manager",
            output="screen",
        ),
    ])

