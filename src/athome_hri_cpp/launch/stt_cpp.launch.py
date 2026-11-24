from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=PathJoinSubstitution([
            FindPackageShare("athome_hri_cpp"),
            "config",
            "params.yaml",
        ]),
        description="Path to ROS 2 parameters YAML file for athome_hri_cpp",
    )

    task_arg = DeclareLaunchArgument(
        "task",
        default_value="all",
        description="Task label for terminal logging (receptionist, help_me_carry, restaurant, all)",
    )

    return LaunchDescription([
        SetEnvironmentVariable("RCUTILS_CONSOLE_OUTPUT_FORMAT", "[{severity}] {message}"),
        SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1"),
        params_file_arg,
        task_arg,
        Node(
            package="athome_hri_cpp",
            executable="stt_engine_node",
            name="stt_engine",
            output="screen",
            emulate_tty=True,
            parameters=[
                LaunchConfiguration("params_file"),
                {"task": LaunchConfiguration("task")},
            ],
        ),
    ])
