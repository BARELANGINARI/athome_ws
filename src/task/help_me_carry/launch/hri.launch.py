from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
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

    task_params_file_arg = DeclareLaunchArgument(
        "task_params_file",
        default_value=PathJoinSubstitution([
            FindPackageShare("help_me_carry"),
            "config",
            "help_me_carry_params.yaml",
        ]),
        description="Path to Help Me Carry task parameters YAML file",
    )

    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_CONSOLE_OUTPUT_FORMAT', '[{severity}] {message}'),
        SetEnvironmentVariable('RCUTILS_COLORIZED_OUTPUT', '1'),
        SetEnvironmentVariable('ATHOME_TASK', 'help_me_carry'),
        params_file_arg,
        task_params_file_arg,
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
            parameters=[LaunchConfiguration("params_file")],
        ),
        Node(
            package="help_me_carry",
            executable="help_me_carry_manager",
            name="help_me_carry_manager",
            output="screen",
            parameters=[LaunchConfiguration("task_params_file")],
        ),
    ])

