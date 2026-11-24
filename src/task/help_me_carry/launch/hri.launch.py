from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    stt_backend_arg = DeclareLaunchArgument(
        "stt_backend",
        default_value="cpp",
        description='STT backend to use: "cpp" (athome_hri_cpp) or "python" (athome_hri).',
    )

    cpp_params_file_arg = DeclareLaunchArgument(
        "cpp_params_file",
        default_value=PathJoinSubstitution([
            FindPackageShare("athome_hri_cpp"),
            "config",
            "params.yaml",
        ]),
        description="Path to ROS 2 parameters YAML file for C++ STT node",
    )

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
            "params.yaml",
        ]),
        description="Path to Help Me Carry task parameters YAML file",
    )

    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_CONSOLE_OUTPUT_FORMAT', '[{severity}] {message}'),
        SetEnvironmentVariable('RCUTILS_COLORIZED_OUTPUT', '1'),
        SetEnvironmentVariable('ATHOME_TASK', 'help_me_carry'),
        stt_backend_arg,
        cpp_params_file_arg,
        params_file_arg,
        task_params_file_arg,
        Node(
            package="athome_hri_cpp",
            executable="stt_engine_node",
            name="stt_engine",
            output="screen",
            emulate_tty=True,
            parameters=[
                LaunchConfiguration("cpp_params_file"),
                {"task": "help_me_carry"},
            ],
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration("stt_backend"), "' == 'cpp'"])
            ),
        ),
        Node(
            package="athome_hri",
            executable="stt_engine_node",
            name="stt_engine",
            output="screen",
            emulate_tty=True,
            parameters=[LaunchConfiguration("params_file")],
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration("stt_backend"), "' == 'python'"])
            ),
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

