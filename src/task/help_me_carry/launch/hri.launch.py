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
            "help_me_carry_params.yaml",
        ]),
        description="Path to Help Me Carry task parameters YAML file",
    )

    enable_hri_logger_arg = DeclareLaunchArgument(
        "enable_hri_logger",
        default_value="true",
        description="Enable automatic HRI conversation logger per trial.",
    )

    hri_log_dir_arg = DeclareLaunchArgument(
        "hri_log_dir",
        default_value="~/athome_hri_logs",
        description="Directory where per-trial HRI log files are saved.",
    )

    enable_visualizer_arg = DeclareLaunchArgument(
        "enable_visualizer",
        default_value="true",
        description="Launch real-time HRI audio & parameter visualizer GUI.",
    )

    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_CONSOLE_OUTPUT_FORMAT', '[{severity}] {message}'),
        SetEnvironmentVariable('RCUTILS_COLORIZED_OUTPUT', '1'),
        SetEnvironmentVariable('ATHOME_TASK', 'help_me_carry'),
        stt_backend_arg,
        cpp_params_file_arg,
        params_file_arg,
        task_params_file_arg,
        enable_hri_logger_arg,
        hri_log_dir_arg,
        enable_visualizer_arg,
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
            parameters=[
                LaunchConfiguration("params_file"),
                {"enable_visualizer": False},
            ],
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
        Node(
            package="athome_hri",
            executable="hri_logger_node",
            name="hri_logger",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "task": "help_me_carry",
                "stt_backend": LaunchConfiguration("stt_backend"),
                "log_dir": LaunchConfiguration("hri_log_dir"),
            }],
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration("enable_hri_logger"), "' == 'true'"])
            ),
        ),
        Node(
            package="athome_hri",
            executable="hri_visualizer_node",
            name="hri_visualizer",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "task": "help_me_carry",
                "stt_backend": LaunchConfiguration("stt_backend"),
                "cpp_params_file": LaunchConfiguration("cpp_params_file"),
                "py_params_file": LaunchConfiguration("params_file"),
            }],
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration("enable_visualizer"), "' == 'true'"])
            ),
        ),
    ])

