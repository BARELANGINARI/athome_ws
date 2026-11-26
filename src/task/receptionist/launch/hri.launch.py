import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_receptionist = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    default_task_params = os.path.join(pkg_receptionist, 'config', 'receptionist_params.yaml')

    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_CONSOLE_OUTPUT_FORMAT', '[{severity}] {message}'),
        SetEnvironmentVariable('RCUTILS_COLORIZED_OUTPUT', '1'),
        SetEnvironmentVariable('ATHOME_TASK', 'receptionist'),
        DeclareLaunchArgument(
            'stt_backend',
            default_value='cpp',
            description='STT backend to use: "cpp" (athome_hri_cpp) or "python" (athome_hri).',
        ),
        DeclareLaunchArgument(
            'cpp_params_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('athome_hri_cpp'),
                'config',
                'params.yaml',
            ]),
            description='Path to ROS 2 parameters YAML file for C++ STT node.',
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('athome_hri'),
                'config',
                'params.yaml',
            ]),
            description='Path to ROS 2 parameters YAML file for HRI nodes (Whisper STT & Piper TTS).',
        ),
        DeclareLaunchArgument(
            'task_params_file',
            default_value=default_task_params,
            description='Path to receptionist task parameters file (FSM & host info).',
        ),
        DeclareLaunchArgument(
            'manager_node',
            default_value='true',
            description='Launch receptionist task manager FSM node.',
        ),
        DeclareLaunchArgument(
            'enable_hri_logger',
            default_value='true',
            description='Enable automatic HRI conversation logger per trial.',
        ),
        DeclareLaunchArgument(
            'hri_log_dir',
            default_value='~/athome_hri_logs',
            description='Directory where per-trial HRI log files are saved.',
        ),
        DeclareLaunchArgument(
            'enable_visualizer',
            default_value='true',
            description='Launch real-time HRI audio & parameter visualizer GUI.',
        ),

        # 1a. Speech-to-Text (ASR) Node - C++ (athome_hri_cpp)
        Node(
            package='athome_hri_cpp',
            executable='stt_engine_node',
            name='stt_engine',
            output='screen',
            emulate_tty=True,
            parameters=[
                LaunchConfiguration('cpp_params_file'),
                {'task': 'receptionist'},
            ],
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('stt_backend'), "' == 'cpp'"])
            ),
        ),

        # 1b. Speech-to-Text (ASR) Node - Python (Faster-Whisper)
        Node(
            package='athome_hri',
            executable='stt_engine_node',
            name='stt_engine',
            output='screen',
            emulate_tty=True,
            parameters=[
                LaunchConfiguration('params_file'),
                {'enable_visualizer': False},
            ],
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('stt_backend'), "' == 'python'"])
            ),
        ),

        # 2. Text-to-Speech (TTS) Node - Piper ONNX
        Node(
            package='athome_hri',
            executable='tts_engine_node',
            name='tts_engine',
            output='screen',
            emulate_tty=True,
            parameters=[LaunchConfiguration('params_file')],
        ),

        # 3. Receptionist Task Manager FSM Node
        Node(
            package='receptionist',
            executable='receptionist_manager',
            name='receptionist_manager',
            output='screen',
            emulate_tty=True,
            parameters=[LaunchConfiguration('task_params_file')],
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('manager_node'), "' == 'true'"])
            ),
        ),

        # 4. Per-Trial HRI Conversation Logger
        Node(
            package='athome_hri',
            executable='hri_logger_node',
            name='hri_logger',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'task': 'receptionist',
                'stt_backend': LaunchConfiguration('stt_backend'),
                'log_dir': LaunchConfiguration('hri_log_dir'),
            }],
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('enable_hri_logger'), "' == 'true'"])
            ),
        ),

        # 5. Real-Time HRI Audio & Parameter Visualizer GUI
        Node(
            package='athome_hri',
            executable='hri_visualizer_node',
            name='hri_visualizer',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'task': 'receptionist',
                'stt_backend': LaunchConfiguration('stt_backend'),
                'cpp_params_file': LaunchConfiguration('cpp_params_file'),
                'py_params_file': LaunchConfiguration('params_file'),
            }],
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('enable_visualizer'), "' == 'true'"])
            ),
        ),
    ])

