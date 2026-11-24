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
    pkg_restaurant = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    default_task_params = os.path.join(pkg_restaurant, 'config', 'restaurant_params.yaml')

    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_CONSOLE_OUTPUT_FORMAT', '[{severity}] {message}'),
        SetEnvironmentVariable('RCUTILS_COLORIZED_OUTPUT', '1'),
        SetEnvironmentVariable('ATHOME_TASK', 'restaurant'),
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
            description='Path to restaurant task parameters file.',
        ),
        DeclareLaunchArgument(
            'manager_node',
            default_value='true',
            description='Launch restaurant task manager FSM node.',
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
                {'task': 'restaurant'},
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
            parameters=[LaunchConfiguration('params_file')],
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

        # 3. Restaurant Task Manager FSM Node
        Node(
            package='restaurant',
            executable='restaurant_manager',
            name='restaurant_manager',
            output='screen',
            emulate_tty=True,
            parameters=[LaunchConfiguration('task_params_file')],
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('manager_node'), "' == 'true'"])
            ),
        ),
    ])
