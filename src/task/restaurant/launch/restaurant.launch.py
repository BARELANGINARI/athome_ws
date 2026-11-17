from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        # Hardware arguments
        DeclareLaunchArgument(
            'hardware',
            default_value='true',
            description='Launch hardware (Nucleo motor driver + RPLiDAR).',
        ),
        DeclareLaunchArgument(
            'stm_ip',
            default_value='192.168.0.4',
            description='STM32 IP address.',
        ),
        DeclareLaunchArgument(
            'navigation',
            default_value='true',
            description='Launch Nav2 navigation stack (SLAM / mapless) and restaurant service node.',
        ),
        DeclareLaunchArgument(
            'slam',
            default_value='true',
            description='Enable real-time SLAM (slam_toolbox) in Nav2 for restaurant task.',
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Launch RViz visualization.',
        ),
        DeclareLaunchArgument(
            'rviz_software',
            default_value='false',
            description='Use software rendering for RViz.',
        ),
        DeclareLaunchArgument(
            'manager',
            default_value='true',
            description='Launch restaurant task manager FSM node in HRI launch.',
        ),
        DeclareLaunchArgument(
            'hri',
            default_value='false',
            description='Launch HRI stack (Faster-Whisper STT + Piper TTS + FSM Task Manager).',
        ),
        DeclareLaunchArgument(
            'camera',
            default_value='false',
            description='Launch ZED camera vision stack for restaurant.',
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('restaurant'),
                    'launch',
                    'hardware.launch.py',
                ])
            ),
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('hardware'), "' == 'true'"])
            ),
            launch_arguments={
                'stm_ip': LaunchConfiguration('stm_ip'),
            }.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('restaurant'),
                    'launch',
                    'motion.launch.py',
                ])
            ),
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('navigation'), "' == 'true'"])
            ),
            launch_arguments={
                'slam': LaunchConfiguration('slam'),
                'rviz': LaunchConfiguration('rviz'),
                'rviz_software': LaunchConfiguration('rviz_software'),
                'task_node': 'true',
            }.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('restaurant'),
                    'launch',
                    'hri.launch.py',
                ])
            ),
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('hri'), "' == 'true'"])
            ),
            launch_arguments={
                'manager_node': LaunchConfiguration('manager'),
            }.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('restaurant'),
                    'launch',
                    'camera.launch.py',
                ])
            ),
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('camera'), "' == 'true'"])
            ),
        ),
    ])
