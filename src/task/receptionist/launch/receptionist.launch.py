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
        # Navigation arguments
        DeclareLaunchArgument(
            'navigation',
            default_value='true',
            description='Launch Nav2 navigation stack with static map and receptionist node.',
        ),
        DeclareLaunchArgument(
            'map',
            default_value=PathJoinSubstitution([
                FindPackageShare('athome_navigation'),
                'maps',
                'map.yaml',
            ]),
            description='Path to static map yaml file from athome_navigation.',
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
            description='Launch receptionist task manager FSM node in HRI launch.',
        ),
        DeclareLaunchArgument(
            'hri',
            default_value='false',
            description='Launch HRI stack (Faster-Whisper STT + Piper TTS + FSM Task Manager).',
        ),
        DeclareLaunchArgument(
            'person_det',
            default_value='true',
            description='Launch person detection and face recognition node.',
        ),
        DeclareLaunchArgument(
            'camera_index',
            default_value='0',
            description='Camera index for person_det.',
        ),

        # 1. Hardware Bringup (Motor + LiDAR)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('receptionist'),
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

        # 2. Navigation Bringup (Nav2 + AMCL on existing Map + Receptionist Service Node + RViz)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('receptionist'),
                    'launch',
                    'navigation.launch.py',
                ])
            ),
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('navigation'), "' == 'true'"])
            ),
            launch_arguments={
                'map': LaunchConfiguration('map'),
                'rviz': LaunchConfiguration('rviz'),
                'rviz_software': LaunchConfiguration('rviz_software'),
                'task_node': 'true',
            }.items(),
        ),

        # 3. Person Detection & Face Recognition (person_det)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('person_det'),
                    'launch',
                    'person_det.launch.py',
                ])
            ),
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('person_det'), "' == 'true'"])
            ),
            launch_arguments={
                'camera_index': LaunchConfiguration('camera_index'),
            }.items(),
        ),

        # 4. HRI & Task Manager FSM Bringup (Speech + FSM)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('receptionist'),
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
    ])
