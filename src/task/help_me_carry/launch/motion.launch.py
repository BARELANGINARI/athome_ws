import os

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
    pkg_carry = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    default_rviz = os.path.join(pkg_carry, 'rviz', 'help_me_carry.rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'slam',
            default_value='true',
            description='Enable online SLAM mapping.',
        ),
        DeclareLaunchArgument(
            'nav2_params_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('help_me_carry'),
                'config',
                'nav2_params.yaml',
            ]),
            description='Path to Nav2 configuration file.',
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
            'rviz_config',
            default_value=default_rviz,
            description='Path to RViz configuration file.',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('athome_navigation'),
                    'launch',
                    'navigation.launch.py',
                ])
            ),
            launch_arguments={
                'slam': LaunchConfiguration('slam'),
                'params_file': LaunchConfiguration('nav2_params_file'),
                'use_sensors': 'false',
                'publish_static_tf': 'false',
                'rviz': LaunchConfiguration('rviz'),
                'rviz_software': LaunchConfiguration('rviz_software'),
                'rviz_config': LaunchConfiguration('rviz_config'),
                'use_following': 'true',
                'initial_x': '0.0',
                'initial_y': '0.0',
                'initial_yaw': '0.0',
            }.items(),
        ),
    ])
