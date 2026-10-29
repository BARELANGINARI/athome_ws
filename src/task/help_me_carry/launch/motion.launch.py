import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_carry = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    default_rviz = os.path.join(pkg_carry, 'rviz', 'help_me_carry.rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'slam',
            default_value='true',
            description='true = aktifkan Online SLAM (slam_toolbox) di dalam Nav2; '
                        'peta dibangun sambil robot gerak.',
        ),
        DeclareLaunchArgument(
            'nav2_params_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('athome_navigation'),
                'config',
                'nav2_params.yaml',
            ]),
            description='Path file parameter Nav2 (planner, controller, costmaps, presets).',
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Tampilkan RViz (map SLAM + costmaps + rute + posisi robot).',
        ),
        DeclareLaunchArgument(
            'rviz_software',
            default_value='false',
            description='Paksa RViz software rendering (tanpa GPU).',
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=default_rviz,
            description='File konfigurasi RViz (.rviz).',
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
