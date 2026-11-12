import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_restaurant = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    default_rviz = os.path.join(pkg_restaurant, 'rviz', 'restaurant.rviz')
    default_task_params = os.path.join(pkg_restaurant, 'config', 'restaurant_params.yaml')

    return LaunchDescription([
        # SLAM argument: defaults to 'true' for unmapped/mapless environment
        DeclareLaunchArgument(
            'slam',
            default_value='true',
            description='Enable real-time SLAM (slam_toolbox) in Nav2 for unmapped environment'
        ),
        DeclareLaunchArgument(
            'nav2_params_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('restaurant'),
                'config',
                'nav2_params.yaml',
            ]),
            description='Path to Nav2 parameter configuration file'
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Launch RViz2 visualization'
        ),
        DeclareLaunchArgument(
            'rviz_software',
            default_value='false',
            description='Force RViz software OpenGL rendering'
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=default_rviz,
            description='Path to RViz configuration file'
        ),
        DeclareLaunchArgument(
            'task_node',
            default_value='true',
            description='Launch restaurant service node'
        ),
        DeclareLaunchArgument(
            'task_params_file',
            default_value=default_task_params,
            description='Path to restaurant task parameters configuration file'
        ),

        # 1. Nav2 Navigation Bringup (SLAM + Planner + Controller)
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
                'use_following': 'false',
                'initial_x': '0.0',
                'initial_y': '0.0',
                'initial_yaw': '0.0',
            }.items(),
        ),

        # 2. Restaurant Node (provides /homing and /go_to_order1 services)
        Node(
            package='restaurant',
            executable='restaurant_node',
            name='restaurant_node',
            output='screen',
            parameters=[LaunchConfiguration('task_params_file')],
            condition=IfCondition(LaunchConfiguration('task_node')),
        ),
    ])
