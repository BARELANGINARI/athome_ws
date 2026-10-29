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
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_receptionist = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    default_rviz = os.path.join(pkg_receptionist, 'rviz', 'receptionist.rviz')
    default_task_params = os.path.join(pkg_receptionist, 'config', 'receptionist_params.yaml')

    return LaunchDescription([
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
            'nav2_params_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('receptionist'),
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
        DeclareLaunchArgument(
            'initial_x',
            default_value='0.0',
            description='Initial robot X pose on map.',
        ),
        DeclareLaunchArgument(
            'initial_y',
            default_value='0.0',
            description='Initial robot Y pose on map.',
        ),
        DeclareLaunchArgument(
            'initial_yaw',
            default_value='0.0',
            description='Initial robot yaw on map (rad).',
        ),
        DeclareLaunchArgument(
            'task_node',
            default_value='true',
            description='Launch receptionist navigation service node.',
        ),
        DeclareLaunchArgument(
            'task_params_file',
            default_value=default_task_params,
            description='Path to receptionist task parameters file.',
        ),

        # 1. Nav2 Bringup (AMCL Localization + Static Map + Planner + Controller)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('athome_navigation'),
                    'launch',
                    'navigation.launch.py',
                ])
            ),
            launch_arguments={
                'slam': 'false',
                'map': LaunchConfiguration('map'),
                'params_file': LaunchConfiguration('nav2_params_file'),
                'use_sensors': 'false',
                'publish_static_tf': 'false',
                'rviz': LaunchConfiguration('rviz'),
                'rviz_software': LaunchConfiguration('rviz_software'),
                'rviz_config': LaunchConfiguration('rviz_config'),
                'use_following': 'false',
                'initial_x': LaunchConfiguration('initial_x'),
                'initial_y': LaunchConfiguration('initial_y'),
                'initial_yaw': LaunchConfiguration('initial_yaw'),
            }.items(),
        ),

        # 2. Receptionist Service Node (provides /receptionist/go_to_door & /receptionist/go_to_living_room)
        Node(
            package='receptionist',
            executable='receptionist_node',
            name='receptionist_node',
            output='screen',
            parameters=[LaunchConfiguration('task_params_file')],
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('task_node'), "' == 'true'"])
            ),
        ),
    ])
