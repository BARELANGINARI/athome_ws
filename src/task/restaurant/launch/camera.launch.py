from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('camera_fps', default_value='30.0', description='Camera frame rate (FPS)'),
        DeclareLaunchArgument('depth_minimum_distance', default_value='30.0', description='Minimum depth sensing distance (cm)'),
        DeclareLaunchArgument('debug_window', default_value='true', description='Show OpenCV preview visualization window'),
        DeclareLaunchArgument('mirror', default_value='false', description='Mirror display preview horizontally'),
        DeclareLaunchArgument('publish_yaw', default_value='true', description='Publish IMU yaw topic'),
        DeclareLaunchArgument('yaw_360', default_value='true', description='Publish IMU yaw in 0-360 deg range (default: true)'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('athome_vision'),
                    'launch',
                    'vision.launch.py',
                ])
            ),
            launch_arguments={
                'task': 'restaurant',
                'camera_fps': LaunchConfiguration('camera_fps'),
                'depth_minimum_distance': LaunchConfiguration('depth_minimum_distance'),
                'debug_window': LaunchConfiguration('debug_window'),
                'mirror': LaunchConfiguration('mirror'),
                'publish_yaw': LaunchConfiguration('publish_yaw'),
                'yaw_360': LaunchConfiguration('yaw_360'),
                'enable_body_tracking': 'true',
                'enable_yolo': 'false',
                'enable_bag': 'false',
                'enable_drinks': 'false',
                'enable_chair': 'false',
            }.items(),
        ),
    ])
