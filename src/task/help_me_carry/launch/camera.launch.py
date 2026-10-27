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
        DeclareLaunchArgument(
            'camera_fps',
            default_value='30.0',
            description='ZED camera framerate (FPS).',
        ),
        DeclareLaunchArgument(
            'image_stride',
            default_value='2',
            description='Publish image every N frames.',
        ),
        DeclareLaunchArgument(
            'yolo_conf',
            default_value='0.20',
            description='YOLO bag detection confidence threshold.',
        ),
        DeclareLaunchArgument(
            'yolo_stride',
            default_value='2',
            description='Run YOLO inference every N frames.',
        ),
        DeclareLaunchArgument(
            'debug_window',
            default_value='true',
            description='Enable OpenCV debug visualization window.',
        ),
        DeclareLaunchArgument(
            'mirror',
            default_value='false',
            description='Mirror preview horizontally.',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('athome_vision'),
                    'launch',
                    'zed.launch.py',
                ])
            ),
            launch_arguments={
                'camera_fps': LaunchConfiguration('camera_fps'),
                'publish_yaw': 'true',
                'publish_image': 'true',
                'image_stride': LaunchConfiguration('image_stride'),
                'enable_body_tracking': 'true',
                'enable_yolo': 'true',
                'yolo_conf': LaunchConfiguration('yolo_conf'),
                'yolo_stride': LaunchConfiguration('yolo_stride'),
                'debug_window': LaunchConfiguration('debug_window'),
                'mirror': LaunchConfiguration('mirror'),
            }.items(),
        ),
    ])
