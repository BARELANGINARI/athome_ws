from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_CONSOLE_OUTPUT_FORMAT', '[{severity}] {message}'),
        DeclareLaunchArgument('camera_fps', default_value='30.0', description='Camera frame rate (FPS)'),
        DeclareLaunchArgument('depth_minimum_distance', default_value='30.0', description='Minimum depth sensing distance (cm)'),
        DeclareLaunchArgument('debug_window', default_value='true', description='Show OpenCV preview visualization window'),
        DeclareLaunchArgument('mirror', default_value='false', description='Mirror display preview horizontally'),
        DeclareLaunchArgument('publish_yaw', default_value='true', description='Publish IMU yaw topic from vision (default: true)'),
        DeclareLaunchArgument('yaw_360', default_value='true', description='Publish IMU yaw in 0-360 deg range (default: true)'),
        DeclareLaunchArgument('enable_drinks', default_value='true', description='Enable drink & object detection (pringles, sugarbox, soda, crackerbox, milk)'),
        DeclareLaunchArgument('pringles_model', default_value='models/pringles.onnx', description='Path to pringles ONNX model'),
        DeclareLaunchArgument('cracker_model', default_value='models/crackerbox.onnx', description='Path to crackerbox ONNX model'),
        DeclareLaunchArgument('soda_model', default_value='models/soda.onnx', description='Path to soda ONNX model'),
        DeclareLaunchArgument('sugarbox_model', default_value='models/sugarbox.onnx', description='Path to sugarbox ONNX model'),
        DeclareLaunchArgument('milk_model', default_value='models/milk.onnx', description='Path to milk ONNX model'),
        DeclareLaunchArgument('drink_min_area', default_value='180.0', description='Minimum bounding box area for drinks & crackers (pixels)'),
        DeclareLaunchArgument('drink_conf', default_value='0.48', description='Confidence threshold for drinks & crackers'),
        DeclareLaunchArgument('drink_max_dist', default_value='2.5', description='Maximum distance (m) to detect drinks & crackers'),

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
                'enable_drinks': LaunchConfiguration('enable_drinks'),
                'enable_chair': 'false',
                'enable_back_detection': 'false',
                'pringles_model': LaunchConfiguration('pringles_model'),
                'cracker_model': LaunchConfiguration('cracker_model'),
                'soda_model': LaunchConfiguration('soda_model'),
                'sugarbox_model': LaunchConfiguration('sugarbox_model'),
                'milk_model': LaunchConfiguration('milk_model'),
                'drink_min_area': LaunchConfiguration('drink_min_area'),
                'drink_conf': LaunchConfiguration('drink_conf'),
                'drink_max_dist': LaunchConfiguration('drink_max_dist'),
            }.items(),
        ),
    ])
