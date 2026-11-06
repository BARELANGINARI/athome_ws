from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch ZED camera with body tracking, person attributes, and YOLO detection for Receptionist."""
    return LaunchDescription([
        DeclareLaunchArgument('camera_fps', default_value='30.0', description='Camera frame rate (FPS)'),
        DeclareLaunchArgument('depth_minimum_distance', default_value='30.0', description='Minimum depth sensing distance (cm)'),
        DeclareLaunchArgument('enable_body_tracking', default_value='true', description='Enable ZED body tracking'),
        DeclareLaunchArgument('enable_yolo', default_value='true', description='Enable YOLO object/bag detection'),
        DeclareLaunchArgument('yolo_conf', default_value='0.25', description='YOLO confidence threshold'),
        DeclareLaunchArgument('yolo_model', default_value='models/yolov8n_sim.onnx', description='Path to YOLO ONNX model'),
        DeclareLaunchArgument('botol_model', default_value='models/botol.onnx', description='Path to botol ONNX model'),
        DeclareLaunchArgument('cracker_model', default_value='models/best_cracker.onnx', description='Path to cracker ONNX model'),
        DeclareLaunchArgument('enable_drinks', default_value='true', description='Enable bottle and cracker detection'),
        DeclareLaunchArgument('chair_model', default_value='', description='Optional path to custom chair ONNX model'),
        DeclareLaunchArgument('enable_chair', default_value='true', description='Enable chair detection and empty/occupied classification'),
        DeclareLaunchArgument('debug_window', default_value='true', description='Show OpenCV preview visualization window'),
        DeclareLaunchArgument('mirror', default_value='false', description='Mirror display preview horizontally'),
        DeclareLaunchArgument('yaw_360', default_value='true', description='Publish IMU yaw in 0-360 deg range (default: true)'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('athome_vision'),
                    'launch',
                    'zed.launch.py',
                ])
            ),
            launch_arguments={
                'task': 'receptionist',
                'camera_fps': LaunchConfiguration('camera_fps'),
                'publish_yaw': 'true',
                'yaw_360': LaunchConfiguration('yaw_360'),
                'enable_body_tracking': LaunchConfiguration('enable_body_tracking'),
                'enable_yolo': LaunchConfiguration('enable_yolo'),
                'yolo_model': LaunchConfiguration('yolo_model'),
                'botol_model': LaunchConfiguration('botol_model'),
                'cracker_model': LaunchConfiguration('cracker_model'),
                'enable_drinks': LaunchConfiguration('enable_drinks'),
                'chair_model': LaunchConfiguration('chair_model'),
                'enable_chair': LaunchConfiguration('enable_chair'),
                'yolo_conf': LaunchConfiguration('yolo_conf'),
                'debug_window': LaunchConfiguration('debug_window'),
                'mirror': LaunchConfiguration('mirror'),
                'depth_minimum_distance': LaunchConfiguration('depth_minimum_distance'),
            }.items(),
        ),
    ])
