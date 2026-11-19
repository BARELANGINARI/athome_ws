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
        DeclareLaunchArgument('enable_body_tracking', default_value='true', description='Enable ZED body tracking for guest/person'),
        DeclareLaunchArgument('enable_chair', default_value='true', description='Enable chair detection and empty/occupied classification'),
        DeclareLaunchArgument('chair_model', default_value='', description='Optional path to custom chair ONNX model'),
        DeclareLaunchArgument('enable_drinks', default_value='true', description='Enable bottle and cracker detection'),
        DeclareLaunchArgument('botol_model', default_value='models/water.onnx', description='Path to water/botol ONNX model'),
        DeclareLaunchArgument('cracker_model', default_value='models/cheezit.onnx', description='Path to cheezit/cracker ONNX model'),
        DeclareLaunchArgument('pringles_model', default_value='models/pringles.onnx', description='Path to pringles ONNX model'),
        DeclareLaunchArgument('sugarbox_model', default_value='models/sugarbox.onnx', description='Path to sugarbox ONNX model'),
        DeclareLaunchArgument('face_model', default_value='models/face_recognition_sface.onnx', description='Path to face recognition ONNX model'),
        DeclareLaunchArgument('face_threshold', default_value='0.363', description='Cosine similarity threshold for face recognition match'),
        DeclareLaunchArgument('drink_min_area', default_value='100.0', description='Minimum bounding box area for drinks & crackers (pixels)'),
        DeclareLaunchArgument('drink_conf', default_value='0.20', description='Confidence threshold for drinks & crackers'),
        DeclareLaunchArgument('enable_bag', default_value='false', description='Enable bag detection (disabled for receptionist)'),
        DeclareLaunchArgument('enable_yolo', default_value='true', description='Enable YOLO backend for chair/objects'),
        DeclareLaunchArgument('yolo_conf', default_value='0.25', description='YOLO confidence threshold'),
        DeclareLaunchArgument('yolo_model', default_value='models/yolov8n_sim.onnx', description='Path to general YOLO ONNX model'),
        DeclareLaunchArgument('debug_window', default_value='true', description='Show OpenCV preview visualization window'),
        DeclareLaunchArgument('mirror', default_value='false', description='Mirror display preview horizontally'),
        DeclareLaunchArgument('publish_yaw', default_value='false', description='Publish IMU yaw from ZED vision (default: false, receptionist uses /yaw from Razor IMU)'),
        DeclareLaunchArgument('yaw_360', default_value='true', description='Publish IMU yaw in 0-360 deg range (default: true)'),
        DeclareLaunchArgument('host_name', default_value='Alex', description='Default host name for pre-registration'),
        DeclareLaunchArgument('host_photo', default_value='models/host.jpg', description='Optional photo of the host to pre-register face'),
        DeclareLaunchArgument('color_model', default_value='models/color_model.onnx', description='Path to clothing color ONNX model'),
        DeclareLaunchArgument('color_labels', default_value='models/color_names.txt', description='Path to clothing color labels file'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('athome_vision'),
                    'launch',
                    'vision.launch.py',
                ])
            ),
            launch_arguments={
                'task': 'receptionist',
                'camera_fps': LaunchConfiguration('camera_fps'),
                'publish_yaw': LaunchConfiguration('publish_yaw'),
                'yaw_360': LaunchConfiguration('yaw_360'),
                'enable_body_tracking': LaunchConfiguration('enable_body_tracking'),
                'enable_yolo': LaunchConfiguration('enable_yolo'),
                'yolo_model': LaunchConfiguration('yolo_model'),
                'enable_bag': LaunchConfiguration('enable_bag'),
                'botol_model': LaunchConfiguration('botol_model'),
                'cracker_model': LaunchConfiguration('cracker_model'),
                'pringles_model': LaunchConfiguration('pringles_model'),
                'sugarbox_model': LaunchConfiguration('sugarbox_model'),
                'face_model': LaunchConfiguration('face_model'),
                'face_threshold': LaunchConfiguration('face_threshold'),
                'drink_min_area': LaunchConfiguration('drink_min_area'),
                'drink_conf': LaunchConfiguration('drink_conf'),
                'enable_drinks': LaunchConfiguration('enable_drinks'),
                'chair_model': LaunchConfiguration('chair_model'),
                'enable_chair': LaunchConfiguration('enable_chair'),
                'yolo_conf': LaunchConfiguration('yolo_conf'),
                'debug_window': LaunchConfiguration('debug_window'),
                'mirror': LaunchConfiguration('mirror'),
                'depth_minimum_distance': LaunchConfiguration('depth_minimum_distance'),
                'host_name': LaunchConfiguration('host_name'),
                'host_photo': LaunchConfiguration('host_photo'),
                'color_model': LaunchConfiguration('color_model'),
                'color_labels': LaunchConfiguration('color_labels'),
            }.items(),
        ),
    ])
