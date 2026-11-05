from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('task', default_value='all', description='Task filter for vision IC logging: help_me_carry, receptionist, or all'),
        DeclareLaunchArgument('camera_fps', default_value='30.0', description='Camera frame rate (FPS)'),
        DeclareLaunchArgument('publish_yaw', default_value='true', description='Publish IMU yaw topic'),
        DeclareLaunchArgument('depth_minimum_distance', default_value='30.0', description='Minimum depth sensing distance (cm)'),
        DeclareLaunchArgument('enable_body_tracking', default_value='true', description='Enable ZED body tracking'),
        DeclareLaunchArgument('enable_yolo', default_value='false', description='Enable YOLO object detection'),
        DeclareLaunchArgument('debug_window', default_value='false', description='Show OpenCV preview window'),
        DeclareLaunchArgument('mirror', default_value='true', description='Mirror display preview horizontally'),
        DeclareLaunchArgument('yolo_model', default_value='', description='Path to YOLO ONNX model'),
        DeclareLaunchArgument('yolo_classes', default_value='models/coco_names.txt', description='Path to COCO class labels file'),
        DeclareLaunchArgument('yolo_conf', default_value='0.25', description='YOLO confidence threshold'),
        DeclareLaunchArgument('yolo_nms', default_value='0.45', description='YOLO NMS IoU threshold'),
        DeclareLaunchArgument('yolo_size', default_value='640', description='YOLO input image dimension'),
        DeclareLaunchArgument('yolo_min_area', default_value='1500.0', description='Minimum bounding box area (pixels)'),
        DeclareLaunchArgument('gender_model', default_value='models/gender_cls.onnx', description='Path to gender ONNX model'),
        DeclareLaunchArgument('color_model', default_value='models/color_model.onnx', description='Path to clothing color ONNX model'),
        DeclareLaunchArgument('color_labels', default_value='models/color_names.txt', description='Path to clothing color labels file'),
        DeclareLaunchArgument('hair_model', default_value='models/hair_model.onnx', description='Path to hair color ONNX model'),
        DeclareLaunchArgument('hair_labels', default_value='models/hair_names.txt', description='Path to hair labels file'),

        Node(
            package='athome_vision',
            executable='zed',
            name='zed',
            output='screen',
            respawn=True,
            respawn_delay=2.0,
            parameters=[{
                'task': LaunchConfiguration('task'),
                'camera_fps': LaunchConfiguration('camera_fps'),
                'publish_yaw': LaunchConfiguration('publish_yaw'),
                'depth_minimum_distance': LaunchConfiguration('depth_minimum_distance'),
                'enable_body_tracking': LaunchConfiguration('enable_body_tracking'),
                'enable_yolo': LaunchConfiguration('enable_yolo'),
                'debug_window': LaunchConfiguration('debug_window'),
                'mirror': LaunchConfiguration('mirror'),
                'yolo_model': LaunchConfiguration('yolo_model'),
                'yolo_classes': LaunchConfiguration('yolo_classes'),
                'yolo_conf': LaunchConfiguration('yolo_conf'),
                'yolo_nms': LaunchConfiguration('yolo_nms'),
                'yolo_size': LaunchConfiguration('yolo_size'),
                'yolo_min_area': LaunchConfiguration('yolo_min_area'),
                'gender_model': LaunchConfiguration('gender_model'),
                'color_model': LaunchConfiguration('color_model'),
                'color_labels': LaunchConfiguration('color_labels'),
                'hair_model': LaunchConfiguration('hair_model'),
                'hair_labels': LaunchConfiguration('hair_labels'),
            }],
        ),
    ])
