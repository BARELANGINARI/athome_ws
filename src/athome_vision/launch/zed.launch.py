from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('camera_fps', default_value='30.0', description='Camera FPS'),
        DeclareLaunchArgument('publish_yaw', default_value='true', description='Publish IMU yaw'),
        DeclareLaunchArgument('publish_image', default_value='true', description='Publish preview image'),
        DeclareLaunchArgument('image_stride', default_value='2', description='Image frame stride'),
        DeclareLaunchArgument('enable_body_tracking', default_value='true', description='Enable body tracking'),
        DeclareLaunchArgument('enable_yolo', default_value='true', description='Enable YOLO bag detection'),
        DeclareLaunchArgument('debug_window', default_value='false', description='Show OpenCV preview window'),
        DeclareLaunchArgument('mirror', default_value='true', description='Flip preview horizontally'),
        DeclareLaunchArgument('yolo_model', default_value='models/yolov8n_sim.onnx', description='Path to YOLO model'),
        DeclareLaunchArgument('yolo_classes', default_value='models/coco_names.txt', description='Path to COCO classes'),
        DeclareLaunchArgument('yolo_conf', default_value='0.20', description='YOLO confidence threshold'),
        DeclareLaunchArgument('yolo_nms', default_value='0.45', description='YOLO NMS threshold'),
        DeclareLaunchArgument('yolo_size', default_value='640', description='YOLO input size'),
        DeclareLaunchArgument('yolo_stride', default_value='2', description='YOLO frame stride'),
        DeclareLaunchArgument('yolo_min_area', default_value='1500.0', description='Min bbox area in px'),

        Node(
            package='athome_vision',
            executable='zed',
            name='zed',
            output='screen',
            parameters=[{
                'camera_fps': LaunchConfiguration('camera_fps'),
                'publish_yaw': LaunchConfiguration('publish_yaw'),
                'publish_image': LaunchConfiguration('publish_image'),
                'image_stride': LaunchConfiguration('image_stride'),
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
                'yolo_stride': LaunchConfiguration('yolo_stride'),
            }],
        ),
    ])
