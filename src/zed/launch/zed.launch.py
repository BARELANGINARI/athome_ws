from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_fps',
            default_value='30.0',
            description='Camera frame rate (FPS)',
        ),
        DeclareLaunchArgument(
            'publish_yaw',
            default_value='true',
            description='Enable publishing of IMU yaw data',
        ),
        DeclareLaunchArgument(
            'publish_image',
            default_value='true',
            description='Publish preview awal /zed/image (BGR, untuk window task)',
        ),
        DeclareLaunchArgument(
            'image_stride',
            default_value='2',
            description='Publish /zed/image tiap N frame (kecilkan delay)',
        ),
        DeclareLaunchArgument(
            'depth_mode',
            default_value='NEURAL_LIGHT',
            description=(
                'NEURAL_LIGHT (default, ringan), NEURAL (paling akurat), atau '
                'ULTRA/PERFORMANCE (sudah deprecated di ZED SDK 5.x)'),
        ),
        DeclareLaunchArgument(
            'body_model',
            default_value='FAST',
            description='FAST (frame halus) atau MEDIUM (keypoint akurat)',
        ),
        DeclareLaunchArgument(
            'enable_body_tracking',
            default_value='true',
            description='Enable ZED body tracking (false = yaw-only, hemat GPU)',
        ),
        DeclareLaunchArgument(
            'front_yaw_offset_deg',
            default_value='180.0',
            description='Body yaw offset for FRONT orientation',
        ),
        DeclareLaunchArgument(
            'orientation_threshold_deg',
            default_value='30.0',
            description='Orientation classification threshold',
        ),
        # YOLO tas (pengganti HSV bola). Person via body tracking.
        # "Tinggal pake": yolov8n_sim.onnx + coco_names.txt sudah di models/
        # (regenerasi via src/zed/models/download_yolo.sh bila perlu).
        DeclareLaunchArgument(
            'enable_yolo',
            default_value='true',
            description='Enable YOLO bag detection (yolov8n_sim.onnx pretrained)',
        ),
        DeclareLaunchArgument(
            'debug_window',
            default_value='false',
            description=(
                'Tampilkan preview OpenCV (bbox person/tas + 34 keypoint). '
                'Default false supaya aman tanpa display.'
            ),
        ),
        DeclareLaunchArgument(
            'mirror',
            default_value='true',
            description='Balik preview kiri-kanan (tangan kanan = kanan layar)',
        ),
        DeclareLaunchArgument(
            'depth_minimum_distance',
            default_value='30.0',
            description='Jarak depth minimum cm ala maen_boss (30 = 30cm)',
        ),
        DeclareLaunchArgument(
            'yolo_model',
            default_value='models/yolov8n_sim.onnx',
            description='Path YOLO ONNX relatif share pkg (atau absolut)',
        ),
        DeclareLaunchArgument(
            'yolo_classes',
            default_value='models/coco_names.txt',
            description='Path nama kelas COCO relatif share pkg (atau absolut)',
        ),
        DeclareLaunchArgument(
            'yolo_conf',
            default_value='0.20',
            description='YOLO confidence threshold (bag)',
        ),
        DeclareLaunchArgument(
            'yolo_nms',
            default_value='0.45',
            description='YOLO NMS threshold',
        ),
        DeclareLaunchArgument(
            'yolo_size',
            default_value='640',
            description='YOLO input size (640 = bawaan export yolov8n)',
        ),
        DeclareLaunchArgument(
            'yolo_stride',
            default_value='2',
            description='Inferensi tiap N frame (hemat CPU); pose depth tiap frame',
        ),
        DeclareLaunchArgument(
            'yolo_min_area',
            default_value='1500.0',
            description='Luas bbox minimum px (buang deteksi kecil/noise)',
        ),

        Node(
            package='zed',
            executable='zed',
            name='zed',
            output='screen',
            parameters=[{
                'camera_fps': LaunchConfiguration('camera_fps'),
                'publish_yaw': LaunchConfiguration('publish_yaw'),
                'publish_image': LaunchConfiguration('publish_image'),
                'image_stride': LaunchConfiguration('image_stride'),
                'depth_mode': LaunchConfiguration('depth_mode'),
                'body_model': LaunchConfiguration('body_model'),
                'enable_body_tracking': LaunchConfiguration('enable_body_tracking'),
                'front_yaw_offset_deg': LaunchConfiguration('front_yaw_offset_deg'),
                'orientation_threshold_deg': LaunchConfiguration('orientation_threshold_deg'),
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
                'depth_minimum_distance': LaunchConfiguration('depth_minimum_distance'),
            }],
        ),
    ])
