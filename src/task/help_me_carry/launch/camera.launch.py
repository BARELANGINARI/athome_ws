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
            description='FPS kamera ZED (30 aman; 15 bila patah-patah).',
        ),
        DeclareLaunchArgument(
            'body_model',
            default_value='MEDIUM',
            description='Model body tracking ZED (MEDIUM akurat / FAST halus).',
        ),
        DeclareLaunchArgument(
            'depth_minimum_distance',
            default_value='30.0',
            description='Depth minimum ZED (cm ala maen_boss, 30 = 30cm).',
        ),
        DeclareLaunchArgument(
            'depth_mode',
            default_value='ULTRA',
            description='Depth ZED: ULTRA (default), NEURAL_LIGHT, atau PERFORMANCE.',
        ),
        DeclareLaunchArgument(
            'image_stride',
            default_value='2',
            description='Publish /zed/image tiap N frame (kecilkan delay).',
        ),
        DeclareLaunchArgument(
            'yolo_conf',
            default_value='0.20',
            description='Confidence threshold YOLO deteksi tas.',
        ),
        DeclareLaunchArgument(
            'yolo_stride',
            default_value='2',
            description='Inferensi YOLO tiap N frame (hemat beban CPU).',
        ),
        DeclareLaunchArgument(
            'debug_window',
            default_value='true',
            description=(
                'Preview OpenCV dari node zed: bbox person/tas, HUD koordinat, '
                'garis pelacak, dan fps. Tekan q untuk keluar. Default true karena '
                'file ini memang khusus kamera. Matikan: '
                'debug_window:=false. Kalau mesin tanpa display, node tetap '
                'jalan, hanya preview-nya yang dimatikan.'
            ),
        ),
        DeclareLaunchArgument(
            'mirror',
            default_value='false',
            description='Balik preview kiri-kanan (tangan kanan = kanan layar).',
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
                'depth_mode': LaunchConfiguration('depth_mode'),
                'body_model': LaunchConfiguration('body_model'),
                'enable_body_tracking': 'true',
                'enable_yolo': 'true',
                'yolo_conf': LaunchConfiguration('yolo_conf'),
                'yolo_stride': LaunchConfiguration('yolo_stride'),
                'debug_window': LaunchConfiguration('debug_window'),
                'mirror': LaunchConfiguration('mirror'),
                'depth_minimum_distance': LaunchConfiguration('depth_minimum_distance'),
            }.items(),
        ),
    ])
