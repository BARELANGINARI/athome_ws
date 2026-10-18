"""Kamera ZED (vision) untuk luggage_task.

Output zed: /zed/person_keypoints, /zed/bag_detections, /zed/yaw_deg, /zed/image.
Body tracking & YOLO WAJIB hidup untuk task (sudah default true).

Preview person (JENDELA LANGSUNG DI NODE zed, tanpa node tambahan):
    ros2 launch luggage_task camera.launch.py
  -> window "zed - person (preview)" langsung muncul.

  Matikan window:  ros2 launch luggage_task camera.launch.py debug_window:=false
  Matikan cermin:   ros2 launch luggage_task camera.launch.py mirror:=false

  Window menampilkan frame kamera + bbox person/tas, 34 keypoint, jarak,
  yaw, dan jumlah deteksi. Orang paling dekat digambar hijau (index 0 di
  /zed/person_keypoints). Tekan q di window untuk keluar.

  Gambar langsung diambil dari frame yang sedang diproses node zed, jadi
  tidak lewat /zed/image dan tidak ada selisih QoS. Kalau mesin tanpa
  display (SSH tanpa X, kontainer), preview otomatis dimatikan - node tetap
  jalan & publish normal.

  PENTING: file ini khusus KAMERA. Kalau mau /goto atau person following,
  itu perlu Nav2 -> pakai navigation.launch.py (yang juga bisa nyalakan
  preview dengan zed_debug_window:=true). JANGAN jalankan keduanya
  bersamaan: hanya 1 proses boleh buka kamera ZED.

JANGAN dijalankan bareng launch lain yang menyalakan kamera sama
(mis. navigation.launch.py / sensors.launch.py): 1 kamera ZED hanya
boleh dibuka 1 proses.

Cara jalan (split-stack):
  Terminal 1: ros2 launch luggage_task hardware.launch.py
  Terminal 2: ros2 launch luggage_task camera.launch.py
  Terminal 3: ros2 launch luggage_task motion.launch.py
"""
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
            default_value='FAST',
            description='Model body tracking ZED (FAST halus / MEDIUM akurat).',
        ),
        DeclareLaunchArgument(
            'depth_minimum_distance',
            default_value='30.0',
            description='Depth minimum ZED (cm ala maen_boss, 30 = 30cm).',
        ),
        DeclareLaunchArgument(
            'depth_mode',
            default_value='NEURAL_LIGHT',
            description=(
                'Depth ZED: NEURAL_LIGHT (default, ringan), NEURAL (akurat), atau '
                'ULTRA/PERFORMANCE (deprecated di SDK 5.x).'),
        ),
        DeclareLaunchArgument(
            'image_stride',
            default_value='2',
            description='Publish /zed/image tiap N frame (kecilkan delay).',
        ),
        DeclareLaunchArgument(
            'debug_window',
            default_value='true',
            description=(
                'Preview OpenCV dari node zed: bbox person/tas, 34 keypoint, '
                'jarak, dan fps. Tekan q untuk keluar. Default true karena '
                'file ini memang khusus kamera. Matikan: '
                'debug_window:=false. Kalau mesin tanpa display, node tetap '
                'jalan, hanya preview-nya yang dimatikan.'
            ),
        ),
        DeclareLaunchArgument(
            'mirror',
            default_value='true',
            description='Balik preview kiri-kanan (tangan kanan = kanan layar).',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('zed'),
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
                'debug_window': LaunchConfiguration('debug_window'),
                'mirror': LaunchConfiguration('mirror'),
                'depth_minimum_distance': LaunchConfiguration('depth_minimum_distance'),
            }.items(),
        ),
    ])
