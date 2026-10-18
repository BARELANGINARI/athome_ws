"""Task 1 Carry My Luggage (Navigasi & Pelacakan).

Orchestrasi node existing:
- zed (kamera): /zed/person_keypoints, /zed/bag_detections, /zed/yaw_deg
- robot_bringup (lidar): /scan
- nucleo (base driver): cmd_vel
Jalankan base driver + sensor dulu (atau via mapping/navigation launch),
lalu task ini. Body tracking zed HARUS aktif (enable_body_tracking:=true).

Cara minimal (request): nucleo + task + nav2 dalam SATU launch:
  ros2 launch luggage_task luggage_task.launch.py nav2:=true nucleo:=true
  - nucleo:=true  = ikutkan nucleo.launch.py (base driver UDP STM32)
  - nav2:=true    = ikutkan navigation.launch.py (sensor lidar+zed + SLAM/Nav2)
  - nav2:=false   = hanya node task (untuk tes gesture-only tanpa robot jalan)

Tuning cukup edit config/params.yaml lalu RESTART node (tanpa build):
node membaca file YAML langsung saat start (param `params_file`).
Path default diturunkan dari lokasi launch file ini sehingga selalu
menunjuk ke src/.../config/params.yaml (hidup, bukan hasil copy build).
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # __file__ symlink-install -> resolve() tembus ke src tree.
    pkg_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    default_yaml = os.path.join(pkg_dir, 'config', 'params.yaml')
    default_rviz = os.path.join(pkg_dir, 'rviz', 'luggage_task.rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_yaml,
            description='Path file YAML params (dibaca node saat start).',
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Tampilkan RViz navigasi',
        ),
        DeclareLaunchArgument(
            'rviz_software',
            default_value='false',
            description='Paksa RViz software rendering (tanpa GPU)',
        ),
        DeclareLaunchArgument(
            'nav2',
            default_value='false',
            description='true = ikutkan navigation.launch.py (Nav2 + sensor, '
                        'dibutuhkan state FOLLOW).',
        ),
        DeclareLaunchArgument(
            'nucleo',
            default_value='true',
            description='true = ikutkan nucleo.launch.py (base driver STM32, '
                        'sub cmd_vel). Minimal: nucleo + task aja.',
        ),
        DeclareLaunchArgument(
            'stm_ip',
            default_value='192.168.0.4',
            description='Diteruskan ke nucleo.launch.py bila nucleo:=true.',
        ),
        DeclareLaunchArgument(
            'depth_minimum_distance',
            default_value='30.0',
            description='Diteruskan ke navigation/sensors/zed bila nav2:=true '
                        '(cm ala maen_boss, 30 = 30cm).',
        ),
        DeclareLaunchArgument(
            'image_stride',
            default_value='2',
            description='Diteruskan ke zed bila nav2:=true '
                        '(publish /zed/image tiap N frame, kecilkan delay).',
        ),
        DeclareLaunchArgument(
            'depth_mode',
            default_value='PERFORMANCE',
            description='Diteruskan ke zed bila nav2:=true '
                        '(PERFORMANCE ringan, atau ULTRA ala maen_boss).',
        ),
        DeclareLaunchArgument(
            'mirror',
            default_value='true',
            description='true (default) = perspektif person: tangan kanan = KANAN.',
        ),
        DeclareLaunchArgument(
            'body_model',
            default_value='FAST',
            description='Diteruskan ke zed bila nav2:=true (FAST halus / MEDIUM akurat).',
        ),
        DeclareLaunchArgument(
            'camera_fps',
            default_value='30.0',
            description='Diteruskan ke zed bila nav2:=true (30 aman; turunkan ke 15 '
                        'bila patah-patah).',
        ),
        DeclareLaunchArgument(
            'enable_yolo',
            default_value='true',
            description='Diteruskan ke zed bila nav2:=true (YOLO tas pretrained).',
        ),
        DeclareLaunchArgument(
            'slam',
            default_value='true',
            description='Diteruskan ke navigation.launch.py bila nav2:=true.',
        ),
        DeclareLaunchArgument(
            'enable_body_tracking',
            default_value='true',
            description='Diteruskan ke navigation.launch.py bila nav2:=true: '
                        'body tracking ZED (task butuh person).',
        ),
        DeclareLaunchArgument(
            'map',
            default_value=PathJoinSubstitution([
                FindPackageShare('navigation'),
                'maps',
                'map.yaml',
            ]),
            description='Diteruskan ke navigation.launch.py bila nav2:=true: '
                        'peta statis (dipakai bila slam:=false).',
        ),
        # RViz task: map (/map) + posisi robot (TF + /odom).
        # Peta muncul bila mapping/navigation jalan (map_server/SLAM);
        # posisi robot dari odom + TF nucleo (Fixed Frame: map).
        # PENTING: dua Node ini HARUS di-visit SEBELUM IncludeLaunchDescription
        # nav2 di bawah. ROS 2 launch TIDAK mengisolasi launch_arguments:
        # Include menerapkan SetLaunchConfiguration di context yang SAMA,
        # sehingga 'rviz': 'false' milik nav2 akan menimpa 'rviz' kita
        # (default 'true') dan RViz task tidak pernah start. JANGAN pindah
        # blok ini ke bawah include.
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            condition=IfCondition(
                PythonExpression([
                    "'",
                    LaunchConfiguration('rviz'),
                    "' == 'true' and '",
                    LaunchConfiguration('rviz_software'),
                    "' == 'false'",
                ])
            ),
            arguments=['-d', default_rviz],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2_sw',
            output='screen',
            condition=IfCondition(
                PythonExpression([
                    "'",
                    LaunchConfiguration('rviz'),
                    "' == 'true' and '",
                    LaunchConfiguration('rviz_software'),
                    "' == 'true'",
                ])
            ),
            arguments=['-d', default_rviz],
            additional_env={
                'LIBGL_ALWAYS_SOFTWARE': '1',
            },
        ),
        # Nucleo base driver (UDP STM32, sub cmd_vel + pub odom/TF).
        # Dipisah dari nav2 agar "nucleo + task aja" bisa satu launch.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('nucleo'),
                    'launch',
                    'nucleo.launch.py',
                ])
            ),
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('nucleo'), "' == 'true'"])
            ),
            launch_arguments={
                'stm_ip': LaunchConfiguration('stm_ip'),
            }.items(),
        ),
        # Nav2 stack (planner + controller + TF map->odom). RViz milik
        # navigation dimatikan agar tidak dobel dengan RViz task ini.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('navigation'),
                    'launch',
                    'navigation.launch.py',
                ])
            ),
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('nav2'), "' == 'true'"])
            ),
            launch_arguments={
                'rviz': 'false',
                'slam': LaunchConfiguration('slam'),
                'enable_body_tracking': LaunchConfiguration('enable_body_tracking'),
                'depth_minimum_distance': LaunchConfiguration('depth_minimum_distance'),
                'image_stride': LaunchConfiguration('image_stride'),
                'depth_mode': LaunchConfiguration('depth_mode'),
                'body_model': LaunchConfiguration('body_model'),
                'camera_fps': LaunchConfiguration('camera_fps'),
                'enable_yolo': LaunchConfiguration('enable_yolo'),
                'map': LaunchConfiguration('map'),
            }.items(),
        ),
        Node(
            package='luggage_task',
            executable='luggage_task_node',
            name='luggage_task',
            output='screen',
            parameters=[{
                'params_file': LaunchConfiguration('params_file'),
                'mirror': LaunchConfiguration('mirror'),
            }],
        ),
    ])
