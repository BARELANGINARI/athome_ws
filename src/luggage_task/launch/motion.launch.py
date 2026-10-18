"""Gerak task luggage (node + SLAM + RViz).

Prasyarat: hardware (nucleo + lidar) & vision (zed) SUDAH jalan
(mis. via hardware.launch.py + camera.launch.py). Launch ini TIDAK
menyalakan hardware maupun kamera.

Isi: luggage_task_node (state machine START -> DETECT_PERSON -> ... ->
FOLLOW via cmd_vel / goal Nav2, baca /zed/* + /scan + TF) + slam_toolbox
(mapping + TF map->odom, default hidup, peta dibangun sambil robot gerak)
+ RViz (default TAMPIL: map SLAM + posisi robot).

Window OpenCV "luggage_task - gesture" (frame + bbox person/tas + arah
gesture) hidupkan bila perlu via debug_window:=true.

Cara jalan (split-stack):
  Terminal 1: ros2 launch luggage_task hardware.launch.py
  Terminal 2: ros2 launch luggage_task camera.launch.py
  Terminal 3: ros2 launch luggage_task motion.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


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
            'mirror',
            default_value='true',
            description='true (default) = perspektif person: tangan kanan = KANAN.',
        ),
        DeclareLaunchArgument(
            'debug_window',
            default_value='false',
            description='true = tampilkan window OpenCV task (frame kamera + '
                        'bbox person/tas + teks gesture). Default mati '
                        '(1 window: RViz).',
        ),
        DeclareLaunchArgument(
            'slam',
            default_value='true',
            description='true = ikutkan slam_toolbox online async (mapping + '
                        'TF map->odom); peta dibangun sambil robot gerak.',
        ),
        DeclareLaunchArgument(
            'slam_params_file',
            default_value=os.path.join(
                get_package_share_directory('mapping'),
                'config',
                'slam.yaml',
            ),
            description='File param slam_toolbox (dibaca langsung node-nya).',
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Tampilkan RViz (map SLAM + posisi robot).',
        ),
        DeclareLaunchArgument(
            'rviz_software',
            default_value='false',
            description='Paksa RViz software rendering (tanpa GPU).',
        ),
        # SLAM: mapping + TF map->odom; peta dibangun sambil robot gerak
        # (butuh /scan + TF odom->base dari hardware.launch.py).
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('slam_toolbox'),
                    'launch',
                    'online_async_launch.py',
                )
            ),
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('slam'), "' == 'true'"])
            ),
            launch_arguments={
                'use_sim_time': 'false',
                'slam_params_file': LaunchConfiguration('slam_params_file'),
            }.items(),
        ),
        # RViz: map SLAM (/map) + posisi robot (TF + /odom).
        # Butuh slam:=true (atau mapping lain) agar /map ada; tanpa /map
        # window tampil KOSONG.
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
        Node(
            package='luggage_task',
            executable='luggage_task_node',
            name='luggage_task',
            output='screen',
            parameters=[{
                'params_file': LaunchConfiguration('params_file'),
                'mirror': LaunchConfiguration('mirror'),
                'debug_window': LaunchConfiguration('debug_window'),
            }],
        ),
    ])
