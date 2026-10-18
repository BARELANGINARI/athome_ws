"""Hardware (base + lidar). JALANKAN HANYA SALAH SATU stack sensor!

Isi: nucleo.launch.py (base driver STM32: sub cmd_vel, pub odom/TF) +
rplidar (lidar /scan full 360 deg) + TF statis base->laser & footprint->base.

JANGAN dijalankan bareng launch lain yang menyalakan hardware sama
(mis. navigation.launch.py / sensors.launch.py): port serial lidar +
driver base HANYA boleh dibuka 1x (duplikat = lidar mati / TF konflik).

Cara jalan (split-stack):
  Terminal 1: ros2 launch luggage_task hardware.launch.py
  Terminal 2: ros2 launch luggage_task camera.launch.py
  Terminal 3: ros2 launch luggage_task motion.launch.py
"""
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
    return LaunchDescription([
        DeclareLaunchArgument(
            'nucleo',
            default_value='true',
            description='true = ikutkan nucleo.launch.py (base driver STM32).',
        ),
        DeclareLaunchArgument(
            'stm_ip',
            default_value='192.168.0.4',
            description='Diteruskan ke nucleo.launch.py bila nucleo:=true.',
        ),
        DeclareLaunchArgument(
            'joy_topic',
            default_value='joy_vel',
            description='Diteruskan ke nucleo.launch.py: kanal override joystick '
                        '(prioritas di atas cmd_vel!). Isi nama topik yang tidak '
                        'ada (mis. joy_disabled) untuk menonaktifkan.',
        ),
        DeclareLaunchArgument(
            'test_topic',
            default_value='test_vel',
            description='Diteruskan ke nucleo.launch.py: kanal override uji manual. '
                        'Isi nama topik yang tidak ada untuk menonaktifkan.',
        ),
        DeclareLaunchArgument(
            'lidar',
            default_value='true',
            description='true = nyalakan rplidar (/scan). false = lidar sudah '
                        'jalan dari launch lain.',
        ),
        DeclareLaunchArgument(
            'lidar_x',
            default_value='0.0',
            description='LiDAR offset X dari base_link (meter).',
        ),
        DeclareLaunchArgument(
            'lidar_y',
            default_value='0.0',
            description='LiDAR offset Y dari base_link (meter).',
        ),
        DeclareLaunchArgument(
            'lidar_z',
            default_value='0.25',
            description='LiDAR offset Z dari base_link (meter).',
        ),
        DeclareLaunchArgument(
            'lidar_yaw',
            default_value='3.14159',
            description='LiDAR yaw offset (radian).',
        ),
        # Nucleo base driver (UDP STM32, sub cmd_vel + pub odom/TF).
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
                'joy_topic': LaunchConfiguration('joy_topic'),
                'test_topic': LaunchConfiguration('test_topic'),
            }.items(),
        ),
        # Lidar RPLIDAR (/scan full 360 deg, param robot_bringup lidar.yaml:
        # serial_port /dev/ttyUSB0, baud 256000).
        Node(
            package='rplidar_ros',
            executable='rplidar_composition',
            name='rplidar_node',
            output='screen',
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('lidar'), "' == 'true'"])
            ),
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare('robot_bringup'),
                    'config',
                    'lidar.yaml',
                ])
            ],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_laser',
            output='screen',
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('lidar'), "' == 'true'"])
            ),
            arguments=[
                LaunchConfiguration('lidar_x'),
                LaunchConfiguration('lidar_y'),
                LaunchConfiguration('lidar_z'),
                LaunchConfiguration('lidar_yaw'),
                '0.0', '0.0',
                'base_link', 'laser_frame',
            ],
        ),
        Node(
            # REP-105: base_footprint = proyeksi base_link di tanah.
            package='tf2_ros',
            executable='static_transform_publisher',
            name='footprint_to_base',
            output='screen',
            arguments=[
                '0', '0', '0',
                '0', '0', '0',
                'base_footprint', 'base_link',
            ],
        ),
    ])
