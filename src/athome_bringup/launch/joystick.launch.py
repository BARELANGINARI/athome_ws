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
    """Launch gabungan: Joystick teleop + driver base motor & lifter (STM32 Nucleo)."""
    return LaunchDescription([
        # Nucleo / Base Driver Options
        DeclareLaunchArgument(
            'launch_nucleo',
            default_value='true',
            description='Jalankan driver hardware base motor dan lifter (athome_base/nucleo)',
        ),
        DeclareLaunchArgument(
            'stm_ip',
            default_value='192.168.0.4',
            description='IP address STM32 Nucleo',
        ),
        DeclareLaunchArgument(
            'stm_port',
            default_value='2002',
            description='Port UDP STM32',
        ),
        DeclareLaunchArgument(
            'use_yaw_topic',
            default_value='false',
            description='Gunakan /zed/yaw_deg jika true; false untuk pakai IMU internal STM32',
        ),
        DeclareLaunchArgument(
            'smoothing_enable',
            default_value='true',
            description='Aktifkan perhalus akselerasi gerakan motor',
        ),

        # Joystick Parameters
        DeclareLaunchArgument(
            'joy_id',
            default_value='0',
            description='Joystick device ID (misal: 0 untuk /dev/input/js0)',
        ),
        DeclareLaunchArgument(
            'cmd_vel_topic',
            default_value='/cmd_vel',
            description='Topic velocity motor',
        ),
        DeclareLaunchArgument(
            'lifter_topic',
            default_value='lifter_target',
            description='Topic perintah posisi lifter (0: bawah/turun, 1: atas/naik)',
        ),
        DeclareLaunchArgument(
            'autorepeat_rate',
            default_value='50.0',
            description='Frekuensi publish joystick (Hz)',
        ),
        DeclareLaunchArgument(
            'joy_timeout',
            default_value='0.2',
            description='Timeout joystick (detik)',
        ),

        # 1. Driver Base Motor & Lifter (STM32 Nucleo via UDP)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('athome_base'),
                    'launch',
                    'nucleo.launch.py',
                ])
            ),
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('launch_nucleo'), "' == 'true'"])
            ),
            launch_arguments={
                'stm_ip': LaunchConfiguration('stm_ip'),
                'stm_port': LaunchConfiguration('stm_port'),
                'use_yaw_topic': LaunchConfiguration('use_yaw_topic'),
                'smoothing_enable': LaunchConfiguration('smoothing_enable'),
                'lifter_cmd_topic': LaunchConfiguration('lifter_topic'),
            }.items(),
        ),

        # 2. Static TF: base_footprint -> base_link (diperlukan odometri dasar)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='footprint_to_base',
            output='screen',
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('launch_nucleo'), "' == 'true'"])
            ),
            arguments=[
                '--frame-id', 'base_footprint',
                '--child-frame-id', 'base_link',
            ],
        ),

        # 3. Joystick Node (Membaca controller USB/Bluetooth)
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen',
            parameters=[{
                'device_id': LaunchConfiguration('joy_id'),
                'deadzone': 0.05,
                'autorepeat_rate': LaunchConfiguration('autorepeat_rate'),
                'sticky_buttons': False,
                'coalesce_interval_ms': 1,
            }],
        ),

        # 4. DS4 Drive Teleop Node (Mengontrol motor & lifter)
        Node(
            package='athome_bringup',
            executable='ds4_drive',
            name='ds4_drive',
            output='screen',
            parameters=[{
                'joy_timeout': LaunchConfiguration('joy_timeout'),
                'lifter_topic': LaunchConfiguration('lifter_topic'),
            }],
            remappings=[
                ('cmd_vel', LaunchConfiguration('cmd_vel_topic')),
            ],
        ),
    ])
