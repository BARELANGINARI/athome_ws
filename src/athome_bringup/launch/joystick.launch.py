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
    """Launch joystick teleop and STM32 Nucleo motor/lifter hardware driver."""
    return LaunchDescription([
        DeclareLaunchArgument('launch_nucleo', default_value='true', description='Launch STM32 Nucleo hardware base and lifter driver'),
        DeclareLaunchArgument('stm_ip', default_value='192.168.0.4', description='STM32 Nucleo IP address'),
        DeclareLaunchArgument('stm_port', default_value='2002', description='STM32 Nucleo UDP port'),
        DeclareLaunchArgument('use_yaw_topic', default_value='false', description='Use /zed/yaw_deg instead of internal IMU'),
        DeclareLaunchArgument('smoothing_enable', default_value='true', description='Enable motor acceleration smoothing'),
        DeclareLaunchArgument('joy_id', default_value='0', description='Joystick device ID (e.g. 0 for /dev/input/js0)'),
        DeclareLaunchArgument('cmd_vel_topic', default_value='/cmd_vel', description='Output velocity topic'),
        DeclareLaunchArgument('lifter_topic', default_value='lifter_target', description='Output lifter target topic (0: down, 1: up)'),
        DeclareLaunchArgument('autorepeat_rate', default_value='50.0', description='Joystick publish rate (Hz)'),
        DeclareLaunchArgument('joy_timeout', default_value='0.2', description='Joystick timeout in seconds'),

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
