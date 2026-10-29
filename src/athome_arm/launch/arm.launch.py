import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch Dynamixel robotic arm motion server."""
    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='/dev/ttyUSB1', description='Dynamixel USB serial device port'),
        DeclareLaunchArgument('baudrate', default_value='1000000', description='Dynamixel communication baud rate'),
        DeclareLaunchArgument(
            'motions_dir',
            default_value=PathJoinSubstitution([FindPackageShare('athome_arm'), 'motions']),
            description='Path to arm motion files directory',
        ),
        Node(
            package='athome_arm',
            executable='arm_motion_server',
            name='arm_motion_server',
            output='screen',
            parameters=[{
                'port': LaunchConfiguration('port'),
                'baudrate': LaunchConfiguration('baudrate'),
                'motions_dir': LaunchConfiguration('motions_dir'),
            }],
        ),
    ])
