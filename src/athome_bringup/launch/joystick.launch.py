
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'joy_id',
            default_value='0',
            description='Joystick device ID (e.g. 0 for js0)',
        ),
        DeclareLaunchArgument(
            'cmd_vel_topic',
            default_value='/cmd_vel',
            description='Output velocity topic',
        ),
        DeclareLaunchArgument(
            'autorepeat_rate',
            default_value='50.0',
            description='Joy publish rate (Hz)',
        ),
        DeclareLaunchArgument(
            'joy_timeout',
            default_value='0.2',
            description='Joystick command timeout (s)',
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
            }],
            remappings=[
                ('cmd_vel', LaunchConfiguration('cmd_vel_topic')),
            ],
        ),
    ])
