import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_receptionist = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    default_params_file = os.path.join(pkg_receptionist, 'config', 'receptionist_params.yaml')

    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_CONSOLE_OUTPUT_FORMAT', '[{severity}] {message}'),
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file,
            description='Path to ROS 2 parameters YAML file for receptionist (contains vision node config)',
        ),
        Node(
            package='athome_vision',
            executable='vision',
            name='vision',
            output='screen',
            respawn=True,
            respawn_delay=2.0,
            parameters=[LaunchConfiguration('params_file')],
        ),
    ])
