import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch Barelang HRI speech and interaction node with YAML parameters."""
    pkg_dir = get_package_share_directory("athome_hri")
    default_params_file = os.path.join(pkg_dir, "config", "params.yaml")

    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_CONSOLE_OUTPUT_FORMAT', '[{severity}] {message}'),
        SetEnvironmentVariable('RCUTILS_COLORIZED_OUTPUT', '1'),
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params_file,
            description="Full path to the ROS 2 parameters YAML file for athome_hri",
        ),
        Node(
            package="athome_hri",
            executable="help_me_carry_hri",
            name="barelang_hri_node",
            output="screen",
            emulate_tty=True,
            parameters=[LaunchConfiguration("params_file")],
        ),
    ])
