import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='help_me_carry',
            executable='help_me_carry_node',
            name='help_me_carry_node',
            output='screen',
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare('help_me_carry'),
                    'config',
                    'params.yaml',
                ])
            ],
        ),
    ])
