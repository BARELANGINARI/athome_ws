import os

from ament_index_python.packages import get_package_share_directory

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
    mapping_share = get_package_share_directory('athome_mapping')
    slam_toolbox_share = get_package_share_directory('slam_toolbox')

    rviz_config = os.path.join(
        mapping_share,
        'rviz',
        'mapping.rviz',
    )
    default_slam_params = os.path.join(
        mapping_share,
        'config',
        'slam.yaml',
    )
    slam_toolbox_launch = os.path.join(
        slam_toolbox_share,
        'launch',
        'online_async_launch.py',
    )
    slam_params_arg = DeclareLaunchArgument(
        'slam_params_file',
        default_value=default_slam_params,
        description='Path to the SLAM Toolbox parameter file',
    )
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Enable RViz visualization',
    )
    rviz_software_arg = DeclareLaunchArgument(
        'rviz_software',
        default_value='false',
        description='Force RViz software rendering',
    )
    sensors_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('athome_bringup'),
                'launch',
                'sensors.launch.py',
            ])
        )
    )
    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(slam_toolbox_launch),
        launch_arguments={
            'use_sim_time': 'false',
            'slam_params_file': LaunchConfiguration('slam_params_file'),
        }.items(),
    )
    rviz = Node(
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
        arguments=['-d', rviz_config],
    )
    rviz_software = Node(
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
        arguments=['-d', rviz_config],
        additional_env={
            'LIBGL_ALWAYS_SOFTWARE': '1',
        },
    )

    return LaunchDescription([
        slam_params_arg,
        rviz_arg,
        rviz_software_arg,
        sensors_launch,
        slam_toolbox,
        rviz,
        rviz_software,
    ])