from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description: lidar + filter + TF + ZED yaw."""
    lidar_yaml = PathJoinSubstitution(
        [FindPackageShare('athome_bringup'), 'config', 'lidar.yaml'])

    return LaunchDescription([
        DeclareLaunchArgument('lidar_x', default_value='0.0', description='LiDAR X offset from base_link (m)'),
        DeclareLaunchArgument('lidar_y', default_value='0.0', description='LiDAR Y offset from base_link (m)'),
        DeclareLaunchArgument('lidar_z', default_value='0.3', description='LiDAR Z offset from base_link (m)'),
        DeclareLaunchArgument('lidar_yaw', default_value='3.14159', description='LiDAR yaw offset (rad)'),
        DeclareLaunchArgument('camera_fps', default_value='30.0', description='ZED camera framerate (FPS)'),
        DeclareLaunchArgument(
            'use_scan_filter',
            default_value='false',
            description='Enable 180 deg angular laser filter',
        ),
        Node(
            package='rplidar_ros',
            executable='rplidar_composition',
            name='rplidar_node',
            output='screen',
            parameters=[lidar_yaml],
            condition=UnlessCondition(LaunchConfiguration('use_scan_filter')),
        ),
        Node(
            package='rplidar_ros',
            executable='rplidar_composition',
            name='rplidar_node',
            output='screen',
            parameters=[lidar_yaml],
            remappings=[('scan', 'scan_raw')],
            condition=IfCondition(LaunchConfiguration('use_scan_filter')),
        ),
        Node(
            package='laser_filters',
            executable='scan_to_scan_filter_chain',
            name='scan_filter',
            output='screen',
            remappings=[
                ('scan', 'scan_raw'),
                ('scan_filtered', 'scan'),
            ],
            parameters=[lidar_yaml],
            condition=IfCondition(LaunchConfiguration('use_scan_filter')),
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='footprint_to_base',
            output='screen',
            arguments=[
                '--frame-id', 'base_footprint',
                '--child-frame-id', 'base_link',
            ],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_laser',
            output='screen',
            arguments=[
                '--x', LaunchConfiguration('lidar_x'),
                '--y', LaunchConfiguration('lidar_y'),
                '--z', LaunchConfiguration('lidar_z'),
                '--yaw', LaunchConfiguration('lidar_yaw'),
                '--pitch', '0.0',
                '--roll', '0.0',
                '--frame-id', 'base_link',
                '--child-frame-id', 'laser_frame',
            ],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([FindPackageShare('athome_vision'), 'launch', 'zed.launch.py'])),
            launch_arguments={
                'camera_fps': LaunchConfiguration('camera_fps'),
                'publish_annotated': 'false',
                'preview': 'false',
                'publish_yaw': 'true',
            }.items(),
        ),
    ])
