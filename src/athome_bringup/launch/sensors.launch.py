from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch LiDAR (RPLiDAR), laser filters, static TF, and ZED camera."""
    lidar_yaml = PathJoinSubstitution(
        [FindPackageShare('athome_bringup'), 'config', 'lidar.yaml'])

    return LaunchDescription([
        DeclareLaunchArgument('lidar_x', default_value='0.0', description='LiDAR X offset from base_link (m)'),
        DeclareLaunchArgument('lidar_y', default_value='0.0', description='LiDAR Y offset from base_link (m)'),
        DeclareLaunchArgument('lidar_z', default_value='0.3', description='LiDAR Z offset from base_link (m)'),
        DeclareLaunchArgument('lidar_yaw', default_value='3.14159', description='LiDAR yaw offset from base_link (rad)'),
        DeclareLaunchArgument('camera_fps', default_value='30.0', description='ZED camera frame rate (FPS)'),
        DeclareLaunchArgument('use_scan_filter', default_value='false', description='Enable laser scan angular filter'),
        DeclareLaunchArgument('enable_body_tracking', default_value='true', description='Enable ZED body tracking'),
        DeclareLaunchArgument('depth_minimum_distance', default_value='30.0', description='Minimum depth sensing distance (cm)'),
        DeclareLaunchArgument('image_stride', default_value='2', description='Publish preview image every N frames'),
        DeclareLaunchArgument('enable_yolo', default_value='true', description='Enable YOLO object detection'),
        DeclareLaunchArgument('debug_window', default_value='false', description='Show OpenCV preview window'),
        DeclareLaunchArgument('mirror', default_value='true', description='Mirror display preview horizontally'),
        DeclareLaunchArgument('yaw_360', default_value='true', description='Publish IMU yaw in 0-360 range (default: true)'),

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
                'publish_yaw': 'true',
                'yaw_360': LaunchConfiguration('yaw_360'),
                'enable_body_tracking': LaunchConfiguration('enable_body_tracking'),
                'depth_minimum_distance': LaunchConfiguration('depth_minimum_distance'),
                'image_stride': LaunchConfiguration('image_stride'),
                'enable_yolo': LaunchConfiguration('enable_yolo'),
                'debug_window': LaunchConfiguration('debug_window'),
                'mirror': LaunchConfiguration('mirror'),
            }.items(),
        ),
    ])
