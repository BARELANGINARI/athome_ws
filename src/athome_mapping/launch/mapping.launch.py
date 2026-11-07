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

    # Launch arguments
    launch_hardware_arg = DeclareLaunchArgument(
        'launch_hardware',
        default_value='true',
        description='Launch low-level base driver (nucleo) and RPLiDAR sensor',
    )
    stm_ip_arg = DeclareLaunchArgument(
        'stm_ip',
        default_value='192.168.0.4',
        description='STM32 Nucleo IP address',
    )
    lidar_port_arg = DeclareLaunchArgument(
        'lidar_port',
        default_value='/dev/ttyUSB1',
        description='RPLiDAR serial port device',
    )
    lidar_x_arg = DeclareLaunchArgument(
        'lidar_x',
        default_value='0.0',
        description='LiDAR X offset from base_link (m)',
    )
    lidar_y_arg = DeclareLaunchArgument(
        'lidar_y',
        default_value='0.0',
        description='LiDAR Y offset from base_link (m)',
    )
    lidar_z_arg = DeclareLaunchArgument(
        'lidar_z',
        default_value='0.25',
        description='LiDAR Z offset from base_link (m)',
    )
    lidar_yaw_arg = DeclareLaunchArgument(
        'lidar_yaw',
        default_value='3.14159',
        description='LiDAR yaw offset from base_link (rad)',
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

    nucleo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('athome_base'),
                'launch',
                'nucleo.launch.py',
            ])
        ),
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('launch_hardware'), "' == 'true'"])
        ),
        launch_arguments={
            'stm_ip': LaunchConfiguration('stm_ip'),
        }.items(),
    )

    rplidar_node = Node(
        package='rplidar_ros',
        executable='rplidar_composition',
        name='rplidar_node',
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('launch_hardware'), "' == 'true'"])
        ),
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('athome_bringup'),
                'config',
                'lidar.yaml',
            ]),
            {
                'serial_port': LaunchConfiguration('lidar_port'),
            },
        ],
    )

    footprint_to_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='footprint_to_base',
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('launch_hardware'), "' == 'true'"])
        ),
        arguments=[
            '--frame-id', 'base_footprint',
            '--child-frame-id', 'base_link',
        ],
    )

    base_to_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_laser',
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('launch_hardware'), "' == 'true'"])
        ),
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
        launch_hardware_arg,
        stm_ip_arg,
        lidar_port_arg,
        lidar_x_arg,
        lidar_y_arg,
        lidar_z_arg,
        lidar_yaw_arg,
        slam_params_arg,
        rviz_arg,
        rviz_software_arg,
        nucleo_launch,
        rplidar_node,
        footprint_to_base,
        base_to_laser,
        slam_toolbox,
        rviz,
        rviz_software,
    ])
