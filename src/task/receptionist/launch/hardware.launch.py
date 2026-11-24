from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
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
    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_CONSOLE_OUTPUT_FORMAT', '[{severity}] {message}'),
        SetEnvironmentVariable('RCUTILS_COLORIZED_OUTPUT', '1'),
        SetEnvironmentVariable('ATHOME_TASK', 'receptionist'),
        DeclareLaunchArgument(
            'nucleo',
            default_value='true',
            description='Launch STM32 Nucleo base motor driver.',
        ),
        DeclareLaunchArgument(
            'stm_ip',
            default_value='192.168.0.4',
            description='STM32 IP address.',
        ),
        DeclareLaunchArgument(
            'joy_topic',
            default_value='joy_vel',
            description='Joystick override velocity topic.',
        ),
        DeclareLaunchArgument(
            'test_topic',
            default_value='test_vel',
            description='Manual test velocity topic.',
        ),
        DeclareLaunchArgument(
            'publish_imu_yaw',
            default_value='true',
            description='Publish Razor IMU yaw to ROS topic.',
        ),
        DeclareLaunchArgument(
            'pub_yaw_topic',
            default_value='/yaw',
            description='Topic name for published IMU yaw.',
        ),
        DeclareLaunchArgument(
            'use_yaw_topic',
            default_value='false',
            description='Use external yaw topic instead of direct STM32/Razor IMU (false = use direct Razor IMU).',
        ),
        DeclareLaunchArgument(
            'yaw_topic',
            default_value='/yaw',
            description='Topic name to subscribe if external yaw is enabled (use /yaw instead of /vision/yaw_deg).',
        ),
        DeclareLaunchArgument(
            'lidar',
            default_value='true',
            description='Launch RPLiDAR node.',
        ),
        DeclareLaunchArgument(
            'lidar_x',
            default_value='0.0',
            description='LiDAR X offset from base_link (m).',
        ),
        DeclareLaunchArgument(
            'lidar_y',
            default_value='0.0',
            description='LiDAR Y offset from base_link (m).',
        ),
        DeclareLaunchArgument(
            'lidar_z',
            default_value='0.25',
            description='LiDAR Z offset from base_link (m).',
        ),
        DeclareLaunchArgument(
            'lidar_yaw',
            default_value='3.14159',
            description='LiDAR yaw offset (rad).',
        ),
        # Motor control driver (UDP to Nucleo)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('athome_base'),
                    'launch',
                    'nucleo.launch.py',
                ])
            ),
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('nucleo'), "' == 'true'"])
            ),
            launch_arguments={
                'task': 'receptionist',
                'stm_ip': LaunchConfiguration('stm_ip'),
                'joy_topic': LaunchConfiguration('joy_topic'),
                'test_topic': LaunchConfiguration('test_topic'),
                'publish_imu_yaw': LaunchConfiguration('publish_imu_yaw'),
                'pub_yaw_topic': LaunchConfiguration('pub_yaw_topic'),
                'use_yaw_topic': LaunchConfiguration('use_yaw_topic'),
                'yaw_topic': LaunchConfiguration('yaw_topic'),
            }.items(),
        ),
        # RPLiDAR node
        Node(
            package='rplidar_ros',
            executable='rplidar_composition',
            name='rplidar_node',
            output='screen',
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('lidar'), "' == 'true'"])
            ),
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare('athome_bringup'),
                    'config',
                    'lidar.yaml',
                ])
            ],
        ),
        # TF: base_link -> laser_frame
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_laser',
            output='screen',
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('lidar'), "' == 'true'"])
            ),
            arguments=[
                LaunchConfiguration('lidar_x'),
                LaunchConfiguration('lidar_y'),
                LaunchConfiguration('lidar_z'),
                LaunchConfiguration('lidar_yaw'),
                '0.0', '0.0',
                'base_link', 'laser_frame',
            ],
        ),
        # TF: base_footprint -> base_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='footprint_to_base',
            output='screen',
            arguments=[
                '0', '0', '0',
                '0', '0', '0',
                'base_footprint', 'base_link',
            ],
        ),
    ])
