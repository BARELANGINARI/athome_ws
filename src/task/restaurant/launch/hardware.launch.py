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
        # Base Motor Driver (STM32 Nucleo) Arguments
        DeclareLaunchArgument(
            'nucleo',
            default_value='true',
            description='Launch STM32 Nucleo base motor driver'
        ),
        DeclareLaunchArgument(
            'stm_ip',
            default_value='192.168.0.4',
            description='STM32 IP address'
        ),
        DeclareLaunchArgument(
            'joy_topic',
            default_value='joy_vel',
            description='Joystick override velocity topic'
        ),
        DeclareLaunchArgument(
            'test_topic',
            default_value='test_vel',
            description='Manual test velocity topic'
        ),
        DeclareLaunchArgument(
            'publish_imu_yaw',
            default_value='true',
            description='Publish Razor IMU yaw to ROS topic'
        ),
        DeclareLaunchArgument(
            'pub_yaw_topic',
            default_value='/yaw',
            description='Topic name for published IMU yaw'
        ),
        # Menggunakan data yaw dari vision (sebelumnya):
        # DeclareLaunchArgument('use_yaw_topic', default_value='true', description='Use external /vision/yaw_deg instead of STM32 yaw'),
        # DeclareLaunchArgument('yaw_topic', default_value='/vision/yaw_deg', description='External yaw topic name'),
        # Menggunakan data dari Razor IMU langsung (STM32) / topic /yaw:
        DeclareLaunchArgument(
            'use_yaw_topic',
            default_value='false',
            description='false = direct Razor IMU from STM32'
        ),
        DeclareLaunchArgument(
            'yaw_topic',
            default_value='/yaw',
            description='Topic name for yaw (default: /yaw)'
        ),

        # RPLiDAR Arguments
        DeclareLaunchArgument(
            'lidar',
            default_value='true',
            description='Launch RPLiDAR sensor node'
        ),
        DeclareLaunchArgument(
            'lidar_port',
            default_value='/dev/ttyUSB1',
            description='RPLiDAR serial port device'
        ),
        DeclareLaunchArgument(
            'lidar_x',
            default_value='0.0',
            description='LiDAR X offset from base_link (m)'
        ),
        DeclareLaunchArgument(
            'lidar_y',
            default_value='0.0',
            description='LiDAR Y offset from base_link (m)'
        ),
        DeclareLaunchArgument(
            'lidar_z',
            default_value='0.25',
            description='LiDAR Z offset from base_link (m)'
        ),
        DeclareLaunchArgument(
            'lidar_yaw',
            default_value='3.14159',
            description='LiDAR yaw offset from base_link (rad)'
        ),

        # 1. Base Controller Driver
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
                'stm_ip': LaunchConfiguration('stm_ip'),
                'joy_topic': LaunchConfiguration('joy_topic'),
                'test_topic': LaunchConfiguration('test_topic'),
                'publish_imu_yaw': LaunchConfiguration('publish_imu_yaw'),
                'pub_yaw_topic': LaunchConfiguration('pub_yaw_topic'),
                'use_yaw_topic': LaunchConfiguration('use_yaw_topic'),
                'yaw_topic': LaunchConfiguration('yaw_topic'),
            }.items(),
        ),

        # 2. RPLiDAR Sensor Node
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
                ]),
                {
                    'serial_port': LaunchConfiguration('lidar_port'),
                },
            ],
        ),

        # 3. Static TF: base_link -> laser_frame
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_laser',
            output='screen',
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('lidar'), "' == 'true'"])
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
        ),

        # 4. Static TF: base_footprint -> base_link
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
    ])
