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
    return LaunchDescription([
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Launch RViz visualization.',
        ),
        DeclareLaunchArgument(
            'rviz_software',
            default_value='false',
            description='Use software rendering for RViz.',
        ),
        DeclareLaunchArgument(
            'nav2',
            default_value='true',
            description='Launch Nav2 navigation stack.',
        ),
        DeclareLaunchArgument(
            'nucleo',
            default_value='true',
            description='Launch STM32 Nucleo base driver.',
        ),
        DeclareLaunchArgument(
            'stm_ip',
            default_value='192.168.0.4',
            description='STM32 IP address.',
        ),
        DeclareLaunchArgument(
            'image_stride',
            default_value='2',
            description='Publish image every N frames.',
        ),
        DeclareLaunchArgument(
            'camera_fps',
            default_value='30.0',
            description='ZED camera framerate (FPS).',
        ),
        DeclareLaunchArgument(
            'enable_yolo',
            default_value='true',
            description='Enable YOLO bag detection.',
        ),
        DeclareLaunchArgument(
            'slam',
            default_value='true',
            description='Enable online SLAM mapping.',
        ),
        DeclareLaunchArgument(
            'enable_body_tracking',
            default_value='true',
            description='Enable ZED body tracking.',
        ),
        DeclareLaunchArgument(
            'enable_person_detection',
            default_value=LaunchConfiguration('enable_body_tracking'),
            description='Enable person detection / body tracking.',
        ),
        DeclareLaunchArgument(
            'map',
            default_value=PathJoinSubstitution([
                FindPackageShare('athome_navigation'),
                'maps',
                'map.yaml',
            ]),
            description='Path to static map yaml file.',
        ),
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
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('athome_navigation'),
                    'launch',
                    'navigation.launch.py',
                ])
            ),
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('nav2'), "' == 'true'"])
            ),
            launch_arguments={
                'rviz': LaunchConfiguration('rviz'),
                'rviz_software': LaunchConfiguration('rviz_software'),
                'slam': LaunchConfiguration('slam'),
                'enable_person_detection': LaunchConfiguration('enable_person_detection'),
                'enable_body_tracking': LaunchConfiguration('enable_body_tracking'),
                'image_stride': LaunchConfiguration('image_stride'),
                'camera_fps': LaunchConfiguration('camera_fps'),
                'enable_yolo': LaunchConfiguration('enable_yolo'),
                'map': LaunchConfiguration('map'),
                'params_file': PathJoinSubstitution([
                    FindPackageShare('help_me_carry'),
                    'config',
                    'nav2_params.yaml',
                ]),
            }.items(),
        ),
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
