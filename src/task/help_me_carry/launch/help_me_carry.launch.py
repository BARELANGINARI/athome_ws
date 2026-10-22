from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Tampilkan RViz di navigation (aktif saat nav2:=true)',
        ),
        DeclareLaunchArgument(
            'rviz_software',
            default_value='false',
            description='Paksa RViz software rendering (tanpa GPU)',
        ),
        DeclareLaunchArgument(
            'nav2',
            default_value='false',
            description='true = ikutkan navigation.launch.py (Nav2 + sensor, '
                        'dibutuhkan state FOLLOW).',
        ),
        DeclareLaunchArgument(
            'nucleo',
            default_value='true',
            description='true = ikutkan nucleo.launch.py (base driver STM32, '
                        'sub cmd_vel). Minimal: nucleo + task aja.',
        ),
        DeclareLaunchArgument(
            'stm_ip',
            default_value='192.168.0.4',
            description='Diteruskan ke nucleo.launch.py bila nucleo:=true.',
        ),
        DeclareLaunchArgument(
            'depth_minimum_distance',
            default_value='30.0',
            description='Diteruskan ke navigation/sensors/zed bila nav2:=true '
                        '(cm ala maen_boss, 30 = 30cm).',
        ),
        DeclareLaunchArgument(
            'image_stride',
            default_value='2',
            description='Diteruskan ke zed bila nav2:=true '
                        '(publish /zed/image tiap N frame, kecilkan delay).',
        ),
        DeclareLaunchArgument(
            'depth_mode',
            default_value='ULTRA',
            description='Diteruskan ke zed bila nav2:=true '
                        '(ULTRA akurat ala maen_boss).',
        ),
        DeclareLaunchArgument(
            'body_model',
            default_value='MEDIUM',
            description='Diteruskan ke zed bila nav2:=true (MEDIUM akurat / FAST halus).',
        ),
        DeclareLaunchArgument(
            'camera_fps',
            default_value='30.0',
            description='Diteruskan ke zed bila nav2:=true (30 aman; turunkan ke 15 '
                        'bila patah-patah).',
        ),
        DeclareLaunchArgument(
            'enable_yolo',
            default_value='true',
            description='Diteruskan ke zed bila nav2:=true (YOLO tas pretrained).',
        ),
        DeclareLaunchArgument(
            'slam',
            default_value='true',
            description='Diteruskan ke navigation.launch.py bila nav2:=true.',
        ),
        DeclareLaunchArgument(
            'enable_body_tracking',
            default_value='true',
            description='Diteruskan ke navigation.launch.py bila nav2:=true: '
                        'body tracking ZED (task butuh person).',
        ),
        DeclareLaunchArgument(
            'enable_person_detection',
            default_value=LaunchConfiguration('enable_body_tracking'),
            description='Diteruskan ke navigation.launch.py bila nav2:=true: '
                        'deteksi orang / body tracking ZED (default true).',
        ),
        DeclareLaunchArgument(
            'map',
            default_value=PathJoinSubstitution([
                FindPackageShare('athome_navigation'),
                'maps',
                'map.yaml',
            ]),
            description='Diteruskan ke navigation.launch.py bila nav2:=true: '
                        'peta statis (dipakai bila slam:=false).',
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
                'depth_minimum_distance': LaunchConfiguration('depth_minimum_distance'),
                'image_stride': LaunchConfiguration('image_stride'),
                'depth_mode': LaunchConfiguration('depth_mode'),
                'body_model': LaunchConfiguration('body_model'),
                'camera_fps': LaunchConfiguration('camera_fps'),
                'enable_yolo': LaunchConfiguration('enable_yolo'),
                'map': LaunchConfiguration('map'),
            }.items(),
        ),
    ])
