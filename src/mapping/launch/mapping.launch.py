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
    mapping_share = get_package_share_directory('mapping')
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

    # Sensors
    sensors_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('robot_bringup'),
                'launch',
                'sensors.launch.py',
            ])
        )
    )

    # SLAM Toolbox
    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(slam_toolbox_launch),
        launch_arguments={
            'use_sim_time': 'false',
            'slam_params_file': LaunchConfiguration('slam_params_file'),
        }.items(),
    )

    # RViz with default rendering
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

    # RViz with software rendering
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

# import os

# from ament_index_python.packages import get_package_share_directory
# from launch import LaunchDescription
# from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
# from launch.conditions import IfCondition
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression

# from launch_ros.actions import Node
# from launch_ros.substitutions import FindPackageShare


# def generate_launch_description():
#     """Generate launch description: sensors + SLAM backend + RViz."""
#     pkg_share = get_package_share_directory('mapping')
#     rviz_config = os.path.join(pkg_share, 'rviz', 'mapping.rviz')
#     default_slam = os.path.join(pkg_share, 'config', 'slam.yaml')
#     slam_toolbox_launch = os.path.join(
#         get_package_share_directory('slam_toolbox'), 'launch',
#         'online_async_launch.py')
#     return LaunchDescription([
#         DeclareLaunchArgument('slam', default_value='toolbox',description='Backend SLAM: toolbox (slam_toolbox) atau hector'),
#         DeclareLaunchArgument('slam_params_file', default_value=default_slam,description='File parameter slam_toolbox'),
#         DeclareLaunchArgument('rviz', default_value='true',description='Tampilkan RViz mapping'),
#         DeclareLaunchArgument('rviz_software', default_value='false',description='Paksa RViz software rendering (tanpa GPU/display)'),
#         IncludeLaunchDescription(
#             PythonLaunchDescriptionSource(
#                 PathJoinSubstitution(
#                     [FindPackageShare('robot_bringup'), 'launch', 'sensors.launch.py'])),
#         ),
#         IncludeLaunchDescription(
#             PythonLaunchDescriptionSource(slam_toolbox_launch),
#             condition=IfCondition(PythonExpression([
#                 "'", LaunchConfiguration('slam'), "' == 'toolbox'"])),
#             launch_arguments={
#                 'use_sim_time': 'false',
#                 'slam_params_file': LaunchConfiguration('slam_params_file'),
#             }.items(),
#         ),
#         Node(
#             package='hector_mapping',
#             executable='hector_mapping',
#             name='hector_mapping',
#             output='screen',
#             condition=IfCondition(PythonExpression([
#                 "'", LaunchConfiguration('slam'), "' == 'hector'"])),
#             parameters=[{
#                 'map_frame': 'map',
#                 'odom_frame': 'odom',
#                 'base_frame': 'base_link',
#                 'scan_topic': '/scan',
#                 'use_tf_scan_transformation': True,
#                 'use_tf_pose_start_estimate': False,
#             }],
#         ),
#         Node(
#             package='rviz2',
#             executable='rviz2',
#             name='rviz2',
#             output='screen',
#             condition=IfCondition(PythonExpression([
#                 "'", LaunchConfiguration('rviz'), "' == 'true' and '",
#                 LaunchConfiguration('rviz_software'), "' == 'false'"])),
#             arguments=['-d', rviz_config],
#         ),
#         Node(
#             package='rviz2',
#             executable='rviz2',
#             name='rviz2_sw',
#             output='screen',
#             condition=IfCondition(PythonExpression([
#                 "'", LaunchConfiguration('rviz'), "' == 'true' and '",
#                 LaunchConfiguration('rviz_software'), "' == 'true'"])),
#             arguments=['-d', rviz_config],
#             additional_env={'LIBGL_ALWAYS_SOFTWARE': '1'},
#         ),
#     ])
