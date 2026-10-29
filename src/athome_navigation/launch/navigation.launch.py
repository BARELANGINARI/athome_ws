import os
import tempfile

import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _resolve_map_file(map_path, navigation_share):
    """Resolve map YAML file path with fallback to source tree."""
    if map_path and os.path.isabs(map_path) and os.path.exists(map_path):
        return os.path.abspath(map_path)

    candidates = []
    if map_path:
        candidates.append(os.path.abspath(map_path))
        candidates.append(os.path.join(navigation_share, map_path))
        candidates.append(os.path.join(navigation_share, 'maps', os.path.basename(map_path)))

    src_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if map_path:
        candidates.append(os.path.join(src_dir, 'maps', os.path.basename(map_path)))
    candidates.append(os.path.join(src_dir, 'maps', 'map.yaml'))
    candidates.append(os.path.join(navigation_share, 'maps', 'map.yaml'))

    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)

    return os.path.abspath(map_path) if map_path else os.path.join(navigation_share, 'maps', 'map.yaml')


def _resolve_params_file(params_path, fallback_share):
    """Resolve Nav2 params YAML file path with fallback to task package."""
    if params_path and os.path.isabs(params_path) and os.path.exists(params_path):
        return os.path.abspath(params_path)

    candidates = [
        os.path.abspath(params_path) if params_path else '',
        os.path.join(fallback_share, 'config', 'nav2_params.yaml'),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)

    return os.path.abspath(params_path) if params_path else os.path.join(fallback_share, 'config', 'nav2_params.yaml')


def _prepare_params(params_file, map_file, x, y, yaw):
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f) or {}

    map_server = params.setdefault('map_server', {}).setdefault('ros__parameters', {})
    map_server['use_sim_time'] = False
    map_server['yaml_filename'] = map_file

    amcl = params.setdefault('amcl', {}).setdefault('ros__parameters', {})
    amcl['use_sim_time'] = False
    amcl['set_initial_pose'] = True
    amcl['initial_pose'] = {
        'x': float(x),
        'y': float(y),
        'z': 0.0,
        'yaw': float(yaw),
    }

    fd, path = tempfile.mkstemp(prefix='nav2_params_', suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.safe_dump(params, f, default_flow_style=False)
    return path


def _launch_setup(context, *args, **kwargs):
    navigation_share = get_package_share_directory('athome_navigation')
    help_me_carry_share = get_package_share_directory('help_me_carry')
    nav2_bringup_share = get_package_share_directory('nav2_bringup')

    rviz_config_arg = context.perform_substitution(LaunchConfiguration('rviz_config'))
    rviz_config = rviz_config_arg if rviz_config_arg else os.path.join(navigation_share, 'rviz', 'navigation.rviz')
    nav2_bringup_launch = os.path.join(
        nav2_bringup_share, 'launch', 'bringup_launch.py')

    raw_map_file = context.perform_substitution(LaunchConfiguration('map'))
    raw_params_file = context.perform_substitution(LaunchConfiguration('params_file'))
    initial_x = context.perform_substitution(LaunchConfiguration('initial_x'))
    initial_y = context.perform_substitution(LaunchConfiguration('initial_y'))
    initial_yaw = context.perform_substitution(LaunchConfiguration('initial_yaw'))
    slam_enabled = context.perform_substitution(LaunchConfiguration('slam')).lower() in ('true', '1')

    map_file = _resolve_map_file(raw_map_file, navigation_share)
    params_file = _resolve_params_file(raw_params_file, help_me_carry_share)

    if slam_enabled:
        print("\033[92m[INFO] [navigation.launch]: Online SLAM mode active (mapless navigation).\033[0m")
    elif not os.path.exists(map_file):
        print(f"\033[91m[ERROR] [navigation.launch]: Map file not found at '{map_file}'!\033[0m")
    else:
        print(f"\033[92m[INFO] [navigation.launch]: Using map: {map_file}\033[0m")

    prepared_params = _prepare_params(
        params_file, map_file, initial_x, initial_y, initial_yaw)

    sensors_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('athome_bringup'),
                'launch',
                'sensors.launch.py',
            ])
        ),
        condition=IfCondition(LaunchConfiguration('use_sensors')),
        launch_arguments={
            'enable_body_tracking': LaunchConfiguration('enable_body_tracking'),
            'depth_minimum_distance': LaunchConfiguration('depth_minimum_distance'),
            'image_stride': LaunchConfiguration('image_stride'),
            'depth_mode': LaunchConfiguration('depth_mode'),
            'body_model': LaunchConfiguration('body_model'),
            'camera_fps': LaunchConfiguration('camera_fps'),
            'enable_yolo': LaunchConfiguration('enable_yolo'),
            'debug_window': LaunchConfiguration('zed_debug_window'),
            'mirror': LaunchConfiguration('zed_mirror'),
        }.items(),
    )
    footprint_to_base_standalone = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='footprint_to_base_standalone',
        output='screen',
        arguments=[
            '--frame-id', 'base_footprint',
            '--child-frame-id', 'base_link',
        ],
        condition=IfCondition(
            PythonExpression([
                "'", LaunchConfiguration('use_sensors'), "' == 'false' and '",
                LaunchConfiguration('publish_static_tf'), "' == 'true'",
            ])
        ),
    )
    base_to_laser_standalone = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_laser_standalone',
        output='screen',
        arguments=[
            '--x', '0.0',
            '--y', '0.0',
            '--z', '0.3',
            '--yaw', '3.14159',
            '--pitch', '0.0',
            '--roll', '0.0',
            '--frame-id', 'base_link',
            '--child-frame-id', 'laser_frame',
        ],
        condition=IfCondition(
            PythonExpression([
                "'", LaunchConfiguration('use_sensors'), "' == 'false' and '",
                LaunchConfiguration('publish_static_tf'), "' == 'true'",
            ])
        ),
    )
    use_composition_val = 'True' if context.perform_substitution(
        LaunchConfiguration('use_composition')).lower() in ('true', '1') else 'False'

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_bringup_launch),
        launch_arguments={
            'slam': 'True' if slam_enabled else 'False',
            'map': map_file,
            'params_file': prepared_params,
            'use_sim_time': 'false',
            'autostart': 'true',
            'use_composition': use_composition_val,
        }.items(),
    )
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        condition=IfCondition(
            PythonExpression([
                "'", LaunchConfiguration('rviz'), "' == 'true' and '",
                LaunchConfiguration('rviz_software'), "' == 'false'",
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
                "'", LaunchConfiguration('rviz'), "' == 'true' and '",
                LaunchConfiguration('rviz_software'), "' == 'true'",
            ])
        ),
        arguments=['-d', rviz_config],
        additional_env={'LIBGL_ALWAYS_SOFTWARE': '1'},
    )
    carry_params = os.path.join(help_me_carry_share, 'config', 'params.yaml')

    carry_node = Node(
        package='help_me_carry',
        executable='help_me_carry_node',
        name='help_me_carry_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_following')),
        parameters=[
            carry_params,
            {'follow_bt_xml': os.path.join(
                help_me_carry_share, 'behavior_trees', 'follow_person.xml')},
        ],
    )

    return [
        sensors_launch,
        footprint_to_base_standalone,
        base_to_laser_standalone,
        nav2_launch,
        carry_node,
        rviz,
        rviz_software,
    ]


def generate_launch_description():
    navigation_share = get_package_share_directory('athome_navigation')
    help_me_carry_share = get_package_share_directory('help_me_carry')

    default_map = os.path.join(navigation_share, 'maps', 'map.yaml')
    default_params = os.path.join(help_me_carry_share, 'config', 'nav2_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('slam', default_value='false', description='Enable real-time SLAM (slam_toolbox)'),
        DeclareLaunchArgument('map', default_value=default_map, description='Path to map YAML file'),
        DeclareLaunchArgument('params_file', default_value=default_params, description='Path to Nav2 params file'),
        DeclareLaunchArgument('use_sensors', default_value='true', description='Launch LiDAR and ZED sensors'),
        DeclareLaunchArgument('publish_static_tf', default_value='true', description='Publish standalone static TF when use_sensors is false'),
        DeclareLaunchArgument('initial_x', default_value='0.0', description='Initial robot X pose on map (m)'),
        DeclareLaunchArgument('initial_y', default_value='0.0', description='Initial robot Y pose on map (m)'),
        DeclareLaunchArgument('initial_yaw', default_value='0.0', description='Initial robot heading on map (rad)'),
        DeclareLaunchArgument('use_composition', default_value='False', description='Use Nav2 composable nodes'),
        DeclareLaunchArgument('rviz', default_value='true', description='Launch RViz2 visualization'),
        DeclareLaunchArgument('rviz_software', default_value='false', description='Force RViz software OpenGL rendering'),
        DeclareLaunchArgument('rviz_config', default_value=default_map.replace('maps/map.yaml', 'rviz/navigation.rviz'), description='Path to RViz configuration file'),
        DeclareLaunchArgument('use_following', default_value='true', description='Launch person following node'),
        DeclareLaunchArgument('enable_person_detection', default_value='true', description='Enable person detection (body tracking alias)'),
        DeclareLaunchArgument('enable_body_tracking', default_value=LaunchConfiguration('enable_person_detection'), description='Enable ZED body tracking'),
        DeclareLaunchArgument('depth_minimum_distance', default_value='30.0', description='Minimum depth sensing distance (cm)'),
        DeclareLaunchArgument('image_stride', default_value='2', description='Publish image every N frames'),
        DeclareLaunchArgument('depth_mode', default_value='PERFORMANCE', description='ZED depth sensing mode'),
        DeclareLaunchArgument('body_model', default_value='FAST', description='ZED body tracking model'),
        DeclareLaunchArgument('camera_fps', default_value='30.0', description='Camera frame rate (FPS)'),
        DeclareLaunchArgument('enable_yolo', default_value='true', description='Enable YOLO object detection'),
        DeclareLaunchArgument('zed_debug_window', default_value='false', description='Show OpenCV preview window'),
        DeclareLaunchArgument('zed_mirror', default_value='true', description='Mirror display preview horizontally'),
        OpaqueFunction(function=_launch_setup),
    ])
