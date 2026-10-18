"""Navigasi berbasis PETA STATIS (AMCL + Nav2 + RViz).

Prasyarat:
    Jalankan base driver Nucleo di terminal terpisah:
        ros2 launch nucleo nucleo.launch.py

Lalu jalankan navigasi di terminal ini:
    ros2 launch navigation navigation.launch.py

Robot melokalisasi diri di peta hasil mapping (maps/map.yaml) dengan AMCL,
lalu Nav2 menerima goal dari RViz (tool "2D Goal Pose") atau dari service /goto/<nama>.
Untuk MEMBUAT peta baru, gunakan paket `mapping` (slam_toolbox), bukan ini.

Alur TF lengkap:
    map --(AMCL)--> odom --(nucleo)--> base_footprint --(static)--> base_link
        --(static)--> laser_frame

Argumen penting:
    use_sensors:=true/false (default true: menyalakan lidar + zed via robot_bringup)
    map:=<path> (default maps/map.yaml)
    initial_x/initial_y/initial_yaw (pose awal robot di peta)
"""
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
    """Resolusi path peta (.yaml) secara robust dengan fallback ke source tree."""
    if map_path and os.path.isabs(map_path) and os.path.exists(map_path):
        return os.path.abspath(map_path)

    candidates = []
    if map_path:
        candidates.append(os.path.abspath(map_path))
        candidates.append(os.path.join(navigation_share, map_path))
        candidates.append(os.path.join(navigation_share, 'maps', os.path.basename(map_path)))

    # Fallback ke folder source tree jika launch dijalankan sebelum/saat build symlink
    src_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if map_path:
        candidates.append(os.path.join(src_dir, 'maps', os.path.basename(map_path)))
    candidates.append(os.path.join(src_dir, 'maps', 'map.yaml'))
    candidates.append(os.path.join(navigation_share, 'maps', 'map.yaml'))

    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)

    return os.path.abspath(map_path) if map_path else os.path.join(navigation_share, 'maps', 'map.yaml')


def _resolve_params_file(params_path, navigation_share):
    """Resolusi path nav2_params.yaml dengan fallback ke source tree."""
    if params_path and os.path.isabs(params_path) and os.path.exists(params_path):
        return os.path.abspath(params_path)

    src_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    candidates = [
        os.path.abspath(params_path) if params_path else '',
        os.path.join(navigation_share, 'config', 'nav2_params.yaml'),
        os.path.join(src_dir, 'config', 'nav2_params.yaml'),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)

    return os.path.abspath(params_path) if params_path else os.path.join(navigation_share, 'config', 'nav2_params.yaml')


def _prepare_params(params_file, map_file, x, y, yaw):
    """Suntikkan initial pose AMCL + path peta ke salinan params Nav2.

    Dua hal penting yang HARUS benar agar peta tampil & navigasi jalan:

    1. map_server.yaml_filename = path peta. Nilai ini ditulis di section
       node-specific `map_server:` sehingga tidak bisa tertimpa default kosong.
       (Kalau dibiarkan kosong, map_server memuat peta kosong -> frame `map`
       tidak pernah ada, AMCL "Waiting for map", RViz "Fixed Frame map does
       not exist".)
    2. amcl.set_initial_pose = true + initial_pose. Tanpa initial pose, AMCL
       tidak pernah mem-publish TF map->odom sehingga frame `map` tetap tidak
       ada meski /map sudah terbit.
    """
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
    navigation_share = get_package_share_directory('navigation')
    nav2_bringup_share = get_package_share_directory('nav2_bringup')

    rviz_config = os.path.join(navigation_share, 'rviz', 'navigation.rviz')
    nav2_bringup_launch = os.path.join(
        nav2_bringup_share, 'launch', 'bringup_launch.py')

    raw_map_file = context.perform_substitution(LaunchConfiguration('map'))
    raw_params_file = context.perform_substitution(LaunchConfiguration('params_file'))
    initial_x = context.perform_substitution(LaunchConfiguration('initial_x'))
    initial_y = context.perform_substitution(LaunchConfiguration('initial_y'))
    initial_yaw = context.perform_substitution(LaunchConfiguration('initial_yaw'))

    map_file = _resolve_map_file(raw_map_file, navigation_share)
    params_file = _resolve_params_file(raw_params_file, navigation_share)

    if not os.path.exists(map_file):
        print(f"\033[91m[ERROR] [navigation.launch]: File peta tidak ditemukan di '{map_file}'!\033[0m")
    else:
        print(f"\033[92m[INFO] [navigation.launch]: Menggunakan peta: {map_file}\033[0m")

    prepared_params = _prepare_params(
        params_file, map_file, initial_x, initial_y, initial_yaw)

    # Sensors: lidar /scan + ZED + static TF (base_footprint/base_link/laser).
    sensors_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('robot_bringup'),
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

    # Standalone TF jika use_sensors:=false (agar rantai TF tetap utuh).
    footprint_to_base_standalone = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='footprint_to_base_standalone',
        output='screen',
        arguments=[
            '--frame-id', 'base_footprint',
            '--child-frame-id', 'base_link',
        ],
        condition=UnlessCondition(LaunchConfiguration('use_sensors')),
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
        condition=UnlessCondition(LaunchConfiguration('use_sensors')),
    )

    # Nav2 dengan peta statis: map_server + AMCL + seluruh stack navigasi.
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_bringup_launch),
        launch_arguments={
            'slam': 'False',
            'map': map_file,
            'params_file': prepared_params,
            'use_sim_time': 'false',
            'autostart': 'true',
            'use_composition': LaunchConfiguration('use_composition'),
        }.items(),
    )

    # RViz (render biasa).
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

    # RViz (software rendering untuk mesin tanpa GPU).
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

    goal_bridge = Node(
        package='navigation',
        executable='goal_pose_bridge.py',
        name='goal_pose_bridge',
        output='screen',
    )

    # Service /goto/<nama> per lokasi di config/places.yaml. Program lain
    # (hri, C++, bash) memanggilnya untuk menyuruh robot ke koordinat itu.
    place_navigator = Node(
        package='navigation',
        executable='place_navigator.py',
        name='place_navigator',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_places')),
        parameters=[{
            'places_file': LaunchConfiguration('places_file'),
            'timeout_sec': LaunchConfiguration('place_timeout_sec'),
            'frame_id': LaunchConfiguration('place_frame_id'),
            'action_name': LaunchConfiguration('place_action_name'),
        }],
    )

    # Person following: service /follow/person + /follow/stop. Membaca pose
    # orang dari /zed/person_keypoints lalu mengirim goal BERGERAK ke Nav2
    # lewat behavior tree khusus (behavior_trees/follow_person.xml).
    person_follower = Node(
        package='navigation',
        executable='person_follower.py',
        name='person_follower',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_following')),
        parameters=[
            os.path.join(navigation_share, 'config', 'person_follow.yaml'),
            # Port BT tidak bisa di-set dari YAML, jadi path-nya dikirim lewat
            # node: field `behavior_tree` pada action NavigateToPose.
            {'follow_bt_xml': os.path.join(
                navigation_share, 'behavior_trees', 'follow_person.xml')},
        ],
    )

    return [
        sensors_launch,
        footprint_to_base_standalone,
        base_to_laser_standalone,
        nav2_launch,
        goal_bridge,
        place_navigator,
        person_follower,
        rviz,
        rviz_software,
    ]


def generate_launch_description():
    navigation_share = get_package_share_directory('navigation')

    default_map = os.path.join(navigation_share, 'maps', 'map.yaml')
    default_params = os.path.join(navigation_share, 'config', 'nav2_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=default_map,
            description='Path peta (.yaml) hasil mapping. Default maps/map.yaml.',
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='File parameter Nav2 (config/nav2_params.yaml).',
        ),
        DeclareLaunchArgument(
            'use_sensors',
            default_value='true',
            description='Aktifkan sensor lidar & kamera ZED via robot_bringup/sensors.launch.py.',
        ),
        DeclareLaunchArgument(
            'initial_x',
            default_value='0.0',
            description='Pose awal robot di peta (x meter, frame map).',
        ),
        DeclareLaunchArgument(
            'initial_y',
            default_value='0.0',
            description='Pose awal robot di peta (y meter, frame map).',
        ),
        DeclareLaunchArgument(
            'initial_yaw',
            default_value='0.0',
            description='Heading awal robot (radian, 0 = +X peta).',
        ),
        DeclareLaunchArgument(
            'use_composition',
            default_value='False',
            description='False = tiap node Nav2 proses sendiri (aman).',
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Tampilkan RViz.',
        ),
        DeclareLaunchArgument(
            'rviz_software',
            default_value='false',
            description='Paksa RViz software rendering.',
        ),
        # Navigasi lokasi bernama: service /goto/<nama>.
        DeclareLaunchArgument(
            'use_places',
            default_value='true',
            description='Aktifkan place_navigator (service /goto/<nama>).',
        ),
        DeclareLaunchArgument(
            'places_file',
            default_value='',
            description='Kosong = config/places.yaml. Bisa override path YAML lain.',
        ),
        DeclareLaunchArgument(
            'place_timeout_sec',
            default_value='90.0',
            description='Batas tunggu robot sampai tujuan (detik).',
        ),
        DeclareLaunchArgument(
            'place_frame_id',
            default_value='map',
            description='Frame koordinat lokasi (harus frame peta global).',
        ),
        DeclareLaunchArgument(
            'place_action_name',
            default_value='navigate_to_pose',
            description='Action Nav2 untuk mengirim goal.',
        ),
        # Person following: service /follow/person + /follow/stop.
        DeclareLaunchArgument(
            'use_following',
            default_value='true',
            description=(
                'Aktifkan person_follower (dynamic object following). '
                'BUTUH zed dengan enable_body_tracking:=true, kalau tidak '
                'topik /zed/person_keypoints selalu NO_PERSON.'
            ),
        ),
        # Passthrough ke sensors/zed.
        DeclareLaunchArgument('enable_body_tracking', default_value='false'),
        DeclareLaunchArgument('depth_minimum_distance', default_value='30.0'),
        DeclareLaunchArgument('image_stride', default_value='2'),
        DeclareLaunchArgument('depth_mode', default_value='NEURAL_LIGHT'),
        DeclareLaunchArgument('body_model', default_value='FAST'),
        DeclareLaunchArgument('camera_fps', default_value='30.0'),
        DeclareLaunchArgument('enable_yolo', default_value='true'),
        DeclareLaunchArgument(
            'zed_debug_window',
            default_value='false',
            description=(
                'Preview kamera OpenCV dari node zed: bbox person/tas, 34 '
                'keypoint, jarak. Tekan q untuk keluar. Default false supaya '
                'aman di mesin tanpa display.'
            ),
        ),
        DeclareLaunchArgument(
            'zed_mirror',
            default_value='true',
            description='Balik preview zed kiri-kanan (tangan kanan = kanan layar).',
        ),
        OpaqueFunction(function=_launch_setup),
    ])
