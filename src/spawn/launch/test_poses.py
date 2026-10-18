"""Satu perintah uji poses.yaml: tabel validasi + publish semua preset berurutan.

  ros2 launch spawn test_poses.py                      # uji lengkap (default)
  ros2 launch spawn test_poses.py publish:=false       # tabel validasi saja
  ros2 launch spawn test_poses.py hold_sec:=1.0        # tiap preset 1 detik
  ros2 launch spawn test_poses.py poses_file:=/path/poses.yaml

Yang dilihat:
  - terminal: tabel STATUS/NAMA/X/Y/YAW/CATATAN per preset (OK/WARN/FAIL),
    dihitung dari config/poses.yaml + peta src/navigation/maps/map.yaml
    (cek preset tidak jatuh di dinding / di luar peta).
  - RViz (bila navigation + AMCL jalan): robot "loncat" antar preset.
  - terminal lain: ros2 topic echo /initialpose
Berhenti sendiri setelah preset terakhir (tidak perlu Ctrl-C).
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _share(pkg):
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory(pkg)
    except Exception:
        return ''


def _default_poses_file():
    share = _share('spawn')
    if share:
        p = os.path.join(share, 'config', 'poses.yaml')
        if os.path.isfile(p):
            return p
    # __file__ symlink-install -> resolve() tembus ke src tree.
    pkg_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return os.path.join(pkg_dir, 'config', 'poses.yaml')


def _default_map_file():
    share = _share('navigation')
    if share:
        p = os.path.join(share, 'maps', 'map.yaml')
        if os.path.isfile(p):
            return p
    pkg_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    cand = os.path.normpath(os.path.join(pkg_dir, '..', 'navigation', 'maps', 'map.yaml'))
    return cand if os.path.isfile(cand) else ''


def _launch(context):
    poses_file = context.perform_substitution(LaunchConfiguration('poses_file')) or _default_poses_file()
    map_file = context.perform_substitution(LaunchConfiguration('map_file')) or _default_map_file()

    hold = float(context.perform_substitution(LaunchConfiguration('hold_sec')))
    period = float(context.perform_substitution(LaunchConfiguration('period_sec')))
    publish = context.perform_substitution(LaunchConfiguration('publish')).lower() == 'true'

    return [
        LogInfo(msg=f'test_poses: poses_file={poses_file} '
                    f'map_file={map_file or "(tanpa peta)"} '
                    f'hold_sec={hold:.1f} period_sec={period:.1f} publish={publish}'),
        Node(
            package='spawn',
            executable='test_poses',
            name='test_poses',
            output='screen',
            parameters=[{
                'poses_file': poses_file,
                'map_file': map_file,
                'hold_sec': hold,
                'period_sec': period,
                'publish': publish,
            }],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'poses_file', default_value='',
            description='Kosong = config/poses.yaml paket ini. Bisa override path lain.'),
        DeclareLaunchArgument(
            'map_file', default_value='',
            description='Kosong = src/navigation/maps/map.yaml (auto). Untuk cek okupansi.'),
        DeclareLaunchArgument(
            'hold_sec', default_value='3.0',
            description='Lama publish tiap preset (detik) sebelum loncat ke preset berikutnya.'),
        DeclareLaunchArgument(
            'period_sec', default_value='0.5',
            description='Jeda antar publish dalam satu preset (detik).'),
        DeclareLaunchArgument(
            'publish', default_value='true',
            description='false = hanya tabel validasi, tanpa publish /initialpose.'),
        OpaqueFunction(function=_launch),
    ])
