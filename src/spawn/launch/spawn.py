"""Spawn robot di koordinat preset (publish /initialpose untuk AMCL).

Pakai:
  ros2 launch spawn spawn.py preset:=1
  ros2 launch spawn spawn.py x:=1.0 y:=1.5 yaw_deg:=0.0   # tanpa preset
  ros2 launch spawn spawn.py preset:=2 duration_sec:=5.0

Preset dibaca dari config/poses.yaml (tambah preset = edit YAML itu saja).
Hanya bermakna di mode peta statis (navigation slam:=false + AMCL):
memindah ESTIMASI pose di peta, bukan robot fisik. Pastikan posisi fisik
robot sesuai koordinat preset saat perintah ini dijalankan.

Uji tanpa Nav2 (cek YAML kebaca + topik kepublish):
  T0: ros2 launch spawn test_poses.py     # tabel validasi SEMUA preset + publish berurutan
  T1: ros2 launch spawn spawn.py preset:=1
  T2: ros2 topic echo /initialpose     # harus muncul x=1.0 y=1.5
Uji nyata: navigation slam:=false jalan + peta tampil di RViz, lalu
jalankan launch ini; robot di RViz pindah ke koordinat preset
(cek juga: ros2 topic echo /amcl_pose).
"""
import os

import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def resolve_and_launch(context):
    poses_file = context.perform_substitution(LaunchConfiguration('poses_file'))
    if not poses_file:
        try:
            from ament_index_python.packages import get_package_share_directory
            cand = os.path.join(get_package_share_directory('spawn'), 'config', 'poses.yaml')
        except Exception:
            cand = ''
        if cand and os.path.isfile(cand):
            poses_file = cand
        else:
            # __file__ symlink-install -> resolve() tembus ke src tree.
            pkg_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            poses_file = os.path.join(pkg_dir, 'config', 'poses.yaml')
    preset = context.perform_substitution(LaunchConfiguration('preset'))
    x_raw = context.perform_substitution(LaunchConfiguration('x'))
    y_raw = context.perform_substitution(LaunchConfiguration('y'))
    yaw_raw = context.perform_substitution(LaunchConfiguration('yaw_deg'))

    if x_raw != '' and y_raw != '':
        # Koordinat eksplisit (tanpa preset).
        x, y = float(x_raw), float(y_raw)
        yaw_deg = float(yaw_raw) if yaw_raw != '' else 0.0
        src = 'argumen x/y'
    else:
        # Ambil dari preset di poses.yaml.
        with open(poses_file, 'r') as f:
            data = yaml.safe_load(f) or {}
        presets = data.get('presets', {}) or {}
        keys = sorted(str(k) for k in presets.keys())
        if preset not in presets:
            raise RuntimeError(
                f"preset '{preset}' tidak ada di {poses_file}. "
                f"Yang tersedia: {keys}. "
                f"Atau pakai koordinat bebas: x:=... y:=... yaw_deg:=...")
        entry = presets[preset] or {}
        x = float(entry.get('x', 0.0))
        y = float(entry.get('y', 0.0))
        yaw_deg = float(entry.get('yaw_deg', 0.0))
        src = f"preset '{preset}' ({poses_file})"

    duration = float(context.perform_substitution(LaunchConfiguration('duration_sec')))
    return [
        LogInfo(msg=f"spawn: {src} -> x={x:.3f} y={y:.3f} yaw={yaw_deg:.1f} deg "
                    f"(publish /initialpose {duration:.1f}s)"),
        Node(
            package='spawn',
            executable='set_pose',
            name='set_pose',
            output='screen',
            parameters=[{
                'x': x,
                'y': y,
                'yaw_deg': yaw_deg,
                'duration_sec': duration,
            }],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'preset',
            default_value='1',
            description='Nama preset di config/poses.yaml (diabaikan bila x+y diisi).',
        ),
        DeclareLaunchArgument(
            'x',
            default_value='',
            description='Kosong = pakai preset. Diisi = koordinat bebas (meter, frame map).',
        ),
        DeclareLaunchArgument(
            'y',
            default_value='',
            description='Kosong = pakai preset. Diisi = koordinat bebas (meter, frame map).',
        ),
        DeclareLaunchArgument(
            'yaw_deg',
            default_value='',
            description='Heading robot derajat (0 = +X peta). Kosong = 0 / ikut preset.',
        ),
        DeclareLaunchArgument(
            'poses_file',
            default_value='',
            description='Kosong = config/poses.yaml paket ini. Bisa override path YAML lain.',
        ),
        DeclareLaunchArgument(
            'duration_sec',
            default_value='10.0',
            description='Lama publish /initialpose berulang (detik, anti-race AMCL).',
        ),
        OpaqueFunction(function=resolve_and_launch),
    ])
