#!/usr/bin/env python3
"""Kalibrasi skala odometry robot dari pengukuran nyata.

Untuk apa
---------
Posisi robot Nav2 datang dari DUA sumber:
  1. /odom  = encoder + IMU  (cepat, tapi MENGALIR)
  2. /amcl_pose = lidar + peta (lambat, tapi benar)

AMCL hanya mengoreksi error /odom secara SLOW. Kalau skala /odom salah,
posisi tetap menyimpang terus, dan itu yang muncul sebagai "error posisi".

Penyebab skala salah hampir selalu salah satu dari:
  pulses_per_rev  : jumlah pulse per putaran roda
  wheel_radius_cm : jari-jari roda (bukan diameter!)
  base_length_cm  : jarak roda belakang ke poros roda depan

Cara pakai
----------
  # 1. Taruh robot di titik berarsir, lalu catat posisi awal
  ros2 run nucleo odom_calibrate --start

  # 2. Dorong robot LURUS 2-3 meter tanpa berputar

  # 3. Catat posisi akhir, lalu hitung koreksinya
  ros2 run nucleo odom_calibrate --end
  ros2 run nucleo odom_calibrate --report

Pakai --start / --end / --report dalam SESI yang sama (data disimpan di
/tmp) supaya tidak salah baca transform.

Setelah dapat angka barunya, jalankan:
  ros2 launch nucleo nucleo.launch.py \
      pulses_per_rev:=<BARU> wheel_radius_cm:=<BARU> base_length_cm:=<BARU>
"""
import argparse
import json
import math
import os
import sys

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

STATE_FILE = '/tmp/odom_calib.json'


class OdomCalibrate(Node):
    def __init__(self, tag):
        super().__init__('odom_calibrate')
        self._tag = tag
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.history = HistoryPolicy.KEEP_LAST
        self.create_subscription(
            Odometry, 'odom', self._on_odom, qos)

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        yaw = math.atan2(
            2 * (msg.pose.pose.orientation.w * msg.pose.pose.orientation.z),
            1 - 2 * msg.pose.pose.orientation.z ** 2)
        state = {}
        if os.path.isfile(STATE_FILE):
            try:
                with open(STATE_FILE) as f:
                    state = json.load(f)
            except Exception:
                state = {}
        state[self._tag] = {'x': p.x, 'y': p.y, 'yaw': yaw}
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        self.get_logger().info(
            f'{self._tag}: x={p.x:+.4f} y={p.y:+.4f} yaw={math.degrees(yaw):+.1f}deg')


def load_state():
    if not os.path.isfile(STATE_FILE):
        return None
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def report(state, args):
    if not state or 'start' not in state or 'end' not in state:
        print('ERROR: belum ada start/end. Jalankan --start lalu --end dulu.')
        return 2
    s, e = state['start'], state['end']
    dx = e['x'] - s['x']
    dy = e['y'] - s['y']
    dist = math.hypot(dx, dy)
    dyaw = math.degrees(math.atan2(math.sin(e['yaw'] - s['yaw']),
                                    math.cos(e['yaw'] - s['yaw'])))
    print()
    print('=' * 62)
    print('  HASIL KALIBRASI ODOMETRY')
    print('=' * 62)
    print(f'  jarak di /odom        : {dist:+.4f} m  (dx={dx:+.4f} dy={dy:+.4f})')
    print(f'  perubahan yaw         : {dyaw:+.2f} deg')
    print()

    if abs(dyaw) > 12.0:
        print('  ⚠️  YAW berubah > 12 deg -> pengukuran TIDAK lurus.')
        print('      Ulangi straighten-robot. Kalibrasi ppr hanya sah')
        print('      kalau arah roda benar-benar lurus.')
        print()
        return 1

    if args.real < 0.05:
        print('  ⚠️  jarak terlalu pendek (< 5 cm).minimal 2-3 m.')
        return 1

    scale = args.real / dist if dist > 1e-6 else 1.0
    print(f'  jarak riil (argumen)  : {args.real:.4f} m')
    print(f'  rasio koreksi         : {scale:.6f}')
    print()
    print('  ---------------------------------------------------------------')
    if scale > 1.0:
        print('  /odom kurang jauh -> radius roda/ppr terlalu kecil.')
    elif scale < 1.0:
        print('  /odom terlalu jauh -> radius roda/ppr terlalu besar.')
    print('  ---------------------------------------------------------------')
    print(f'  pulses_per_rev_baru = {args.ppr} x {scale:.6f} = {args.ppr * scale:.1f}')
    print()
    print(f'  Jalankan:')
    print(f'  ros2 launch nucleo nucleo.launch.py pulses_per_rev:={args.ppr * scale:.1f}')
    print()
    print('  Kalau YAW masih menyimpang (robot berputar padahal seharusnya')
    print('  lurus), kalibrasi base_length_cm + wheel_radius_cm:')
    print('    rasio_base_new = rasio_base_lama x (dyaw_riil / dyaw_odom)')
    print(f'    dyaw_odom saat ini = {dyaw:+.2f} deg')
    print('=' * 62)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start', action='store_true',
                    help='catat posisi awal (robot di titik berarsir)')
    ap.add_argument('--end', action='store_true',
                    help='catat posisi akhir (setelah robot maju lurus)')
    ap.add_argument('--report', action='store_true',
                    help='hitung & tampilkan parameter barunya')
    ap.add_argument('--real', type=float, default=2.0,
                    help='jarak riil yang ditempuh dalam meter (default 2.0)')
    ap.add_argument('--ppr', type=float, default=7500.0,
                    help='nilai pulses_per_rev yang sedang dipakai')
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    if not (args.start or args.end or args.report):
        ap.print_help()
        return 2

    if args.report:
        return report(load_state(), args)

    rclpy.init(args=None)
    tag = 'start' if args.start else 'end'
    node = OdomCalibrate(tag)
    # Tunggu satu pesan odom, lalu keluar supaya tidak menggantung.
    end = node.get_clock().now().nanoseconds + 10_000_000_000
    while rclpy.ok() and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.2)
        if os.path.isfile(STATE_FILE):
            st = load_state() or {}
            if tag in st:
                break
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())