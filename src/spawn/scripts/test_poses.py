#!/usr/bin/env python3
"""Uji poses.yaml end-to-end: validasi tabel + publish SEMUA preset berurutan.

Dijalankan lewat launch (satu perintah):
  ros2 launch spawn test_poses.py
  ros2 launch spawn test_poses.py hold_sec:=2.0 publish:=false   # tabel saja
  ros2 launch spawn test_poses.py poses_file:=/path/poses.yaml

Urutan kerja:
  1. Baca poses.yaml -> tabel status tiap preset (OK/WARN/FAIL + cek di peta).
     Bila ada FAIL, node berhenti dengan error (exit code 1) tanpa publish.
  2. Publish tiap preset ke /initialpose berurutan selama hold_sec, jadi di
     RViz (2D Pose Estimate marker / posisi robot AMCL) terlihat "loncat"
     antar preset, dan `ros2 topic echo /initialpose` menampilkan isinya.
Tanpa Nav2/AMCL yang jalan, publish tetap aman (hanya topik biasa).
"""
import sys
import time

import pose_check
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile


def yaw_to_quat(yaw_deg: float):
    import math
    yaw = math.radians(yaw_deg)
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class TestPoses(Node):
    def __init__(self):
        super().__init__('test_poses')
        self.declare_parameter('poses_file', '')
        self.declare_parameter('map_file', '')
        self.declare_parameter('hold_sec', 3.0)
        self.declare_parameter('period_sec', 0.5)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('topic', '/initialpose')
        self.declare_parameter('publish', True)

    def run(self) -> int:
        poses_file = str(self.get_parameter('poses_file').value)
        map_file = str(self.get_parameter('map_file').value)
        hold = float(self.get_parameter('hold_sec').value)
        period = float(self.get_parameter('period_sec').value)
        frame_id = str(self.get_parameter('frame_id').value)
        topic = str(self.get_parameter('topic').value)
        do_publish = bool(self.get_parameter('publish').value)

        presets, err = pose_check.load_presets(poses_file)
        if err:
            self.get_logger().error(f'{err}')
            return 1
        map_info = pose_check.load_map(map_file)
        rows, counts = pose_check.evaluate(presets, map_info)
        report = pose_check.format_report(rows, counts, poses_file, map_info)
        for line in report.splitlines():
            (self.get_logger().warning if line.startswith('FAIL') else self.get_logger().info)(line)
        if counts['FAIL']:
            self.get_logger().error(f'uji GAGAL: {counts["FAIL"]} preset FAIL, '
                                    f'perbaiki poses.yaml dulu (tidak ada yang dipublish)')
            return 1
        if not do_publish:
            self.get_logger().info('publish:=false -> selesai di tahap validasi.')
            return 0

        pub = self.create_publisher(PoseWithCovarianceStamped, topic, QoSProfile(depth=10))
        names = [r['name'] for r in rows]
        for i, r in enumerate(rows, 1):
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = frame_id
            msg.pose.pose.position.x = r['x']
            msg.pose.pose.position.y = r['y']
            qx, qy, qz, qw = yaw_to_quat(r['yaw'])
            msg.pose.pose.orientation.z = qz
            msg.pose.pose.orientation.w = qw
            cov = [0.0] * 36
            cov[0], cov[7], cov[35] = 0.25, 0.25, 0.0685
            msg.pose.covariance = cov

            self.get_logger().info(
                f'[{i}/{len(rows)}] preset "{r["name"]}" -> x={r["x"]:.3f} y={r["y"]:.3f} '
                f'yaw={r["yaw"]:.1f} deg, publish {topic} {hold:.1f}s '
                f'(RViz: robot harus pindah ke koordinat ini)')
            t_end = time.time() + hold
            n = 0
            while time.time() < t_end and rclpy.ok():
                msg.header.stamp = self.get_clock().now().to_msg()
                pub.publish(msg)
                n += 1
                time.sleep(period)
            self.get_logger().info(f'   -> {n}x publish OK')
        self.get_logger().info(f'test_poses SELESAI: semua {len(names)} preset '
                               f'({", ".join(names)}) ter-publish ke {topic}')
        return 0


def main() -> int:
    rclpy.init()
    node = TestPoses()
    try:
        rc = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return rc


if __name__ == '__main__':
    sys.exit(main())
