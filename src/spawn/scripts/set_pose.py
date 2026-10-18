#!/usr/bin/env python3
"""Publish /initialpose sekali jalan (untuk AMCL di mode peta statis).

Dipanggil oleh launch spawn.py (atau manual):
  ros2 run spawn set_pose --ros-args -p x:=1.0 -p y:=1.5 -p yaw_deg:=0.0

Publish diulang tiap 0.5 dtk selama duration_sec (default 10 dtk) agar tidak
hilang bila AMCL belum fully up saat pesan pertama dikirim, lalu exit.
"""
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile


def yaw_to_quat(yaw_deg: float):
    yaw = math.radians(yaw_deg)
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def main() -> int:
    rclpy.init()
    node = Node('set_pose')
    node.declare_parameter('x', 0.0)
    node.declare_parameter('y', 0.0)
    node.declare_parameter('yaw_deg', 0.0)
    node.declare_parameter('frame_id', 'map')
    node.declare_parameter('topic', '/initialpose')
    node.declare_parameter('duration_sec', 10.0)
    node.declare_parameter('period_sec', 0.5)

    x = float(node.get_parameter('x').value)
    y = float(node.get_parameter('y').value)
    yaw_deg = float(node.get_parameter('yaw_deg').value)
    frame_id = str(node.get_parameter('frame_id').value)
    topic = str(node.get_parameter('topic').value)
    duration = float(node.get_parameter('duration_sec').value)
    period = float(node.get_parameter('period_sec').value)

    qx, qy, qz, qw = yaw_to_quat(yaw_deg)
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = frame_id
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.position.z = 0.0
    msg.pose.pose.orientation.x = qx
    msg.pose.pose.orientation.y = qy
    msg.pose.pose.orientation.z = qz
    msg.pose.pose.orientation.w = qw
    # Kovarians ala default AMCL (x/y 0.5m, yaw ~15deg) agar partikel menyebar
    # wajar lalu konvergen setelah robot bergerak / scan masuk.
    cov = [0.0] * 36
    cov[0] = 0.25
    cov[7] = 0.25
    cov[35] = 0.0685
    msg.pose.covariance = cov

    pub = node.create_publisher(PoseWithCovarianceStamped, topic, QoSProfile(depth=10))
    node.get_logger().info(
        f'set_pose: publish {topic} x={x:.3f} y={y:.3f} yaw={yaw_deg:.1f} deg '
        f'({duration:.0f}s @ {period:.1f}s, frame={frame_id})')
    t_end = time.time() + duration
    n = 0
    while time.time() < t_end and rclpy.ok():
        msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(msg)
        n += 1
        time.sleep(period)
    node.get_logger().info(f'set_pose: selesai ({n}x publish), exit.')
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
