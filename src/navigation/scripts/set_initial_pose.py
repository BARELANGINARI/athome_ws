#!/usr/bin/env python3
"""Kirim initial pose ke AMCL tanpa RViz (CLI).

Kenapa perlu: tool "2D Pose Estimate" di RViz kadang tidak bereaksi (klik
tidak Register, atau RViz jalan dengan software rendering sehingga event
mouse tidak sampai). Rantai `RViz -> /initialpose -> AMCL` sudah diuji
berfungsi; kalau RViz yang bermasalah, script ini jalur Alternatif yang
pasti jalan.

Sifatnya persis sama dengan tool RViz: publish
`geometry_msgs/msg/PoseWithCovarianceStamped` ke `/initialpose` dengan
frame `map`. AMCL akan menerima dan memfilter ulang_partikel ke lokasi
baru.

Cara pakai
----------
# pindah ke koordinat tertentu (meter, frame peta)
ros2 run navigation set_initial_pose --x 1.5 --y -0.5 --yaw 0.785

# ikutkan dengan "2D Goal Pose" supaya robot langsung GOAL ke situ
ros2 run navigation set_initial_pose --x 1.5 --y -0.5 --yaw 0.785 --go

# tanpa argumen: tombol ENTER untuk mengulang一遍遍, ctrl+C keluar
ros2 run navigation set_initial_pose

`--yaw` dalam RADIAN. 0 = +X peta, 1.5708 = +Y peta, -1.5708 = -Y peta.
"""
import argparse
import math
import sys

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node

# Covariance yang dipakai tool RViz "2D Pose Estimate".
DEFAULT_COVARIANCE = [
    0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891945200942,
]


class InitialPoseSetter(Node):
    def __init__(self):
        super().__init__('set_initial_pose')
        self._pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)
        self._client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

    def build_msg(self, x: float, y: float, yaw: float, frame: str):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        msg.pose.covariance = DEFAULT_COVARIANCE
        return msg

    def spin_for(self, seconds: float):
        deadline = self.get_clock().now().nanoseconds + int(seconds * 1e9)
        while self.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for_subscriber(self, timeout: float = 8.0) -> bool:
        """Tunggu subscriber muncul SEBELUM publish.

        Ini penting: kalau pesan dipublish sebelum subscriber terdeteksi,
        DDS tidak akan mengirim ulang sample yang sudah hilang -- pesannya
        hilang permanen dan AMCL tidak pernah melihatnya.
        """
        deadline = self.get_clock().now().nanoseconds + int(timeout * 1e9)
        while self.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._pub.get_subscription_count() > 0:
                return True
        return False

    def go(self, x: float, y: float, yaw: float, frame: str):
        if not self._client.wait_for_server(timeout_sec=30.0):
            self.get_logger().error(
                'action /navigate_to_pose tidak aktif '
                '(nav2 belum jalan / belum selesai inisialisasi)')
            return False
        pose = PoseStamped()
        pose.header.frame_id = frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        goal = NavigateToPose.Goal()
        goal.pose = pose
        fut = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        handle = fut.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('goal ditolak Nav2')
            return False
        self.get_logger().info(
            f'goal dikirim ke ({x:.2f}, {y:.2f}) yaw {yaw:.3f} rad')
        return True


def parse_args(argv):
    p = argparse.ArgumentParser(
        description='Kirim initial pose ke AMCL tanpa RViz.')
    p.add_argument('--x', type=float, default=None, help='x meter, frame peta')
    p.add_argument('--y', type=float, default=None, help='y meter, frame peta')
    p.add_argument('--yaw', type=float, default=0.0,
                   help='heading radian (0=+X, 1.5708=+Y, -1.5708=-Y)')
    p.add_argument('--frame', default='map', help='frame tujuan (default map)')
    p.add_argument('--go', action='store_true',
                   help='setelah pose dikirim, kirim goal navigate_to_pose')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.x is None or args.y is None:
        print(__doc__)
        print('ERROR: --x dan --y wajib diisi.\n')
        return 2

    rclpy.init(args=None)
    node = InitialPoseSetter()
    try:
        # 1. Tunggu subscriber dulu (AMCL) supaya pesan tidak hilang.
        subs = node.wait_for_subscriber()
        if not subs:
            node.get_logger().warn(
                'TIDAK ADA subscriber /initialpose dalam 8 detik. '
                'Nav2/AMCL belum aktif? Cek: ros2 lifecycle get /amcl')
        else:
            node.get_logger().info('subscriber /initialpose ditemukan')

        # 2. Publish, lalu Publishing ulang beberapa kali supaya aman dari
        #    sample yang masih hilang. AMCL cukup menerima satu yang sah.
        msg = node.build_msg(args.x, args.y, args.yaw, args.frame)
        for _ in range(5):
            msg.header.stamp = node.get_clock().now().to_msg()
            node._pub.publish(msg)
            node.spin_for(0.25)

        node.get_logger().info(
            f'initial pose dikirim ke ({args.x:.2f}, {args.y:.2f}) '
            f'yaw {args.yaw:.3f} rad, frame {args.frame}')

        if args.go:
            node.go(args.x, args.y, args.yaw, args.frame)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())