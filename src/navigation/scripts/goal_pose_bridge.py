#!/usr/bin/env python3
"""Jembatan topic -> action untuk Navigasi Nav2.

RViz toolbar "2D Goal Pose" mempublikasikan geometry_msgs/PoseStamped ke
topik `/goal_pose`, sedangkan Nav2 menerima tujuan lewat action
`navigate_to_pose` (bukan topik). Node ini menyambungkan keduanya, sehingga
klik "2D Goal Pose" di RViz langsung menyuruh robot bergerak.

Alternatif tanpa node ini: pakai tool "Nav2 Goal" di RViz yang memang
mengirim action Nav2 secara langsung.
"""
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose


class GoalPoseBridge(Node):
    def __init__(self):
        super().__init__('goal_pose_bridge')
        self._client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.create_subscription(PoseStamped, 'goal_pose', self._on_goal, 10)
        self.get_logger().info(
            'goal_pose_bridge siap: /goal_pose -> action navigate_to_pose')

    def _on_goal(self, msg: PoseStamped):
        p = msg.pose.position
        self.get_logger().info(
            f'goal diterima: x={p.x:.2f} y={p.y:.2f} frame={msg.header.frame_id or "?"}')
        if not self._client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(
                'action navigate_to_pose belum tersedia (bt_navigator belum aktif)')
            return
        goal = NavigateToPose.Goal()
        goal.pose = msg
        self._client.send_goal_async(goal)


def main(args=None):
    rclpy.init(args=args)
    node = GoalPoseBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
