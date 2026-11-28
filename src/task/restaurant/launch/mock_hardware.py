#!/usr/bin/env python3
"""Mock hardware node (Navigation, Vision, Arm) for HRI-only testing of restaurant_manager."""

import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from std_srvs.srv import Trigger


class MockHardwareNode(Node):
    """Provides mocked Trigger services and Vision topics for restaurant FSM testing."""

    NAV_SERVICES = (
        '/go_to_order',
        '/go_to_order1',
        '/deliver_order',
        '/homing',
    )

    ACTION_SERVICES = (
        '/find_barman',
        '/vision/wave_hand',
        '/vision/camera_capture_face',
        '/arm/ambil_order',
        '/arm/raising_arm',
        '/arm/release_gripper',
        '/arm/letakkan_tas',
        '/arm/free',
        '/arm/relax',
    )

    def __init__(self):
        super().__init__('restaurant_mock_hardware')

        self.declare_parameter('nav_delay_sec', 2.0)
        self.declare_parameter('action_delay_sec', 0.5)
        self.declare_parameter('gesture', 'WAVE_LEFT')
        self.declare_parameter('order_pose', 'order:1.5,0.0,0.0')
        self.declare_parameter('person_label', 'person:200,100,150,300|1.5,0.0,0.0|id=1')
        self.declare_parameter(
            'drink_detections',
            'coke:100,200,50,80|0.6,0.1,0.7;'
            'water:300,200,50,80|0.6,0.0,0.7;'
            'juice:500,200,50,80|0.6,-0.1,0.7',
        )
        self.declare_parameter('bag_detections', '')
        self.declare_parameter('publish_rate_hz', 5.0)

        self._cb_group = ReentrantCallbackGroup()
        self._services = []

        for srv_name in self.NAV_SERVICES:
            self._register_trigger(srv_name, is_nav=True)
        for srv_name in self.ACTION_SERVICES:
            self._register_trigger(srv_name, is_nav=False)

        self._gesture_pub = self.create_publisher(String, '/vision/gesture', 10)
        self._order_pub = self.create_publisher(String, '/vision/order', 10)
        self._person_pub = self.create_publisher(
            String, '/vision/person_label', qos_profile_sensor_data
        )
        self._drink_pub = self.create_publisher(
            String, '/vision/drink_detections', qos_profile_sensor_data
        )
        self._bag_pub = self.create_publisher(
            String, '/vision/bag_detections', qos_profile_sensor_data
        )

        self._name_sub = self.create_subscription(
            String,
            '/vision/set_person_name',
            lambda msg: self.get_logger().info(f'[MOCK VISION] /vision/set_person_name: "{msg.data}"'),
            10,
        )
        self._expr_sub = self.create_subscription(
            String,
            '/robot/expression',
            lambda msg: self.get_logger().info(f'[MOCK FACE] /robot/expression: "{msg.data}"'),
            10,
        )
        self._hmi_sub = self.create_subscription(
            String,
            '/hmi/customer_display',
            lambda msg: self.get_logger().info(f'[MOCK HMI] /hmi/customer_display: "{msg.data}"'),
            10,
        )
        self._arm_target_sub = self.create_subscription(
            String,
            '/arm/target_point',
            lambda msg: self.get_logger().info(f'[MOCK ARM] /arm/target_point: "{msg.data}"'),
            10,
        )

        rate_hz = max(0.5, float(self.get_parameter('publish_rate_hz').value))
        self._timer = self.create_timer(1.0 / rate_hz, self._publish_vision)

        self.get_logger().info(
            'Restaurant Mock Hardware ready (12 Trigger services + 5 Vision topics).'
        )

    def _register_trigger(self, srv_name: str, is_nav: bool):
        def handler(_req, res):
            param_name = 'nav_delay_sec' if is_nav else 'action_delay_sec'
            delay = max(0.0, float(self.get_parameter(param_name).value))
            self.get_logger().info(f'[MOCK] {srv_name} called (delay={delay:.2f}s)...')
            if delay > 0.0:
                time.sleep(delay)
            res.success = True
            if srv_name == '/vision/wave_hand':
                order_pose = str(self.get_parameter('order_pose').value).strip()
                res.message = order_pose if order_pose else 'order:1.5,0.0,0.0'
            else:
                res.message = f'[MOCK] {srv_name} done'
            self.get_logger().info(f'[MOCK] {srv_name} -> success ({res.message})')
            return res

        srv = self.create_service(
            Trigger, srv_name, handler, callback_group=self._cb_group
        )
        self._services.append(srv)

    def _publish_vision(self):
        for pub, param_name in (
            (self._gesture_pub, 'gesture'),
            (self._order_pub, 'order_pose'),
            (self._person_pub, 'person_label'),
            (self._drink_pub, 'drink_detections'),
            (self._bag_pub, 'bag_detections'),
        ):
            val = str(self.get_parameter(param_name).value)
            if val:
                msg = String()
                msg.data = val
                pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MockHardwareNode()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
