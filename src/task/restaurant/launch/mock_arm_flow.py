#!/usr/bin/env python3
"""Standalone arm-only flow tester mirroring restaurant_manager.cpp (no HRI, Nav, or Vision)."""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


class MockArmFlowNode(Node):
    """Drives athome_arm services through the exact Restaurant FSM arm sequence."""

    def __init__(self):
        super().__init__('restaurant_mock_arm_flow')

        self.declare_parameter('mode', 'auto')
        self.declare_parameter('start_stage', 'grasp')
        self.declare_parameter('order1_name', 'coke')
        self.declare_parameter('order2_name', 'water')
        self.declare_parameter('target_point', '0.6,0.1,0.7')
        self.declare_parameter('startup_delay_sec', 3.0)
        self.declare_parameter('between_grasp_and_delivery_sec', 2.0)
        self.declare_parameter('countdown_sec', 3.0)

        self._mode = str(self.get_parameter('mode').value).strip().lower()
        self._start_stage = str(self.get_parameter('start_stage').value).strip().lower()
        self._order1 = str(self.get_parameter('order1_name').value).strip() or 'coke'
        self._order2 = str(self.get_parameter('order2_name').value).strip() or 'water'
        self._target_point = str(self.get_parameter('target_point').value).strip()
        self._startup_delay = max(0.0, float(self.get_parameter('startup_delay_sec').value))
        self._between_delay = max(
            0.0, float(self.get_parameter('between_grasp_and_delivery_sec').value)
        )
        self._countdown_sec = max(0.0, float(self.get_parameter('countdown_sec').value))

        self._target_pub = self.create_publisher(String, '/arm/target_point', 10)
        self._ambil_order_client = self.create_client(Trigger, '/arm/ambil_order')
        self._raising_arm_client = self.create_client(Trigger, '/arm/raising_arm')
        self._release_gripper_client = self.create_client(Trigger, '/arm/release_gripper')
        self._relax_client = self.create_client(Trigger, '/arm/relax')
        self._free_client = self.create_client(Trigger, '/arm/free')

    def _wait_step(self, label: str) -> None:
        if self._mode != 'step':
            return
        try:
            input(f'\n[STEP] Tekan Enter untuk lanjut -> {label} ... ')
        except EOFError:
            self.get_logger().warn(
                '[STEP] stdin tidak interaktif (EOF), beralih otomatis ke mode auto.'
            )
            self._mode = 'auto'

    def _call_trigger(self, client, srv_name: str, timeout_sec: float) -> tuple[bool, str]:
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(f'[SERVICE] Service {srv_name} belum tersedia.')
            return False, 'Service not ready'

        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)

        if not future.done():
            self.get_logger().error(f'[ARM] {srv_name} timeout (> {timeout_sec:.1f}s).')
            return False, 'Service response timeout'

        try:
            res = future.result()
            ok = bool(res.success)
            msg = str(res.message)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'[ARM] {srv_name} error: {exc}')
            return False, str(exc)

        if ok:
            self.get_logger().info(f'[ARM] {srv_name} selesai (ok=true, msg="{msg}")')
        else:
            self.get_logger().warn(f'[ARM] {srv_name} gagal (ok=false, msg="{msg}")')
        return ok, msg

    def run(self) -> None:
        self.get_logger().info(
            f'=== Mock Arm Flow dimulai (mode={self._mode}, start_stage={self._start_stage}, '
            f'order1="{self._order1}", order2="{self._order2}") ==='
        )
        if self._startup_delay > 0.0:
            self.get_logger().info(
                f'Menunggu {self._startup_delay:.1f}s agar arm_motion_server siap...'
            )
            time.sleep(self._startup_delay)

        if self._start_stage == 'grasp':
            self.get_logger().info('[STATE] GRASP_ORDER1')
            self._wait_step('Publish /arm/target_point & panggil /arm/ambil_order')
            if self._target_point:
                msg = String()
                msg.data = self._target_point
                self._target_pub.publish(msg)
                self.get_logger().info(
                    f'[ARM TARGET] Published /arm/target_point: ({self._target_point})'
                )
            self._call_trigger(self._ambil_order_client, '/arm/ambil_order', timeout_sec=25.0)
            self.get_logger().info(
                f'[TTS SIM] "Thank you. Please place {self._order2} on the tray in front of me."'
            )
            if self._mode == 'step':
                self._wait_step('Lanjut ke tahap delivery (GREET_DELIVERY_CUSTOMER)')
            elif self._between_delay > 0.0:
                self.get_logger().info(
                    f'[WAIT] Simulasi jeda menuju delivery ({self._between_delay:.1f}s)...'
                )
                time.sleep(self._between_delay)
        elif self._start_stage != 'delivery':
            self.get_logger().warn(
                f'start_stage "{self._start_stage}" tidak dikenal, langsung menjalankan delivery.'
            )

        self.get_logger().info('[STATE] GREET_DELIVERY_CUSTOMER')
        self.get_logger().info('[TTS SIM] "Hello, here is your order."')

        self.get_logger().info('[STATE] WAIT_1S_BEFORE_RELEASE_PROMPT (1.0s)')
        time.sleep(1.0)

        self.get_logger().info('[STATE] PROMPT_RELEASE_ORDER1')
        self._wait_step('Jalankan MOTION 1 (/arm/raising_arm)')
        self.get_logger().info('[ARM] Raising arm via /arm/raising_arm...')
        self._call_trigger(self._raising_arm_client, '/arm/raising_arm', timeout_sec=20.0)

        self.get_logger().info(
            f'[TTS SIM] "For {self._order1}, please hold your hand under my gripper. '
            'Releasing on one, two, three."'
        )
        if self._countdown_sec > 0.0:
            step_sleep = self._countdown_sec / 3.0
            for word in ('one', 'two', 'three'):
                time.sleep(step_sleep)
                self.get_logger().info(f'[TTS COUNTDOWN] ... {word}')

        self.get_logger().info('[STATE] RELEASE_GRIPPER_AND_TORQUE_OFF')
        self._wait_step('Jalankan MOTION 2 (/arm/release_gripper)')
        self.get_logger().info('[ARM] Releasing gripper via /arm/release_gripper...')
        self._call_trigger(self._release_gripper_client, '/arm/release_gripper', timeout_sec=20.0)

        self._wait_step('Relax gripper (/arm/relax, fallback /arm/free)')
        relax_client = self._relax_client
        relax_srv_name = '/arm/relax'
        if not relax_client.service_is_ready() and self._free_client.service_is_ready():
            relax_client = self._free_client
            relax_srv_name = '/arm/free'
        self.get_logger().info(f'[ARM] Relaxing arm via {relax_srv_name}...')
        self._call_trigger(relax_client, relax_srv_name, timeout_sec=10.0)

        self.get_logger().info('[STATE] PROMPT_TAKE_ORDER2_TRAY')
        self.get_logger().info(
            f'[TTS SIM] "For {self._order2}, please take it from my tray."'
        )
        self.get_logger().info('=== Mock Arm Flow selesai ===')


def main(args=None):
    rclpy.init(args=args)
    node = MockArmFlowNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
