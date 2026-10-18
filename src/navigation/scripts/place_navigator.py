#!/usr/bin/env python3
"""Node navigasi lokasi bernama untuk peta statis Nav2.

Untuk setiap key di `config/places.yaml`, node ini membuat SATU ROS service:

    /goto/<nama>        tipe std_srvs/srv/Trigger

Jadi mis. key `ruang_tamu` menghasilkan service `/goto/ruang_tamu`.

Cara pakai
----------
 Dari terminal:
     ros2 service call /goto/ruang_tamu std_srvs/srv/Trigger "{}"

 Dari program Python:
     from std_srvs.srv import Trigger
     client = node.create_client(Trigger, '/goto/ruang_tamu')
     client.wait_for_service(timeout_sec=2.0)
     future = client.call_async(Trigger.Request())
     # future.result().success == True  ->  robot sudah sampai
     # (call_async TIDAK memblokir; node lain tetap responsif)

 Dari program C++:
     auto c = this->create_client<std_srvs::srv::Trigger>("/goto/ruang_tamu");
     c->wait_for_service(1s);
     c->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());

Perilaku
--------
* Memakai peta statis + AMCL, jadi goal dikirim ke action `navigate_to_pose`
  milik Nav2 (`bt_navigator`). Path otomatis tampil di RViz pada topic `/plan`.
* Pemblokiran: service baru selesai dijawab setelah robot SAMPAI, timeout
  90 detik, atau gagal. Di luar itu balas `success=false` + alasannya.
* Hanya satu goal aktif pada satu waktu. Panggilan saat sedang jalan langsung
  ditolak (`success=false`) supaya tidak menumpuk perintah.
* Tambah/ubah lokasi cukup edit `config/places.yaml`, tanpa ubah kode.

Catatan penting soal executor
-----------------------------
Service di sini sengaja memblokir (menunggu robot sampai tujuan). Kalau paket
ini dieksekusi dengan SingleThreadedExecutor, callback pemblokir akan membuat
node BUDAH sehingga hasil action tidak pernah diproses -> deadlock. Karena itu
node ini menjalankan executor-nya sendiri (MultiThreadedExecutor) di main().
"""
import math
import os
import re
import threading

import yaml

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

# Nama ROS hanya boleh huruf/angka/underscore dan tidak boleh diawali angka.
VALID_NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def default_places_file() -> str:
    """Lokasi config/places.yaml: share dir (hasil build) atau source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('navigation')
    except Exception:
        share = ''
    if share:
        candidate = os.path.join(share, 'config', 'places.yaml')
        if os.path.isfile(candidate):
            return candidate
    # symlink-install: realpath(__file__) tembus ke src tree.
    pkg_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return os.path.join(pkg_dir, 'config', 'places.yaml')


def load_places(path: str):
    """-> (places: dict, error: str|None)."""
    if not os.path.isfile(path):
        return {}, f'file tidak ada: {path}'
    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        return {}, f'YAML rusak di {path}: {e}'
    presets = data.get('presets')
    if not isinstance(presets, dict) or not presets:
        return {}, f'{path} tidak punya blok "presets" yang tidak kosong'
    return presets, None


def yaw_to_quat(yaw_deg: float):
    yaw = math.radians(yaw_deg)
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class PlaceNavigator(Node):
    """Expose /goto/<nama> per lokasi, lalu kirim NavigateToPose ke Nav2."""

    def __init__(self):
        super().__init__('place_navigator')

        self.declare_parameter('places_file', '')
        self.declare_parameter('timeout_sec', 90.0)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('action_name', 'navigate_to_pose')
        self.declare_parameter('action_server_wait_sec', 30.0)

        self._timeout = float(self.get_parameter('timeout_sec').value)
        self._frame_id = str(self.get_parameter('frame_id').value)
        self._action_name = str(self.get_parameter('action_name').value)
        self._server_wait = float(self.get_parameter('action_server_wait_sec').value)

        places_file = str(self.get_parameter('places_file').value) or default_places_file()
        places, err = load_places(places_file)
        if err:
            raise RuntimeError(err)

        self._client = ActionClient(self, NavigateToPose, self._action_name)

        # Lock non-blokir: dipakai sebagai penanda "sudah ada goal aktif".
        self._busy = threading.Lock()
        # Callback group reentrant + executor multi-thread supaya callback yang
        # memblokir tidak mematikan pemrosesan hasil action.
        self._group = ReentrantCallbackGroup()

        # NOTE: jangan pakai nama '_services' -> itu atribut internal rclpy
        # (Node._services) dan akan menimpanya.
        self._place_services = {}
        skipped = []
        for name, entry in places.items():
            name = str(name)
            if not VALID_NAME.match(name):
                skipped.append(name)
                continue
            self._place_services[name] = self.create_service(
                Trigger,
                f'/goto/{name}',
                self._make_callback(name, entry or {}),
                callback_group=self._group,
            )

        self.get_logger().info(
            f'places_file={places_file} -> {len(self._place_services)} lokasi: '
            + ', '.join(sorted(self._place_services)))
        if skipped:
            self.get_logger().warn(
                'key di-skip (nama tidak valid sebagai nama ROS service): '
                + ', '.join(skipped))
        if not self._place_services:
            raise RuntimeError(f'tidak ada lokasi valid di {places_file}')

    def _make_callback(self, name: str, entry: dict):
        def callback(_request, response):
            return self._on_goto(name, entry, response)
        return callback

    def _on_goto(self, name, entry, response):
        if not self._busy.acquire(blocking=False):
            response.success = False
            response.message = f'sibok: robot masih menuju lokasi lain'
            return response

        try:
            response.success, response.message = self._navigate(name, entry)
        except Exception as e:  # jangan sampai service mati
            self.get_logger().error(f'/goto/{name} error: {e}')
            response.success = False
            response.message = f'error internal: {e}'
        finally:
            self._busy.release()
        return response

    def _navigate(self, name, entry):
        try:
            x = float(entry['x'])
            y = float(entry['y'])
        except (KeyError, TypeError, ValueError) as e:
            return False, f'lokasi "{name}" tidak punya x/y yang valid ({e})'
        try:
            yaw_deg = float(entry.get('yaw_deg', 0.0))
        except (TypeError, ValueError):
            yaw_deg = 0.0

        if not self._client.wait_for_server(timeout_sec=self._server_wait):
            return False, (
                f'action {self._action_name} tidak aktif '
                f'(nav2 belum jalan / belum selesai inisialisasi)')

        qx, qy, qz, qw = yaw_to_quat(yaw_deg)
        pose = PoseStamped()
        pose.header.frame_id = self._frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        goal = NavigateToPose.Goal()
        goal.pose = pose

        self.get_logger().info(
            f'/goto/{name}: x={x:.2f} y={y:.2f} yaw={yaw_deg:.1f}deg '
            f'(timeout {self._timeout:.0f}s)')

        send_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            return False, f'goal "{name}" ditolak Nav2'

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=self._timeout)
        if not result_future.done():
            handle.cancel_goal_async()
            return False, (
                f'timeout {self._timeout:.0f}s, belum sampai di "{name}"')

        status = result_future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            return True, f'sampai di "{name}" (x={x:.2f}, y={y:.2f})'
        if status == GoalStatus.STATUS_CANCELED:
            return False, f'perjalanan ke "{name}" dibatalkan'
        return False, f'gagal mencapai "{name}" (status {status})'


def main(args=None):
    rclpy.init(args=args)
    node = PlaceNavigator()
    # WAJIB multi-thread: service memblokir sambil menunggu robot sampai tujuan.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()