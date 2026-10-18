#!/usr/bin/env python3
"""Person following: robot mengikuti orang yang terdeteksi ZED.

Dynamic object following di atas Nav2. Orang dianggap objek DINAMIS karena
pose-nya berubah terus; karena itu goal tidak dikirim sekali, tapi terus
diperbarui ke dalam behaviour tree.

Cara kerja
----------
1. Subscribe `/zed/person_keypoints` (std_msgs/String) dari paket `zed`.
   Ambil orang ke-`target_index` (default 0 = paling dekat).
2. Ubah posisi 3D orang ke frame `map` lewat TF `base_link` -> `map`.
   Data zed sudah relatif robot (x depan, y kiri, z atas).
3. Hitung jarak d = pusat robot -> pusat orang, lalu terapkan deadband:
     d > follow_start_distance  ->  publish pose ke `/goal_update`
     d <= follow_stop_distance  ->  diam (jalur sudah dipotong oleh
                                    TruncatePath di behaviour tree)
     antara keduanya            ->  zona mati, tidak ada aksi
4. BT `behavior_trees/follow_person.xml` menySubscribe `/goal_update`
   lewat node `GoalUpdater`, lalu re-plan dan memotong jalur
   `path_truncate_distance` meter sebelum orang. Hasilnya robot menjaga
   jarak dan terus mengejar selama orang bergerak.

Cara pakai
----------
 Dari terminal:
     ros2 service call /follow/person std_srvs/srv/Trigger "{}"
     ros2 service call /follow/stop   std_srvs/srv/Trigger "{}"
     ros2 topic echo /follow/state

 Dari program Python:
     from std_srvs.srv import Trigger
     c = node.create_client(Trigger, '/follow/person')
     c.wait_for_service(timeout_sec=2.0)
     fut = c.call_async(Trigger.Request())
     # fut.result().success == True  ->  following diminta (langsung)
     # TIPUAN: tidak blocking, sama seperti /goto. following jalan di
     #         background sampai /follow/stop atau auto-stop.

 Dari program C++:
     auto c = std::make_shared<rclcpp::Client<std_srvs::srv::Trigger>>(
         node, "/follow/person");
     c->wait_for_service(1s);
     c->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());

State machine
-------------
    IDLE  --/follow/person-->  WAITING_PERSON
    WAITING_PERSON  --orang terlihat-->  FOLLOWING
    FOLLOWING  --d <= stop / hilang / terlalu jauh / /follow/stop-->  IDLE
    WAITING_PERSON --timeout-->  IDLE

Kenapa /follow/person tidak memblokir (berbeda dari /goto/<nama>)
------------------------------------------------------------------
`/goto/<nama>` memblokir sampai robot SAMPAI, karena ada titik akhir.
Following tidak punya "sampai" -- orang terus bergerak, jadi following bisa
berjalan tanpa henti. Karena itu /follow/person langsung menjawab sukses dan
meneruskan pekerjaan ke background. Yang berubah hanya jawabannya, bukan
cara_nav2-nya.

Opsi "tunggu orang"
-------------------
Kalau dipanggil saat belum ada orang, node tetap menjawab sukses lalu masuk
ke WAITING_PERSON sampai `wait_person_timeout`. Ini lebih praktis daripada
menolak: pemanggil tidak perluAddition logic menunggu sensor.

Catatan penting soal executor
-----------------------------
Recommends MultiThreadedExecutor: ada service, subscription, timer 1 Hz, dan
pendingan hasil action yang semuanya jalan bersamaan. Dengan
SingleThreadedExecutor, callback yang menunggu hasil action akan membuat node
BUNDEH.
"""
import math
import os
import threading

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String
from std_srvs.srv import Trigger

from tf2_geometry_msgs import do_transform_pose_stamped
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
# Di ROS 2 Jazzy kelas exception TF ada langsung di tf2_ros (bukan
# tf2_ros.exceptions seperti di Humble).
from tf2_ros import (
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformException,
)

IDLE = 'IDLE'
WAITING_PERSON = 'WAITING_PERSON'
FOLLOWING = 'FOLLOWING'

NO_PERSON = 'NO_PERSON'


def parse_persons(data: str):
    """Parse /zed/person_keypoints -> list dict orang yang punya koordinat valid.

    Format (dari paket zed):
        person:x,y,w,h|x,y,z|yaw,rel|kpt=...;person:...
    Orang dipisah ';', urut dari yang paling dekat. Nilai `nan` (depth gagal)
    justru dibuang supaya tidak dipakai sebagai goal.
    """
    out = []
    if not data or data == NO_PERSON:
        return out
    for chunk in data.split(';'):
        chunk = chunk.strip()
        if not chunk.startswith('person:'):
            continue
        fields = chunk.split('|')
        # parts[0] = person:x,y,w,h | parts[1] = x,y,z | parts[2] = yaw,rel
        if len(fields) < 3:
            continue
        try:
            xyz = [float(v) for v in fields[1].split(',')]
            yaw_rel = [float(v) for v in fields[2].split(',')]
        except ValueError:
            continue
        if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
            continue  # depth tidak terbaca -> abaikan orang ini
        out.append({
            'x': xyz[0],
            'y': xyz[1],
            'z': xyz[2],
            'yaw': yaw_rel[0] if yaw_rel else 0.0,
        })
    return out


def yaw_to_quat(yaw: float):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def default_follow_bt() -> str:
    """Lokasi behavior_trees/follow_person.xml.

    Dual path supaya `ros2 run navigation person_follower.py` bisa jalan
    tanpa launch: share dir (hasil build) dulu, lalu source tree. Pola ini
    sama dengan default_places_file() di place_navigator.py.
    """
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('navigation')
    except Exception:
        share = ''
    if share:
        candidate = os.path.join(share, 'behavior_trees', 'follow_person.xml')
        if os.path.isfile(candidate):
            return candidate
    # symlink-install: realpath(__file__) tembus ke src tree.
    pkg_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return os.path.join(pkg_dir, 'behavior_trees', 'follow_person.xml')


class PersonFollower(Node):
    """Expose /follow/person + /follow/stop, kirim pose dinamis ke Nav2."""

    def __init__(self):
        super().__init__('person_follower')

        self.declare_parameter('person_topic', '/zed/person_keypoints')
        self.declare_parameter('goal_update_topic', '/goal_update')
        self.declare_parameter('state_topic', '/follow/state')
        # Preview following di RViz: orang target, garis robot->orang, dan
        # titik goal yang dikirim ke GoalUpdater. Tanpa ini Following
        # "buta": tidak terlihat siapa yang jadi target atau jaraknya berapa.
        self.declare_parameter('marker_topic', '/follow/markers')
        self.declare_parameter('target_pose_topic', '/follow/target')
        self.declare_parameter('navigate_action', '/navigate_to_pose')
        self.declare_parameter('frame_map', 'map')
        self.declare_parameter('frame_base', 'base_link')
        self.declare_parameter('follow_start_distance', 0.60)
        self.declare_parameter('follow_stop_distance', 0.50)
        self.declare_parameter('path_truncate_distance', 0.50)
        self.declare_parameter('max_follow_range', 3.0)
        self.declare_parameter('wait_person_timeout', 0.0)
        self.declare_parameter('max_lost_time', 3.0)
        self.declare_parameter('update_hz', 3.0)
        self.declare_parameter('target_index', 0)
        self.declare_parameter('follow_bt_xml', '')
        self.declare_parameter('action_server_wait_sec', 3.0)
        # Jeda antar percobaan kirim goal saat Nav2 belum siap. Following
        # otomatis dilanjutkan, jadi tidak perlu panggil /follow/person lagi.
        self.declare_parameter('send_retry_sec', 3.0)

        # Fusi LiDAR (/scan) & Peta Statis (/map)
        self.declare_parameter('enable_lidar_fusion', True)
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('lidar_gate_radius', 0.50)
        self.declare_parameter('lidar_min_points', 3)
        self.declare_parameter('lidar_max_staleness', 0.5)
        self.declare_parameter('map_cost_threshold', 50)
        self.declare_parameter('wall_margin_cells', 1)

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self._person_topic = str(p('person_topic'))
        self._goal_update_topic = str(p('goal_update_topic'))
        self._state_topic = str(p('state_topic'))
        self._marker_topic = str(p('marker_topic'))
        self._target_topic = str(p('target_pose_topic'))
        self._action_name = str(p('navigate_action'))
        self._frame_map = str(p('frame_map'))
        self._frame_base = str(p('frame_base'))
        self._d_start = float(p('follow_start_distance'))
        self._d_stop = float(p('follow_stop_distance'))
        self._max_range = float(p('max_follow_range'))
        self._wait_timeout = float(p('wait_person_timeout'))
        self._max_lost = float(p('max_lost_time'))
        self._target_index = int(p('target_index'))
        self._bt_xml = str(p('follow_bt_xml'))
        self._server_wait = float(p('action_server_wait_sec'))
        self._send_retry_sec = float(p('send_retry_sec'))

        self._enable_lidar = bool(p('enable_lidar_fusion'))
        self._scan_topic = str(p('scan_topic'))
        self._map_topic = str(p('map_topic'))
        self._lidar_gate_radius = float(p('lidar_gate_radius'))
        self._lidar_min_points = int(p('lidar_min_points'))
        self._lidar_max_staleness = float(p('lidar_max_staleness'))
        self._map_cost_threshold = int(p('map_cost_threshold'))
        self._wall_margin_cells = int(p('wall_margin_cells'))

        update_hz = float(p('update_hz'))
        if update_hz <= 0.0:
            update_hz = 3.0

        if self._d_stop > self._d_start:
            raise RuntimeError(
                f'follow_stop_distance ({self._d_stop}) harus <= '
                f'follow_start_distance ({self._d_start})')
        if not self._bt_xml:
            # Fallback biar `ros2 run` tanpa launch tetap bisa dipakai.
            self._bt_xml = default_follow_bt()
            self.get_logger().warn(
                'follow_bt_xml kosong, memakai fallback: '
                f'{self._bt_xml}')
        if not os.path.isfile(self._bt_xml):
            raise RuntimeError(f'behavior tree tidak ditemukan: {self._bt_xml}')

        # --- state ---
        self._state = IDLE
        self._lock = threading.Lock()
        self._goal_handle = None
        self._last_person_seen = None   # wall time orang terakhir terlihat
        self._waiting_since = None
        self._latest_person_map = None  # (x, y, yaw) di frame map
        # ID marker RViz (dipakai juga saat DELETE).
        self._marker_person = 1
        self._marker_line = 2
        self._marker_text = 3
# _send_follow_goal() bisa memblokir sampai action_server_wait_sec detik
        # menunggu Nav2. Topik /zed/person_keypoints datang terus (bisa 30 Hz), jadi
        # tanpa penanda ini tiap pesan akan membuka thread baru yang sama-sama
        # menunggu -- exhausting executor.
        self._sending = False
        self._last_goal_pub = 0.0

        # Data LiDAR & Map
        self._latest_scan = None
        self._last_scan_time = 0.0
        self._cached_laser_tf = None
        self._cached_laser_tf_time = 0.0
        self._latest_map = None

        self._group = ReentrantCallbackGroup()
        self._client = ActionClient(self, NavigateToPose, self._action_name)

        # TF base_link -> map. Buffer harus hidup di thread yang spun, makanya
        # dibuat di sini (bukan di main) supaya ikut ter-spin.
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # QoS latching untuk state: subscriber baru (mis. `ros2 topic echo
        # /follow/state --once`) langsung menerima state terkini tanpa harus
        # menunggu perubahan berikutnya.
        state_qos = QoSProfile(depth=1)
        state_qos.reliability = ReliabilityPolicy.RELIABLE
        state_qos.history = HistoryPolicy.KEEP_LAST
        state_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        # QoS untuk /goal_update: standar reliable queue depth 10 agar cocok
        # langsung dengan GoalUpdater Nav2.
        goal_update_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._goal_update_pub = self.create_publisher(
            PoseStamped, self._goal_update_topic, goal_update_qos)
        self._state_pub = self.create_publisher(
            String, self._state_topic, state_qos)

        # Preview following. QoS latching juga: subscriber baru (mis. RViz
        # yang baru dibuka) langsung melihat target terakhir.
        self._marker_pub = self.create_publisher(
            MarkerArray, self._marker_topic, state_qos)
        self._target_pub = self.create_publisher(
            PoseStamped, self._target_topic, state_qos)

        self._person_sub = self.create_subscription(
            String, self._person_topic, self._on_person_msg, 10,
            callback_group=self._group)

        if self._enable_lidar:
            self._scan_sub = self.create_subscription(
                LaserScan, self._scan_topic, self._on_scan_msg,
                qos_profile_sensor_data,
                callback_group=self._group)

            map_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._map_sub = self.create_subscription(
                OccupancyGrid, self._map_topic, self._on_map_msg,
                map_qos,
                callback_group=self._group)

        self._start_srv = self.create_service(
            Trigger, '/follow/person', self._on_start,
            callback_group=self._group)
        self._stop_srv = self.create_service(
            Trigger, '/follow/stop', self._on_stop,
            callback_group=self._group)

        self._update_timer = self.create_timer(
            1.0 / update_hz, self._publish_goal_update,
            callback_group=self._group)
        self._watchdog = self.create_timer(
            1.0, self._check_watchdog, callback_group=self._group)

        self.get_logger().info(
            f'person_follower siap. {self._person_topic} -> '
            f'{self._goal_update_topic}; deadband '
            f'{self._d_stop:.2f}..{self._d_start:.2f} m; '
            f'bt={self._bt_xml}')
        self._publish_state()

    # ------------------------------------------------------------------ #
    # util                                                                #
    # ------------------------------------------------------------------ #

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _publish_state(self):
        msg = String()
        msg.data = self._state
        self._state_pub.publish(msg)

    def _set_state(self, state: str):
        if self._state == state:
            return
        self._state = state
        self.get_logger().info(f'state -> {state}')
        self._publish_state()

    def _on_scan_msg(self, msg: LaserScan):
        self._latest_scan = msg
        self._last_scan_time = self._now()

    def _on_map_msg(self, msg: OccupancyGrid):
        self._latest_map = msg
        self.get_logger().info(
            f'Peta statis diterima untuk LiDAR fusion: {msg.info.width}x{msg.info.height} '
            f'res={msg.info.resolution:.3f}m origin=({msg.info.origin.position.x:.2f}, {msg.info.origin.position.y:.2f})',
            throttle_duration_sec=30.0)

    def _get_laser_to_base_tf(self, laser_frame: str):
        if not laser_frame or laser_frame == self._frame_base:
            return (0.0, 0.0, 0.0)
        now = self._now()
        if self._cached_laser_tf is not None and (now - self._cached_laser_tf_time) < 5.0:
            return self._cached_laser_tf
        try:
            t = self._tf_buffer.lookup_transform(
                self._frame_base, laser_frame, rclpy.time.Time())
            tx = t.transform.translation.x
            ty = t.transform.translation.y
            q = t.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            self._cached_laser_tf = (tx, ty, yaw)
            self._cached_laser_tf_time = now
            return self._cached_laser_tf
        except Exception as exc:
            self.get_logger().warn(
                f'Belum dapat TF {self._frame_base} <- {laser_frame}: {exc}',
                throttle_duration_sec=5.0,
                skip_first=True)
            return None

    def _is_static_obstacle_in_map(self, xm: float, ym: float) -> bool:
        """True jika koordinat (xm, ym) di frame map menabrak dinding/rintangan statis."""
        map_msg = self._latest_map
        if map_msg is None:
            return False
        info = map_msg.info
        res = info.resolution
        if res <= 0.0:
            return False

        ox = info.origin.position.x
        oy = info.origin.position.y
        oq = info.origin.orientation
        siny = 2.0 * (oq.w * oq.z + oq.x * oq.y)
        cosy = 1.0 - 2.0 * (oq.y * oq.y + oq.z * oq.z)
        oyaw = math.atan2(siny, cosy)

        dx = xm - ox
        dy = ym - oy
        if abs(oyaw) > 1e-4:
            c = math.cos(-oyaw)
            s = math.sin(-oyaw)
            u = dx * c - dy * s
            v = dx * s + dy * c
        else:
            u = dx
            v = dy

        col0 = int(math.floor(u / res))
        row0 = int(math.floor(v / res))

        w = info.width
        h = info.height
        data = map_msg.data
        margin = self._wall_margin_cells
        thresh = self._map_cost_threshold

        # Di luar batas peta dianggap rintangan/tidak valid
        if col0 < 0 or col0 >= w or row0 < 0 or row0 >= h:
            return True

        for dr in range(-margin, margin + 1):
            r = row0 + dr
            if 0 <= r < h:
                row_off = r * w
                for dc in range(-margin, margin + 1):
                    c = col0 + dc
                    if 0 <= c < w:
                        val = data[row_off + c]
                        if val > thresh:
                            return True
        return False

    def _refine_with_lidar(self, zx: float, zy: float, robot_x: float, robot_y: float, robot_yaw: float):
        """Memperhalus posisi orang (zx, zy) di base_link menggunakan pantulan dinamis LiDAR.

        Pantulan yang menabrak dinding atau rintangan pada peta statis (/map) disaring.
        Hanya pantulan di area bebas / dinamis yang diagregasi (median cluster).
        """
        if not self._enable_lidar:
            return (zx, zy, False)

        scan = self._latest_scan
        if scan is None:
            return (zx, zy, False)

        now = self._now()
        if (now - self._last_scan_time) > self._lidar_max_staleness:
            return (zx, zy, False)

        laser_tf = self._get_laser_to_base_tf(scan.header.frame_id)
        if laser_tf is None:
            return (zx, zy, False)

        ltx, lty, lyaw = laser_tf
        cos_lyaw = math.cos(lyaw)
        sin_lyaw = math.sin(lyaw)

        d_cam = math.hypot(zx, zy)
        gate_r = self._lidar_gate_radius
        r_min_gate = max(0.05, d_cam - gate_r)
        r_max_gate = d_cam + gate_r
        gate_r_sq = gate_r * gate_r

        ranges = scan.ranges
        n_ranges = len(ranges)
        if n_ranges == 0:
            return (zx, zy, False)

        angle_min = scan.angle_min
        angle_inc = scan.angle_increment
        scan_rmin = scan.range_min
        scan_rmax = scan.range_max

        cos_ryaw = math.cos(robot_yaw)
        sin_ryaw = math.sin(robot_yaw)

        dynamic_points = []

        for i in range(n_ranges):
            r = ranges[i]
            if not (math.isfinite(r) and scan_rmin <= r <= scan_rmax):
                continue
            if r < r_min_gate or r > r_max_gate:
                continue

            alpha = angle_min + i * angle_inc
            xl = r * math.cos(alpha)
            yl = r * math.sin(alpha)

            # Transform ke base_link
            xb = ltx + xl * cos_lyaw - yl * sin_lyaw
            yb = lty + xl * sin_lyaw + yl * cos_lyaw

            dx = xb - zx
            dy = yb - zy
            if (dx * dx + dy * dy) > gate_r_sq:
                continue

            # Transform ke map frame untuk cek terhadap peta statis
            xm = robot_x + xb * cos_ryaw - yb * sin_ryaw
            ym = robot_y + xb * sin_ryaw + yb * cos_ryaw

            if self._is_static_obstacle_in_map(xm, ym):
                continue

            dynamic_points.append((xb, yb))

        if len(dynamic_points) < self._lidar_min_points:
            return (zx, zy, False)

        xs = [p[0] for p in dynamic_points]
        ys = [p[1] for p in dynamic_points]
        xs.sort()
        ys.sort()
        mid = len(xs) // 2
        med_x = xs[mid] if len(xs) % 2 == 1 else (xs[mid - 1] + xs[mid]) / 2.0
        med_y = ys[mid] if len(ys) % 2 == 1 else (ys[mid - 1] + ys[mid]) / 2.0

        return (med_x, med_y, True)

    def _person_pose_in_map(self, person):
        """-> (x, y, yaw_robot_to_person, dist, fused) di frame map, atau None."""
        try:
            trans = self._tf_buffer.lookup_transform(
                self._frame_map, self._frame_base, rclpy.time.Time())
        except (LookupException, ConnectivityException,
                ExtrapolationException, TransformException) as exc:
            # Rate-limited: pesan ini muncul tiap frame ZED bisa 30 Hz.
            # Tanpa throttle, log jadi penuh dan menutupi penyebab sebenarnya.
            self.get_logger().warn(
                f'TF {self._frame_map} <- {self._frame_base} belum ada, '
                f'following tidak bisa jalan: {exc}\n'
                f'  -> ini masalah lokalisasi, bukan person_follower. '
                f'Cek: ros2 run tf2_ros tf2_echo {self._frame_map} '
                f'{self._frame_base}',
                throttle_duration_sec=5.0,
                skip_first=True,
            )
            self._tf_missing_logged = True
            return None
        self._tf_missing_logged = False

        rx = trans.transform.translation.x
        ry = trans.transform.translation.y
        rq = trans.transform.rotation
        siny_cosp = 2.0 * (rq.w * rq.z + rq.x * rq.y)
        cosy_cosp = 1.0 - 2.0 * (rq.y * rq.y + rq.z * rq.z)
        ryaw = math.atan2(siny_cosp, cosy_cosp)

        # Koordinat orang dari ZED (di base_link)
        zx = person['x']
        zy = person['y']

        # Perhalus dengan LiDAR jika pantulan dinamis di area bebas ditemukan
        bx, by, fused = self._refine_with_lidar(zx, zy, rx, ry, ryaw)

        pose_base = PoseStamped()
        pose_base.header.frame_id = self._frame_base
        pose_base.header.stamp = self.get_clock().now().to_msg()
        # Data zed sudah relatif robot: x depan, y kiri, z atas.
        # Cocok dengan REP-103 base_link (X forward, Y left, Z up).
        pose_base.pose.position.x = bx
        pose_base.pose.position.y = by
        # z diabaikan: Nav2 plan 2D, dan z dari kamera kurang akurat untuk
        # collision check. Yang penting x dan y.
        pose_base.pose.position.z = 0.0
        pose_base.pose.orientation.w = 1.0

        pose_map = do_transform_pose_stamped(pose_base, trans)
        mx = pose_map.pose.position.x
        my = pose_map.pose.position.y
        dx = mx - rx
        dy = my - ry
        dist = math.hypot(dx, dy)
        yaw = math.atan2(dy, dx)
        return (mx, my, yaw, dist, fused)

    # ------------------------------------------------------------------ #
    # callback: data dari zed                                             #
    # ------------------------------------------------------------------ #

    def _on_person_msg(self, msg: String):
        try:
            self._handle_person_msg(msg)
        except Exception as exc:  # noqa: BLE001
            # Sensor camera bisa mengirim apa saja (frame dropout, depth
            # NaN, string terpotong). Satu pesan rusak tidak boleh
            # menjatuhkan node -- kalau hilang, following mati diam-diam.
            self.get_logger().error(
                f'gagal proses /zed/person_keypoints: {exc}')

    def _handle_person_msg(self, msg: String):
        people = parse_persons(msg.data)
        if not people:
            return
        if self._state == IDLE:
            return  # tidak ada yang minta following

        idx = min(self._target_index, len(people) - 1)
        located = self._person_pose_in_map(people[idx])
        if located is None:
            return

        x, y, yaw, dist, fused = located
        now = self._now()
        self._last_person_seen = now

        # Dari WAITING ke FOLLOWING begitu orang pertama terlihat.
        if self._state == WAITING_PERSON:
            # Nav2 belum tentu siap saat /follow/person dipanggil (bt_navigator
            # masih configure, atau controller_server belum active karena lidar
            # belum scan). Kalau gagal sekali lalu state langsung IDLE, semua
            # pesan person berikutnya akan di-return di atas dan following TIDAK
            # PERNAH pulih. Jadi: coba berkala sampai berhasil.
            if now < self._next_send_attempt:
                return
            self._next_send_attempt = now + self._send_retry_sec
            if not self._acquire_send():
                return
            try:
                mode_str = 'LiDAR+ZED' if fused else 'ZED'
                self.get_logger().info(
                    f'orang terdeteksi ({mode_str}) di ({x:.2f}, {y:.2f}), '
                    f'jarak {dist:.2f} m -> mulai following')
                ok = self._send_follow_goal(x, y, yaw)
            finally:
                self._sending = False
            if not ok:
                # Tetap di WAITING_PERSON supaya otomatis dicoba lagi.
                # Jangan set IDLE: itu membuat following mati permanen.
                self.get_logger().warn(
                    'goal following belum terkirim, akan dicoba lagi '
                    f'{self._send_retry_sec:.0f} detik sekali. '
                    'Pastikan Nav2 aktif: ros2 lifecycle get /bt_navigator')
                try:
                    self._publish_markers(x, y, yaw, dist, fused)
                except Exception:
                    pass
                self._latest_person_map = (x, y, yaw, now, fused)
                return
            self._set_state(FOLLOWING)

        self._latest_person_map = (x, y, yaw, now, fused)
        # Update goal langsung dikirim bila sedang FOLLOWING (dibatasi maks ~4 Hz agar responsif)
        if self._state == FOLLOWING and (now - self._last_goal_pub) >= 0.25:
            self._publish_goal_update()

    # ------------------------------------------------------------------ #
    # service                                                             #
    # ------------------------------------------------------------------ #

    def _on_start(self, _req, resp):
        with self._lock:
            if self._state != IDLE:
                resp.success = False
                resp.message = f'sudah jalan ({self._state}), panggil /follow/stop dulu'
                return resp
            self._state = WAITING_PERSON
            self._waiting_since = self._now()
            self._last_person_seen = None
            self._latest_person_map = None
            self._next_send_attempt = 0.0

        resp.success = True
        robot = self._robot_pose_map()
        if robot is None:
            # Kasus paling sering: lokalisasi belum jadi -> tidak ada transform
            # map ke robot -> pose orang tak bisa dihitung. Katakan sekarang,
            # jangan sampai pengguna menunggu 60 detik tanpa tahu kenapa.
            resp.message = (
                f'PERINGATAN: TF {self._frame_map} <- {self._frame_base} '
                'belum ada, jadi following tidak akan jalan. AMCL belum '
                'terlocalisasi (cek ros2 topic hz /scan). Panggil ulang nanti.')
            self.get_logger().error(resp.message)
        else:
            timeout_str = f'timeout {self._wait_timeout:.0f}s' if self._wait_timeout > 0.0 else 'tanpa batas waktu'
            resp.message = (
                f'menunggu orang ({timeout_str}). Kalau robot sudah diam di '
                f'{self._frame_map} tapi following tetap tidak jalan, cek '
                'log: "goal following belum terkirim".')
        self.get_logger().info(resp.message)
        self._publish_state()
        return resp

    def _on_stop(self, _req, resp):
        with self._lock:
            was = self._state
            self._state = IDLE
            handle = self._goal_handle
            self._goal_handle = None

        if handle is not None:
            handle.cancel_goal_async()
            self.get_logger().info('goal following dibatalkan')
        self._clear_markers()
        resp.success = True
        resp.message = f'following dihentikan (dari {was})'
        self._publish_state()
        return resp

    # ------------------------------------------------------------------ #
    # action                                                              #
    # ------------------------------------------------------------------ #

    def _acquire_send(self) -> bool:
        """Ambil penanda 'sedang mengirim goal'; False kalau sudah ada."""
        with self._lock:
            if self._sending:
                return False
            self._sending = True
            return True

    def _send_follow_goal(self, x: float, y: float, yaw: float) -> bool:
        if not self._client.wait_for_server(timeout_sec=self._server_wait):
            self.get_logger().error(
                f'action {self._action_name} tidak aktif '
                '(nav2 belum jalan / belum selesai inisialisasi)')
            return False

        pose = PoseStamped()
        pose.header.frame_id = self._frame_map
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.x, pose.pose.orientation.y, \
            pose.pose.orientation.z, pose.pose.orientation.w = yaw_to_quat(yaw)

        goal = NavigateToPose.Goal()
        goal.pose = pose
        # Field inilah yang memb lettingkan /follow hidup berdampingan dengan
        # /goto: BT dipilih per-goal, jadi /goto tetap pakai BT miliknya.
        goal.behavior_tree = self._bt_xml

        send_future = self._client.send_goal_async(
            goal, feedback_callback=self._on_feedback)
        ev = threading.Event()
        send_future.add_done_callback(lambda _: ev.set())
        if not ev.wait(timeout=10.0):
            self.get_logger().error('timeout menunggu penerimaan goal dari Nav2')
            return False
        handle = send_future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('goal following ditolak Nav2')
            return False

        self._goal_handle = handle
        self.get_logger().info(
            f'following aktif: ({x:.2f}, {y:.2f}) trunc={self._d_stop:.2f} m')

        result_future = handle.get_result_async()
        result_future.add_done_callback(self._on_result)
        return True

    def _on_feedback(self, _msg):
        """Feed back tidak dipakai; hanya menjaga goal tetap hidup."""
        pass

    def _on_result(self, future):
        status = future.result().status if future.result() else None
        self._goal_handle = None
        if self._state == FOLLOWING:
            if status == GoalStatus.STATUS_CANCELED:
                self.get_logger().info('goal following dibatalkan')
                self._set_state(IDLE)
            else:
                self.get_logger().warn(
                    f'Nav2 goal selesai/aborted (status {status}), kembali ke WAITING_PERSON untuk re-track target...')
                self._set_state(WAITING_PERSON)

    # ------------------------------------------------------------------ #
    # timer: kirim pose terbaru                                          #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # preview (RViz)                                                     #
    # ------------------------------------------------------------------ #

    def _clear_markers(self):
        """Kosongkan marker supaya tidak ada sisa saat stop / IDLE."""
        self._marker_pub.publish(MarkerArray())  # hapus semua marker
        for mid in (self._marker_person, self._marker_line, self._marker_text):
            m = Marker()
            m.header.frame_id = self._frame_map
            m.header.stamp = self.get_clock().now().to_msg()
            m.id = mid
            m.action = Marker.DELETE
            self._marker_pub.publish(MarkerArray(markers=[m]))

    def _publish_markers(self, x: float, y: float, yaw: float, dist: float, fused: bool = False):
        """Tampilkan di RViz: orang target + garis + teks jarak & state."""
        stamp = self.get_clock().now().to_msg()
        robot = self._robot_pose_map()

        # 1. Orang target: bola (hijau jika LiDAR fused, biru jika ZED only)
        person = Marker()
        person.header.frame_id = self._frame_map
        person.header.stamp = stamp
        person.id = self._marker_person
        person.ns = 'follow_target'
        person.type = Marker.SPHERE
        person.action = Marker.ADD
        person.pose.position.x = x
        person.pose.position.y = y
        person.pose.position.z = 0.9
        person.pose.orientation.z = math.sin(yaw / 2.0)
        person.pose.orientation.w = math.cos(yaw / 2.0)
        person.scale.x = 0.45
        person.scale.y = 0.45
        person.scale.z = 0.45
        if fused:
            person.color = ColorRGBA(r=0.0, g=0.90, b=0.40, a=0.90)
        else:
            person.color = ColorRGBA(r=0.20, g=0.40, b=1.00, a=0.90)
        person.lifetime = Duration(seconds=2.0).to_msg()

        line = Marker()
        line.header.frame_id = self._frame_map
        line.header.stamp = stamp
        line.id = self._marker_line
        line.ns = 'follow_line'
        line.type = Marker.LINE_LIST
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.04
        if fused:
            line.color = ColorRGBA(r=0.10, g=1.00, b=0.60, a=0.85)
        else:
            line.color = ColorRGBA(r=0.20, g=0.80, b=1.00, a=0.75)
        line.lifetime = Duration(seconds=2.0).to_msg()
        if robot is not None:
            line.points = [
                Point(x=robot[0], y=robot[1], z=0.25),
                Point(x=x, y=y, z=0.25),
            ]

        # 2. Teks: jarak sekarang + batas deadband + status sensor
        sensor_tag = 'LIDAR+CAM' if fused else 'CAM ONLY'
        near = dist <= self._d_stop
        far = dist > self._d_start
        label = (
            f'FOLLOW [{sensor_tag}] d={dist:.2f}m  '
            f'[stop<={self._d_stop:.2f} start>{self._d_start:.2f}]'
        )
        if near:
            label += '  DEKAT: diam'
        elif far:
            label += '  DEKATKAN'
        else:
            label += '  zona mati'
        text = Marker()
        text.header.frame_id = self._frame_map
        text.header.stamp = stamp
        text.id = self._marker_text
        text.ns = 'follow_text'
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = x
        text.pose.position.y = y
        text.pose.position.z = 1.45
        text.pose.orientation.w = 1.0
        text.scale.z = 0.28
        if fused:
            text.color = ColorRGBA(r=0.20, g=1.00, b=0.50, a=1.0)
        else:
            text.color = ColorRGBA(
                r=1.0 if far else 0.2, g=1.0 if far else 0.9, b=0.2, a=1.0)
        text.text = label
        text.lifetime = Duration(seconds=2.0).to_msg()

        self._marker_pub.publish(
            MarkerArray(markers=[person, line, text]))

        pose = PoseStamped()
        pose.header.frame_id = self._frame_map
        pose.header.stamp = stamp
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        self._target_pub.publish(pose)

    def _robot_pose_map(self):
        """Posisi robot di frame map, atau None."""
        try:
            trans = self._tf_buffer.lookup_transform(
                self._frame_map, self._frame_base, rclpy.time.Time())
        except (LookupException, ConnectivityException,
                ExtrapolationException, TransformException):
            return None
        return (trans.transform.translation.x, trans.transform.translation.y)

    # ------------------------------------------------------------------ #

    def _publish_goal_update(self):
        # WAITING_PERSON ikut: preview perlu menampilkan target begitu ada,
        # walau goal belum bisa dikirim (Nav2 belum siap).
        if self._state not in (FOLLOWING, WAITING_PERSON) \
                or self._latest_person_map is None:
            return
        x, y, yaw, stamp, *rest = self._latest_person_map
        fused = rest[0] if rest else False

        # Hanya FOLLOWING yang boleh mengirim goal ke BT.
        send_goal = (self._state == FOLLOWING)

        # Jarak dihitung ulang dari transform terbaru supaya tidak basi.
        pose = PoseStamped()
        pose.header.frame_id = self._frame_map
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.x, pose.pose.orientation.y, \
            pose.pose.orientation.z, pose.pose.orientation.w = yaw_to_quat(yaw)

        # SELALU kirim posisi terbaru orang ke Nav2 GoalUpdater segera
        if send_goal:
            self._goal_update_pub.publish(pose)
            self._last_goal_pub = self._now()

        # Marker untuk RViz visualisasi
        dist = 0.0
        try:
            trans = self._tf_buffer.lookup_transform(
                self._frame_map, self._frame_base, rclpy.time.Time())
            dist = math.hypot(
                x - trans.transform.translation.x,
                y - trans.transform.translation.y)
        except (LookupException, ConnectivityException,
                ExtrapolationException, TransformException):
            pass

        try:
            self._publish_markers(x, y, yaw, dist, fused)
        except Exception as exc:
            self.get_logger().warn(
                f'Gagal publish marker RViz: {exc}',
                throttle_duration_sec=5.0)

    # ------------------------------------------------------------------ #
    # timer: pengaman                                                     #
    # ------------------------------------------------------------------ #

    def _check_watchdog(self):
        now = self._now()

        if self._state == WAITING_PERSON:
            if self._wait_timeout > 0.0 and self._waiting_since is not None and \
                    now - self._waiting_since > self._wait_timeout:
                self.get_logger().warn(
                    f'tidak ada orang dalam {self._wait_timeout:.0f}s, '
                    'following dibatalkan')
                self._on_stop(None, Trigger.Response())
            return

        if self._state != FOLLOWING:
            return

        if self._last_person_seen is None:
            return
        lost = now - self._last_person_seen
        if self._max_lost > 0.0 and lost > self._max_lost:
            self.get_logger().warn(
                f'tidak ada deteksi orang selama {lost:.1f}s '
                f'(> {self._max_lost:.0f}s) -> jeda, menunggu orang terlihat kembali...')
            with self._lock:
                handle = self._goal_handle
                self._goal_handle = None
                self._state = WAITING_PERSON
                self._waiting_since = now
            if handle is not None:
                handle.cancel_goal_async()
            self._publish_state()
            return

        if self._latest_person_map is None:
            return
        x, y, _yaw, _stamp, *_ = self._latest_person_map
        try:
            trans = self._tf_buffer.lookup_transform(
                self._frame_map, self._frame_base, rclpy.time.Time())
        except (LookupException, ConnectivityException,
                ExtrapolationException, TransformException):
            return
        dist = math.hypot(
            x - trans.transform.translation.x,
            y - trans.transform.translation.y)
        if dist > self._max_range:
            self.get_logger().warn(
                f'orang terlalu jauh ({dist:.2f} m > '
                f'{self._max_range:.2f} m) -> stop following')
            self._on_stop(None, Trigger.Response())


def main(args=None):
    rclpy.init(args=args)
    node = PersonFollower()
    # WAJIB multi-thread: service, subscription, timer, dan pendingan hasil
    # action jalan bersamaan.
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