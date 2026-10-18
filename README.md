# at-home-ws (ROS 2 Jazzy)

## Folder structure
```
src/                # ROS packages only (colcon)
├── nucleo/          # robot movement 3 motor (ala NucleoServer: main.cpp + src/Kinematic)
│   ├── main.cpp                  # program utama (node + loop, MKiri/MKanan/MBelakang)
│   ├── launch/nucleo.launch.py   # use_yaw_topic:=true yaw_topic:=/zed/yaw_deg
│   ├── src/Kinematic.cpp, udpSender.cpp, udpServer.cpp  # lib murni
│   └── include/*.h(pp) flat      # Kinematic.hpp, nucleo_node.hpp, udpSender.h, udpServer.h
├── zed/             # kamera: yaw + body tracking + YOLO tas (ala zed_ws: src/main.cpp)
│   ├── src/main.cpp              # SATU file: struct + vision murni + YOLO tas + node + main
│   ├── include/deploy/...        # header TensorRT opsional (OFF default)
│   ├── launch/zed.launch.py      # enable_body_tracking:=true
│   └── models/yolov8n_sim.onnx + coco_names.txt  # YOLO tas (bag), tinggal pake
│   # publish: /zed/yaw_deg, /zed/bag_detections, /zed/person_keypoints
├── robot_bringup/   # shared sensors (do not run manually)
│   ├── launch/sensors.launch.py  # auto-included by mapping/navigation
│   │   ├── sensor 1: RPLiDAR (rplidar_node -> /scan, full 360°, ~14 Hz)
│   │   ├── static TF: base_link -> laser_frame, base_footprint -> base_link
│   │   └── sensor 2: ZED camera (yaw + body tracking + YOLO tas)
│   └── config/lidar.yaml  # lidar driver params (filter 180° dilepas)
├── mapping/         # room mapping (requires teleop)
│   ├── launch/mapping.launch.py  # includes sensors + slam_toolbox (toolbox only)
│   ├── config/slam.yaml          # travel thresholds 0.15, update 2 s
│   └── rviz/mapping.rviz
├── navigation/      # point-to-point using the map
│   ├── launch/navigation.launch.py  # include sensors + nav2_bringup (+ rviz args)
│   ├── config/nav2_params.yaml
│   ├── maps/  # SINGLE source of truth: map.yaml + map.pgm (via map_saver_cli)
│   ├── maps_backup/  # arsip peta lama (tidak ikut ter-install)
│   └── rviz/navigation.rviz
├── luggage_task/    # Task 1 Carry My Luggage (state machine flowchart)
│   ├── launch/  # split-stack: hardware (base+lidar), camera (zed),
│   │   │       # motion (task+SLAM+RViz); + luggage_task.launch.py (1 perintah, via Nav2)
│   ├── config/params.yaml  # SATU file param (27 kunci, section luggage_task)
│   │   # (tuning: edit yaml + restart launch, TANPA build ulang)
│   ├── rviz/luggage_task.rviz         # Fixed Frame: map (/map + TF + /odom + /scan)
│   ├── include/luggage_task/task.hpp  # tipe + state + deklarasi logika
│   └── src/task/task_node.cpp, task_logic.cpp  # node ROS + state machine murni
docs/               # RUNNING.md, REQUIREMENTS.md, ARCHITECTURE.md
```

## Run
Full instructions: `docs/RUNNING.md`. Dependencies/hardware: `docs/REQUIREMENTS.md`. Arsitektur: `docs/ARCHITECTURE.md`.

### 0. One-time setup (per machine)
```bash
cd ~/Documents/at-home-ws
bash src/zed/models/setup_onnxruntime.sh   # sekali saja: ONNX Runtime C++ v1.22 (YOLO zed)
colcon build --symlink-install
source install/setup.bash   # required in EVERY new terminal
sudo apt install -y ros-jazzy-rplidar-ros
sudo usermod -aG dialout $USER   # then LOGOUT + LOGIN (lidar USB access)
```
> Setup script pasang ke `/opt/onnxruntime` (butuh sudo) atau fallback
> `$HOME/onnxruntime` (tanpa sudo); CMake menemukan keduanya otomatis,
> atau via `ONNXRUNTIME_ROOT=<dir>`. Butuh juga ZED SDK + CUDA
> (header `cuda.h`) — paket `zed` tidak bisa build tanpa keduanya.
> Workspace habis dipindah/copy dari path lain (error CMakeCache
> menunjuk path lama)? Bersihkan dulu: `rm -rf build install log`,
> lalu build ulang.

### 1. Mapping
```bash
# T1 base driver:
ros2 launch nucleo nucleo.launch.py use_yaw_topic:=true yaw_topic:=/zed/yaw_deg
# T2 mapping:
ros2 launch mapping mapping.launch.py
# T3 teleop:
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p speed:=0.6 -p turn:=1.5
```
Check: `ros2 topic hz /scan` (~15 Hz), `/map` (~0.5 Hz), `/odom` (~33 Hz).

### 2. Save the map
```bash
ros2 run nav2_map_server map_saver_cli -f $(pwd)/src/navigation/maps/map --ros-args -p save_map_timeout:=10.0
```
JANGAN save ke root workspace (`-f map` dari root): `map.pgm`/`map.yaml`
nyasar di luar `src/` dan tidak ter-install. Satu-satunya lokasi peta
adalah `src/navigation/maps/` (`luggage_task` me-reuse peta ini via
`nav2:=true`, tidak punya `maps/` sendiri). Peta lama diarsipkan di
`src/navigation/maps_backup/`.

### 3. Navigation (SLAM: mapping sambil bernavigasi)
```bash
# T1 base driver:
ros2 launch nucleo nucleo.launch.py
# T2 nav (default slam:=true: peta dibangun + dipakai Nav2 bersamaan):
ros2 launch navigation navigation.launch.py
# Peta statis + AMCL (bila sudah punya peta bagus):
ros2 launch navigation navigation.launch.py slam:=false map:=$(pwd)/src/navigation/maps/map.yaml
```
Kirim tujuan via RViz tombol 2D Goal Pose.

### 4. Luggage task
Syarat: kamera ZED di USB3 + lidar di `/dev/ttyUSB0` + base `nucleo` menyala.
`source install/setup.bash` di tiap terminal. Boolean huruf kecil (`true`).
Mode saat ini: **gesture-dulu** (`gesture_only: true` di
`src/luggage_task/config/params.yaml`) — deteksi menunjuk, belum lanjut
ke SEARCH_BAG/FOLLOW.

**Split-stack (3 terminal, rekomen):**
```bash
# T1 base + lidar:
ros2 launch luggage_task hardware.launch.py
# T2 kamera (vision):
ros2 launch luggage_task camera.launch.py
# T3 task + SLAM + RViz:
ros2 launch luggage_task motion.launch.py
# T4 opsional HRI:
ros2 launch luggage_task hri.launch.py
```

**Satu launch (alternatif): task + nav2 + nucleo sekaligus**
```bash
ros2 launch luggage_task luggage_task.launch.py nav2:=true nucleo:=true rviz:=false enable_body_tracking:=true
```
- `nucleo:=true` = ikutkan base driver STM32 (sub `cmd_vel`, pub odom/TF).
- `nav2:=true` = ikutkan `navigation.launch.py` (lidar + zed + SLAM/Nav2).
- `rviz:=false` = tanpa RViz (pakai `true` bila mau lihat peta).
- `stm_ip:=192.168.0.X` bila IP STM32 beda.
- `depth_minimum_distance:=30.0` = 30cm ala maen_boss (units CENTIMETER).

**Frame patah-patah?** VGA memang kecil, yang berat itu prosesnya. Ringankan
bila perlu: `body_model:=FAST`, `depth_mode:=PERFORMANCE`, `image_stride:=3`
(publish tiap 3 frame), `camera_fps:=30.0` (30 = nilai aman; turunkan ke 15
bila firmware mendukung). Pastikan tidak ada proses lain yang buka kamera
(1 kamera = 1 proses). Butuh akurat (bukan halus):
`body_model:=MEDIUM enable_yolo:=true`.

**Cek hidup (terminal lain):**
```bash
ros2 topic hz /scan /zed/yaw_deg /zed/person_keypoints /zed/image  # harap ~14/~/30 Hz
ros2 topic echo /luggage/status     # state machine task
ros2 topic echo /luggage/target_dir # "kiri"/"kanan"/"bawah"
ros2 topic echo /zed/person_keypoints  # "person:x,y,w,h|x,y,z|yaw,rel|kpt=...;..." (urut TERDEKAT dulu)
```

**Window preview (yang nampilin = `luggage_task`, `zed` hanya publish):**
- `zed` publish `/zed/image` (gambar kamera asli/BGR, native 672×376) + keypoints.
- `luggage_task` subs lalu tampilkan window `luggage_task - gesture`:
  preview + kotak hijau (orang TERDEKAT) + 34 titik keypoint mentah ZED +
  teks overlay kiri-atas (`Gesture : ...`, `distance : ...`).
- Hitam + tulisan `menunggu /zed/image ...` = gambar belum masuk (cek zed jalan
  / `publish_image:=true`).
- Window cuma tampil `BAWAH KIRI` / `BAWAH KANAN` (= **perspektif person**:
  tangan kanan = `KANAN`) + distance. Cek terminal ada
  `task start: ... mirror=true ...`. Matikan: `mirror:=false`.
- Delay preview terasa? `depth_mode` default sudah `PERFORMANCE` (ringan);
  balik ke `ULTRA` ala maen_boss bila perlu. `image_stride:=3` untuk
  publish tiap 3 frame.


**Tuning param (TANPA build):** semua di `src/luggage_task/config/params.yaml`
(27 kunci, tiap baris ada komentar `##`). Ubah → restart launch.
Full task (lanjut SEARCH_BAG→FOLLOW): set `gesture_only: false` + restart.

**Cara main (operator) — mode gesture-dulu:**
1. Berdiri di depan robot (orang PALING DEKAT = yang terdeteksi).
2. Tunjuk diagonal bawah ke kiri/kanan → window `luggage_task - gesture` tampil
   `BAWAH KIRI` / `BAWAH KANAN` + jarak.

**Cara main (operator):**
1. Berdiri di depan robot menghadap kamera → `DETECT_PERSON` → `PERSON_POSE`.
2. Tunjuk diagonal bawah ke arah tas (tangan kiri/kanan) → `DETECT_GESTURE` → `TARGET_DIR`.
3. Robot memutar menghadap tas (`ROTATE_TO_BAG`), lalu maju sampai 0.5 m darimu dan berhenti (`APPROACH`).
4. Membelakangi robot → `WAIT_TURN` → `TRACK`.
5. Berjalan pelan — robot mengikuti jaga jarak ~0.5 m (`FOLLOW`) sampai durasi habis / `goal_reached` → `DONE`.

Lalu di RViz: klik **2D Pose Estimate**, drag di posisi robot sebenarnya
(wajib untuk AMCL). Langkah ini TIDAK bisa dilewat: AMCL baru menerbitkan
TF `map->odom` SETELAH menerima pose awal. Tanpa ini RViz selamanya
"Fixed Frame map does not exist" + "no map received" walau semua node
hijau. Alternatif via terminal:
```bash
ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{header: {frame_id: 'map'}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}, covariance: [0.25,0,0,0,0,0, 0,0.25,0,0,0,0, 0,0,0.068,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0.068]}}"
```
Verifikasi TF hidup: `ros2 run tf2_ros tf2_echo map odom` harus
menampilkan `Translation: [...]` (bukan `Waiting for transform`).
Varian lain: `slam:=true` (peta SLAM live), `map:=/path/map.yaml` (peta
lain), tanpa arg (task saja, tanpa peta).
> PENTING: nilai boolean launch **case-sensitive, harus huruf kecil semua**:
> `nav2:=true` BUKAN `nav2:=True`. Dengan huruf kapital, kondisi
> `IfCondition` tidak terpenuhi sehingga stack Nav2/sensor/kamera
> **diam-diam tidak dijalankan** (di log hanya muncul `luggage_task_node`
> + `rviz2`, tanpa node nav2/zed/lidar) lalu task ABORT karena
> `/zed/person_keypoints` dan `/scan` tidak datang.
Butuh topik: `/zed/person_keypoints`, `/zed/bag_detections`, `/zed/yaw_deg`,
`/scan`, `/luggage/goal_reached` (opsional).
Output: `cmd_vel`, `/luggage/status`, `/luggage/target_dir`.
Alur state: START → DETECT_PERSON → PERSON_POSE → DETECT_GESTURE →
TARGET_DIR → SEARCH_BAG → BAG_POSE → ROTATE_TO_BAG → APPROACH (0.5 m) →
WAIT_TURN → TRACK → FOLLOW (jaga 0.5 m) → DONE/ABORT.
Yang tampil di mana:
- **Peta `.pgm` + posisi robot** → RViz (`/map` + TF `map->odom` dari
  AMCL/SLAM). File `.pgm` simpanan hanya dipakai bila `slam:=false`.
  Bare launch (tanpa arg) TIDAK menampilkan peta (tidak ada yang publish
  `/map` — makanya RViz log "Message Filter dropping ... frame 'odom'").
  Tampilan RViz task disederhanakan: hanya panel Displays + toolbar
  (Selection/Tool Properties/Views disembunyikan).

Semua arg launch punya default, jadi perintah paling minimal (tanpa
params sama sekali) cukup:
```bash
ros2 launch luggage_task luggage_task.launch.py
```
`params_file` default ke `src/luggage_task/config/params.yaml`; kalau file
tidak terbaca node tetap jalan pakai default bawaan kode. Override satu
param tanpa edit file: `ros2 run luggage_task luggage_task_node --ros-args -p follow_speed:=0.35`
(node langsung, tanpa RViz/Nav2 — sensor/nav2 jalankan terpisah).

Deteksi mentah zed (bbox + 3D + nama, dibaca `luggage_task`):
```bash
ros2 topic echo /zed/bag_detections    # "bag:x,y,w,h|x,y,z|conf;..."
ros2 topic echo /zed/person_keypoints  # "person:x,y,w,h|x,y,z|yaw,rel|kpt=...;..."
```
SATU proses saja yang buka kamera (jangan jalankan `zed.launch.py`
standalone bersamaan dengan `nav2:=true` / mapping / navigation).
Catatan run pertama: ZED SDK sekali saja download model body tracking
(~80 MB) + optimasi untuk GPU (beberapa menit, node belum publish topik
selama ini) ke `/usr/local/zed/resources/`; run berikutnya langsung jalan.

Troubleshooting cepat:
- `[ABORT] input tidak datang: /zed/person_keypoints /scan` → sensor belum
  publish. Cek: proses zombie (T0), kabel USB, `/dev/ttyUSB0` + grup
  `dialout` (relogin bila baru ditambah), model AI masih download/optimasi.
- `ZED open failed: CAMERA STREAM FAILED TO START` → kamera dipakai proses
  lain (lihat T0) atau tercolok port USB2 (pindah ke USB3).
- `Couldn't load critics ... FollowPath` + `Failed to bring up` → keterbatasan
  yang diketahui di `nav2_params.yaml` (controller_server); state FOLLOW /
  Nav2 driving belum bisa, state lain task tetap jalan.
- `Couldn't transform ... base_footprint` (amcl) → sudah diperbaiki via
  static TF `base_footprint -> base_link` di `sensors.launch.py`. Kalau masih
  muncul, berarti TF `map->odom` belum ada (Nav2 belum aktif) atau 2D Pose
  Estimate belum di-set.

### Options
- No GPU/display: add `rviz_software:=true`. Headless: add `rviz:=false`.
- Mapping/nav menjalankan ZED hemat (`enable_body_tracking:=false`, yaw-only);
  launcher `luggage_task` default ON (`true`) karena task butuh person.
  Standalone: `ros2 launch zed zed.launch.py enable_body_tracking:=true`.
- Run mapping or navigation, never both. `sensors.launch.py` is auto-included.
- Restart the launch after editing any launch/config/yaml.
