# navigation

Navigasi berbasis **peta statis** (AMCL + Nav2) untuk robot beroda.
Robot memuat peta hasil mapping, melokalisasi dirinya, lalu bisa dikirim ke
koordinat mana pun di peta — lewat RViz (2D Goal Pose) atau lewat service
ROS `/goto/<nama>`.

> Untuk membuat peta baru, pakai paket `mapping` (slam_toolbox). Paket ini
> hanya **menggunakan** peta, tidak membuat peta.

---

## Isi paket

| Path | Keterangan |
|---|---|
| `maps/map.yaml`, `maps/map.pgm` | Peta statis yang dipakai. Ganti file di sini untuk memakai peta lain. |
| `config/nav2_params.yaml` | Parameter Nav2 (AMCL, costmap, planner, controller). |
| `config/places.yaml` | **Daftar lokasi bernama** untuk service `/goto/<nama>`. |
| `config/person_follow.yaml` | Parameter person following (jarak, timeout, topik). |
| `behavior_trees/follow_person.xml` | BT person following (dipilih per-goal, bukan default). |
| `launch/navigation.launch.py` | Launcher utama: sensor + peta + Nav2 + RViz + service lokasi + person following. |
| `scripts/place_navigator.py` | Node yang membuat service `/goto/<nama>`. |
| `scripts/person_follower.py` | Node yang membuat service `/follow/person` + `/follow/stop`. |
| `scripts/goal_pose_bridge.py` | Jembatan `2D Goal Pose` (RViz) → action `NavigateToPose`. |
| `scripts/check_places.py` | Validasi koordinat lokasi terhadap peta (tanpa ROS). |
| `scripts/set_initial_pose.py` | Initial pose dari terminal (alternatif tool `2D Pose Estimate` RViz). |
| `scripts/check_system.sh` | Diagnosa cepat: node, TF, `/scan`, `/odom`, AMCL, costmap, konflik `cmd_vel`. |
| `rviz/navigation.rviz` | Konfigurasi RViz (peta, costmap, path, particle AMCL). |

---

## Robot: Omni 3 Roda

Robot ini **omni 3 roda (Kiwi, sudut 30°/150°/270°)** — bisa bergerak lateral
(`vy`) dan berputar di tempat. Sudah dikonfirmasi dari inverse kinematics
`nucleo/src/Kinematic.cpp`.

Konsekuensinya di `config/nav2_params.yaml`:

| Parameter | Nilai | Arti |
|---|---|---|
| `amcl.robot_model_type` | `OmniMotionModel` | Wajib untuk omni |
| `FollowPath.min_vel_y` / `max_vel_y` | `-0.15` / `0.15` | strafe kiri / kanan |
| `FollowPath.acc_lim_y` / `decel_lim_y` | `2.5` / `-2.5` | percepatan lateral |
| `min_y_velocity_threshold` | `0.02` | **harus < max_vel_y** |
| `velocity_smoother.*[1]` | `±0.15`, `±2.5` | harus sinkron dengan DWB |

> Kalau `velocity_smoother` tidak sinkron dengan DWB, smoother akan memotong
> perintah lateral dan robot tidak akan pernah bergerak menyamping.

---

## Lokalisasi: scan harus nempel ke peta

Robot memperbarui posisi bukan dari odom saja, tapi **odom dikoreksi AMCL memakai
pembacaan lidar terhadap peta**. Kalau scan terlihat bergeser dari peta,
berarti AMCL tidak cukup yakin (drift).

Tiga parameter yang paling menentukan:

| Parameter | Nilai | Arti |
|---|---|---|
| `laser_likelihood_max_dist` | `6.0` | Jarak return laser yang dipakai matching ke peta. **Paling penting.** Peta 8.4×7.65 m — dengan nilai 2.0 hanya ~20% peta yang dilihat AMCL, sehingga drift tidak terkoreksi. |
| `max_beams` | `120` | Jumlah sinar laser per update. Makin banyak = scan makin nempel, CPU naik. |
| `update_min_d` / `update_min_a` | `0.15` / `0.1` | Jarak/putaran minimum sebelum AMCL mengoreksi pose. Makin kecil = koreksi lebih sering. |

Karena robot ini **omni 3 roda**, `alpha3` (noise translasi) dan `alpha4`
(noise rotasi) dinaikkan — roda omni lebih banyak slip dibanding differential,
terutama saat bergerak lateral.

### Kalau scan masih bergeser
1. Cek dengan `ros2 run navigation check_system` — bagian **4b** melaporkan
   confidence AMCL. Confidence rendah = lokalisasi tidak yakin.
2. Pastikan pose awal benar (klik **2D Pose Estimate** di RViz, atau set
   `initial_x/y`).
3. **Kalibrasi odometry** (lihat [Kalibrasi Odometry](#kalibrasi-odometry-error-posisi)
   di bawah) — ini penyebab paling sering "error posisi" yang menetap.
   AMCL mengoreksi drift odom, tapi kalau **skala** odom salah, AMCL akan
   terus mengoreksi dan hasilnya tetap bergeser.
4. `/zed/yaw_deg` jadi sumber yaw odom. Kalau ZED IMU tidak aktif, yaw
   jatuh ke STM32 — pastikan `/zed/yaw_deg` benar-benar terbit
   (`ros2 topic hz /zed/yaw_deg`).

> Catatan tradeoff: mengaktifkan gerak lateral (vy) membuat encoder omni
> **lebih slip**, jadi odom jadi kurang akurat. Karena itu AMCL harus
> dikonfigurasi lebih agresif supaya bisa mengoreksinya.

---

## Kalibrasi Odometry (error posisi)

Posisi robot Nav2 datang dari dua sumber:

| Sumber | Kelebihan | Kekurangan |
|---|---|---|
| `/odom` (encoder + IMU) | cepat, halus | **mengalir** |
| `/amcl_pose` (lidar + peta) | akurat | lambat, hanya koreksi lokal |

Kalau **skala `/odom` salah**, posisi menyimpang terus dan itu muncul
sebagai error posisi. Penyebabnya hampir selalu salah satu dari tiga
konstanta di bawah — yang sebelumnya **hardcoded** dan tidak bisa diubah:

| Parameter | Arti | Default |
|---|---|---|
| `pulses_per_rev` | pulse encoder per putaran roda | `7500.0` |
| `wheel_radius_cm` | jari-jari roda (**bukan** diameter) | `5.0` |
| `base_length_cm` | jarak roda belakang ke poros roda depan | `20.8` |

### Cara kalibrasi

1. Taruh robot di titik berarsir, lurus dan catat posisi awal:

   ```bash
   ros2 run nucleo odom_calibrate --start
   ```

2. Dorong robot **lurus** 2–3 meter tanpa berputar.

3. Catat posisi akhir lalu minta hitungan:

   ```bash
   ros2 run nucleo odom_calibrate --end
   ros2 run nucleo odom_calibrate --report --real 2.0 --ppr 7500
   ```

4. Jalankan dengan nilai baru:

   ```bash
   ros2 launch nucleo nucleo.launch.py pulses_per_rev:=6976.7
   ```

Ulangi sampai jarak tempuh di `/odom` cocok dengan jarak riil dalam
~2%. Kalau `/odom` **kurang jauh** → naikkan `pulses_per_rev`.

### Kalau robot berputar padahal seharusnya lurus

Itu masalah geometri roda, bukan skala jarak:

- `wheel_radius_cm` terlalu besar → robot berputar melingkar saat berjalan
- `base_length_cm` tidak sesuai → radius putaran salah, robot
  melenceng saat berputar

Ukur radius putar nyata (mis. 1 m) lalu bandingkan dengan yang dibaca
`/odom`, lalu sesuaikan rasio `base_length_cm`.

### Verifikasi cepat tanpa robot

```bash
ros2 topic echo /odom --field pose.pose.position.x   # saat maju lurus
ros2 topic echo /amcl_pose --field pose.pose.position # koreksi AMCL
```

Kalau `/amcl_pose` terus mengoreksi arah yang **sama** (selalu +x
atau selalu −x), berarti skala `/odom` masih salah — bukan gangguan AMCL.

---

## Teleop (hanya untuk tes debug)

Teleop **tidak dipakai saat navigation jalan**. Kalau perlu tes motor
manual, `nucleo` punya channel override berjenjang:

```
STM_TO (cmd_vel)  ←  CMD_JOY (joy_vel, age<50ms)  ←  TEST (test_vel)
```

Jangan teleop ke `/cmd_vel` — topik itu dimiliki Nav2
(`controller_server → velocity_smoother → collision_monitor → cmd_vel`).
Kalau Nav2 ikut jalan, ada dua publisher di `/cmd_vel` → perintah bercampur.

```bash
# Tes motor manual (Nav2 dimatikan / belum jalan):
ros2 launch nucleo nucleo.launch.py
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=joy_vel
```

`joy_vel` punya prioritas di atas `cmd_vel`, jadi tetap aman kalau Nav2
kebetulan hidup. Key mode holonomic (tahan **Shift**): `Shift+J`/`Shift+L`
geser kiri/kanan, `Shift+U`/`Shift+O` diagonal, `I`/`'` maju/mundur,
`U` `J` `L` `O` `M` putar, `K` stop.

---

## Collision & obstacle

### Geometri dasar
```
radius robot = sqrt(base^2 + r_roda^2) = sqrt(20.8² + 5²) cm = 21.4 cm
              -> robot_radius: 0.22  (dengan margin)
lebar lorong  = 90 cm -> dinding ada di 0.45 m dari pusat robot
```

### Aturan main lorong 90 cm
```
robot_radius + inflation_radius        <  0.45   ->  0.22 + 0.20 = 0.42 ✅
collision_monitor Stop.radius          <  0.45   ->  0.30 ✅
collision_monitor Slow.radius          <  0.45   ->  0.40 ✅
```

Nilai lama (`inflation_radius: 0.55`, `Slow.radius: 0.60`) membuat
`0.22 + 0.55 = 0.77 > 0.45` — lorong **terhalang total**, robot tidak
bisa lewat. Selain itu `Slow.radius: 0.60 > 0.45` membuat robot selalu
melambat di lorong yang sebenarnya lega.

### Parameter collision

| Parameter | Nilai | Arti |
|---|---|---|
| `robot_radius` (L & G) | `0.22` | Radius robot. **Harus ≥ radius nyata** dan **sama** di local & global. |
| `inflation_radius` (L & G) | `0.20` | Buffer biaya di sekitar dinding. Determines apakah lorong bisa dilewati. |
| `cost_scaling_factor` (L & G) | `6.0` | Makin besar → biaya jatuh lebih cepat → robot lebih mau lewat lorong sempit. |
| `Stop.radius` | `0.30` | Jarak stop total. Harus < jarak dinding lorong. |
| `Stop.min_points` | `5` | Toleransi noise lidar (lebih tinggi = tidak gampang false stop). |
| `Slow.radius` | `0.40` | Jarak mulai melambat. |
| `Slow.slowdown_ratio` | `0.4` | Max 40% kecepatan di zona Slow. |
| `collision_monitor.scan.min_height` | `0.10` | Obstacle < 10 cm diabaikan. Turunkan ke `0.05` untuk aba-aba. |
| `BaseObstacle.scale` (DWB) | `0.2` | Dorongan Hindari obstacle saat navigasi. Default Nav2 0.02 terlalu kecil. |
| `collision_monitor.visualize` | `True` | **Lingkaran merah terlihat di RViz** — dipakai untuk kalibrasi. |

### Kalau lorong lebih sempit / lebih lebar
- **Lebih sempit** (mis. 70 cm): `inflation_radius` → `0.10`, `Stop.radius` → `0.25`, `Slow.radius` → `0.35`
- **Lebih lebar** (mis. 120 cm): boleh naikkan `inflation_radius` → `0.35`, `Slow.radius` → `0.50`

### Cara kalibrasi cepat
1. Jalankan `ros2 launch navigation navigation.launch.py` dengan RViz.
2. Lihat `visualize: True` — lingkaran merah (Stop) dan oranye (Slow) terlihat.
3. Naikkan **sampai** `robot_radius + inflation_radius < jarak dinding` (ukur dari `Grid` di RViz).
4. Kalau ada benda yang **tidak** memicu padahal seharusnya (mis. kaki kursi), turunkan `scan.min_height`.

---

## Diagnosa cepat

Kalau navigasi_error atau robot "tidak update posisinya" di RViz:

```bash
ros2 run navigation check_system
```

Yang diperiksa:

| Cek | Kalau GAGAL |
|---|---|
| Node (`nucleo`, `amcl`, `planner_server`, ...) | Ada proses yang belum jalan |
| `/scan` Hz | Lidar mati / `serial_port` salah / `scan_mode` |
| `/odom` Hz | `nucleo` tidak jalan atau pulse STM32 tidak masuk |
| **TF `map → odom`** | **Penyebab utama display beku** — AMCL belum terlokalisasi |
| TF `odom → base_footprint` | `nucleo` tidak mempublish TF |
| AMCL active + `amcl_pose` | Robot belum di-set pose awal (klik **2D Pose Estimate**) |
| Costmap terbit | Costmap belum aktif / peta belum termuat |
| Publisher `/cmd_vel` > 1 | Konflik teleop vs Nav2 (lihat bagian Teleop) |

> **Syarat mutlak agar RViz menampilkan robot & scan:**
> TF `map → odom` harus ada. Tanpa itu, `base_link` dan `laser_frame`
> (yang ada di bawah `odom`) tidak bisa ditransformasi ke frame `map`,
> jadi robot maupun scan tampak "beku" meskipun datanya sebenarnya ada.

---

## Cara menjalankan

```bash
source ~/Documents/athome-ws/install/setup.bash

# Terminal 1 — base driver Nucleo (UDP STM32 + odom/TF odom->base_footprint)
ros2 launch nucleo nucleo.launch.py

# Terminal 2 — peta + AMCL + Nav2 + RViz + /goto
ros2 launch navigation navigation.launch.py
```

Tunggu ±10–20 detik sampai peta muncul di RViz dan AMCL aktif.

**Syarat agar robot bergerak:**
- `nucleo` jalan di terminal terpisah (Sumber TF `odom -> base_footprint` dan topik `/odom`)
- TF `base_footprint -> base_link` aktif (otomatis disediakan navigation launch)
- AMCL terlokalisasi (bisa diisi otomatis dari `initial_x/y`, atau klik
  **2D Pose Estimate** di RViz)
- Peta ada di `maps/`

### Kalau robot tidak di titik (0,0) saat start
```bash
ros2 launch navigation navigation.launch.py initial_x:=1.0 initial_y:=1.5 initial_yaw:=0.0
```

### Argument launch
| Argumen | Default | Keterangan |
|---|---|---|
| `map` | `maps/map.yaml` | Path file peta (.yaml). Otomatis fallback ke source tree jika belum terpasang. |
| `params_file` | `config/nav2_params.yaml` | Parameter Nav2. |
| `use_sensors` | `true` | `true` = jalankan lidar & kamera ZED via `robot_bringup/sensors.launch.py`. |
| `initial_x`, `initial_y`, `initial_yaw` | `0.0` | Pose awal robot di peta (meter & radian). |
| `use_composition` | `False` | `False` = tiap node Nav2 proses sendiri. |
| `rviz` | `true` | Tampilkan RViz. |
| `rviz_software` | `false` | Paksa software rendering (tanpa GPU). |
| `rviz_software` | `false` | Paksa software rendering (tanpa GPU). |

---

## Daftar Lokasi (`/goto/<nama>`)

Lokasi didefinisikan di `config/places.yaml`. Tiap key menjadi **satu service ROS**.

| Service | x (m) | y (m) | yaw (°) | Keterangan |
|---|---|---|---|---|
| `/goto/kamar_tidur` | 1.00 | 1.50 | 0 | Titik kumpul tamu |
| `/goto/dapur` | 3.50 | 2.30 | 0 | Area timur-utara |
| `/goto/kamar` | 2.15 | -2.15 | 0 | Area selatan |
| `/goto/teras` | -1.60 | 0.00 | 180 | Sisi barat, menghadap -X |
| `/goto/pojok_timur` | 5.55 | 2.15 | 90 | Ujung timur |
| `/goto/lorong` | 2.00 | 0.00 | 0 | Lorong tengah |
| `/goto/lantai_utara` | 0.90 | 2.50 | 0 | Area utara |
| `/goto/lantai_timur` | 3.90 | 2.30 | 0 | Area timur |
| `/goto/start` | 0.00 | 0.00 | 0 | Titik awal mapping |

Semua koordinat ber-**frame `map`**, satuan meter. `yaw_deg` 0 = menghadap +X peta.

> Koordinat ini **draft** — sesuaikan dengan denah asli kamu (lihat bagian
> Menambah/Mengubah Lokasi).

---

## Cara memakai `/goto`

### 1. Dari terminal (paling gampang, untuk tes)
```bash
ros2 service call /goto/kamar_tidur std_srvs/srv/Trigger "{}"
```
Robot bergerak ke sana, path tampil di RViz, terminal **menunggu** sampai selesai:
```
requester: making request: std_srvs.srv.Trigger_Request()

response:
std_srvs.srv.Trigger_Response(success=True, message='sampai di "kamar_tidur" (x=1.00, y=1.50)')
```

Cek daftar service yang tersedia:
```bash
ros2 service list | grep goto
```

### 2. Dari program Python
```python
from std_srvs.srv import Trigger

client = node.create_client(Trigger, '/goto/kamar_tidur')
if client.wait_for_service(timeout_sec=2.0):
    future = client.call_async(Trigger.Request())   # tidak memblokir
    future.add_done_callback(lambda f: print(f.result().success, f.result().message))
```

### 3. Dari program C++
```cpp
#include "std_srvs/srv/trigger.hpp"

auto client = this->create_client<std_srvs::srv::Trigger>("/goto/kamar_tidur");
client->wait_for_service(1s);
client->async_send_request(
    std::make_shared<std_srvs::srv::Trigger::Request>());
// di callback: response->success == true saat robot sudah sampai
```

### 4. Dari RViz (klik langsung)
Pakai tool **Nav2 Goal** atau **2D Goal Pose** di toolbar RViz — robot langsung
dihitung dan goal dikirim ke Nav2.

---

## Perilaku Penting

1. **Pemblokiran.** Service baru dijawab setelah robot **sampai** (maks
   `place_timeout_sec`, default 90 detik). Timeout → goal dibatalkan, balas
   `success=False`.
2. **Satu goal aktif.** Panggil lokasi lain saat robot masih jalan → langsung
   ditolak:
   ```
   success=False, message='sibok: robot masih menuju lokasi lain'
   ```
   Tidak ada antrean.
3. **Path di RViz.** Setiap goal di-plan Nav2 → topic `/plan`. Di RViz tampil
   sebagai display **Global Plan** (merah) dan **Local Plan** (biru).
4. **Lokalisasi.** `/goto` hanya berhasil kalau AMCL sudah terlokalisasi.
   Kalau belum, robot tidak bergerak though service dipanggil.

---

## Person Following (`/follow/person`)

Dynamic object following: robot mengikuti **orang** yang dideteksi ZED, bukan
koordinat tetap. Berdampingan dengan `/goto/<nama>` tanpa bentrok.

### Panggil

```bash
ros2 service call /follow/person std_srvs/srv/Trigger "{}"   # mulai
ros2 service call /follow/stop   std_srvs/srv/Trigger "{}"   # berhenti
ros2 topic echo /follow/state   # IDLE / WAITING_PERSON / FOLLOWING
```

Dari program (Python / C++) panggil persis seperti `/goto/<nama>` — jangan lupa
`wait_for_service()` dulu. Contohnya ada di docstring `scripts/person_follower.py`.

### Prasyarat ⚠️

```bash
ros2 launch navigation navigation.launch.py enable_body_tracking:=true
```

Default `enable_body_tracking` adalah **`false`**, dan kalau begitu topiknya
selalu berisi `NO_PERSON` sehingga following tidak akan pernah mulai.

### Cara kerja

```
/zed/person_keypoints ──▶ person_follower ──▶ /goal_update ──▶ GoalUpdater (BT)
   (String, zed)            hitung jarak          (PoseStamped)      │
   orang terdekat          + deadband                             ▼
   (index 0)                                        ComputePathToPose
                                                                    ▼
                                                    TruncatePath 0.50 m
                                                                    ▼
                                              FollowPath ──▶ DWB ──▶ /cmd_vel
```

Orang dibaca dari `/zed/person_keypoints`, diambil yang **paling dekat**
(`target_index: 0`). Posisinya sudah relatif robot (`x` depan, `y` kiri,
`z` atas) lalu diubah ke frame `map` lewat TF.

Goal **tidak dikirim sekali**, tapi terus diperbarui (default 2 Hz) lewat
`/goal_update`. Inilah yang membuat robot benar-benar mengikuti orang yang
berjalan, bukan berhenti di posisi terakhir.

### Deadband jarak

Parameter di `config/person_follow.yaml` (jarak = pusat robot → pusat orang):

| Nilai | Arti |
|---|---|
| `follow_stop_distance: 0.50` | Orang sedekat ini → robot berhenti mendekat |
| `follow_start_distance: 0.60` | Orang sejauh ini → robot mulai mendekat |
| `path_truncate_distance: 0.50` | Ujung jalur dipotong 0.50 m sebelum orang |

Celah `0.50..0.60` adalah **zona mati** yang mencegah robot maju–rem–maju–rem.
Nilai `0.50` sengaja dibuat berimpit dengan `collision_monitor`
(`Stop.radius: 0.30` + radius badan orang ~0.25), jadi navigasi global menahan
dulu di 0.50 m dan collision_monitor jadi jaring pengaman kedua.

### Auto-stop

Following berhenti sendiri kalau:

- tidak ada deteksi orang selama `max_lost_time` (3 s)
- orang menjauh lebih dari `max_follow_range` (3 m)
- tidak ada orang sama sekali dalam `wait_person_timeout` (60 s) saat menunggu

### Keterbatasan kecepatan

`max_vel_x` di `nav2_params.yaml` adalah **0.14 m/s**, sedangkan orang berjalan
normal ~1.3 m/s. Following karena itu hanya berhasil untuk orang yang **diam atau
berjalan sangat pelan** (< 0.14 m/s). Menaikkan `max_vel_x` untuk mengejar orang
berjalanrous mengubah tuning navigasi yang sudah disetel, jadi tidak dilakukan
otomatis.

### Nonaktifkan

```bash
ros2 launch navigation navigation.launch.py use_following:=false
```

---

## Menambah / Mengubah Lokasi

Tambah blok baru di `config/places.yaml`, lalu restart launch:

```yaml
presets:
  gudang: {x: 4.0, y: 1.0, yaw_deg: 90.0}     # -> /goto/gudang
```

Otomatis terbentuk service `/goto/gudang`. **Tidak perlu ubah kode.**

### Aturan nama key
- Boleh: huruf, angka, underscore (`dapur2`, `lantai_atas`, `gudang_belakang`)
- **Tidak boleh** diawali angka, tidak boleh spasi/strip (`1`, `ruang tamu`,
  `ruang-tamu`)

Key yang tidak valid akan **dilewati** dan muncul sebagai peringatan di log —
tidak membuat node crash.

### Validasi koordinat
Sebelum dipakai, cek koordinat tidak menabrak dinding / di luar peta:
```bash
ros2 run navigation check_places
```
Outputnya tabel status `OK` / `WARN` / `FAIL` per lokasi.

---

## Troubleshooting

| Gejala | Penyebab & Solusi |
|---|---|
| `ros2 service list` tidak ada `/goto/*` | `place_navigator` tidak jalan. Cek log launch, atau jalankan dengan `use_places:=true`. |
| `success=False, message='action navigate_to_pose tidak aktif'` | Nav2 belum siap. Tunggu sampai `Managed nodes are active` di log. |
| `success=False, message='sibok ...'` | Masih ada goal aktif sebelumnya. Tunggu robot berhenti. |
| Robot diam, tidak bergerak | AMCL belum terlokalisasi. Klik **2D Pose Estimate** di RViz atau set `initial_x/y`. |
| `Fixed Frame [map] does not exist` di RViz | TF `map -> odom` belum ada. Pastikan `nucleo` jalan dan AMCL aktif. |
| Peta tidak muncul di RViz | Map display butuh **Transient Local** QoS (sudah diset di `rviz/navigation.rviz`). |
| Goal ditolak / planning gagal | Lokasi berada di area dinding atau unknown. Cek dengan `check_places`. |
| `ros2 service list` tidak ada `/follow/*` | `person_follower` tidak jalan. Jalankan dengan `use_following:=true`. |
| `/follow/person` sukses tapi robot diam | `/zed/person_keypoints` masih `NO_PERSON`. ZED harus jalan dengan **`enable_body_tracking:=true`** (default launch adalah `false`). |
| `person_follower` selalu `WAITING_PERSON` | Topic orang kosong: cek `ros2 topic echo /zed/person_keypoints`, dan `ros2 topic hz /zed/person_keypoints`. |
| Robot tidak bergerak padahal orang terdeteksi | `max_vel_x` 0.14 m/s hanya bisa menyusul orang yang diam atau < 0.14 m/s. |
| Robot menabrak / terlalu dekat orang | Naikkan `follow_stop_distance` di `config/person_follow.yaml`, atau turunkan `Stop.radius` di `nav2_params.yaml`. |
| Tombol `2D Pose Estimate` ada tapi klik tidak bereaksi | Jalur `/initialpose → AMCL` sudah diverifikasi normal, jadi ini masalah interaksi RViz. Bypass dengan `set_initial_pose` (lihat `## Menetapkan Posisi Awal`). |

---

## Menetapkan Posisi Awal (bypass RViz)

Tool `2D Pose Estimate` di RViz sesekali tidak bereaksi — klik tidak memicu
apa-apa. Rantai `RViz → /initialpose → AMCL` sendiri sudah diverifikasi
normal, jadi untuk menyetel posisi awal ada jalur terminal yang tidak
bergantung GUI:

```bash
# pindah ke koordinat tertentu (meter, frame peta)
ros2 run navigation set_initial_pose --x 2.5 --y -1.25 --yaw 0.785

# sekalian kirim goal supaya robot langsung bergerak ke sana
ros2 run navigation set_initial_pose --x 2.5 --y -1.25 --yaw 0.785 --go
```

`--yaw` dalam **radian**: `0` = +X peta, `1.5708` = +Y, `-1.5708` = -Y.

Untuk memastikan apakah RViz benar-benar tidak mengirim apa pun, jalankan:

```bash
ros2 topic echo /initialpose
```

Kalau topik ini **tidak keluar apa-apa** saat Anda klik di RViz, berarti
masalahnya di sisi RViz. Kalau keluar tapi robot tetap di tempat /
lompat balik, berarti AMCL menerimanya tetapi tidak cocok dengan `/scan`.

---

## Catatan Teknis

- **TF yang diharapkan:**
  ```
  map --(AMCL)--> odom --(nucleo)--> base_footprint --> base_link --> laser_frame
  ```
- `place_navigator` memakai `MultiThreadedExecutor` + `ReentrantCallbackGroup`
  karena service-nya memblokir menunggu robot sampai. Dengan single-thread
  executor akan terjadi deadlock.
- Goal dikirim via action Nav2 `navigate_to_pose` (`bt_navigator`).
- `/follow/person` memakai field `behavior_tree` pada action yang sama, jadi BT
  dipilih **per-goal**. `/goto/<nama>` tetap memakai `default_nav_to_pose_bt_xml`
  dan tidak terpengaruh `behavior_trees/follow_person.xml`.
- `person_follower` juga memakai `MultiThreadedExecutor` karena ada service,
  subscription, timer, dan pendingan hasil action sekaligus.
- Tampilan RViz: peta, costmap global & lokal, path global & lokal, sebaran
  partikel AMCL, dan pose AMCL.