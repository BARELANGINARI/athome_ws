# restaurant

Package task **Restaurant** (RoboCup@Home) untuk robot Barelang Inari. Mengatur layanan pelayan restoran otonom di lingkungan tanpa peta awal (*mapless / real-time SLAM*): mendeteksi pelanggan yang melambaikan tangan, bernavigasi ke meja pelanggan, merekam data wajah, mencatat dua pesanan pelanggan, kembali ke barman di titik origin, mengambil pesanan di nampan/gripper, mengantarkan kembali ke pelanggan, dan mengulang untuk pelanggan berikutnya.

---

## 📋 Fitur Utama
- **Mapless / Real-time SLAM Navigation**: Bernavigasi secara dinamis menggunakan SLAM Toolbox tanpa memerlukan peta statis sebelum lomba.
- **Deteksi Lambaian Tangan (`/vision/wave_hand` / `/vision/order`)**: Robot mendeteksi pelanggan yang memanggil dengan melambaikan tangan dan mendekat ke posisi aman di depannya.
- **Heading Alignment**: Robot menyelaraskan arah hadap robot tepat ke posisi bounding box orang menggunakan feedback visual kamera.
- **Perekaman Wajah & Interaksi Suara**: Mengambil foto/fitur wajah pelanggan untuk validasi saat pengantaran, mencatat pesanan 1 & 2, serta konfirmasi suara.
- **Pengantaran Pesanan & Kendali Gripper**: Melepaskan objek pertama dari gripper dengan aba-aba hitungan mundur, dan menginstruksikan pelanggan mengambil objek kedua dari baki (*tray*).
- **Konfigurasi Terpusat YAML**: Seluruh parameter jarak henti, sudut sweep rotasi, gain alignment, watchdog timeout, dan alur FSM tersimpan di `config/restaurant_params.yaml`.

---

## 🚀 Cara Menjalankan

### 1. Build Package
```bash
cd ~/Documents/athome_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select restaurant --symlink-install
source install/setup.bash
```

### 2. Multi-Pane Grid (Terminator Layout 2x2)
Metode peluncuran utama untuk pengujian dan kompetisi:
```bash
./src/task/restaurant/launch/run.sh
```

Untuk menghentikan semua proses:
```bash
./src/task/restaurant/launch/stop.sh
# Atau:
./src/task/restaurant/launch/run.sh stop
```

### 3. Peluncuran Modular (Manual per Terminal)
```bash
# Terminal 1: Hardware (STM32 & RPLiDAR)
ros2 launch restaurant hardware.launch.py

# Terminal 2: Navigasi SLAM & Restaurant Node
ros2 launch restaurant motion.launch.py

# Terminal 3: HRI & FSM Task Manager
ros2 launch restaurant hri.launch.py

# Terminal 4: Vision (ZED Camera, Wave Detection & Face Recognition)
ros2 launch restaurant camera.launch.py
```

### 4. Peluncuran All-in-One
```bash
ros2 launch restaurant restaurant.launch.py hri:=true camera:=true
```

---

## ⚙️ Konfigurasi Parameter YAML (`config/restaurant_params.yaml`)

Semua nilai dapat diubah langsung tanpa perlu kompilasi ulang C++:

```yaml
restaurant_node:
  ros__parameters:
    origin_x: 0.0
    origin_y: 0.0
    origin_yaw: 0.0
    standoff_distance: 0.85              # Jarak aman berhenti di depan pelanggan (m)
    search_sweep_angle_deg: 90.0         # Sudut putar sweep saat mencari pelanggan (deg)
    search_angular_speed: 0.35           # Kecepatan putar sweep (rad/s)
    align_to_person_enabled: true        # Luruskan hadap robot ke bbox orang
    align_p_gain: 0.6                    # P-gain koreksi arah hadap
    camera_image_width: 672.0            # Resolusi lebar kamera (ZED VGA)

restaurant_manager:
  ros__parameters:
    customer_standoff_distance: 0.85
    object_standoff_distance: 0.45
    nav_timeout_sec: 120.0               # Timeout navigasi Nav2
    max_customers: 2                     # Jumlah pelanggan yang dilayani dalam 1 sesi
    global_timeout_sec: 900.0            # Batas waktu total task (15 menit)
    dialog_reprompt_sec: 12.0            # Timeout reprompt dialog jika pelanggan diam
    wave_reprompt_sec: 10.0              # Interval ajakan melambaikan tangan
    nav_retry_cooldown_sec: 2.0          # Jeda retry navigasi jika gagal
```

---

## 📡 Service ROS 2
* `ros2 service call /search std_srvs/srv/Trigger {}`: Memutar robot untuk mencari lambaian tangan pelanggan.
* `ros2 service call /go_to_order std_srvs/srv/Trigger {}`: Navigasi menuju koordinat pelanggan yang melambai.
* `ros2 service call /homing std_srvs/srv/Trigger {}`: Navigasi kembali ke titik asal (barman).
* `ros2 service call /deliver_order std_srvs/srv/Trigger {}`: Mengantarkan pesanan kembali ke meja pelanggan.
