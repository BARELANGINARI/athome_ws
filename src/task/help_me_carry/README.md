# help_me_carry

Package task **Help Me Carry** (RoboCup@Home) untuk Barelang Inari. Mengatur interaksi suara (HRI), pelacakan dan pengikutan operator (`help_me_carry_node`), deteksi gestur penunjukan tas, manipulasi lengan robot untuk mengangkat tas, serta navigasi otonom kembali ke titik awal (*autonomous homing*) dengan fitur antri jika terdapat halangan orang.

---

## 📋 Fitur Utama
- **Person Following & Kalman Filter**: Mengikuti operator menggunakan Nav2 dan pengontrol spasial anti-hijack yang diperkuat Kalman Filter dan fusi LiDAR 360°.
- **FSM Task Manager (`help_me_carry_manager`)**: Mengatur alur penuh dari sapaan wakeword, konfirmasi awal, following, pengenalan titik tujuan ("arrived"), penunjukan tas, pemanggilan service lengan, hingga kembali ke titik awal.
- **Homing Queueing & Obstacle Avoidance**: Robot mampu mendeteksi antrian orang saat kembali ke titik awal, merayap pelan (*crawl*), atau berhenti menunggu hingga jalur bersih.
- **Konfigurasi Terpusat YAML**: Seluruh parameter jarak, kecepatan, gain PID, watchdog timeout, dan durasi grace period dikonfigurasi melalui `config/params.yaml`.

---

## 🚀 Cara Menjalankan

### 1. Build Package
```bash
cd ~/Documents/athome_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select help_me_carry --symlink-install
source install/setup.bash
```

### 2. Multi-Pane Grid (Terminator Layout)
Metode paling direkomendasikan untuk kompetisi atau pengujian penuh:
```bash
./src/task/help_me_carry/launch/run.sh
```

Untuk menghentikan seluruh proses:
```bash
./src/task/help_me_carry/launch/stop.sh
```

### 3. Peluncuran Modular (Terminal Terpisah)
```bash
# Terminal 1: Hardware (STM32 & RPLiDAR)
ros2 launch help_me_carry hardware.launch.py

# Terminal 2: Vision & Kamera ZED
ros2 launch help_me_carry camera.launch.py

# Terminal 3: Navigasi SLAM & Following Controller
ros2 launch help_me_carry motion.launch.py

# Terminal 4: HRI Voice & Task Orchestrator FSM
ros2 launch help_me_carry hri.launch.py

# Terminal 5: Manipulator Lengan Robot
ros2 launch athome_arm arm.launch.py
```

### 4. Peluncuran All-in-One
```bash
ros2 launch help_me_carry help_me_carry.launch.py
```

---

## ⚙️ Konfigurasi Parameter YAML
Parameter task Help Me Carry tersimpan di `config/params.yaml`:
* **`help_me_carry_node`**:
  * `follow_stop_distance`: Jarak henti di belakang operator (default: `0.85 m`).
  * `follow_start_distance`: Jarak mulai bergerak maju (default: `1.00 m`).
  * `max_vx`, `max_vy`, `max_wz`: Batas kecepatan robot.
  * `kp_dist`, `kp_yaw`, `kd_yaw`: Gain kendali PID pengikutan.
  * `homing_queue_enabled`: Aktifkan perilaku antri saat homing (default: `true`).
* **`help_me_carry_manager`**:
  * `global_timeout_sec`: Batas waktu total tugas (default: `600.0 s`).
  * `operator_timeout_sec`: Batas waktu hilang operator sebelum recovery (default: `5.0 s`).
  * `follow_grace_period_sec`: Masa tenggang di awal follow (default: `2.0 s`).
  * `arrived_grace_period_sec`: Jeda waktu abaikan kata "arrived" di awal follow (default: `5.0 s`).

---

## 📡 Service ROS 2
* `ros2 service call /follow/person std_srvs/srv/Trigger {}`: Mulai melacak dan mengikuti orang.
* `ros2 service call /follow/stop std_srvs/srv/Trigger {}`: Berhenti mengikuti orang.
* `ros2 service call /task/step_back std_srvs/srv/Trigger {}`: Gerakan mundur aman dari orang/objek.
* `ros2 service call /task/align_bag std_srvs/srv/Trigger {}`: Menghadap ke arah tas yang terdeteksi.
* `ros2 service call /homing std_srvs/srv/Trigger {}`: Navigasi kembali ke titik start.
