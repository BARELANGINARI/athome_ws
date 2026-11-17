# athome_bringup

Package bringup sensor dan antarmuka hardware robot Barelang Inari, meliputi RPLiDAR 2D 360°, filter laser, transform TF statis sensor, dan kendali manual menggunakan Joystick (Gamepad).

---

## 📋 Fitur Utama
- **Sensor Bringup (`sensors.launch.py`)**:
  - Driver RPLiDAR (`rplidar_node`) untuk pemindaian 360°.
  - Filter sudut laser (`laser_filters`) untuk memangkas sudut blind spot sasis robot.
  - Transform statis TF (`base_link` -> `laser_frame`).
- **Teleoperasi Joystick (`joystick.launch.py`)**:
  - Pembacaan input joystick melalui `joy_node`.
  - Pemetaan tombol dan joystick ke perintah kecepatan `/cmd_vel` melalui `teleop_twist_joy`.

---

## 🚀 Cara Menjalankan

### 1. Build Package
```bash
cd ~/Documents/athome_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select athome_bringup --symlink-install
source install/setup.bash
```

### 2. Menjalankan Sensor LiDAR
Pastikan perangkat LiDAR terhubung di `/dev/ttyUSB0` (atau berikan akses `sudo chmod 666 /dev/ttyUSB0`):
```bash
ros2 launch athome_bringup sensors.launch.py
```

Argumen opsional:
```bash
# Menyesuaikan port serial atau baudrate LiDAR
ros2 launch athome_bringup sensors.launch.py serial_port:=/dev/ttyUSB1 serial_baudrate:=115200
```

### 3. Menjalankan Kendali Manual Joystick (Gamepad)
Hubungkan gamepad via USB / Bluetooth, lalu jalankan:
```bash
ros2 launch athome_bringup joystick.launch.py
```

Untuk mengubah perangkat gamepad (default: `/dev/input/js0`):
```bash
ros2 launch athome_bringup joystick.launch.py dev:=/dev/input/js1
```

---

## ⚙️ Konfigurasi Parameter
Parameter LiDAR tersimpan pada:
- `config/lidar.yaml`: Pengaturan serial port, frame id, inversi sudut, dan baudrate.
