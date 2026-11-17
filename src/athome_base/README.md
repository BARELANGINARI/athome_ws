# athome_base

Package driver dasar untuk robot mobile Barelang Inari. Mengatur komunikasi jaringan UDP tingkat rendah dengan mikrokontroler STM32 Nucleo untuk kontrol roda penggerak (kinematika roda omnidirectional/mecanum/diff), pembacaan encoder odometri, dan penerbitan transform TF `odom -> base_link`.

---

## 📋 Fitur Utama
- **Komunikasi UDP Kecepatan Tinggi**: Mengirimkan perintah kecepatan (`/cmd_vel`) ke STM32 Nucleo dan menerima feedback sensor/odometri secara real-time.
- **Odometry & TF Publisher**: Menghitung posisi robot dan mempublikasikan topik `/odom` serta transform koordinat `odom` -> `base_link`.
- **Kinematika Robot**: Konversi kecepatan linier $(v_x, v_y)$ dan angular $(\omega_z)$ ke putaran masing-masing roda robot.

---

## 🚀 Cara Menjalankan

### 1. Build Package
```bash
cd ~/Documents/athome_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select athome_base --symlink-install
source install/setup.bash
```

### 2. Jalankan Driver Nucleo Base
```bash
ros2 launch athome_base nucleo.launch.py
```

### 3. Mengubah Parameter IP STM32
Jika IP STM32 berbeda dari default (`192.168.0.4`):
```bash
ros2 launch athome_base nucleo.launch.py stm_ip:=192.168.0.5
```

---

## 📡 Topik & Komunikasi
- **Subscribed Topics**:
  - `/cmd_vel` (`geometry_msgs/msg/Twist`): Perintah kecepatan gerak robot.
- **Published Topics**:
  - `/odom` (`nav_msgs/msg/Odometry`): Data odometri posisi dan kecepatan robot.
  - `/tf` (`tf2_msgs/msg/TFMessage`): Transform `odom` ke `base_link`.
