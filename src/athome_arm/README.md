# athome_arm

Package ROS 2 untuk kontrol lengan robot dan adaptive gripper pada robot **Barelang Inari** (RoboCup@Home). Package ini menyediakan teaching motion recording/playback, kalibrasi servo Dynamixel AX-18A, adaptive grasping, dan ROS 2 service server untuk integrasi ke sistem HRI dan task manager.

---

## 1. Prasyarat & Instalasi Dependensi

### A. Dynamixel SDK (C++)
Package ini membutuhkan library C++ Dynamixel SDK dari ROBOTIS:

```bash
git clone https://github.com/ROBOTIS-GIT/DynamixelSDK.git
cd DynamixelSDK/c++/build/linux64    # gunakan linux_sbc bila di Raspberry Pi / Jetson
make
sudo make install
```

### B. Dependensi Paket ROS 2
Pastikan paket ROS 2 berikut sudah terinstal di sistem:

```bash
sudo apt update
sudo apt install -y \
  ros-$ROS_DISTRO-rclcpp \
  ros-$ROS_DISTRO-std-srvs \
  ros-$ROS_DISTRO-std-msgs \
  ros-$ROS_DISTRO-ament-index-cpp
```

### C. Izin Akses Serial Port
Pastikan user memiliki akses ke port serial `/dev/ttyUSB*`:

```bash
sudo usermod -aG dialout $USER
```
*(Logout dan login kembali agar izin grup aktif).*

---

## 2. Cara Kompilasi (Build)

Jalankan perintah berikut di root workspace ROS 2:

```bash
cd ~/athome_ws
colcon build --packages-select athome_arm
source install/setup.bash
```

---

## 3. Cara Menjalankan

### A. Menjalankan ROS 2 Service Node (Arm Motion Server)
Node ini memutar motion hasil teaching secara otomatis ketika menerima request service.

* **Via Launch File:**
  ```bash
  ros2 launch athome_arm arm.launch.py
  ```
  *Parameter opsional:*
  * `port:=/dev/ttyUSB1` (default: `/dev/ttyUSB1`)
  * `baudrate:=1000000` (default: `1000000`)

* **Uji Coba Memanggil Gerakan `ambil_tas`:**
  ```bash
  ros2 service call /arm/ambil_tas std_srvs/srv/Trigger {}
  ```

---

### B. Program Utilitas Mandiri (CLI Tools)

1. **Teaching & Recording Gerakan (`learn_motion`):**
   Program interaktif di terminal untuk merekam pose/stream gerakan dan menyimpannya ke folder `motions/`:
   ```bash
   ros2 run athome_arm learn_motion
   ```
   *Perintah interaktif utama:* `record`, `add`, `play <nama>`, `save <nama>`, `list`.

2. **Kalibrasi Zero & Limit Servo (`calibrate_arm`):**
   Program untuk memandu kalibrasi posisi 0 dan batas sudut setiap servo Dynamixel:
   ```bash
   ros2 run athome_arm calibrate_arm
   ```

3. **Uji Coba Adaptive Gripper (`adaptive_gripper`):**
   Uji coba penutupan gripper adaptif berbasis deteksi beban torsi (load sensing):
   ```bash
   ros2 run athome_arm adaptive_gripper
   ```

---

## 4. Struktur Direktori

```text
src/athome_arm/
├── CMakeLists.txt                # Build configuration ament_cmake
├── package.xml                   # Manifest dependensi ROS 2
├── README.md                     # Panduan penggunaan
├── include/athome_arm/
│   └── calibration_data.h        # Data sudut kalibrasi servo
├── src/
│   ├── arm_motion_server.cpp     # ROS 2 Service Node pemutar file motion
│   ├── learn_motion.cpp          # CLI offline untuk teaching gerakan
│   ├── calibrate_arm.cpp         # CLI kalibrasi servo
│   └── adaptive_gripper.cpp      # CLI uji adaptive grasping
├── motions/
│   └── ambil_tas                 # File rekaman gerakan stream 50Hz
└── launch/
    └── arm.launch.py             # Launch file ROS 2
```

