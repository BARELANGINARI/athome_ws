# athome_navigation

Package navigasi otonom berbasis ROS 2 Navigation Stack (Nav2) untuk robot Barelang Inari. Mengatur lokalisasi (AMCL / SLAM Toolbox), perencanaan jalur global (Nav2 Planner), penghindaran rintangan dinamis (Nav2 Costmaps), dan pelacakan lintasan (Nav2 Controller / RPP / DWB).

---

## 📋 Fitur Utama
- **Navigasi dengan Peta Statis (AMCL)**: Menggunakan peta 2D yang telah disimpan (`maps/map.yaml`) dan sensor fusion LiDAR untuk lokalisasi presisi.
- **Navigasi Mapless / Real-time SLAM**: Dapat dijalankan tanpa peta statis (`slam:=true`), robot langsung memetakan dan bernavigasi secara dinamis (sangat berguna untuk task *Restaurant*).
- **Penghindaran Rintangan Real-time**: Local costmap dan global costmap mengintegrasikan pembacaan LiDAR 360° untuk menghindari orang, furnitur, dan halangan tak terduga.
- **Integrasi RViz2**: Menyediakan visualisasi jalur global (`/plan`), costmap footprint, goal pose, dan partikel lokalisasi robot.

---

## 🚀 Cara Menjalankan

### 1. Build Package
```bash
cd ~/Documents/athome_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select athome_navigation --symlink-install
source install/setup.bash
```

### 2. Navigasi Menggunakan Peta Statis (Default)
Pastikan peta telah disimpan di `maps/map.yaml`, lalu jalankan:
```bash
ros2 launch athome_navigation navigation.launch.py
```

Jika file peta berada di lokasi lain:
```bash
ros2 launch athome_navigation navigation.launch.py map:=/path/to/my_map.yaml
```

### 3. Navigasi dengan SLAM Aktif (Tanpa Peta Awal)
Untuk skenario ruangan yang belum dipetakan:
```bash
ros2 launch athome_navigation navigation.launch.py slam:=true
```

### 4. Argumen Peluncuran yang Sering Digunakan
- `slam`: `true` untuk SLAM real-time, `false` untuk peta statis + AMCL (default: `false`).
- `map`: Path ke file peta `.yaml` (default: `athome_navigation/maps/map.yaml`).
- `rviz`: Tampilkan RViz2 (default: `true`).
- `rviz_software`: Gunakan OpenGL software rendering jika GPU driver bermasalah (default: `false`).
- `initial_x`, `initial_y`, `initial_yaw`: Menentukan perkiraan posisi awal robot di peta (default: `0.0`).
- `use_sensors`: Menyalakan RPLiDAR otomatis (default: `false`).
- `nucleo`: Menyalakan driver base STM32 otomatis (default: `false`).

Contoh peluncuran lengkap mandiri:
```bash
ros2 launch athome_navigation navigation.launch.py \
  map:=$(ros2 pkg prefix --share athome_navigation)/maps/map.yaml \
  use_sensors:=true \
  nucleo:=true \
  stm_ip:=192.168.0.4
```
