# athome_mapping

Package pemetaan (mapping) 2D Simultaneous Localization and Mapping (SLAM) untuk robot Barelang Inari menggunakan `slam_toolbox`. Digunakan untuk membuat peta occupancy grid sebelum menjalankan task navigasi otonom.

---

## 📋 Fitur Utama
- **SLAM Toolbox 2D**: Menghasilkan peta 2D real-time dengan algoritma graph SLAM berbasis scan-matching dan deteksi loop-closure.
- **Visualisasi RViz**: Membuka antarmuka grafis RViz yang menampilkan map, laser scan, dan pose robot saat pemetaan berlangsung.
- **Konfigurasi Fleksibel**: Parameter SLAM diatur melalui file `config/slam.yaml`.

---

## 🚀 Cara Menjalankan

### 1. Build Package
```bash
cd ~/Documents/athome_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select athome_mapping --symlink-install
source install/setup.bash
```

### 2. Memulai Proses Mapping (Membuat Peta)
Pastikan robot dan sensor telah menyala, atau jalankan mapping all-in-one:
```bash
ros2 launch athome_mapping mapping.launch.py
```

Argumen opsional:
```bash
# Jika sensor & base sudah menyala di terminal terpisah
ros2 launch athome_mapping mapping.launch.py use_sensors:=false (atau launch_hardware:=false)

# Mengatur IP mikrokontroler STM32
ros2 launch athome_mapping mapping.launch.py stm_ip:=192.168.0.4

# Menjalankan tanpa visualisasi RViz
ros2 launch athome_mapping mapping.launch.py rviz:=false
```

### 3. Menggerakkan Robot Saat Pemetaan
Buka terminal baru dan kendalikan robot dengan joystick atau keyboard:
```bash
# Menggunakan joystick:
ros2 launch athome_bringup joystick.launch.py

# Atau menggunakan keyboard:
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### 4. Menyimpan Peta (Map Saver)
Setelah seluruh ruangan terpetakan dengan baik di RViz, simpan peta ke direktori tujuan (misal ke `athome_navigation/maps`):
```bash
ros2 run nav2_map_server map_saver_cli -f ~/Documents/athome_ws/src/athome_navigation/maps/map
```
Perintah di atas akan menghasilkan dua file: `map.yaml` dan `map.pgm`.
