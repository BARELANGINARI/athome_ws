# athome_mapping

2D SLAM indoor mapping package using `slam_toolbox` and RPLiDAR.

## Panduan Mapping

### Opsi 1: Satu Perintah All-in-One (Paling Praktis)
Perintah ini otomatis menyalakan driver motor Nucleo, RPLiDAR, static TF, SLAM Toolbox, dan RViz2:
```bash
ros2 launch athome_mapping mapping.launch.py
```
> **Catatan:** Jika port RPLiDAR berbeda (default `/dev/ttyUSB1`), bisa dioperkan via argument:
> ```bash
> ros2 launch athome_mapping mapping.launch.py lidar_port:=/dev/ttyUSB0
> ```

---

### Opsi 2: Multi-Terminal (Hardware Terpisah)
Jika driver hardware sudah dinyalakan di terminal lain:

**Terminal 1 (Hardware):**
```bash
ros2 launch receptionist hardware.launch.py
# atau:
# ros2 launch help_me_carry hardware.launch.py
```

**Terminal 2 (Mapping & RViz):**
```bash
ros2 launch athome_mapping mapping.launch.py launch_hardware:=false
```

---

### Menggerakkan Robot Saat Mapping (Teleoperasi)
Buka terminal baru untuk mengemudikan robot menyusuri ruangan:
```bash
# Menggunakan joystick:
ros2 launch athome_bringup joystick.launch.py

# Atau menggunakan keyboard:
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

---

### Menyimpan Peta (Map Saver)
Setelah ruangan selesai dipetakan dengan rapi di RViz:
```bash
cd ~/Documents/athome-ws
ros2 run nav2_map_server map_saver_cli -f src/athome_navigation/maps/map
```
Perintah ini akan memperbarui file `map.yaml` dan `map.pgm` di dalam `src/athome_navigation/maps/`.
