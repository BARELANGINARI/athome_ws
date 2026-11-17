# athome_vision

Package Computer Vision utama untuk robot Barelang Inari berbasis **Stereolabs ZED SDK** dan **ONNX Runtime (YOLO & Deep Learning)** dengan akselerasi GPU NVIDIA CUDA.

---

## 📋 Fitur Utama
* **Human Body Tracking (BODY_34)**: Pelacakan kerangka tubuh 3D (34 keypoints) untuk operator following, gesture recognition, dan estimasi jarak.
* **Proximity Bounding Box & Bag Filtering**:
  - Filter jarak deteksi orang (`person_max_dist`, default: `2.8 m`) dan tas (`bag_max_dist`, default: `2.5 m`). Objek di luar batas tidak di-bbox dan tidak dipublikasikan ke controller.
  - Memilih maksimal 2 paper bag terdekat (`max_bags = 2`) dengan proyeksi 3D centroid & fallback median window cloud.
  - Garis visualisasi oranye-merah dari titik tengah tas ke titik tengah bawah kamera `(320, 480)` dan tanda lingkaran centroid.
* **Person Attributes Classification**: Klasifikasi gender (DeepFace), warna pakaian, dan warna rambut (`/vision/person_attributes`) untuk task Receptionist.
* **Pointing Gesture Detection**: Deteksi tangan operator menunjuk ke lantai (`/vision/gesture`: `RIGHT`, `LEFT`, `NONE`) untuk task Help Me Carry.
* **Waving Hand Detection**: Deteksi lambaian tangan pelanggan (`WAVE_LEFT`, `WAVE_RIGHT`) untuk memanggil robot pada task Restaurant.
* **Person Facing Backward**: Deteksi arah hadap orang membelakangi kamera (`/vision/person_back`) berbasis orientasi bahu dan pinggul.
* **Chair Occupancy Tracking**: Pelacakan kursi dan status terisi/kosong (`/vision/chair_detections`, `/vision/empty_seat`).
* **YAML Centralized Configuration**: Semua parameter threshold deteksi, model path, dan konfigurasi task disimpan di `config/vision_params.yaml`.

---

## 🚀 Cara Menjalankan

### 1. Build Package
> *Catatan: Package ini membutuhkan ZED SDK dan CUDA Toolkit yang terinstal pada PC robot.*
```bash
cd ~/Documents/athome_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select athome_vision --symlink-install
source install/setup.bash
```

### 2. Download Model Tambahan (Jika Diperlukan)
```bash
bash src/athome_vision/scripts/download_deepface_gender.sh
bash src/athome_vision/scripts/download_sface.sh
```

### 3. Menjalankan Node Vision Berdasarkan Task

```bash
# Mode umum (semua task aktif):
ros2 launch athome_vision vision.launch.py

# Mode Help Me Carry:
ros2 launch athome_vision vision.launch.py task:=help_me_carry

# Mode Receptionist:
ros2 launch athome_vision vision.launch.py task:=receptionist

# Mode Restaurant:
ros2 launch athome_vision vision.launch.py task:=restaurant
```

### 4. Menampilkan Preview Window (GUI OpenCV)
```bash
ros2 launch athome_vision vision.launch.py debug_window:=true
```

### 5. Mengatur Jarak Deteksi via Launch Arguments
```bash
ros2 launch athome_vision vision.launch.py person_max_dist:=2.5 bag_max_dist:=2.0
```

---

## ⚙️ Konfigurasi Parameter YAML
Parameter lengkap tersimpan di:
- `config/vision_params.yaml`

Dapat diedit langsung tanpa perlu kompilasi ulang kode C++.

---

## 📡 Topik ROS 2
* `/vision/yaw_deg` (`std_msgs/msg/Float64`): Sudut hadap kamera ZED dalam derajat (0 - 360°).
* `/vision/gesture` (`std_msgs/msg/String`): Gestur menunjuk (`RIGHT`, `LEFT`, `NONE`) atau lambaian (`WAVE_LEFT`, `WAVE_RIGHT`).
* `/vision/order` (`std_msgs/msg/String`): Koordinat pelanggan yang melambaikan tangan (`"x, y"` dalam meter).
* `/vision/person_back` (`std_msgs/msg/String`): Koordinat orang yang membelakangi robot (`"x, y"` dalam meter).
* `/vision/person_label` (`std_msgs/msg/String`): Koordinat 3D dan skeleton orang terdekat yang memenuhi ambang batas jarak.
* `/vision/recognized_name` (`std_msgs/msg/String`): Nama orang yang dikenali wajahnya dari database.
* `/vision/person_attributes` (`std_msgs/msg/String`): Atribut visual (`gender, shirt color, hair color`).
* `/vision/bag_detections` (`std_msgs/msg/String`): Koordinat 3D tas terdekat yang terdeteksi.
