# person_det

Paket C++ untuk deteksi orang, estimasi pose keypoint, klasifikasi orientasi hadap, gender, warna pakaian, gaya rambut, dan identifikasi wajah (*Face Re-ID*).

Dikonversi sepenuhnya dari Python ke **C++ (ROS 2 Jazzy & ONNX Runtime)** untuk performa tinggi, stabilitas, dan latensi rendah.

---

## 1. Fitur Utama

1. **YOLOv8-Pose (17 Keypoints):** Deteksi orang dan estimasi kerangka pose 2D (*nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles*).
2. **Klasifikasi Orientasi Tubuh (*Body Orientation*):** Menentukan arah hadap orang:
   - `FRONT` (Menghadap depan / ke kamera)
   - `BACK` (Membelakangi kamera)
   - `LEFT SIDE` (Menghadap ke kiri)
   - `RIGHT SIDE` (Menghadap ke kanan)
   - `SIDE`
3. **Klasifikasi Gender (*Gender Classification*):** Klasifikasi `"male"` / `"female"` dari crop tubuh bagian atas.
4. **Klasifikasi Warna Pakaian (*Clothing Color*):** Klasifikasi 46 warna pakaian dari crop torso.
5. **Klasifikasi Rambut (*Hair Style*):** Klasifikasi 5 kategori rambut (`black hair`, `blond hair`, `completely bald`, `red hair`, `white hair`) dari crop kepala.
6. **Face Re-ID / Memory:** Pencocokan embedding wajah via *cosine similarity* dan penyimpanan ke `face_memory.json` saat nama tamu diterima.
7. **Deteksi Orang Terdekat (*Nearest Person*):** Otomatis memilih kandidat terdekat berdasarkan area bounding box terbesar untuk dipublikasikan ke topik ROS 2.

---

## 2. Struktur Direktori Bersih

```text
src/person_det/
├── CMakeLists.txt                    # CMake build configuration (ament_cmake)
├── package.xml                       # ROS 2 package manifest
├── README.md                         # Dokumentasi lengkap
├── include/
│   └── person_det/
│       └── person_detection.hpp      # Header C++ (Structs, Enums, & PersonDetector class)
├── src/
│   ├── person_detection.cpp          # Implementasi engine ONNX, NMS, & klasifikasi
│   ├── person_det_node.cpp           # ROS 2 Node (Sub: sound_event, guest_name | Pub: detected_person)
│   └── person_detection_cli.cpp      # Standalone CLI tool (pengujian kamera tanpa ROS 2)
├── launch/
│   └── person_det.launch.py          # Launch file ROS 2
└── models/
    ├── yolov8n-pose.onnx             # Model ONNX pose keypoints
    ├── gender_cls.onnx               # Model ONNX gender
    ├── color_model.onnx              # Model ONNX warna pakaian
    ├── hair_model.onnx               # Model ONNX rambut
    ├── color_names.txt               # Daftar 46 kelas warna
    ├── hair_names.txt                # Daftar 5 kelas rambut
    └── face_memory.json              # Basis data embedding tamu tersimpan
```

---

## 3. Cara Kompilasi (Build)

```bash
cd ~/Documents/athome-ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select msgs person_det
source install/setup.bash
```

---

## 4. Cara Menjalankan

### A. Menjalankan ROS 2 Node (Via Launch File)

```bash
ros2 launch person_det person_det.launch.py
```

Parameter yang dapat disesuaikan:
* `camera_index` (default: `2`): Index kamera V4L2.
* `conf_threshold` (default: `0.6`): Ambang batas kepercayaan deteksi.
* `debug_window` (default: `true`): Tampilkan jendela preview GUI OpenCV.

Contoh dengan parameter:
```bash
ros2 launch person_det person_det.launch.py camera_index:=0 conf_threshold:=0.5
```

---

### B. Menjalankan Standalone CLI (Pengujian Langsung Tanpa ROS 2)

Dapat dijalankan langsung untuk menguji kamera dan deteksi tanpa memerlukan core ROS:

```bash
# ros2 run person_det person_detection [camera_index] [models_dir]
ros2 run person_det person_detection 2
```
*Tekan tombol `q` atau `ESC` pada jendela preview untuk keluar.*

---

## 5. Antarmuka ROS 2 (Topik)

### Subscriptions:
* `/hri/sound_event` (`std_msgs/msg/String`):
  Menerima event suara. Jika menerima `"doorbell"`, deteksi otomatis aktif (`START`).
* `/hri/guest_name` (`std_msgs/msg/String`):
  Menerima nama tamu untuk disimpan ke memori wajah (`face_memory.json`).

### Publications:
* `/vision/detected_person` (`msgs/msg/PersonDetected`):
  Mempublikasikan data orang terdekat:
  - `name`: Nama tamu yang cocok (atau kosong jika belum dikenal).
  - `gender`: `"male"` / `"female"` / `"unknown"`.
  - `shirt_color`: Warna pakaian (misal: `"black"`, `"blue"`, `"white"`, dll).

