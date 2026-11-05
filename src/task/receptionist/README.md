# Receptionist Task Package

Paket tugas **Receptionist** untuk robot **Barelang Inari** (Politeknik Negeri Batam) pada kompetisi **RoboCup@Home**.

Paket ini mengatur alur Finite State Machine (FSM) otonom untuk menyambut dua orang tamu di pintu masuk, melakukan registrasi identitas & wajah (*Face Re-ID*), mengekstrak atribut pakaian & gender, menanyakan minuman & hobi di *beverage area*, memandu ke kursi kosong di *living room*, serta memperkenalkan kedua tamu berdasarkan kesamaan minat dan ciri visual.

---

## 1. Arsitektur Subsistem & FSM

```
+---------------------------------------------------------------------------------+
|                               receptionist_manager                              |
|                          (Finite State Machine Utama)                           |
+--------+---------------------+-------------------+---------------------+--------+
         |                     |                   |                     |
         v                     v                   v                     v
   [ Navigasi ]           [ Visual/Pose ]       [ Suara/HRI ]        [ Display/LCD ]
 receptionist_node         athome_vision          athome_hri            athome_face
    (Nav2 AMCL)             (ZED Camera)       (Whisper & Piper)     (Animasi Wajah)
```

### Tahapan Siklus FSM:
1. **Startup & Navigasi ke Pintu Masuk:**
   * Node `receptionist_manager` otomatis menunggu service `/receptionist/go_to_door` siap, lalu robot langsung bernavigasi ke koordinat pintu masuk (`door`).
2. **Siklus Tamu 1 (Guest 1):**
   * **Greeting:** Robot tiba di pintu (`WAIT_GUEST`) dan menyambut tamu saat terdeteksi.
   * **Dialog Identitas:** Menanyakan nama tamu (`ASK_NAME` / `WAIT_NAME`).
   * **Dialog Minat:** Menanyakan hobi/minat tamu (`ASK_INTEREST` / `WAIT_INTEREST`).
   * **Beverage Area:** Memandu tamu ke area minuman (`NAV_TO_BEVERAGE`), menanyakan minuman favorit (`ASK_DRINK`), dan mengecek ketersediaan minuman di meja.
   * **Living Room & Kursi:** Memandu ke ruang tamu (`NAV_TO_LIVING_ROOM`), mendeteksi kursi kosong via `/zed/empty_seat`, dan mempersilakan tamu duduk (`OFFER_SEAT`).
3. **Siklus Tamu 2 (Guest 2):**
   * Robot kembali ke pintu masuk (`NAV_TO_DOOR`) dan mengulangi proses penyambutan untuk Tamu 2.
4. **Perkenalan & Evaluasi Kemiripan:**
   * Robot menuju tengah ruangan (`NAV_TO_LIVING_ROOM_CENTER`).
   * **Perkenalan Tamu 1 ke Tamu 2:** Mengucapkan nama, ciri visual (warna pakaian & gender), minuman favorit, dan hobi Tamu 1.
   * **Perkenalan Tamu 2 ke Tamu 1:** Mengenalkan data Tamu 2 kepada Tamu 1.
   * **Similarity Check:** Mengevaluasi kesamaan minat antara kedua tamu atau host.
   * **Selesai:** Robot menyelesaikan seluruh rangkaian tugas (`COMPLETED`).

---

## 2. Struktur Direktori

```text
src/task/receptionist/
├── CMakeLists.txt                    # Konfigurasi build ROS 2 C++
├── package.xml                       # Dependensi paket (rclcpp, nav2, dll)
├── README.md                         # Dokumentasi panduan penggunaan
├── config/
│   ├── receptionist_params.yaml      # Parameter koordinat waypoint & profil host
│   └── nav2_params.yaml              # Konfigurasi Nav2 & costmaps
├── launch/
│   ├── receptionist.launch.py        # Launch all-in-one (Hardware + Nav2 + Manager)
│   ├── navigation.launch.py          # Launch stack Nav2 + receptionist_node + manager
│   ├── hardware.launch.py            # Launch STM32 Nucleo motor driver + RPLiDAR
│   └── hri.launch.py                 # Launch ASR (Faster-Whisper) & TTS (Piper)
├── src/
│   ├── receptionist_node.cpp         # Service provider waypoint navigasi Nav2
│   └── receptionist_manager.cpp      # Node FSM Task Manager utama
└── rviz/
    └── receptionist.rviz             # Konfigurasi visualisasi RViz2
```

---

## 3. Kompilasi (Build)

Pastikan dependensi paket kustom `athome_hri` dan `receptionist` ikut dikompilasi:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select receptionist athome_hri
source install/setup.bash
```

---

## 4. Panduan Menjalankan Sistem (3 Terminal Modular Stack)

Jalankan perintah berikut pada terminal terpisah (belum termasuk ekspresi wajah/face):

### Terminal 1: Hardware Bringup (Motor Base & RPLiDAR)
Menjalankan komunikasi UDP motor base STM32 Nucleo dan sensor RPLiDAR:
```bash
source install/setup.bash
ros2 launch receptionist hardware.launch.py
```

### Terminal 2: Navigation Bringup (Nav2, AMCL, Map, RViz & Receptionist Node)
Menjalankan stack navigasi Nav2 dengan peta statis, RViz visualisasi, dan service waypoint `/receptionist/go_to_door` & `/receptionist/go_to_living_room`:
```bash
source install/setup.bash
ros2 launch receptionist navigation.launch.py
```

### Terminal 3: HRI & Task Manager FSM (Voice Engine & State Machine)
Menjalankan modul suara Faster-Whisper STT, Piper TTS, dan node FSM utama `receptionist_manager` yang mengatur alur percakapan serta koordinasi tugas:
```bash
source install/setup.bash
ros2 launch receptionist hri.launch.py
```

---

### Terminal 4 *(Opsional)*: Layar Animasi Wajah Robot (LCD 7 Inci)
Menampilkan animasi ekspresi mata dan status interaksi pada layar HDMI 7 inci robot:
```bash
source install/setup.bash
ros2 launch athome_face face.launch.py
```

### Alternatif All-in-One:
Jika ingin menjalankan seluruh sistem (hardware + nav2 + vision + fsm) dalam satu terminal:
```bash
source install/setup.bash
ros2 launch receptionist receptionist.launch.py hri:=true
```

---

## 5. Konfigurasi Parameter

Parameter waypoint dan profil host diatur pada file [`config/receptionist_params.yaml`](config/receptionist_params.yaml):

```yaml
receptionist_node:
  ros__parameters:
    # Koordinat navigasi (x, y, yaw dalam radian)
    door:
      x: 0.0
      y: 0.0
      yaw: 0.0
    beverage_area:
      x: 1.0
      y: 0.5
      yaw: 0.0
    living_room:
      x: 1.5
      y: 0.0
      yaw: 0.0
    living_room_center:
      x: 2.0
      y: 0.0
      yaw: 0.0
    global_frame: "map"

receptionist_manager:
  ros__parameters:
    host:
      name: "Alex"
      drink: "Coke"
      interest: "Robotics"
    camera_width: 672
```

---

## 6. Antarmuka ROS 2 (Topics & Services)

### Layanan Navigasi (std_srvs/srv/Trigger):
* `/receptionist/go_to_door`: Memerintahkan robot ke posisi pintu masuk.
* `/receptionist/go_to_beverage_area`: Memerintahkan robot ke area minuman.
* `/receptionist/go_to_living_room`: Memerintahkan robot ke area ruang tamu.
* `/receptionist/go_to_living_room_center`: Memerintahkan robot ke titik tengah ruang tamu.

### Topik Publikasi:
* `/hri/tts_say` (`std_msgs/msg/String`): Kalimat ucapan robot yang dikirim ke TTS engine.
* `/receptionist/status` (`std_msgs/msg/String`): Broadcast status navigasi robot.

### Topik Berlangganan:
* `/hri/stt_transcript` (`std_msgs/msg/String`): Teks transkripsi hasil ucapan tamu dari STT engine.
* `/hri/tts_done` (`std_msgs/msg/String`): Konfirmasi selesai bicara dari TTS engine.
* `/zed/person_label` (`std_msgs/msg/String`): Data deteksi orang dari kamera ZED.
* `/zed/drink_detections` (`std_msgs/msg/String`): Data deteksi botol/minuman di meja.
* `/zed/empty_seat` (`std_msgs/msg/String`): Posisi kursi kosong (`left`, `center`, `right`).
