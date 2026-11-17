# receptionist

Package task **Receptionist** (RoboCup@Home) untuk robot Barelang Inari. Mengatur penyambutan tamu di pintu masuk, percakapan pengenalan nama & minat, pengantaran ke meja minuman (*beverage area*), deteksi ketersediaan minuman, pengantaran ke ruang tamu (*living room*), deteksi kursi kosong, dan perkenalan tamu kepada Host maupun tamu sebelumnya.

---

## 📋 Fitur Utama
- **Interaksi Suara NLU & Dialog Manager**: Menanyakan nama, minat/hobi, dan minuman favorit tamu, mengonfirmasi dengan pertanyaan *yes/no*, dan menyimpan profil ke memori.
- **Navigasi Multi-Waypoint**: Navigasi otonom Nav2 menuju waypoint `door`, `beverage_area`, dan `living_room`.
- **Deteksi & Alokasi Kursi Kosong (`/search_chair`)**: Robot memutar badan untuk mendeteksi kursi terdekat via YOLO, mengidentifikasi kursi yang masih kosong, dan menunjuk ke kursi menggunakan lengan robot.
- **Pengenalan Wajah & Host Pre-Registration**: Mengenali Host dari foto awal atau registrasi langsung, serta mendeskripsikan atribut visual tamu (gender, warna baju, warna rambut).
- **Konfigurasi Terpusat YAML**: Koordinat waypoint, parameter deteksi kursi, profil host, serta seluruh watchdog timeout dialog dan navigasi dikonfigurasi melalui `config/receptionist_params.yaml`.

---

## 🚀 Cara Menjalankan

### 1. Build Package
```bash
cd ~/Documents/athome_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select receptionist --symlink-install
source install/setup.bash
```

### 2. Multi-Pane Grid (Terminator Layout 2x2)
Metode peluncuran utama untuk pengujian dan kompetisi:
```bash
./src/task/receptionist/launch/run.sh
```

Untuk menghentikan semua proses:
```bash
./src/task/receptionist/launch/stop.sh
# Atau:
./src/task/receptionist/launch/run.sh stop
```

### 3. Peluncuran Modular (Manual per Terminal)
```bash
# Terminal 1: Hardware (STM32 & RPLiDAR)
ros2 launch receptionist hardware.launch.py

# Terminal 2: Navigasi Nav2 & Peta Statis
ros2 launch receptionist motion.launch.py

# Terminal 3: HRI (Whisper STT, Piper TTS & Task Manager)
ros2 launch receptionist hri.launch.py

# Terminal 4: Vision (ZED Camera & Face Recognition)
ros2 launch receptionist camera.launch.py

# Terminal 5 (Opsional): Tampilan Wajah Robot
ros2 launch athome_face face.launch.py
```

### 4. Peluncuran All-in-One
```bash
ros2 launch receptionist receptionist.launch.py hri:=true camera:=true
```

---

## ⚙️ Konfigurasi Parameter YAML (`config/receptionist_params.yaml`)

```yaml
receptionist_node:
  ros__parameters:
    door: { x: 0.5, y: 0.0, yaw: 180.0 }
    beverage_area: { x: 1.9, y: 1.9, yaw: 270.0 }
    living_room: { x: 2.0, y: 0.0, yaw: 270.0 }
    search_chair_angular_speed: 0.35
    search_chair_timeout_sec: 40.0

receptionist_manager:
  ros__parameters:
    host:
      name: "Alex"
      drink: "Coke"
      interest: "Robotics"
      gender: "male"
      shirt_color: "black"
      hair_color: "black"
    camera_width: 672
    global_timeout_sec: 600.0         # Batas waktu total task (10 menit)
    dialog_reprompt_sec: 12.0         # Waktu ulang tanya jika tamu diam
    wait_guest_prompt_sec: 8.0        # Waktu ajakan berdiri di depan robot
    vision_initial_settle_sec: 1.2    # Jeda kamera stabil setelah berhenti
    min_sit_wait_sec: 4.0             # Jeda minimal memberi tamu waktu duduk
    person_freshness_sec: 2.0         # Toleransi masa berlaku deteksi visual
    nav_retry_cooldown_sec: 2.0       # Jeda retry navigasi jika gagal
```

---

## 📡 Service ROS 2
* `ros2 service call /receptionist/go_to_door std_srvs/srv/Trigger {}`: Navigasi ke pintu masuk.
* `ros2 service call /receptionist/go_beverage_area std_srvs/srv/Trigger {}`: Navigasi ke meja minuman.
* `ros2 service call /receptionist/go_to_living_room std_srvs/srv/Trigger {}`: Navigasi ke ruang tamu.
* `ros2 service call /search_chair std_srvs/srv/Trigger {}`: Cari dan hadap ke kursi kosong.

---

## 👤 Pra-Registrasi & Pengenalan Host

Robot dapat mengenali Host sejak awal sebelum misi dimulai menggunakan foto wajah, sekaligus mendeskripsikan atribut Host ke tamu.

### 1. Menyiapkan Foto Wajah Host
Letakkan file foto wajah Host (format `.jpg` atau `.png`, tampak depan jelas) di:
```bash
src/athome_vision/models/host.jpg
```

### 2. Mengatur Profil & Deskripsi Host
Sesuaikan profil Host pada `src/task/receptionist/config/receptionist_params.yaml` di blok `receptionist_manager`.

### 3. Pendaftaran Wajah Host Langsung (Tanpa Foto)
Jika registrasi langsung di lokasi:
1. Posisikan Host berdiri di depan kamera ZED (~1.2 – 1.8 meter).
2. Jalankan script pendaftaran:
   ```bash
   ./src/task/receptionist/launch/register_host.sh Alex
   ```
