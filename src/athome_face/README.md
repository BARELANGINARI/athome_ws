# athome_face

Package ekspresi wajah robot (Robot Face Interface) untuk Barelang Inari. Mengatur animasi mata dan ekspresi emosi robot pada layar LCD kepala robot menggunakan antarmuka Web GUI berbasis HTML5 Canvas dan WebSocket bridge.

---

## 📋 Fitur Utama
- **Web-based Facial Expression**: Tampilan ekspresi wajah animasi interaktif (senang, netral, berkedip, berbicara, bingung, dll.).
- **HTTP & WebSocket Server**: Melayani web GUI lokal pada port HTTP 8080 dan streaming status ekspresi via WebSocket port 8766.
- **Kiosk Mode Support**: Mendukung peluncuran otomatis Chromium/Firefox dalam mode layar penuh (borderless fullscreen kiosk mode) di monitor LCD robot.
- **Sinkronisasi Otomatis**: Secara otomatis menggerakkan mulut/ekspresi ketika robot sedang berbicara via topik `/hri/tts_say` atau mendengarkan interaksi.
- **Kitchen Order HUD Card**: Menampilkan pop-up overlay kartu pesanan & foto capture pelanggan (`/hmi/customer_display`) saat robot menuju kitchen / barman pada task Restaurant.

---

## 🚀 Cara Menjalankan

### 1. Build Package
```bash
cd ~/Documents/athome_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select athome_face --symlink-install
source install/setup.bash
```

### 2. Menjalankan Server Face
```bash
ros2 launch athome_face face.launch.py
```

### 3. Membuka Otomatis di Browser / Layar LCD Robot (Kiosk Mode)
```bash
ros2 launch athome_face face.launch.py browser:=true kiosk:=true
```

Atau akses manual melalui browser di: `http://localhost:8080`.

---

## 🧪 Menguji Perubahan Ekspresi
Kirim pesan string ke topik `/robot/expression`:
```bash
# Ekspresi senang
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'happy'}"

# Ekspresi netral
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'neutral'}"

# Ekspresi terkejut / bingung
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'surprised'}"

### Menguji Tampilan Kartu Pesanan Pelanggan (Kitchen HUD)
```bash
# Tampilkan kartu pesanan pelanggan
ros2 topic pub --once /hmi/customer_display std_msgs/msg/String \
  "{data: '{\"action\":\"show\",\"customer_idx\":1,\"name\":\"Alex\",\"order1\":\"Coke\",\"order2\":\"Chips\",\"image_url\":\"captures/customer_latest.jpg\"}'}"

# Sembunyikan kartu kembali ke animasi wajah
ros2 topic pub --once /hmi/customer_display std_msgs/msg/String "{data: '{\"action\":\"hide\"}'}"
```
```
