# athome_face_display

Package ekspresi wajah robot (Robot Face Interface) untuk **Barelang Inari** (RoboCup@Home). Mengatur animasi mata dan ekspresi emosi robot pada layar LCD kepala robot menggunakan antarmuka Web GUI berbasis HTML5 Canvas/SVG dan WebSocket bridge.

---

## 📋 Fitur Utama
- **Web-based Facial Expression**: Tampilan ekspresi wajah animasi interaktif (senang, netral, berkedip, berbicara, mendengarkan, bingung, dll.).
- **HTTP & WebSocket Server**: Melayani web GUI lokal pada port HTTP 8080 dan streaming status ekspresi via WebSocket port 8766.
- **Kiosk Mode Support**: Mendukung peluncuran otomatis Chromium/Firefox dalam mode layar penuh (borderless fullscreen kiosk mode) di monitor LCD robot.
- **Sinkronisasi Otomatis**: Secara otomatis menggerakkan mulut/ekspresi ketika robot sedang berbicara via topik `/hri/tts_say` atau mendengarkan interaksi.
- **Kitchen Order HUD Card**: Menampilkan pop-up overlay kartu pesanan & foto capture pelanggan (`/hmi/customer_display`) saat robot menuju kitchen / barman pada task Restaurant.

---

## 🚀 Cara Menjalankan

### 1. Build Package
```bash
cd ~/Documents/athome-ws
colcon build --packages-select athome_face_display --symlink-install
source install/setup.bash
```

### 2. Menjalankan Server Face
```bash
ros2 launch athome_face_display face.launch.py
```
Akses manual melalui browser di: `http://localhost:8080`.

### 3. Membuka Otomatis di Layar LCD Robot (Kiosk Mode Fullscreen)
```bash
ros2 launch athome_face_display face.launch.py browser:=true kiosk:=true
```

Argumen launch:
- `http_port` (default: `8080`): Port HTTP untuk Web UI.
- `ws_port` (default: `8766`): Port WebSocket bridge.
- `browser` (default: `false`): Buka otomatis browser saat launch (`true`/`false`).
- `kiosk` (default: `true`): Mode fullscreen borderless (`true`/`false`).
- `browser_type` (default: `auto`): Pilihan browser (`auto`, `firefox`, `chromium`, `google-chrome`).

---

## 🧪 Menguji Perubahan Ekspresi via CLI

Format perintah dasar:
```bash
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: '<EKSPRESI>'}"
```

### Contoh Perintah Cepat
```bash
# Senang / Menyapa (GREETING / HAPPY)
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'HAPPY'}"

# Mendengarkan (LISTENING - Head Tilt, Listening Arcs, Audio Dots)
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'LISTENING'}"

# Berbicara (SPEAKING - Equalizer Wave Mouth)
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'SPEAKING'}"

# Fokus Mengikuti Operator (FOLLOWING)
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'FOLLOWING'}"

# Mencari / Scanning (SEARCHING)
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'SEARCHING'}"

# Penasaran / Menunggu Isyarat (CURIOUS)
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'CURIOUS'}"

# Mengambil Objek / Bekerja (WORKING / GRABBING)
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'WORKING'}"

# Berhasil / Sukses (SUCCESS / WIN)
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'SUCCESS'}"

# Peringatan / Kehilangan Operator (LOST / ALERT)
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'LOST'}"

# Kembali ke Netral / Normal (IDLE)
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'IDLE'}"
```

---

## 💬 Menampilkan Balon Percakapan / Subtitle (`/hri/tts_say`)

```bash
# Gunakan petik tunggal di luar agar tanda seru (!) tidak memicu history error pada Bash:
ros2 topic pub --once /hri/tts_say std_msgs/msg/String '{data: "Halo, saya Barelang Inari!"}'
```

---

## 🍽️ Menguji Tampilan Kartu Pesanan Pelanggan (Kitchen HUD)

```bash
# Tampilkan kartu pesanan pelanggan
ros2 topic pub --once /hmi/customer_display std_msgs/msg/String \
  "{data: '{\"action\":\"show\",\"customer_idx\":1,\"name\":\"Alex\",\"order1\":\"Coke\",\"order2\":\"Chips\",\"image_url\":\"captures/customer_latest.jpg\"}'}"

# Sembunyikan kartu kembali ke animasi wajah
ros2 topic pub --once /hmi/customer_display std_msgs/msg/String "{data: '{\"action\":\"hide\"}'}"
```
