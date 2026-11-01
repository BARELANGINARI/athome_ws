# athome_face

Antarmuka mandiri ekspresi & emoji robot untuk layar **HDMI LCD 7 inch capacitive** pada robot **Barelang Inari** (RoboCup@Home).

Sistem ini dibuat **100% mandiri (terisolasi)** tanpa memodifikasi atau mengganggu kode modul HRI, navigasi, atau hardware yang ada.

---

## 1. Fitur Utama

* **Animasi Vektor Halus (60 FPS):** Menggunakan SVG + CSS3 animations yang ringan, tidak membebani CPU/GPU Mini-PC.
* **10 Ekspresi Robotik:**
  1. `IDLE`: Mata biru cyan rileks, kedip alami acak (3–5 detik), gerakan bernapas lembut.
  2. `SEARCHING`: Scanning radar dan lirikan mata kiri-kanan mencari orang.
  3. `GREETING`: Mata tersenyum bulan sabit (^_^) + rona pipi merah muda (*blush*).
  4. `LISTENING`: Pupil melebar + animasi gelombang suara saat mendengar ucapan.
  5. `SPEAKING`: Gerakan bibir/gelombang audio aktif saat robot berbicara (TTS).
  6. `FOLLOWING`: Tatapan mantap fokus ke depan dengan sudut alis siap sedia.
  7. `CURIOUS`: Satu alis terangkat tinggi + ikon tanda tanya `?` (tanya tas/gesture).
  8. `GRABBING`: Tatapan tajam melirik ke bawah saat tangan mengambil objek.
  9. `SUCCESS`: Mata bintang emas (*_*) + senyum lebar saat tugas selesai (*Task Completed*).
  10. `LOST`: Mata terkejut/khawatir bernuansa amber/oranye + tanda seru `!`.
* **Interaktivitas Layar Sentuh (*Capacitive Touch*):**
  * Sentuh layar di titik mana saja: Mata robot otomatis melirik ke arah jari Anda.
  * Sentuh bagian pipi: Robot berkedip genit (*cute wink*) dan tersipu malu.
* **Testing Panel & Hotkeys:** Tersedia tombol panel di layar dan shortcut keyboard untuk demo cepat.

---

## 2. Cara Uji Coba Cepat (Standalone / Tanpa ROS 2)

Anda bisa langsung membuka dan menguji animasi wajah di browser tanpa harus menyalakan ROS 2:

Buka file berikut langsung dengan browser (Chrome / Firefox / Edge):
```bash
google-chrome src/athome_face/web/index.html
# atau
firefox src/athome_face/web/index.html
```

### Tombol Shortcut Keyboard:
* **`1`** : IDLE / Normal
* **`2`** : SEARCHING (Mencari Orang)
* **`3`** : GREETING (Menyambut)
* **`4`** : LISTENING (Mendengar)
* **`5`** : SPEAKING (Bicara)
* **`6`** : FOLLOWING (Mengikuti)
* **`7`** : CURIOUS (Penasaran / Tunggu Gesture)
* **`8`** : GRABBING (Ambil Tas)
* **`9`** : SUCCESS (Selesai / Menang)
* **`0`** : LOST (Orang Hilang / Peringatan)
* **`Space`** : Kedip manual
* **`T`** : Sembunyikan / Tampilkan panel menu testing di bawah

---

## 3. Menjalankan dengan ROS 2

### A. Build Package
Dari root workspace:
```bash
cd ~/Documents/athome-ws
colcon build --packages-select athome_face
source install/setup.bash
```

### B. Launch Node
Jalankan node antarmuka wajah:
```bash
# Mode server biasa (bisa dibuka di browser http://localhost:8080)
ros2 launch athome_face face.launch.py

# Atau langsung otomatis buka browser full-screen kiosk di layar HDMI:
ros2 launch athome_face face.launch.py browser:=true
```

---

## 4. Mengubah Ekspresi via CLI Terminal

Setelah node aktif, Anda bisa mengganti ekspresi wajah dari terminal menggunakan topik `/robot/expression`:

```bash
# Menyambut operator
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'GREETING'}"

# Mencari orang
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'SEARCHING'}"

# Mengikuti orang
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'FOLLOWING'}"

# Penasaran (menunggu gesture tas)
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'CURIOUS'}"

# Sukses / selesai
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'SUCCESS'}"

# Orang hilang / peringatan
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'LOST'}"

# Kembali normal
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'IDLE'}"
```

