``` bash 

### Multi-Pane (Terminator Layout 2x2 Grid dalam 1 Layar):
./src/task/receptionist/launch/run.sh

# Hentikan semua node:
./src/task/receptionist/launch/stop.sh
# Atau:
./src/task/receptionist/launch/run.sh stop

### Modular (Manual per terminal)
ros2 launch receptionist hardware.launch.py
ros2 launch receptionist motion.launch.py
ros2 launch receptionist hri.launch.py
ros2 launch receptionist camera.launch.py
ros2 launch athome_face face.launch.py

### All-in-One (Semua di 1 terminal):
ros2 launch receptionist receptionist.launch.py hri:=true

### Service
ros2 service call /receptionist/go_to_door std_srvs/srv/Trigger {}
ros2 service call /receptionist/go_beverage_area std_srvs/srv/Trigger {}
ros2 service call /receptionist/go_to_living_room std_srvs/srv/Trigger {}
ros2 service call /search_chair std_srvs/srv/Trigger {}

# Service Menghadap Orang (Gaze / Face Person):
ros2 service call /receptionist/host std_srvs/srv/Trigger {}
ros2 service call /receptionist/guest1 std_srvs/srv/Trigger {}
ros2 service call /receptionist/guest2 std_srvs/srv/Trigger {}
ros2 service call /receptionist/adjust_guest std_srvs/srv/Trigger {}
ros2 service call /receptionist/approach_guest std_srvs/srv/Trigger {}

# Service Rotasi 90 Derajat (Kiri dan Kanan) & Reset Yaw Living Room:
ros2 service call /receptionist/left std_srvs/srv/Trigger {}
ros2 service call /receptionist/right std_srvs/srv/Trigger {}
ros2 service call /receptionist/reset_living_room_yaw std_srvs/srv/Trigger {}

### buat inisialisasi pose yaw robot dimap pakai rad
0∘ ≈ 0.0 rad 
90∘ ≈ 1.57 rad
180∘ ≈ 3.14 rad
270∘ / −90∘ ≈ −1.57 rad

### ini cara cek yaw robot
ros2 run tf2_ros tf2_echo map base_link
```

---

## Pra-Registrasi & Pengenalan Host

Robot dapat mengenali Host sejak awal sebelum misi dimulai menggunakan foto wajah, sekaligus mendeskripsikan atribut Host (pakaian, minuman favorit, minat) ke tamu saat escort ke ruang tamu.

### 1. Mengambil / Menyiapkan Foto Wajah Host

Foto Host harus disimpan di `src/athome_vision/models/host.jpg`. Anda dapat menggunakan salah satu dari 3 cara berikut:

#### Cara A: Ambil Foto Langsung via Kamera (Live Capture)
Gunakan script berikut (pastikan `camera.launch.py` / node `vision` **tidak sedang berjalan**, karena `/dev/video0` dipakai eksklusif oleh node `vision`):
```bash
cd ~/Documents/athome-ws
./src/task/receptionist/launch/capture_host.sh --camera
```
Atau lewat menu interaktif:
```bash
cd ~/Documents/athome-ws
./src/task/receptionist/launch/capture_host.sh
# Pilih [1] Capture Host dari Kamera (Live Camera)
```
- Jendela preview kamera akan terbuka dan mendeteksi wajah secara visual (kotak hijau).
- Posisikan wajah Host di tengah frame.
- Tekan **`[SPASI]`** atau **`[ENTER]`** untuk mengambil foto.
- Script akan otomatis memotong (crop) area wajah, me-resize ke `112x112`, dan menyimpannya ke:
  - `src/athome_vision/models/host.jpg`
  - `models/host.jpg`
  - `install/athome_vision/share/athome_vision/models/host.jpg` (jika folder `install` tersedia)
- Tekan **`[Q]`** atau **`[ESC]`** jika ingin membatalkan.

*(Opsional: jika menggunakan kamera indeks lain, sertakan nomor indeks setelah `--camera`, misal: `./src/task/receptionist/launch/capture_host.sh --camera 1`)*

#### Cara B: Import dari File Foto yang Sudah Ada
Jika Anda memiliki file foto Host (misalnya dari HP/flashdisk), atau saat `camera.launch.py` sedang berjalan, gunakan Fitur 2 untuk auto-crop wajah:
```bash
cd ~/Documents/athome-ws
./src/task/receptionist/launch/capture_host.sh --photo /path/ke/foto_host.jpg
```
Atau langsung:
```bash
cd ~/Documents/athome-ws
./src/task/receptionist/launch/capture_host.sh /path/ke/foto_host.jpg
```
Script akan otomatis mencari wajah pada foto tersebut, melakukan crop, menyimpannya ke `host.jpg`, dan mengirim notifikasi ke topic `/vision/load_host_photo` jika node `vision` sedang aktif.

#### Cara C: Salin Manual File Foto
Letakkan file foto wajah Host (format `.jpg` atau `.png`, wajah tampak depan jelas) secara manual di:
```bash
src/athome_vision/models/host.jpg
```
> **Catatan:** Pastikan `camera.launch.py` menggunakan file parameter yang sesuai (`params_file`, default: `src/task/receptionist/config/receptionist_params.yaml`).

### 2. Mengatur Profil & Deskripsi Host
Buka file konfigurasi `src/task/receptionist/config/receptionist_params.yaml`, lalu sesuaikan profil Host pada bagian `receptionist_manager` **dan** `vision`:
```yaml
receptionist_manager:
  ros__parameters:
    host:
      name: "Yesaya"           # Nama Host yang duduk di living room
      drink: "Coke"
      interest: "Robotics"
      gender: "male"           # "male" atau "female"
      shirt_color: "black"     # Warna pakaian/baju host
      hair_color: "black"      # Warna rambut host

vision:
  ros__parameters:
    face_model: "models/arcface_w600k_r50.onnx"
    face_threshold: 0.40
    host_name: "Yesaya"        # Harus sama dengan receptionist_manager.host.name
    host_photo: "models/host.jpg"
    host_shirt_color: ""       # Kosongkan agar tidak veto jika warna baju/pencahayaan berubah
    host_gender: ""
    host_hair_color: ""
```

### 3. Verifikasi Saat Menjalankan Kamera
Saat `camera.launch.py` (atau `run.sh`) dijalankan, periksa terminal kamera. Fitur aktif jika muncul log hijau:
```text
[RECEPTIONIST] Host 'Yesaya' registered from photo (...x..., shirt: ..., gender: ..., gallery: 3 templates)
```
> Jika muncul `photo pending self-learn`, artinya wajah di `host.jpg` tidak terdeteksi—ulangi capture dengan pencahayaan lebih jelas.

Begitu Host terdeteksi di ruang tamu, robot akan otomatis mengidentifikasinya sebagai Host dan memperkenalkan tamu kepadanya.

### 4. Alternatif: Registrasi Host Langsung (Tanpa Foto)
Jika tidak membawa foto host sebelum lomba/uji coba, Anda dapat mendaftarkan wajah Host secara langsung saat kamera menyala:
1. Posisikan Host berdiri di depan kamera ZED (~1.2 – 1.8 meter).
2. Jalankan script pendaftaran (tanpa tanda kurung siku `[ ]`, default: `Revaldo`):
   ```bash
   ./src/task/receptionist/launch/register_host.sh NamaHost
   # Contoh: ./src/task/receptionist/launch/register_host.sh Yesaya
   ```

