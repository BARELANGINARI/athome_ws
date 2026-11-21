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

# Service Rotasi 90 Derajat (Kiri dan Kanan):
ros2 service call /receptionist/left std_srvs/srv/Trigger {}
ros2 service call /receptionist/right std_srvs/srv/Trigger {}

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

### 1. Menyiapkan Foto Wajah Host
Letakkan file foto wajah Host (format `.jpg` atau `.png`, wajah tampak depan jelas) di:
```bash
src/athome_vision/models/host.jpg
```
> **Catatan:** Anda juga dapat menggunakan foto di lokasi lain dengan menyertakan argumen launch:
> `ros2 launch receptionist camera.launch.py host_name:=Alex host_photo:=/path/ke/foto_host.jpg`

### 2. Mengatur Profil & Deskripsi Host
Buka file konfigurasi `src/task/receptionist/config/receptionist_params.yaml`, lalu sesuaikan profil Host pada bagian `receptionist_manager`:
```yaml
receptionist_manager:
  ros__parameters:
    host:
      name: "Alex"
      drink: "Coke"
      interest: "Robotics"
      gender: "male"          # "male" atau "female"
      shirt_color: "black"     # Warna pakaian/baju host
      hair_color: "black"      # Warna rambut host
```

### 3. Verifikasi Saat Menjalankan Kamera
Saat `camera.launch.py` (atau `run.sh`) dijalankan, periksa terminal kamera. Fitur aktif jika muncul log hijau:
```text
[Host Pre-Register] Berhasil mendaftarkan Host 'Alex' dari file foto: .../models/host.jpg (640x640 px)
```
Begitu Host terdeteksi di ruang tamu, robot akan otomatis mengidentifikasinya sebagai Host dan memperkenalkan tamu kepadanya.

### 4. Alternatif: Registrasi Host Langsung (Tanpa Foto)
Jika tidak membawa foto host sebelum lomba/uji coba, Anda dapat mendaftarkan wajah Host secara langsung saat kamera menyala:
1. Posisikan Host berdiri di depan kamera ZED (~1.2 – 1.8 meter).
2. Jalankan script pendaftaran:
   ```bash
   ./src/task/receptionist/launch/register_host.sh [NamaHost]
   # Contoh: ./src/task/receptionist/launch/register_host.sh Alex
   ```

