## cara menjalankan program

```bash
ros2 launch restaurant hardware.launch.py
ros2 launch restaurant motion.launch.py
```

## kamera / deteksi lambaian tangan

Jalankan vision dalam mode restaurant agar node mendeteksi orang yang mengangkat/melambaikan tangan (kiri atau kanan) dan otomatis mempublikasikan koordinat orang terdekat yang melambai ke `/zed/order1`:

```bash
ros2 launch restaurant camera.launch.py
# atau langsung via athome_vision:
ros2 launch athome_vision zed.launch.py task:=restaurant enable_yolo:=false enable_bag:=false enable_drinks:=false enable_chair:=false
```

Gesture yang dipublikasikan ke `/zed/gesture`: `WAVE_LEFT`, `WAVE_RIGHT`, atau `NONE`.

## service
```bash
ros2 service call /homing std_srvs/srv/Trigger {}
ros2 service call /go_to_order1 std_srvs/srv/Trigger {}
```



### Navigasi ke Pelanggan (Avoid Obstacle & Safe Distance)
1. **Input Posisi Orang dari Kamera:**
   Kamera/Node mempublikasikan koordinat orang ke topik `/zed/order1` (`std_msgs/msg/String`), contoh:
   ```bash
   ros2 topic pub --once /zed/order1 std_msgs/msg/String "{data: '2.0, 0.5'}"
   ```
2. **Perintahkan Robot Menghampiri Pelanggan:**
   ```bash
   ros2 service call /go_to_order1 std_srvs/srv/Trigger {}
   ```
   *Robot akan menghitung jarak henti aman (0.85m di depan orang), mencari jalur aman melewati/menghindari meja & rintangan menggunakan Nav2, lalu berhenti tepat menghadap pelanggan.*

### Service Homing (Kembali ke Titik Awal Start)
```bash

```
