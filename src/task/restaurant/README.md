## cara menjalankan program

```bash
ros2 launch restaurant hardware.launch.py
ros2 launch restaurant motion.launch.py
```

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
