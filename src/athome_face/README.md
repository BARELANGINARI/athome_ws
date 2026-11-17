### 1. Jalankan Node & Buka di Browser Manual
```bash
ros2 launch athome_face face.launch.py

Buka browser dan akses: http://localhost:8080
```

### 2. Jalankan Langsung dengan Kiosk Browser (HDMI LCD Layar Penuh)
```bash
ros2 launch athome_face face.launch.py browser:=true kiosk:=true
```

### Contoh Perintah Cepat

```bash
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'HAPPY'}"
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'FOLLOWING'}"
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'SEARCHING'}"
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'CURIOUS'}"
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'WORKING'}"
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'SUCCESS'}"
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'LOST'}"
ros2 topic pub --once /robot/expression std_msgs/msg/String "{data: 'IDLE'}"
```

### 2. Menampilkan Balon Percakapan / Subtitle (`/hri/tts_say`)
```bash
ros2 topic pub --once /hri/tts_say std_msgs/msg/String '{data: "Halo, saya Barelang Inari!"}'
```
