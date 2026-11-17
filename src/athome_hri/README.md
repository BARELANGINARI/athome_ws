# athome_hri

Package Human-Robot Interaction (HRI) untuk robot Barelang Inari. Mengatur interaksi suara dua arah antara manusia dan robot menggunakan **Faster-Whisper** untuk Speech-to-Text (STT) dan **Piper ONNX** untuk Text-to-Speech (TTS).

---

## 📋 Fitur Utama
- **Speech-to-Text (`stt_engine_node`)**:
  - Pengenalan ucapan cepat dan akurat menggunakan model Faster-Whisper offline.
  - Dilengkapi Voice Activity Detection (VAD) berbasis Silero/WebRTC untuk memotong keheningan audio.
  - Menerbitkan transkripsi teks pada topik `/hri/stt_transcript`.
- **Text-to-Speech (`tts_engine_node`)**:
  - Sintesis suara lokal real-time menggunakan Piper ONNX neural voice model.
  - Mendengarkan instruksi teks dari topik `/hri/tts_say`.
  - Menerbitkan notifikasi `/hri/tts_done` setelah ucapan selesai diputar di speaker robot, memungkinkan FSM task menyinkronkan dialog tanpa tumpang tindih (*barge-in prevention*).

---

## 🚀 Cara Menjalankan

### 1. Build Package
```bash
cd ~/Documents/athome_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select athome_hri --symlink-install
source install/setup.bash
```

### 2. Menjalankan HRI Bringup (STT + TTS Bersamaan)
```bash
ros2 launch athome_hri hri.launch.py
```

### 3. Menjalankan Node Secara Terpisah
Jika ingin menjalankan hanya Text-to-Speech:
```bash
ros2 run athome_hri tts_engine_node --ros-args --params-file src/athome_hri/config/params.yaml
```

Jika ingin menjalankan hanya Speech-to-Text:
```bash
ros2 run athome_hri stt_engine_node --ros-args --params-file src/athome_hri/config/params.yaml
```

---

## 🧪 Pengujian Topik HRI

### Menguji Suara Robot (TTS):
```bash
ros2 topic pub --once /hri/tts_say std_msgs/msg/String "{data: 'Hello, I am Barelang Inari robot.'}"
```

### Memonitor Pendengaran Robot (STT):
Bicara ke mikrofon robot, lalu amati output transkripsi:
```bash
ros2 topic echo /hri/stt_transcript
```

---

## ⚙️ Konfigurasi Parameter
Parameter audio, model, dan perangkat mikrofon/speaker dikonfigurasi pada:
- `config/params.yaml`:
  - `stt_engine`: Model whisper size (`tiny.en`, `base.en`), VAD threshold, device index mikrofon.
  - `tts_engine`: Path model Piper ONNX, volume, device output audio speaker.
