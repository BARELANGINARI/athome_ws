# athome_hri_cpp

Implementasi **C++17** untuk *Speech Recognition* (`stt_engine_node`) pada robot RoboCup@Home sebagai pengganti *drop-in* modul Python `athome_hri/stt_engine_node`. Topik ROS 2 (`/hri/stt_transcript`, `/hri/tts_say`, `/hri/tts_done`) identik sehingga seluruh task (`receptionist`, `restaurant`, `help_me_carry`) dapat langsung menggunakannya tanpa perubahan kode.

## Arsitektur Pipeline

```text
PortAudio Mic (16 kHz / 48 kHz -> 16 kHz)
  │
  ▼ [Thread Capture]
Adaptive Noise-Floor VAD (300–3400 Hz Bandpass + Spectral Flatness + Max 7.5s Cap)
  │
  ▼ [Thread ASR Worker]
AudioEnhancer (CPU: 120 Hz Zero-Phase HPF + DeepFilterNet2 / RNNoise / Wiener Spectral)
  │
  ▼
Silero Neural VAD (ONNX Runtime CPU, 1 thread, 0 VRAM)
  │
  ▼
whisper.cpp (CUDA RTX 2060 / CPU, temperature=0.0, no_context=true, entropy_thold)
  │
  ▼
Post-Processing (N-Gram Deduplication + Hallucination Blacklist) ──▶ /hri/stt_transcript
```

## 1. Instalasi Dependensi (`whisper.cpp` CUDA + Model)

Jalankan script berikut satu kali pada mesin robot:
```bash
chmod +x src/athome_hri_cpp/scripts/setup_deps.sh
./src/athome_hri_cpp/scripts/setup_deps.sh
```
*(Jika folder `models/` dibuat setelah instalasi, tambahkan `install(DIRECTORY models/ ...)` atau arahkan `whisper_model_path` dan `stt_silero_vad_model_path` di `config/params.yaml` ke path absolut modelnya.)*

## 2. Build Package

```bash
colcon build --packages-select athome_hri_cpp
source install/setup.bash
```

## 3. Pengujian Mandiri

Terminal 1:
```bash
ros2 launch athome_hri_cpp stt_cpp.launch.py
```

Terminal 2:
```bash
ros2 topic echo /hri/stt_transcript
```

## 4. Integrasi ke Launch File Task (`hri.launch.py`)

Ketiga task (`receptionist`, `help_me_carry`, `restaurant`) sudah terhubung ke `athome_hri_cpp` secara default melalui argumen `stt_backend`:
```bash
# Default (C++ STT):
ros2 launch receptionist hri.launch.py
ros2 launch help_me_carry hri.launch.py
ros2 launch restaurant hri.launch.py

# Switch kembali ke Python STT untuk perbandingan:
ros2 launch receptionist hri.launch.py stt_backend:=python
ros2 launch help_me_carry hri.launch.py stt_backend:=python
ros2 launch restaurant hri.launch.py stt_backend:=python
```

## 5. Benchmark Perbandingan Python vs C++

1. Jalankan task dengan **Python STT**, lalu rekam metrik CPU, RAM, VRAM, dan transkrip selama 60 detik:
```bash
ros2 launch receptionist hri.launch.py stt_backend:=python
# Terminal lain:
python3 src/athome_hri_cpp/scripts/benchmark_stt_compare.py record --label python --out /tmp/bench_py.json --duration 60
```
2. Hentikan, jalankan task yang sama dengan **C++ STT**, lalu rekam dengan durasi dan skenario suara yang sama:
```bash
ros2 launch receptionist hri.launch.py stt_backend:=cpp
# Terminal lain:
python3 src/athome_hri_cpp/scripts/benchmark_stt_compare.py record --label cpp --out /tmp/bench_cpp.json --duration 60
```
3. Tampilkan tabel perbandingan:
```bash
python3 src/athome_hri_cpp/scripts/benchmark_stt_compare.py compare /tmp/bench_py.json /tmp/bench_cpp.json
```

## 6. GUI Environment Tuner (Kalibrasi & Live Tuning per Lokasi)

Gunakan GUI tuner untuk mengkalibrasi parameter terbaik di setiap ruangan/arena:
```bash
python3 src/athome_hri_cpp/scripts/stt_tuner_gui.py
```
Alur cepat: **Start Monitor** -> **1. Measure Noise (5s)** -> **2. Measure Voice (5s)** -> **Auto-Calculate Params** -> **Save to YAML** (atau uji live ke `/stt_engine` dengan **Apply Live**).
