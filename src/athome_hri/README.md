# athome_hri

Human-Robot Interaction (HRI) package for Barelang Inari robot, integrating speech recognition, speech synthesis, and an interactive Web GUI.

## Features
* **Speech-to-Text (STT):** Offline transcription using Faster-Whisper.
* **Text-to-Speech (TTS):** Natural speech generation using Piper ONNX neural voices.
* **Audio Speech Queue:** Asynchronous, non-blocking speech playback pipeline.
* **Intent Recognition:** Rule-based NLU for wakeword detection, follow commands, and arrival triggers.
* **Web Dashboard:** Real-time WebSocket-based operator UI (served over HTTPS/WSS).

## Running

```bash
ros2 launch athome_hri hri.launch.py
```

Open a web browser and navigate to `https://localhost:8443` (or the robot's IP address) to view the live dashboard and mic controls.

## ROS 2 Topics & Services
* **Subscriptions:**
  * `/zed/person_label`: Detects operator presence.
  * `/zed/gesture`: Pointing gesture input (`LEFT`/`RIGHT`).
  * `/hri/tts_say`: Custom text input to speak immediately.
* **Publications:**
  * `/hri/event`: HRI state events (`WAKEWORD_DETECTED`, `FOLLOW_COMMAND`, `ARRIVED_COMMAND`).
* **Triggers Called:**
  * `/follow/person`, `/follow/stop`, `/task/align_bag`, `/arm/ambil_tas`, `/homing`.
