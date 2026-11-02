# Barelang Inari — RoboCup@Home Workspace

ROS 2 workspace for the **Barelang Inari** service robot (RoboCup@Home).

## Packages

```text
src/
├── athome_base/          # STM32 Nucleo 3-omni motor base controller (UDP)
├── athome_bringup/       # Sensors (RPLiDAR), static TF, and joystick launch
├── athome_mapping/       # 2D SLAM mapping (slam_toolbox)
├── athome_navigation/    # Nav2 navigation stack and saved maps
├── athome_vision/        # ZED camera: body tracking, YOLO bag detection, gesture
├── athome_arm/           # Dynamixel AX-18A robotic arm motion server & CLI tools
├── athome_hri/           # Speech interaction (Faster-Whisper, Piper TTS) & Web GUI
└── task/
    ├── help_me_carry/    # Person following, bag alignment, and homing
    └── receptionist/     # Waypoint navigation (door & living room)
```

## Build

```bash
cd ~/athome_ws
colcon build --symlink-install
source install/setup.bash
```

## Quick Start

### 1. Teleoperation & Sensors
```bash
# Launch LIDAR and base sensors
ros2 launch athome_bringup sensors.launch.py

# Joystick teleop
ros2 launch athome_bringup joystick.launch.py
```

### 2. 2D SLAM Mapping
```bash
# Start mapping node and RViz
ros2 launch athome_mapping mapping.launch.py

# Save generated map
ros2 run nav2_map_server map_saver_cli -f src/athome_navigation/maps/map
```

### 3. Robotic Arm (Dynamixel AX-18A)

#### Motion Server & Services
```bash
# Start arm service server (default port /dev/ttyUSB1, baudrate 1000000)
ros2 launch athome_arm arm.launch.py port:=/dev/ttyUSB1 baudrate:=1000000

# Trigger gerak mengambil tas
ros2 service call /arm/ambil_tas std_srvs/srv/Trigger {}

# Trigger gerak meletakkan tas
ros2 service call /arm/letakkan_tas std_srvs/srv/Trigger {}
```

#### Standalone CLI Tools (Kalibrasi & Teaching)
```bash
# Teaching gerak servo interaktif (record, play, save)
ros2 run athome_arm learn_motion

# Kalibrasi titik nol dan batas servo
ros2 run athome_arm calibrate_arm

# Uji coba cengkeraman gripper adaptif (torque sensing)
ros2 run athome_arm adaptive_gripper
```

### 4. Help Me Carry Task

#### Opsi A: All-in-One Bringup
```bash
ros2 launch help_me_carry help_me_carry.launch.py
```

#### Opsi B: Modular Stack (Direkomendasikan untuk Debugging / 4 Terminal)
```bash
# Terminal 1 — Hardware base motor STM32 & RPLiDAR
ros2 launch help_me_carry hardware.launch.py

# Terminal 2 — Kamera ZED (deteksi orang, gesture, & YOLO tas)
ros2 launch help_me_carry camera.launch.py

# Terminal 3 — SLAM, Nav2, follower tracker node & RViz
ros2 launch help_me_carry motion.launch.py

# Terminal 4 — HRI suara (Faster-Whisper, Piper TTS) & Task Manager
ros2 launch help_me_carry hri.launch.py
# (Opsional custom YAML): ros2 launch help_me_carry hri.launch.py params_file:=/path/to/custom_params.yaml
```

#### Alur Perintah Suara (Voice Interaction)
1. **Wakeword**: Ucapkan *"Hello robot"* saat berdiri di depan robot.
2. **Start Follow**: Ucapkan *"Follow me"* untuk memulai *person following*.
3. **Arrived**: Ucapkan *"Arrived"* / *"We are here"* / *"Berhenti"* saat sampai di tujuan (tersedia jeda aman 5 detik di awal follow).
4. **Pointing Bag**: Tunjuk tas target (*kiri* atau *kanan*) untuk diambil oleh lengan robot.


### 5. Receptionist Task
```bash
ros2 launch receptionist receptionist.launch.py
```
