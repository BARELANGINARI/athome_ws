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

### 3. Robotic Arm Motion Server
```bash
# Start arm service server
ros2 launch athome_arm arm.launch.py

# Trigger bag pickup motion
ros2 service call /arm/ambil_tas std_srvs/srv/Trigger {}
```

### 4. Help Me Carry Task
```bash
# Run modular stack (4 terminals):
ros2 launch help_me_carry hardware.launch.py   # Base motor & LIDAR
ros2 launch help_me_carry camera.launch.py     # ZED vision & gesture
ros2 launch help_me_carry motion.launch.py     # SLAM, Nav2 & tracker
ros2 launch help_me_carry hri.launch.py        # Voice & Web dashboard
```

### 5. Receptionist Task
```bash
ros2 launch receptionist receptionist.launch.py
```
