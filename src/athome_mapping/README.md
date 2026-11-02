# athome_mapping

2D SLAM indoor mapping package using `slam_toolbox` and RPLiDAR.

## Usage

### 1. Launch Hardware & Sensors
```bash
ros2 launch athome_bringup sensors.launch.py
```

### 2. Launch SLAM Node & RViz
```bash
ros2 launch athome_mapping mapping.launch.py
```

### 3. Teleoperate Robot
Drive the robot around to map the environment:
```bash
ros2 launch athome_bringup joystick.launch.py
# or using keyboard:
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### 4. Save Map
```bash
cd ~/athome_ws
ros2 run nav2_map_server map_saver_cli -f src/athome_navigation/maps/map
```
This generates `map.yaml` and `map.pgm` inside `src/athome_navigation/maps/`.
