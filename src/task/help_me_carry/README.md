# help_me_carry

RoboCup@Home **Help Me Carry** task package for Barelang Inari robot.

Features Kalman Filter person tracking with occlusion dead-reckoning, spatial gating, pointing gesture alignment, robotic arm pickup, and autonomous homing.

## Launch Options

### Multi-Pane (Terminator Layout 2x2 Grid dalam 1 Layar)
```bash
./src/task/help_me_carry/launch/run.sh

# Hentikan semua node:
./src/task/help_me_carry/launch/stop.sh
# Atau:
./src/task/help_me_carry/launch/run.sh stop
```

### Modular Stack (Recommended for Testing)
Run across 4 terminals:
```bash
# Terminal 1 — STM32 Nucleo motor base & RPLiDAR
ros2 launch help_me_carry hardware.launch.py

# Terminal 2 — ZED camera (person tracking, bag YOLO, pointing gesture)
ros2 launch help_me_carry camera.launch.py

# Terminal 3 — Online SLAM, Nav2, follower node & RViz
ros2 launch help_me_carry motion.launch.py

# Terminal 4 — HRI speech interaction & Web dashboard
ros2 launch help_me_carry hri.launch.py
```

### All-in-One
```bash
ros2 launch help_me_carry help_me_carry.launch.py
```

## ROS 2 Services

| Service | Type | Description |
|---|---|---|
| `/follow/person` | `std_srvs/srv/Trigger` | Start following the locked operator |
| `/follow/stop` | `std_srvs/srv/Trigger` | Stop following |
| `/task/align_bag` | `std_srvs/srv/Trigger` | Align robot towards the bag indicated by gesture (`LEFT`/`RIGHT`) |
| `/task/step_back` | `std_srvs/srv/Trigger` | Move backward until person distance reaches 2.0m or a bag is detected |
| `/task/rotate_safe` | `std_srvs/srv/Trigger` | In-place rotation towards the most spacious open area oriented towards origin |
| `/arm/ambil_tas` | `std_srvs/srv/Trigger` | Trigger arm trajectory to pick up bag |
| `/homing` | `std_srvs/srv/Trigger` | Autonomous homing: auto in-place rotate if facing away, wait & update map on dynamic obstacles |

### CLI Examples
```bash
ros2 service call /follow/person std_srvs/srv/Trigger {}
ros2 service call /follow/stop std_srvs/srv/Trigger {}
ros2 service call /task/step_back std_srvs/srv/Trigger {}
# Alias: ros2 service call /help_me_carry/step_back std_srvs/srv/Trigger {}
ros2 service call /task/rotate_safe std_srvs/srv/Trigger {}
# Alias: ros2 service call /help_me_carry/rotate_safe std_srvs/srv/Trigger {}
ros2 service call /task/align_bag std_srvs/srv/Trigger {}
ros2 service call /homing std_srvs/srv/Trigger {}
```

## Configuration

Parameters can be adjusted in [config/params.yaml](config/params.yaml) without rebuilding:
* `stop_distance`: Following distance behind operator (default: `1.0` m).
* `occlusion_timeout`: Dead-reckoning duration during target occlusion (default: `1.5` s).
* `gating_threshold`: Distance validation gate for target re-lock (default: `1.2` m).
* `step_back_person_distance`: Target separation to person before stopping backward motion (default: `2.0` m).
* `step_back_speed`: Backward linear speed magnitude (default: `0.15` m/s).
* `step_back_timeout`: Maximum duration before stopping backward motion (default: `15.0` s).
