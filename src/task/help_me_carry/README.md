# help_me_carry

RoboCup@Home **Help Me Carry** task package for Barelang Inari robot.

Features Kalman Filter person tracking with occlusion dead-reckoning, spatial gating, pointing gesture alignment, robotic arm pickup, and autonomous homing.

## Launch Options

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
| `/arm/ambil_tas` | `std_srvs/srv/Trigger` | Trigger arm trajectory to pick up bag |
| `/homing` | `std_srvs/srv/Trigger` | Return robot autonomously to starting pose `(0, 0)` |

### CLI Examples
```bash
ros2 service call /follow/person std_srvs/srv/Trigger {}
ros2 service call /follow/stop std_srvs/srv/Trigger {}
ros2 service call /task/align_bag std_srvs/srv/Trigger {}
ros2 service call /homing std_srvs/srv/Trigger {}
```

## Configuration

Parameters can be adjusted in [config/params.yaml](config/params.yaml) without rebuilding:
* `stop_distance`: Following distance behind operator (default: `1.0` m).
* `occlusion_timeout`: Dead-reckoning duration during target occlusion (default: `1.5` s).
* `gating_threshold`: Distance validation gate for target re-lock (default: `1.2` m).
