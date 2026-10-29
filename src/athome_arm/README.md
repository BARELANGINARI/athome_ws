# athome_arm

ROS 2 package for controlling the Dynamixel AX-18A robotic arm and adaptive gripper on **Barelang Inari** (RoboCup@Home).

## Prerequisites

* **Dynamixel SDK (C++):**
  ```bash
  git clone https://github.com/ROBOTIS-GIT/DynamixelSDK.git
  cd DynamixelSDK/c++/build/linux64
  make && sudo make install
  ```
* **Serial Port Permissions:**
  ```bash
  sudo usermod -aG dialout $USER
  ```

## Build

```bash
cd ~/athome_ws
colcon build --packages-select athome_arm
source install/setup.bash
```

## Running

### 1. ROS 2 Motion Server Node
Runs the playback service for taught arm motions:
```bash
ros2 launch athome_arm arm.launch.py port:=/dev/ttyUSB1 baudrate:=1000000
```

**Trigger bag pickup motion:**
```bash
ros2 service call /arm/ambil_tas std_srvs/srv/Trigger {}
```

### 2. Standalone CLI Tools

* **Teach & Record Motion (`learn_motion`):**
  ```bash
  ros2 run athome_arm learn_motion
  ```
  *Interactive commands:* `record`, `add`, `play <name>`, `save <name>`, `list`.

* **Calibrate Servos (`calibrate_arm`):**
  ```bash
  ros2 run athome_arm calibrate_arm
  ```

* **Test Adaptive Gripper (`adaptive_gripper`):**
  ```bash
  ros2 run athome_arm adaptive_gripper
  ```

## Directory Structure

```text
src/athome_arm/
├── CMakeLists.txt
├── package.xml
├── README.md
├── include/
│   └── athome_arm/
│       └── calibration_data.h    # Servo calibration values
├── src/
│   ├── arm_motion_server.cpp     # ROS 2 service node for motion playback
│   ├── learn_motion.cpp          # Offline CLI for motion teaching
│   ├── calibrate_arm.cpp         # CLI for servo zero & limit calibration
│   └── adaptive_gripper.cpp      # CLI for adaptive torque grasping
├── motions/
│   └── ambil_tas                 # Recorded motion trajectory
└── launch/
    └── arm.launch.py             # Launch file
```
