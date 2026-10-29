# athome_bringup

Sensor bringup, static transforms, and teleoperation launch files for the Barelang Inari robot.

## Launch Files

### 1. Sensors (`sensors.launch.py`)
Brings up the RPLiDAR driver and broadcasts static coordinate frames:
```bash
ros2 launch athome_bringup sensors.launch.py
```
* **Nodes Launched:**
  * `rplidar_node`: Reads LiDAR serial data on `/dev/ttyUSB0` (115200 baud) and publishes `/scan`.
  * `static_transform_publisher`: `base_link -> laser_frame` and `base_footprint -> base_link`.

### 2. Teleoperation (`joystick.launch.py`)
Brings up joystick teleoperation using DualShock 4 / gamepad:
```bash
ros2 launch athome_bringup joystick.launch.py
```
* **Topic Output:** Publishes velocity commands to `/joy_vel` or `/cmd_vel`.
