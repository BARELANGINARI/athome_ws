# athome_base

Low-level base controller for Barelang Inari's 3-wheel omnidirectional mobile platform. Communicates with the STM32 Nucleo board via UDP.

## Features
* **Omnidirectional Kinematics:** Computes individual motor PWM values for 3 omni wheels from `cmd_vel`.
* **Odometry & TF:** Publishes `odom -> base_footprint` transforms and `/odom` topic.
* **External Yaw Fusion:** Supports fusing camera yaw (`/zed/yaw_deg`) for drift-free heading.
* **Velocity Slew Rate Smoothing:** Limits linear and angular accelerations to prevent wheel slip.

## Running

```bash
ros2 launch athome_base nucleo.launch.py
```

### Key Parameters
* `stm_ip`: STM32 Nucleo IP address (default: `192.168.0.4`).
* `use_yaw_topic`: Enable external `/zed/yaw_deg` fusion (default: `true`).
* `smoothing_enable`: Enable acceleration smoothing (default: `true`).
