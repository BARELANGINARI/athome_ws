### Multi-Pane (Terminator Layout 5-Pane Grid dalam 1 Layar)
```bash
./src/task/help_me_carry/launch/run.sh
./src/task/help_me_carry/launch/stop.sh
```

### Modular Stack (Recommended for Testing)
```bash
ros2 launch help_me_carry hardware.launch.py
ros2 launch help_me_carry camera.launch.py
ros2 launch help_me_carry motion.launch.py
ros2 launch help_me_carry hri.launch.py
ros2 launch athome_arm arm.launch.py
```

### All-in-One
```bash
ros2 launch help_me_carry help_me_carry.launch.py
```

## ROS 2 Services
```bash
ros2 service call /follow/person std_srvs/srv/Trigger {}
ros2 service call /follow/stop std_srvs/srv/Trigger {}
ros2 service call /task/step_back std_srvs/srv/Trigger {}
ros2 service call /task/rotate_safe std_srvs/srv/Trigger {}
ros2 service call /task/align_bag std_srvs/srv/Trigger {}
ros2 service call /homing std_srvs/srv/Trigger {}
ros2 service call /lifter/naik std_srvs/srv/Trigger {}
ros2 service call /lifter/turun std_srvs/srv/Trigger {}
```
