### Modular 
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

### CLI Examples
```bash
# Person Following Services
ros2 service call /follow/person std_srvs/srv/Trigger {}
ros2 service call /follow/stop std_srvs/srv/Trigger {}
ros2 action send_goal /follow/person_action help_me_carry/action/FollowPerson "{}"

# Bag Alignment & Homing
ros2 service call /task/align_bag std_srvs/srv/Trigger {}
ros2 service call /homing std_srvs/srv/Trigger {}

# Lifter Services
ros2 service call /lifter/naik std_srvs/srv/Trigger {}
ros2 service call /lifter/turun std_srvs/srv/Trigger {}
```