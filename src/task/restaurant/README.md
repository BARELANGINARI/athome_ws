## cara menjalankan program

```bash
ros2 launch restaurant hardware.launch.py
ros2 launch restaurant motion.launch.py
ros2 launch restaurant camera.launch.py
```

## Service
```bash
ros2 service call /search std_srvs/srv/Trigger {}
ros2 service call /go_to_order std_srvs/srv/Trigger {}
ros2 service call /homing std_srvs/srv/Trigger {}
```