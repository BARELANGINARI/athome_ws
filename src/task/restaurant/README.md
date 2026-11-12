## cara menjalankan program

```bash
ros2 launch restaurant hardware.launch.py
ros2 launch restaurant motion.launch.py
```

### Service Homing (Kembali ke Titik Awal Start)
```bash
ros2 service call /homing std_srvs/srv/Trigger {}
```
