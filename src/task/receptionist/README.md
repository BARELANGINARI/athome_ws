```bash
# 1. Jalankan Hardware (Motor Nucleo + RPLiDAR)
ros2 launch receptionist hardware.launch.py

# 2. Jalankan Navigasi & Service Receptionist (Nav2 + AMCL + Node Service)
ros2 launch receptionist navigation.launch.py

# opsional. All-in-One
ros2 launch receptionist receptionist.launch.py

# --- Panggilan Service Navigasi ---
# Navigasi ke Pintu (Door)
ros2 service call /receptionist/go_to_door std_srvs/srv/Trigger

# Navigasi ke Ruang Tamu (Living Room)
ros2 service call /receptionist/go_to_living_room std_srvs/srv/Trigger

# --- Panggilan Service Lifter Base ---
# Lifter Naik
ros2 service call /lifter/naik std_srvs/srv/Trigger {}

# Lifter Turun
ros2 service call /lifter/turun std_srvs/srv/Trigger {}
```
