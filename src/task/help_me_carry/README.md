```bash
# Terminal 1 — Hardware Base STM32 Nucleo & RPLiDAR (/dev/ttyUSB2):
ros2 launch help_me_carry hardware.launch.py

# Terminal 2 — Penglihatan Kamera ZED (Orang & Tas):
ros2 launch help_me_carry camera.launch.py

# Terminal 3 — Navigasi (Online SLAM, Nav2, Person Following, RViz):
ros2 launch help_me_carry motion.launch.py

# Terminal 4 — HRI Antarmuka Suara & Web Dashboard:
ros2 launch help_me_carry hri.launch.py


# CARA PEMANGGILAN SERVICE VIA CLI:
 1. Person Following (Mulai mengikuti operator):
   ros2 service call /follow/person std_srvs/srv/Trigger {}

 2. Person Stop (Hentikan following):
   ros2 service call /follow/stop std_srvs/srv/Trigger {}

 3. Align to Bag (Sejajarkan hadap robot ke tas sesuai gesture vision lalu stop)
   ros2 service call /task/align_bag std_srvs/srv/Trigger {}

 4. Homing (Kembali ke titik awal 0,0 Nav2):
   ros2 service call /homing std_srvs/srv/Trigger {}
