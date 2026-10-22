## Folder structure
```
src/                # ROS packages only (colcon)
├── athome_base/       # robot movement 3 motor (ala NucleoServer: main.cpp + src/Kinematic)
│   ├── main.cpp                  # program utama (node + loop, MKiri/MKanan/MBelakang)
│   ├── launch/nucleo.launch.py   # use_yaw_topic:=true yaw_topic:=/zed/yaw_deg
│   ├── src/Kinematic.cpp, udpSender.cpp, udpServer.cpp  # lib murni
│   └── include/*.h(pp) flat      # Kinematic.hpp, nucleo_node.hpp, udpSender.h, udpServer.h
├── athome_vision/     # kamera: yaw + body tracking + YOLO tas (ala zed_ws: src/main.cpp)
│   ├── src/main.cpp              # SATU file: struct + vision murni + YOLO tas + node + main
│   ├── include/deploy/...        # header TensorRT opsional (OFF default)
│   ├── launch/zed.launch.py      # enable_body_tracking:=true
│   └── models/yolov8n_sim.onnx + coco_names.txt  # YOLO tas (bag), tinggal pake
│   # publish: /zed/yaw_deg, /zed/bag_detections, /zed/person_keypoints
├── athome_bringup/    # shared sensors (do not run manually)
│   ├── launch/sensors.launch.py  # auto-included by mapping/navigation
│   │   ├── sensor 1: RPLiDAR (rplidar_node -> /scan, full 360°, ~14 Hz)
│   │   ├── static TF: base_link -> laser_frame, base_footprint -> base_link
│   │   └── sensor 2: ZED camera (yaw + body tracking + YOLO tas)
│   └── config/lidar.yaml  # lidar driver params (filter 180° dilepas)
├── athome_mapping/    # room mapping (requires teleop)
│   ├── launch/mapping.launch.py  # includes sensors + slam_toolbox (toolbox only)
│   ├── config/slam.yaml          # travel thresholds 0.15, update 2 s
│   └── rviz/mapping.rviz
├── athome_navigation/ # point-to-point using the map
│   ├── launch/navigation.launch.py  # include sensors + nav2_bringup (+ rviz args)
│   ├── config/nav2_params.yaml
│   ├── maps/  # SINGLE source of truth: map.yaml + map.pgm (via map_saver_cli)
│   ├── maps_backup/  # arsip peta lama (tidak ikut ter-install)
│   └── rviz/navigation.rviz
├── athome_hri/        # Antarmuka suara (Faster-Whisper STT, Piper TTS) & Web Dashboard
└── task/
    └── help_me_carry/ # Task Help Me Carry (Person Following & Return to Origin)
        ├── config/params.yaml             # Parameter terpadu person_tracker & person_follower
        ├── launch/                        # Split-stack: hardware, camera, motion, hri
        ├── rviz/help_me_carry.rviz        # Visualisasi map SLAM, Nav2 costmaps & tracking
        └── src/
            ├── person_tracker.cpp         # Vision target tracker (keypoints -> /follow/target_base)
            ├── person_follower.cpp        # Nav2 motion controller (ideal mode, watchdog, TF guardian)
            └── return_to_origin.cpp       # CLI C++ otonom balik ke titik awal (0,0)
```
