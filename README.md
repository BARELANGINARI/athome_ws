## Workspace Structure

```text
src/
├── athome_base/          # Low-level base controller (3-omniwheel drive via STM32 Nucleo)
│   ├── include/          # Kinematic.hpp, nucleo_node.hpp, udpSender.h, udpServer.h, icecream.hpp
│   ├── launch/
│   │   └── nucleo.launch.py   # Nucleo UDP bridge & base odometry node
│   ├── src/              # Kinematic.cpp, udpSender.cpp, udpServer.cpp
│   └── main.cpp          # Nucleo base driver node entrypoint
│
├── athome_bringup/       # Shared sensors and teleoperation bringup
│   ├── config/
│   │   └── lidar.yaml    # RPLiDAR driver configuration
│   ├── launch/
│   │   ├── sensors.launch.py   # RPLiDAR + ZED Camera + static transforms
│   │   └── joystick.launch.py  # Joystick / gamepad teleoperation
│   └── scripts/
│       ├── bag_kinematics      # Bag handling kinematics utility
│       └── ds4_drive           # DualShock 4 controller driver
│
├── athome_vision/        # ZED camera, YOLO bag detection, and body tracking
│   ├── include/          # icecream.hpp
│   ├── launch/
│   │   └── zed.launch.py # ZED camera driver with YOLO & body tracking
│   ├── models/
│   │   ├── coco_names.txt      # COCO class labels
│   │   └── yolov8n_sim.onnx    # ONNX model for bag detection
│   └── src/
│       └── main.cpp      # Vision node (/zed/yaw_deg, /zed/bag_detections, /zed/person_keypoints)
│
├── athome_mapping/       # 2D SLAM mapping using SLAM Toolbox
│   ├── config/
│   │   └── slam.yaml     # SLAM Toolbox online async mapping parameters
│   ├── launch/
│   │   └── mapping.launch.py   # SLAM Toolbox mapping with RViz
│   └── rviz/
│       └── mapping.rviz  # Mapping RViz configuration
│
├── athome_navigation/    # Nav2 point-to-point autonomous navigation
│   ├── launch/
│   │   └── navigation.launch.py # Nav2 bringup with map server and RViz
│   ├── maps/
│   │   ├── map.yaml      # Saved occupancy grid map metadata
│   │   └── map.pgm       # Saved occupancy grid map image
│   └── rviz/
│       └── navigation.rviz # Navigation RViz configuration
│
├── athome_hri/           # Human-Robot Interaction engines (STT, TTS, Web Dashboard)
│   ├── athome_hri/       # Python modules (intent_recognizer, stt_engine, tts_engine, web_server)
│   ├── launch/
│   │   └── hri.launch.py # HRI speech and web dashboard launch
│   ├── model/            # Piper TTS ONNX model and voice configs
│   └── src/              # Web dashboard frontend and certificates
│
└── task/
    └── help_me_carry/    # RoboCup@Home Help Me Carry competition task
        ├── behavior_trees/
        │   └── follow_person.xml    # Nav2 behavior tree for person following
        ├── config/
        │   ├── nav2_params.yaml     # Nav2 stack parameters for task
        │   └── params.yaml          # Task parameters (person tracking, bag alignment)
        ├── include/
        │   └── icecream.hpp         # IceCream debugging header
        ├── launch/
        │   ├── help_me_carry.launch.py # Master monolithic launch
        │   ├── hardware.launch.py      # Base driver & LiDAR bringup
        │   ├── camera.launch.py        # ZED camera & bag detection
        │   ├── motion.launch.py        # Nav2 navigation stack & SLAM
        │   ├── hri.launch.py           # HRI voice & web server
        │   └── task.launch.py          # Help Me Carry task controller
        ├── rviz/
        │   └── help_me_carry.rviz   # Task RViz configuration
        └── src/
            ├── barelang_hri.py         # HRI voice & task manager node
            ├── help_me_carry_node.cpp  # Task manager (tracking, following, bag alignment)
            └── return_to_origin.cpp    # Autonomous return to origin / homing node
```