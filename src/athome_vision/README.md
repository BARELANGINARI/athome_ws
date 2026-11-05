# athome_vision

Computer vision package powered by Stereolabs ZED SDK and ONNX YOLO object detection for the Barelang Inari robot.

## Features
* **Human Body Tracking (BODY_34):** 3D skeleton keypoint tracking for operator following & guest tracking.
* **Person Attributes Classification:** Classifies gender, clothing color, and hair color (`/zed/person_attributes`) for Receptionist.
* **Pointing Gesture Detection:** Detects operator arm pointing to the floor (`/zed/gesture`: `RIGHT`, `LEFT`, `NONE`) for Help Me Carry.
* **YOLO Luggage Detection:** Detects luggage bags from RGB frames and projects 3D spatial coordinates using the point cloud.
* **Camera Yaw:** Publishes drift-free heading on `/zed/yaw_deg` for mobile base orientation compensation.

## Running

```bash
# Default (tampilkan semua log IC):
ros2 launch athome_vision zed.launch.py

# Mode Help Me Carry:
ros2 launch athome_vision zed.launch.py task:=help_me_carry

# Mode Receptionist:
ros2 launch athome_vision zed.launch.py task:=receptionist
```

### Published Topics
* `/zed/yaw_deg` (`std_msgs/msg/Float64`): Camera yaw angle in degrees.
* `/zed/gesture` (`std_msgs/msg/String`): Operator pointing gesture (`RIGHT`, `LEFT`, `NONE`).
* `/zed/person_label` (`std_msgs/msg/String`): Detected person 3D coordinates and skeleton keypoints.
* `/zed/person_attributes` (`std_msgs/msg/String`): Detected visual attributes (`gender, shirt color, hair color`).
* `/zed/bag_detections` (`std_msgs/msg/String`): Detected luggage bags with 3D positions and confidence scores.
