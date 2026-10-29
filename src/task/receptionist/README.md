# receptionist

RoboCup@Home **Receptionist** task package for Barelang Inari robot.

Handles autonomous waypoint navigation using Nav2 (AMCL) to greet guests at the entrance door and guide them to the living room.

## Launch Options

### Modular Stack
```bash
# Terminal 1 — Hardware base & RPLiDAR
ros2 launch receptionist hardware.launch.py

# Terminal 2 — Nav2 map navigation & receptionist node
ros2 launch receptionist navigation.launch.py
```

### All-in-One
```bash
ros2 launch receptionist receptionist.launch.py
```

## Navigation Services

Trigger waypoints via CLI:
```bash
# Navigate to entrance door
ros2 service call /receptionist/go_to_door std_srvs/srv/Trigger {}

# Navigate to living room
ros2 service call /receptionist/go_to_living_room std_srvs/srv/Trigger {}
```

## Configuration

Waypoint coordinates (*x, y, yaw*) are configured in [config/receptionist_params.yaml](config/receptionist_params.yaml):
* `door`: Position of entrance door.
* `living_room`: Position of living room destination.
