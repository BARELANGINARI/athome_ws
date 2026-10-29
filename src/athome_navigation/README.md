# athome_navigation

Autonomous 2D navigation package using the ROS 2 Navigation Stack (Nav2) and AMCL localization.

## Prerequisites
Ensure the robot base and sensors are active before launching navigation:
```bash
ros2 launch athome_bringup sensors.launch.py
ros2 launch athome_base nucleo.launch.py
```

## Running Navigation
```bash
ros2 launch athome_navigation navigation.launch.py
```

### Launch Arguments
* `map`: Path to the map YAML file (default: `maps/map.yaml`).
* `rviz`: Start RViz visualization (default: `true`).
* `use_sim_time`: Use simulation clock (default: `false`).

## Maps
The default map is located in `maps/`:
* `map.yaml`: Map resolution, origin, and metadata.
* `map.pgm`: Occupancy grid image.
