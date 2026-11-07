``` bash 

### Modular
ros2 launch receptionist hardware.launch.py
ros2 launch receptionist navigation.launch.py
ros2 launch receptionist hri.launch.py
ros2 launch receptionist camera.launch.py
ros2 launch athome_face face.launch.py

### All-in-One:
ros2 launch receptionist receptionist.launch.py hri:=true

### Service
ros2 service call /receptionist/go_to_door std_srvs/srv/Trigger {}
ros2 service call /receptionist/go_beverage_area std_srvs/srv/Trigger {}
ros2 service call /receptionist/go_to_living_room std_srvs/srv/Trigger {}
ros2 service call /search_chair std_srvs/srv/Trigger {}

