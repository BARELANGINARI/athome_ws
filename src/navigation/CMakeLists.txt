cmake_minimum_required(VERSION 3.8)
project(navigation)
find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/goal_pose_bridge.py
  scripts/place_navigator.py
  scripts/person_follower.py
  DESTINATION lib/${PROJECT_NAME}
)
# Validator koordinat: cek places.yaml terhadap peta (tanpa ROS).
install(PROGRAMS scripts/check_places.py
  DESTINATION lib/${PROJECT_NAME} RENAME check_places)
# Initial pose dari terminal (alternatif tool 2D Pose Estimate RViz).
install(PROGRAMS scripts/set_initial_pose.py
  DESTINATION lib/${PROJECT_NAME} RENAME set_initial_pose)
# Diagnosa cepat rantai navigasi (TF / scan / odom / AMCL / costmap).
install(PROGRAMS scripts/check_system.sh
  DESTINATION lib/${PROJECT_NAME} RENAME check_system)

install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY maps/  DESTINATION share/${PROJECT_NAME}/maps)
install(DIRECTORY rviz/  DESTINATION share/${PROJECT_NAME}/rviz)
# Behaviour tree person following (dipilih per-goal lewat field behavior_tree).
install(DIRECTORY behavior_trees/ DESTINATION share/${PROJECT_NAME}/behavior_trees)
install(FILES README.md DESTINATION share/${PROJECT_NAME})
ament_package()
