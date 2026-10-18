cmake_minimum_required(VERSION 3.8)
project(spawn)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
# Script set_pose: bisa dipanggil via ros2 run maupun Node di spawn.py.
# RENAME melepas ekstensi .py supaya nama executable = 'set_pose' (bukan 'set_pose.py').
install(PROGRAMS scripts/set_pose.py
  DESTINATION lib/${PROJECT_NAME} RENAME set_pose)
# Uji poses.yaml: check_poses (tabel, tanpa ROS) + test_poses (publish semua preset).
install(PROGRAMS scripts/check_poses.py
  DESTINATION lib/${PROJECT_NAME} RENAME check_poses)
install(PROGRAMS scripts/test_poses.py
  DESTINATION lib/${PROJECT_NAME} RENAME test_poses)
# pose_check.py = modul bersama (di-import check_poses/test_poses, 1 folder).
install(FILES scripts/pose_check.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()
