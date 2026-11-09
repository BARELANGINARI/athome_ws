cmake_minimum_required(VERSION 3.8)
project(help_me_carry)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Threads REQUIRED)

include_directories(include)

add_executable(help_me_carry_node src/help_me_carry_node.cpp)
ament_target_dependencies(help_me_carry_node
  rclcpp
  rclcpp_action
  ament_index_cpp
  nav2_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  std_srvs
  tf2
  tf2_ros
)
target_link_libraries(help_me_carry_node Threads::Threads)
install(TARGETS help_me_carry_node DESTINATION lib/${PROJECT_NAME})

add_executable(tracking_controller src/help_me_carry_node.cpp)
ament_target_dependencies(tracking_controller
  rclcpp
  rclcpp_action
  ament_index_cpp
  nav2_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  std_srvs
  tf2
  tf2_ros
)
target_link_libraries(tracking_controller Threads::Threads)
install(TARGETS tracking_controller DESTINATION lib/${PROJECT_NAME})

add_executable(return_to_origin src/return_to_origin.cpp)
add_executable(help_me_carry_manager src/help_me_carry_manager.cpp)
ament_target_dependencies(return_to_origin
  rclcpp
  rclcpp_action
  geometry_msgs
  std_srvs
  nav2_msgs
)
install(TARGETS return_to_origin DESTINATION lib/${PROJECT_NAME})
ament_target_dependencies(help_me_carry_manager
  rclcpp
  std_msgs
  std_srvs
)
install(TARGETS help_me_carry_manager DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  launch/run.sh
  launch/stop.sh
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY behavior_trees/ DESTINATION share/${PROJECT_NAME}/behavior_trees)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY rviz/ DESTINATION share/${PROJECT_NAME}/rviz)

ament_package()
