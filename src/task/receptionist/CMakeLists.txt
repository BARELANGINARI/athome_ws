cmake_minimum_required(VERSION 3.8)
project(receptionist)

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

add_executable(receptionist_node src/receptionist_node.cpp)
ament_target_dependencies(receptionist_node
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
target_link_libraries(receptionist_node Threads::Threads)
install(TARGETS receptionist_node DESTINATION lib/${PROJECT_NAME})

# Install directories
install(DIRECTORY
  launch
  config
  rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
