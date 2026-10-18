cmake_minimum_required(VERSION 3.8)
project(luggage_task)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED)

# Task 1 Carry My Luggage: full task (task.hpp + task/task_logic.cpp +
# task/task_node.cpp). task_logic = parse + taktik + state machine flowchart
# (tanpa ROS), task_node = node ROS (baca topik mentah zed -> tulis
# cmd_vel/Nav2).
add_library(luggage_task_lib
  src/task/task_logic.cpp)
target_include_directories(luggage_task_lib PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(luggage_task_lib ${OpenCV_LIBS})
ament_target_dependencies(luggage_task_lib
  rclcpp rclcpp_action rcl_interfaces ament_index_cpp std_msgs std_srvs geometry_msgs
  sensor_msgs nav2_msgs tf2 tf2_ros)

add_executable(luggage_task_node src/task/task_node.cpp)
target_link_libraries(luggage_task_node luggage_task_lib ${OpenCV_LIBS})
target_include_directories(luggage_task_node PRIVATE ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(luggage_task_node
  rclcpp rclcpp_action rcl_interfaces ament_index_cpp std_msgs std_srvs geometry_msgs
  sensor_msgs nav2_msgs tf2 tf2_ros)

install(TARGETS luggage_task_lib luggage_task_node
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY rviz/ DESTINATION share/${PROJECT_NAME}/rviz)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_dependencies(
  rclcpp rclcpp_action rcl_interfaces ament_index_cpp std_msgs geometry_msgs
  sensor_msgs nav_msgs nav2_msgs tf2 tf2_ros)

ament_package()
