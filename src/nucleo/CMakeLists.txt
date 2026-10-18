cmake_minimum_required(VERSION 3.8)
project(nucleo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

if(NOT DEFINED ICECREAM_INCLUDE_DIR AND DEFINED ENV{HOME})
  set(ICECREAM_INCLUDE_DIR "$ENV{HOME}/Documents/icecream-cpp")
endif()
find_path(ICECREAM_HPP_DIR NAMES icecream.hpp HINTS ${ICECREAM_INCLUDE_DIR} NO_DEFAULT_PATH)
if(NOT ICECREAM_HPP_DIR)
  message(FATAL_ERROR "icecream.hpp tidak ketemu di ${ICECREAM_INCLUDE_DIR}. "
    "Clone https://github.com/renatoGarcia/icecream-cpp ke ~/Documents/icecream-cpp "
    "atau build dengan -DICECREAM_INCLUDE_DIR=<path>")
endif()

include_directories(include ${ICECREAM_HPP_DIR})

# Lib murni ala NucleoServer/src (tanpa ROS): kinematik 3 motor + UDP.
add_library(kinematic src/Kinematic.cpp)
target_include_directories(kinematic PUBLIC include)

add_library(udpSender src/udpSender.cpp)
target_include_directories(udpSender PUBLIC include)

add_library(udpServer src/udpServer.cpp)
target_include_directories(udpServer PUBLIC include)
target_link_libraries(udpServer udpSender)

# Program utama ala NucleoServer/main.cpp: node + loop di main.cpp.
add_executable(nucleo main.cpp)
target_include_directories(nucleo PRIVATE include)
target_link_libraries(nucleo kinematic udpServer udpSender)
ament_target_dependencies(nucleo rclcpp geometry_msgs nav_msgs std_msgs tf2_ros)

install(TARGETS nucleo kinematic udpServer udpSender
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY scripts/ DESTINATION share/${PROJECT_NAME}/scripts
        USE_SOURCE_PERMISSIONS)
install(PROGRAMS scripts/odom_calibrate.py
        DESTINATION lib/${PROJECT_NAME}
        RENAME odom_calibrate)

ament_package()
