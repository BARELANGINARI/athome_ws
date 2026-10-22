cmake_minimum_required(VERSION 3.8)
project(athome_base)

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
find_path(ICECREAM_HPP_DIR NAMES icecream.hpp HINTS ${ICECREAM_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include NO_DEFAULT_PATH)
if(NOT ICECREAM_HPP_DIR)
  message(FATAL_ERROR "icecream.hpp not found")
endif()

include_directories(include ${ICECREAM_HPP_DIR})

add_library(kinematic src/Kinematic.cpp)
target_include_directories(kinematic PUBLIC include)

add_library(udpSender src/udpSender.cpp)
target_include_directories(udpSender PUBLIC include)

add_library(udpServer src/udpServer.cpp)
target_include_directories(udpServer PUBLIC include)
target_link_libraries(udpServer udpSender)

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

ament_package()
