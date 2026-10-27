cmake_minimum_required(VERSION 3.8)
project(athome_arm)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(Threads REQUIRED)

find_path(DYNAMIXEL_SDK_INCLUDE_DIR NAMES dynamixel_sdk.h
  PATHS /usr/local/include/dynamixel_sdk /usr/include/dynamixel_sdk
)
find_library(DYNAMIXEL_SDK_LIB NAMES dxl_x64_cpp
  PATHS /usr/local/lib /usr/lib
)

include_directories(
  include
  include/athome_arm
  ${DYNAMIXEL_SDK_INCLUDE_DIR}
)

# 1. Arm Motion Server (ROS 2 Service Node)
add_executable(arm_motion_server src/arm_motion_server.cpp)
ament_target_dependencies(arm_motion_server
  rclcpp
  std_msgs
  std_srvs
  ament_index_cpp
)
target_link_libraries(arm_motion_server
  ${DYNAMIXEL_SDK_LIB}
  Threads::Threads
)

# 2. Learn Motion CLI (Standalone Teaching Tool)
add_executable(learn_motion src/learn_motion.cpp)
target_link_libraries(learn_motion
  ${DYNAMIXEL_SDK_LIB}
  Threads::Threads
)

# 3. Calibrate Arm (Standalone Calibration Tool)
add_executable(calibrate_arm src/calibrate_arm.cpp)
target_link_libraries(calibrate_arm
  ${DYNAMIXEL_SDK_LIB}
  Threads::Threads
)

# 4. Adaptive Gripper (Standalone Gripper Tool)
add_executable(adaptive_gripper src/adaptive_gripper.cpp)
target_link_libraries(adaptive_gripper
  ${DYNAMIXEL_SDK_LIB}
  Threads::Threads
)

# Install targets
install(TARGETS
  arm_motion_server
  learn_motion
  calibrate_arm
  adaptive_gripper
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY motions/
  DESTINATION share/${PROJECT_NAME}/motions
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()

