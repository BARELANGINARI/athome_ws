cmake_minimum_required(VERSION 3.8)
project(msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AsrResult.msg"
  "msg/NluResult.msg"
  "msg/Say.msg"
  "msg/TtsStatus.msg"
  "msg/DialogState.msg"
  "msg/PersonDetected.msg"
)

ament_package()
