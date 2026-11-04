cmake_minimum_required(VERSION 3.12)
project(person_det LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(OpenCV REQUIRED)

# ZED SDK & CUDA Discovery
if(EXISTS "/usr/local/cuda/bin/nvcc")
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda" CACHE PATH "CUDA toolkit root" FORCE)
  set(CUDAToolkit_ROOT "/usr/local/cuda" CACHE PATH "CUDAToolkit root" FORCE)
endif()
find_package(ZED 3 QUIET)
find_package(CUDAToolkit QUIET)
if(ZED_FOUND AND CUDAToolkit_FOUND)
  add_definitions(-DHAVE_ZED=1)
  message(STATUS "ZED SDK found! Enabling native ZED camera & IMU support in person_det.")
endif()

# ONNX Runtime Discovery
if(DEFINED ENV{ONNXRUNTIME_ROOT} AND NOT ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()
if(NOT ONNXRUNTIME_ROOT)
  foreach(_cand "/opt/onnxruntime" "$ENV{HOME}/onnxruntime")
    if(EXISTS "${_cand}/include/onnxruntime_cxx_api.h")
      set(ONNXRUNTIME_ROOT "${_cand}")
      break()
    endif()
  endforeach()
endif()
find_library(ONNXRUNTIME_LIB onnxruntime
  PATHS "${ONNXRUNTIME_ROOT}/lib" /opt/onnxruntime/lib "$ENV{HOME}/onnxruntime/lib"
  NO_DEFAULT_PATH)
find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h
  PATHS "${ONNXRUNTIME_ROOT}/include" /opt/onnxruntime/include "$ENV{HOME}/onnxruntime/include"
  NO_DEFAULT_PATH)
if(NOT ONNXRUNTIME_LIB OR NOT ONNXRUNTIME_INCLUDE)
  message(FATAL_ERROR "ONNX Runtime not found in /opt/onnxruntime or $ENV{HOME}/onnxruntime")
endif()
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE}")
message(STATUS "ONNX Runtime lib: ${ONNXRUNTIME_LIB}")

# 1. Core Person Detection Engine Library
add_library(person_det_lib
  src/person_detection.cpp
)
target_include_directories(person_det_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE}
)
target_link_libraries(person_det_lib
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_LIB}
)
get_filename_component(ONNXRUNTIME_LIBDIR "${ONNXRUNTIME_LIB}" DIRECTORY)
set_target_properties(person_det_lib PROPERTIES
  INSTALL_RPATH "${ONNXRUNTIME_LIBDIR}"
  BUILD_WITH_INSTALL_RPATH TRUE
)

# 2. ROS 2 Person Detection Node
add_executable(person_det_node
  src/person_det_node.cpp
)
target_link_libraries(person_det_node
  person_det_lib
)
if(ZED_FOUND AND CUDAToolkit_FOUND)
  target_include_directories(person_det_node PRIVATE
    ${ZED_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  target_link_libraries(person_det_node
    ${ZED_LIBRARIES}
    CUDA::cudart
  )
endif()
ament_target_dependencies(person_det_node
  rclcpp
  std_msgs
  sensor_msgs
  msgs
  ament_index_cpp
)
set_target_properties(person_det_node PROPERTIES
  INSTALL_RPATH "${ONNXRUNTIME_LIBDIR}"
  BUILD_WITH_INSTALL_RPATH TRUE
)

# 3. Standalone CLI Tool
add_executable(person_detection
  src/person_detection_cli.cpp
)
target_link_libraries(person_detection
  person_det_lib
)
set_target_properties(person_detection PROPERTIES
  INSTALL_RPATH "${ONNXRUNTIME_LIBDIR}"
  BUILD_WITH_INSTALL_RPATH TRUE
)

# Install
install(TARGETS person_det_lib person_det_node person_detection
  EXPORT export_person_det
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
)

ament_export_include_directories(include)
ament_export_libraries(person_det_lib)
ament_export_targets(export_person_det)

ament_package()
