cmake_minimum_required(VERSION 3.12)

project(athome_vision LANGUAGES CXX)

if(EXISTS "/usr/local/cuda/bin/nvcc")
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda" CACHE PATH "CUDA toolkit root (pinned)" FORCE)
  set(CUDAToolkit_ROOT "/usr/local/cuda" CACHE PATH "CUDAToolkit root (pinned)" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(OpenCV REQUIRED)

find_package(ZED 3 REQUIRED)
option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)
if(NOT LINK_SHARED_ZED AND MSVC)
  message(FATAL_ERROR "LINK_SHARED_ZED OFF")
endif()

find_package(CUDAToolkit REQUIRED)

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
  message(FATAL_ERROR
    "ONNX Runtime not found")
endif()
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIB}")

add_executable(zed src/main.cpp)
target_include_directories(zed PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${ZED_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE}
)
ament_target_dependencies(zed
  rclcpp std_msgs sensor_msgs ament_index_cpp)
target_link_libraries(zed
  ${CUDA_cudart_LIBRARY}
  ${OpenCV_LIBS}
  ${ZED_LIBRARIES}
  ${ONNXRUNTIME_LIB}
)
get_filename_component(ONNXRUNTIME_LIBDIR "${ONNXRUNTIME_LIB}" DIRECTORY)
set_target_properties(zed PROPERTIES
  INSTALL_RPATH "${ONNXRUNTIME_LIBDIR}"
  BUILD_WITH_INSTALL_RPATH TRUE)
install(TARGETS zed
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY models/ DESTINATION share/${PROJECT_NAME}/models)

ament_package()
