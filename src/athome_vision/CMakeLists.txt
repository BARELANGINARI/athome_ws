cmake_minimum_required(VERSION 3.12)

project(athome_vision LANGUAGES CXX)

# Suppress dev warnings from external OpenCV5/ZED CMake configs using FindCUDA and exec_program
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()
if(POLICY CMP0153)
  cmake_policy(SET CMP0153 OLD)
endif()

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
find_package(std_srvs REQUIRED)
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

add_executable(vision src/main.cpp)
target_include_directories(vision PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${ZED_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE}
)
ament_target_dependencies(vision
  rclcpp std_msgs std_srvs ament_index_cpp)
target_link_libraries(vision
  ${CUDA_cudart_LIBRARY}
  ${OpenCV_LIBS}
  ${ZED_LIBRARIES}
  ${ONNXRUNTIME_LIB}
)
get_filename_component(ONNXRUNTIME_LIBDIR "${ONNXRUNTIME_LIB}" DIRECTORY)
set_target_properties(vision PROPERTIES
  INSTALL_RPATH "${ONNXRUNTIME_LIBDIR}"
  BUILD_WITH_INSTALL_RPATH TRUE)

add_executable(record src/record.cpp)
target_include_directories(record PRIVATE
  ${ZED_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(record
  ${OpenCV_LIBS}
  ${ZED_LIBRARIES}
)

install(TARGETS vision record
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
set(ARCFACE_MODEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/models/receptionist/arcface_w600k_r50.onnx")
if(NOT EXISTS "${ARCFACE_MODEL_PATH}")
  message(STATUS "ArcFace model not found at ${ARCFACE_MODEL_PATH}, running install_arcface_model.sh...")
  execute_process(
    COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/scripts/install_arcface_model.sh"
    RESULT_VARIABLE _arcface_res
  )
  if(NOT _arcface_res EQUAL 0)
    message(WARNING "Failed to auto-install ArcFace model. Run src/athome_vision/scripts/install_arcface_model.sh manually.")
  endif()
endif()

install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY models/ DESTINATION share/${PROJECT_NAME}/models)

ament_package()
