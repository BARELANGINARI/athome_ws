cmake_minimum_required(VERSION 3.16)
project(athome_hri_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(Threads REQUIRED)

# ------------------------------------------------------------------------------
# 1. PortAudio (Required for local microphone capture)
# ------------------------------------------------------------------------------
find_path(PORTAUDIO_INCLUDE_DIR portaudio.h
  PATHS /usr/include /usr/local/include)
find_library(PORTAUDIO_LIBRARY portaudio
  PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
if(NOT PORTAUDIO_INCLUDE_DIR OR NOT PORTAUDIO_LIBRARY)
  message(FATAL_ERROR "PortAudio not found! Install via: sudo apt install portaudio19-dev")
endif()
message(STATUS "[athome_hri_cpp] PortAudio: ${PORTAUDIO_LIBRARY}")

# ------------------------------------------------------------------------------
# 2. whisper.cpp (Optional at configure time; run scripts/setup_deps.sh for CUDA)
# ------------------------------------------------------------------------------
set(WHISPER_CPP_ROOT "" CACHE PATH "Path to whisper.cpp install or build root")
if(DEFINED ENV{WHISPER_CPP_ROOT} AND NOT WHISPER_CPP_ROOT)
  set(WHISPER_CPP_ROOT "$ENV{WHISPER_CPP_ROOT}")
endif()
set(_WHISPER_SEARCH_ROOTS
  "${WHISPER_CPP_ROOT}"
  "/opt/whisper.cpp"
  "$ENV{HOME}/whisper.cpp"
  "/usr/local"
)
find_path(WHISPER_INCLUDE_DIR whisper.h
  PATHS ${_WHISPER_SEARCH_ROOTS}
  PATH_SUFFIXES include include/whisper)
find_library(WHISPER_LIBRARY whisper
  PATHS ${_WHISPER_SEARCH_ROOTS}
  PATH_SUFFIXES lib lib64 build/src build)

set(HAVE_WHISPER_CPP FALSE)
if(WHISPER_INCLUDE_DIR AND WHISPER_LIBRARY)
  set(HAVE_WHISPER_CPP TRUE)
  get_filename_component(WHISPER_LIBDIR "${WHISPER_LIBRARY}" DIRECTORY)
  set(_WHISPER_COMPANION_LIBS "")
  foreach(_comp ggml ggml-base ggml-cpu ggml-cuda)
    find_library(_FOUND_${_comp} ${_comp}
      PATHS "${WHISPER_LIBDIR}" "${WHISPER_LIBDIR}/../ggml/src" /usr/local/lib
      NO_DEFAULT_PATH)
    if(_FOUND_${_comp})
      list(APPEND _WHISPER_COMPANION_LIBS "${_FOUND_${_comp}}")
    endif()
  endforeach()
  message(STATUS "[athome_hri_cpp] whisper.cpp FOUND: ${WHISPER_LIBRARY}")
else()
  message(WARNING
    "[athome_hri_cpp] whisper.cpp NOT found (WHISPER_CPP_ROOT). "
    "Building without whisper.cpp backend. Run scripts/setup_deps.sh to install whisper.cpp CUDA.")
endif()

# ------------------------------------------------------------------------------
# 3. ONNX Runtime C++ (Optional: enables Silero Neural VAD on CPU)
# ------------------------------------------------------------------------------
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime root")
if(DEFINED ENV{ONNXRUNTIME_ROOT} AND NOT ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  PATHS "${ONNXRUNTIME_ROOT}" /opt/onnxruntime "$ENV{HOME}/onnxruntime" /usr/local
  PATH_SUFFIXES include include/onnxruntime)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
  PATHS "${ONNXRUNTIME_ROOT}" /opt/onnxruntime "$ENV{HOME}/onnxruntime" /usr/local
  PATH_SUFFIXES lib lib64)

set(HAVE_ONNXRUNTIME FALSE)
if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
  set(HAVE_ONNXRUNTIME TRUE)
  get_filename_component(ONNXRUNTIME_LIBDIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
  message(STATUS "[athome_hri_cpp] ONNX Runtime FOUND (Silero Neural VAD enabled): ${ONNXRUNTIME_LIBRARY}")
else()
  message(STATUS "[athome_hri_cpp] ONNX Runtime not found; Silero Neural VAD disabled (using Adaptive DSP VAD).")
endif()

# ------------------------------------------------------------------------------
# 4. Optional Native Denoisers: DeepFilterNet2 (libdf) & RNNoise (librnnoise)
# ------------------------------------------------------------------------------
find_path(DEEPFILTER_INCLUDE_DIR deep_filter.h
  PATHS /opt/deepfilternet "$ENV{HOME}/DeepFilterNet" /usr/local /usr
  PATH_SUFFIXES include libDF/include)
find_library(DEEPFILTER_LIBRARY df
  PATHS /opt/deepfilternet "$ENV{HOME}/DeepFilterNet" /usr/local /usr
  PATH_SUFFIXES lib lib64 target/release)

set(HAVE_DEEPFILTERNET FALSE)
if(DEEPFILTER_INCLUDE_DIR AND DEEPFILTER_LIBRARY)
  set(HAVE_DEEPFILTERNET TRUE)
  get_filename_component(DEEPFILTER_LIBDIR "${DEEPFILTER_LIBRARY}" DIRECTORY)
  message(STATUS "[athome_hri_cpp] DeepFilterNet2 C API (libdf) FOUND: ${DEEPFILTER_LIBRARY}")
endif()

find_path(RNNOISE_INCLUDE_DIR rnnoise.h
  PATHS /usr/include /usr/local/include /opt/rnnoise/include)
find_library(RNNOISE_LIBRARY rnnoise
  PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /opt/rnnoise/lib)

set(HAVE_RNNOISE FALSE)
if(RNNOISE_INCLUDE_DIR AND RNNOISE_LIBRARY)
  set(HAVE_RNNOISE TRUE)
  message(STATUS "[athome_hri_cpp] RNNoise (librnnoise) FOUND: ${RNNOISE_LIBRARY}")
endif()

# ------------------------------------------------------------------------------
# Target: stt_engine_node
# ------------------------------------------------------------------------------
add_executable(stt_engine_node
  src/audio_enhancer.cpp
  src/audio_capture.cpp
  src/neural_vad.cpp
  src/whisper_asr.cpp
  src/stt_engine_node.cpp
)

target_include_directories(stt_engine_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PORTAUDIO_INCLUDE_DIR}
)

ament_target_dependencies(stt_engine_node
  rclcpp
  std_msgs
  ament_index_cpp
)

target_link_libraries(stt_engine_node
  ${PORTAUDIO_LIBRARY}
  Threads::Threads
)

set(_EXTRA_RPATHS "")

if(HAVE_WHISPER_CPP)
  target_compile_definitions(stt_engine_node PRIVATE HAVE_WHISPER_CPP=1)
  target_include_directories(stt_engine_node PRIVATE ${WHISPER_INCLUDE_DIR})
  target_link_libraries(stt_engine_node ${WHISPER_LIBRARY} ${_WHISPER_COMPANION_LIBS})
  list(APPEND _EXTRA_RPATHS "${WHISPER_LIBDIR}")
endif()

if(HAVE_ONNXRUNTIME)
  target_compile_definitions(stt_engine_node PRIVATE HAVE_ONNXRUNTIME=1)
  target_include_directories(stt_engine_node PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(stt_engine_node ${ONNXRUNTIME_LIBRARY})
  list(APPEND _EXTRA_RPATHS "${ONNXRUNTIME_LIBDIR}")
endif()

if(HAVE_DEEPFILTERNET)
  target_compile_definitions(stt_engine_node PRIVATE HAVE_DEEPFILTERNET=1)
  target_include_directories(stt_engine_node PRIVATE ${DEEPFILTER_INCLUDE_DIR})
  target_link_libraries(stt_engine_node ${DEEPFILTER_LIBRARY})
  list(APPEND _EXTRA_RPATHS "${DEEPFILTER_LIBDIR}")
endif()

if(HAVE_RNNOISE)
  target_compile_definitions(stt_engine_node PRIVATE HAVE_RNNOISE=1)
  target_include_directories(stt_engine_node PRIVATE ${RNNOISE_INCLUDE_DIR})
  target_link_libraries(stt_engine_node ${RNNOISE_LIBRARY})
endif()

if(_EXTRA_RPATHS)
  list(REMOVE_DUPLICATES _EXTRA_RPATHS)
  string(JOIN ":" _RPATH_STR ${_EXTRA_RPATHS})
  set_target_properties(stt_engine_node PROPERTIES
    INSTALL_RPATH "${_RPATH_STR}"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
endif()

target_compile_options(stt_engine_node PRIVATE -O3 -Wall -Wextra -Wpedantic)

install(TARGETS stt_engine_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY scripts/ DESTINATION share/${PROJECT_NAME}/scripts
  USE_SOURCE_PERMISSIONS)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models")
  install(DIRECTORY models/ DESTINATION share/${PROJECT_NAME}/models)
endif()

ament_package()
