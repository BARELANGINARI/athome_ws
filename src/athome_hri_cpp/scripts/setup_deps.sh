#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Setup Dependencies for athome_hri_cpp (whisper.cpp CUDA + GGUF + Silero VAD)
# Target Hardware: NUC11PHi7 (Ubuntu 24.04 + RTX 2060 Mobile 6 GB + i7-1165G7)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "${SCRIPT_DIR}")"
MODELS_DIR="${PKG_DIR}/models"
WHISPER_PREFIX="/opt/whisper.cpp"
WHISPER_BUILD_TMP="/tmp/whisper_cpp_build"

echo "=== [1/4] Installing system build & PortAudio dependencies ==="
sudo apt-get update
sudo apt-get install -y git cmake build-essential curl portaudio19-dev

echo "=== [2/4] Building and installing whisper.cpp (CUDA + Shared Libs) ==="
rm -rf "${WHISPER_BUILD_TMP}"
git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git "${WHISPER_BUILD_TMP}"

CUDA_FLAGS=""
if command -v nvcc >/dev/null 2>&1 || [ -x /usr/local/cuda/bin/nvcc ]; then
  echo "CUDA detected -> building whisper.cpp with GGML_CUDA=ON"
  CUDA_FLAGS="-DGGML_CUDA=ON"
else
  echo "CUDA (nvcc) not found -> building whisper.cpp in CPU mode"
  CUDA_FLAGS="-DGGML_CUDA=OFF"
fi

cmake -S "${WHISPER_BUILD_TMP}" -B "${WHISPER_BUILD_TMP}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX="${WHISPER_PREFIX}" \
  ${CUDA_FLAGS}

cmake --build "${WHISPER_BUILD_TMP}/build" -j4
sudo cmake --install "${WHISPER_BUILD_TMP}/build"
rm -rf "${WHISPER_BUILD_TMP}"

echo "=== [3/4] Downloading Whisper large-v3-turbo-q5_0 GGUF model ==="
mkdir -p "${MODELS_DIR}"
WHISPER_MODEL_FILE="${MODELS_DIR}/ggml-large-v3-turbo-q5_0.bin"
if [ ! -f "${WHISPER_MODEL_FILE}" ]; then
  curl -L --progress-bar \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin" \
    -o "${WHISPER_MODEL_FILE}"
else
  echo "Already exists: ${WHISPER_MODEL_FILE}"
fi

echo "=== [4/4] Downloading Silero VAD v5 ONNX model ==="
SILERO_FILE="${MODELS_DIR}/silero_vad.onnx"
if [ ! -f "${SILERO_FILE}" ]; then
  curl -L --progress-bar \
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx" \
    -o "${SILERO_FILE}"
else
  echo "Already exists: ${SILERO_FILE}"
fi

echo ""
echo "=== Setup Complete ==="
echo "whisper.cpp installed at : ${WHISPER_PREFIX}"
echo "Models directory         : ${MODELS_DIR}"
echo "Next step: build package with:"
echo "  colcon build --packages-select athome_hri_cpp"
