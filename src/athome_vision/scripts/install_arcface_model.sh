#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/models"
TARGET_MODEL="${MODEL_DIR}/arcface_w600k_r50.onnx"
LOCAL_CACHE="${HOME}/.insightface/models/buffalo_l/w600k_r50.onnx"
URL="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

mkdir -p "${MODEL_DIR}"

if [ -f "${TARGET_MODEL}" ] && [ -s "${TARGET_MODEL}" ]; then
  echo "[install_arcface_model] Model already exists: ${TARGET_MODEL}"
  exit 0
fi

if [ -f "${LOCAL_CACHE}" ] && [ -s "${LOCAL_CACHE}" ]; then
  echo "[install_arcface_model] Copying from local InsightFace cache: ${LOCAL_CACHE}"
  cp "${LOCAL_CACHE}" "${TARGET_MODEL}"
  exit 0
fi

echo "[install_arcface_model] Downloading ArcFace (buffalo_l) from InsightFace releases..."
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

if command -v curl >/dev/null 2>&1; then
  curl -fL --retry 3 -o "${TMP_DIR}/buffalo_l.zip" "${URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -q --show-progress -O "${TMP_DIR}/buffalo_l.zip" "${URL}"
else
  echo "[install_arcface_model] ERROR: Neither curl nor wget found. Please install curl or copy w600k_r50.onnx manually to ${TARGET_MODEL}." >&2
  exit 1
fi

if command -v unzip >/dev/null 2>&1; then
  unzip -j -q "${TMP_DIR}/buffalo_l.zip" "w600k_r50.onnx" -d "${TMP_DIR}"
elif command -v python3 >/dev/null 2>&1; then
  python3 -c "import zipfile; z=zipfile.ZipFile('${TMP_DIR}/buffalo_l.zip'); open('${TMP_DIR}/w600k_r50.onnx','wb').write(z.read('w600k_r50.onnx'))"
else
  echo "[install_arcface_model] ERROR: Neither unzip nor python3 found to extract buffalo_l.zip." >&2
  exit 1
fi

mv "${TMP_DIR}/w600k_r50.onnx" "${TARGET_MODEL}"
echo "[install_arcface_model] Successfully installed: ${TARGET_MODEL}"
