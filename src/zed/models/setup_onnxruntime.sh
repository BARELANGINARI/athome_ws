#!/usr/bin/env bash
# Setup satu-kali ONNX Runtime C++ (dibutuhkan zed YOLO chair+bag).
# Mengunduh tarball resmi Microsoft (~8 MB) lalu mengekstraknya.
# Default tujuan: /opt/onnxruntime (butuh sudo) atau $HOME/onnxruntime.
# CMake menemukannya otomatis di kedua lokasi, atau via ONNXRUNTIME_ROOT.
#
#   bash src/zed/models/setup_onnxruntime.sh            # coba /opt dulu
#   ONNXRUNTIME_ROOT=$HOME/onnxruntime colcon build --symlink-install
set -euo pipefail

VERSION="1.22.0"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/onnxruntime-linux-x64-${VERSION}.tgz"

if [ -f /opt/onnxruntime/include/onnxruntime_cxx_api.h ]; then
  echo "OK: /opt/onnxruntime sudah ada, skip."
  exit 0
fi
if [ -f "$HOME/onnxruntime/include/onnxruntime_cxx_api.h" ]; then
  echo "OK: \$HOME/onnxruntime sudah ada, skip."
  echo "Build dengan: ONNXRUNTIME_ROOT=\$HOME/onnxruntime colcon build --symlink-install"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "Download ONNX Runtime v${VERSION}..."
curl -sL -o "$TMP/ort.tgz" "$URL"
echo "Ekstrak..."

DEST="/opt/onnxruntime"
if mkdir -p "$DEST" 2>/dev/null && [ -w "$DEST" ]; then
  tar xzf "$TMP/ort.tgz" -C "$TMP"
  cp -r "$TMP/onnxruntime-linux-x64-${VERSION}/"* "$DEST/"
  echo "OK: terpasang di $DEST"
  echo "Build seperti biasa: colcon build --symlink-install"
else
  echo "(tanpa sudo: pasang ke \$HOME/onnxruntime)"
  DEST="$HOME/onnxruntime"
  mkdir -p "$DEST"
  tar xzf "$TMP/ort.tgz" -C "$TMP"
  cp -r "$TMP/onnxruntime-linux-x64-${VERSION}/"* "$DEST/"
  echo "OK: terpasang di $DEST"
  echo "Build dengan: ONNXRUNTIME_ROOT=\$HOME/onnxruntime colcon build --symlink-install"
fi
