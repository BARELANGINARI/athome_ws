#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
DEST="${1:-$PKG_DIR/models/face_recognition_sface.onnx}"
URL="https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx"

mkdir -p "$(dirname "$DEST")"

if [[ -f "$DEST" ]]; then
  echo "SFace model already exists: $DEST"
  exit 0
fi

echo "Downloading OpenCV SFace face recognition ONNX model (~37 MB)..."
curl -L --fail --retry 3 --progress-bar -o "$DEST.tmp" "$URL"
mv "$DEST.tmp" "$DEST"
echo "Saved: $DEST"
