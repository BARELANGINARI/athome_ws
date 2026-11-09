#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
DEST="${1:-$PKG_DIR/models/gender_model_weights.onnx}"
URL="https://huggingface.co/serengil/deepface/resolve/main/gender_model_weights.onnx"

mkdir -p "$(dirname "$DEST")"

if [[ -f "$DEST" ]]; then
  echo "Gender model already exists: $DEST"
  exit 0
fi

echo "Downloading DeepFace gender ONNX model (~512 MB)..."
curl -L --fail --retry 3 --progress-bar -o "$DEST.tmp" "$URL"
mv "$DEST.tmp" "$DEST"
echo "Saved: $DEST"
