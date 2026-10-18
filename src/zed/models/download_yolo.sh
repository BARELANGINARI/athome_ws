#!/usr/bin/env bash
# Regenerasi yolov8n_sim.onnx bila belum ada / ingin varian lain.
# Normalnya TIDAK perlu dijalankan: yolov8n_sim.onnx (YOLOv8n COCO
# pretrained, 13 MB) sudah termasuk di folder ini ("tinggal pake").
#
# Alur regenerasi (butuh PyTorch, sekali saja di mesin bertenaga):
#   1. pip install ultralytics onnx
#      yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12
#   2. pip install "onnx==1.16.2" "onnxsim==0.4.36"
#      python -m onnxsim yolov8n.onnx yolov8n_sim.onnx
# Langkah 2 WAJIB: OpenCV 4.6 (bawaan Ubuntu 24.04) tidak bisa membaca
# ONNX YOLO mentah; node memakai ONNX Runtime sehingga file simplified
# tetap jalan (telah diverifikasi: person/bus/chair/couch/suitcase).
# Setelah regenerasi: colcon build --symlink-install.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

OUT="yolov8n_sim.onnx"
if [ -f "$OUT" ]; then
  echo "OK: $OUT sudah ada ($(du -h "$OUT" | cut -f1)), skip."
  echo "Hapus dulu bila ingin regenerasi."
  exit 0
fi

echo "Install ultralytics + onnx (bila belum ada)..."
pip install -q ultralytics onnx 2>&1 | tail -n 2 || true

echo "Export yolov8n.pt (COCO pretrained) -> ONNX..."
yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12 2>&1 | tail -n 3

echo "Simplify (wajib untuk kompatibilitas)..."
pip install -q "onnx==1.16.2" "onnxsim==0.4.36" 2>&1 | tail -n 1 || true
python3 -m onnxsim yolov8n.onnx "$OUT" 2>&1 | tail -n 2

if [ ! -f "$OUT" ]; then
  echo "ERROR: $OUT tidak terbentuk (cek log di atas)" >&2
  exit 1
fi
echo "OK: $(du -h "$OUT" | cut -f1) tersimpan di $DIR/$OUT"
echo "Lanjut: colcon build --symlink-install (dari ws root)"
