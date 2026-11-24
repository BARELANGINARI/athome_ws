#!/usr/bin/env bash
# ==============================================================================
# Script Pengambil / Pengimpor Foto Host (Revaldo)
# Penggunaan:
#   1. Ambil lewat Webcam / Kamera: ./capture_host.sh
#   2. Import dari file foto yang ada: ./capture_host.sh /path/ke/foto_revaldo.jpg
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/capture_host.py"

chmod +x "${PYTHON_SCRIPT}"

if [ -n "$1" ]; then
    python3 "${PYTHON_SCRIPT}" "$1"
else
    python3 "${PYTHON_SCRIPT}"
fi
