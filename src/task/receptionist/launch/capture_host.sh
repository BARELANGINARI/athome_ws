#!/usr/bin/env bash
# ==============================================================================
# Script Registrasi Foto Wajah Host (Receptionist Task)
#
# FITUR 1: Ambil foto Host secara live dari kamera:
#   ./capture_host.sh --camera
#
# FITUR 2: Baca / deteksi wajah Host dari file foto yang ada:
#   ./capture_host.sh --photo /path/ke/foto.jpg
#   (atau: ./capture_host.sh /path/ke/foto.jpg)
#
# MENU INTERAKTIF:
#   ./capture_host.sh
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/capture_host.py"

chmod +x "${PYTHON_SCRIPT}"
python3 "${PYTHON_SCRIPT}" "$@"
