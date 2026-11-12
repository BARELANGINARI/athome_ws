#!/usr/bin/env bash
# ==============================================================================
# Script Pra-Registrasi Wajah Host (Opsi A) Sebelum Task Receptionist Dimulai
# Penggunaan: ./register_host.sh [NamaHost] (default: Alex)
# ==============================================================================

HOST_NAME="${1:-Alex}"

echo "===================================================================="
echo " [Receptionist] Mendaftarkan Wajah Host: \"${HOST_NAME}\""
echo " 1. Pastikan node kamera sudah menyala (camera.launch.py)"
echo " 2. Pastikan Host berada di depan kamera ZED (~1.2 - 1.8m)"
echo "===================================================================="

ros2 topic pub --once /zed/set_person_name std_msgs/msg/String "{data: '${HOST_NAME}'}"

echo ""
echo "Perintah pendaftaran dikirim!"
echo "Periksa jendela terminal kamera ZED:"
echo "Wajah Host berhasil terkunci jika muncul pesan:"
echo "  [Locked face & clothing for \"${HOST_NAME}\"]"
echo "===================================================================="
