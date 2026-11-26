#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# Jika user mengetik: ./run.sh stop
if [ "$1" == "stop" ]; then
    "${SCRIPT_DIR}/stop.sh"
    exit 0
fi

export DISPLAY="${DISPLAY:-:0}"
export WS_DIR="${WS_DIR}"
CONFIG_FILE="${SCRIPT_DIR}/../config/terminator.config"

cd "${WS_DIR}"

if command -v terminator >/dev/null 2>&1; then
    echo "=========================================================="
    echo " Meluncurkan Receptionist dalam Layout Grid (Terminator)..."
    echo " [Kiri Atas   ] Hardware   (STM32 & RPLiDAR)"
    echo " [Tengah Atas ] Navigation (Nav2 & RViz)"
    echo " [Kanan Atas  ] Face       (athome_face_display)"
    echo " [Kiri Bawah  ] Camera     (Vision & IMU)"
    echo " [Tengah Bawah] HRI        (Whisper & Piper)"
    echo " [Kanan Bawah ] Arm        (Dynamixel Arm Motion Server)"
    echo "=========================================================="

    if [ -f "$CONFIG_FILE" ]; then
        terminator --working-directory="${WS_DIR}" -u -g "$CONFIG_FILE" -l receptionist &
    else
        terminator --working-directory="${WS_DIR}" -u -l receptionist &
    fi

    echo "Terminator layout Receptionist berhasil diluncurkan!"
else
    echo "[WARN] Terminator tidak ditemukan, menggunakan GNOME Terminal..."
    SETUP="cd \"${WS_DIR}\" && source /opt/ros/jazzy/setup.bash && source install/setup.bash"
    gnome-terminal --tab --working-directory="${WS_DIR}" --title="Hardware" -- bash -c "${SETUP} && ros2 launch receptionist hardware.launch.py; exec bash"
    sleep 1
    gnome-terminal --tab --working-directory="${WS_DIR}" --title="Navigation" -- bash -c "${SETUP} && sleep 2 && ros2 launch receptionist motion.launch.py; exec bash"
    sleep 1
    gnome-terminal --tab --working-directory="${WS_DIR}" --title="Camera" -- bash -c "${SETUP} && sleep 3 && ros2 launch receptionist camera.launch.py; exec bash"
    sleep 1
    gnome-terminal --tab --working-directory="${WS_DIR}" --title="HRI" -- bash -c "${SETUP} && sleep 4 && ros2 launch receptionist hri.launch.py; exec bash"
    sleep 1
    gnome-terminal --tab --working-directory="${WS_DIR}" --title="Arm" -- bash -c "${SETUP} && sleep 5 && ros2 launch athome_arm arm.launch.py task:=receptionist; exec bash"
    sleep 1
    gnome-terminal --tab --working-directory="${WS_DIR}" --title="Face" -- bash -c "${SETUP} && sleep 6 && ros2 launch athome_face_display face.launch.py; exec bash"
fi
