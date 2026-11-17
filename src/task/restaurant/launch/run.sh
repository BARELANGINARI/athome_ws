#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# Jika user mengetik: ./run.sh stop
if [ "$1" == "stop" ]; then
    "${SCRIPT_DIR}/stop.sh"
    exit 0
fi

export DISPLAY="${DISPLAY:-:0}"
CONFIG_FILE="${SCRIPT_DIR}/../config/terminator.config"

if command -v terminator >/dev/null 2>&1; then
    echo "=========================================================="
    echo " Meluncurkan Restaurant Task dalam Layout 2x2 (Terminator)..."
    echo " [Kiri Atas ] Hardware   (STM32 & RPLiDAR)"
    echo " [Kanan Atas] Navigation (Nav2 SLAM & Restaurant Node)"
    echo " [Kiri Bawah] Camera     (ZED Vision & Gesture/Face)"
    echo " [Kanan Bwh ] HRI        (Whisper, Piper & Manager)"
    echo "=========================================================="

    if [ -f "$CONFIG_FILE" ]; then
        terminator -u -g "$CONFIG_FILE" -l restaurant &
    else
        terminator -u -l restaurant &
    fi

    echo "Terminator layout 2x2 berhasil diluncurkan!"
else
    echo "[WARN] Terminator tidak ditemukan, menggunakan GNOME Terminal..."
    SETUP="cd \"${WS_DIR}\" && source /opt/ros/jazzy/setup.bash && source install/setup.bash"
    gnome-terminal --tab --working-directory="${WS_DIR}" --title="Hardware" -- bash -c "${SETUP} && ros2 launch restaurant hardware.launch.py; exec bash"
    sleep 1
    gnome-terminal --tab --working-directory="${WS_DIR}" --title="Navigation" -- bash -c "${SETUP} && sleep 2 && ros2 launch restaurant motion.launch.py; exec bash"
    sleep 1
    gnome-terminal --tab --working-directory="${WS_DIR}" --title="Camera" -- bash -c "${SETUP} && sleep 3 && ros2 launch restaurant camera.launch.py; exec bash"
    sleep 1
    gnome-terminal --tab --working-directory="${WS_DIR}" --title="HRI" -- bash -c "${SETUP} && sleep 4 && ros2 launch restaurant hri.launch.py; exec bash"
fi
