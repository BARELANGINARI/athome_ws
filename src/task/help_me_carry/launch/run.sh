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
    echo " Meluncurkan Help Me Carry dalam Layout Grid (Terminator)..."
    echo " [Kiri Atas   ] Hardware (STM32 & RPLiDAR)"
    echo " [Kanan Atas  ] Motion   (SLAM, Nav2, Follower & RViz)"
    echo " [Kiri Bawah  ] Camera   (ZED Vision & IMU)"
    echo " [Tengah Bawah] HRI      (Speech & Web Dashboard)"
    echo " [Kanan Bawah ] Arm      (Dynamixel Arm Motion Server)"
    echo "=========================================================="

    if [ -f "$CONFIG_FILE" ]; then
        terminator -u -g "$CONFIG_FILE" -l help_me_carry &
    else
        terminator -u -l help_me_carry &
    fi

    echo "Terminator layout Help Me Carry berhasil diluncurkan!"
else
    echo "[WARN] Terminator tidak ditemukan, menggunakan GNOME Terminal..."
    SETUP="cd \"${WS_DIR}\" && source /opt/ros/jazzy/setup.bash && source install/setup.bash"

    gnome-terminal --tab --working-directory="${WS_DIR}" --title="Hardware" -- bash -c "${SETUP} && ros2 launch help_me_carry hardware.launch.py; exec bash"
    sleep 1
    gnome-terminal --tab --working-directory="${WS_DIR}" --title="Camera" -- bash -c "${SETUP} && sleep 2 && ros2 launch help_me_carry camera.launch.py; exec bash"
    sleep 1
    gnome-terminal --tab --working-directory="${WS_DIR}" --title="Motion" -- bash -c "${SETUP} && sleep 3 && ros2 launch help_me_carry motion.launch.py; exec bash"
    sleep 1
    gnome-terminal --tab --working-directory="${WS_DIR}" --title="HRI" -- bash -c "${SETUP} && sleep 4 && ros2 launch help_me_carry hri.launch.py; exec bash"
    sleep 1
    gnome-terminal --tab --working-directory="${WS_DIR}" --title="Arm" -- bash -c "${SETUP} && sleep 5 && ros2 launch athome_arm arm.launch.py; exec bash"
fi
