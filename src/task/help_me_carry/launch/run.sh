#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

if [ "$1" == "stop" ]; then
    "${SCRIPT_DIR}/stop.sh"
    exit 0
fi

export DISPLAY="${DISPLAY:-:0}"
SETUP="cd \"${WS_DIR}\" && source /opt/ros/jazzy/setup.bash && source install/setup.bash"

if command -v terminator >/dev/null 2>&1; then
    echo "=========================================================="
    echo " Meluncurkan Help Me Carry dalam Layout 2x2 (Terminator)..."
    echo " [Kiri Atas ] Hardware   (STM32 & RPLiDAR)"
    echo " [Kanan Atas] Motion     (SLAM, Nav2, Follower & RViz)"
    echo " [Kiri Bawah] Camera     (ZED Vision & IMU)"
    echo " [Kanan Bwh ] HRI        (Speech & Web Dashboard)"
    echo "=========================================================="

    CONFIG_FILE="$(mktemp /tmp/help_me_carry_terminator.XXXXXX.config)"
    cat > "$CONFIG_FILE" <<EOF
[global_config]
  title_transmit_fg_color = "#ffffff"
  title_transmit_bg_color = "#1a73e8"
  title_inactive_fg_color = "#cccccc"
  title_inactive_bg_color = "#333333"
[keybindings]
[profiles]
  [[default]]
    scrollback_infinite = True
[layouts]
  [[default]]
    [[[window0]]]
      type = Window
      parent = ""
    [[[child1]]]
      type = Terminal
      parent = window0
  [[help_me_carry]]
    [[[window0]]]
      type = Window
      parent = ""
      title = "Help Me Carry (2x2 Grid)"
    [[[paned0]]]
      type = VPaned
      parent = window0
      ratio = 0.5
    [[[paned_top]]]
      type = HPaned
      parent = paned0
      order = 0
      ratio = 0.5
    [[[paned_bottom]]]
      type = HPaned
      parent = paned0
      order = 1
      ratio = 0.5
    [[[terminal_hw]]]
      type = Terminal
      parent = paned_top
      order = 0
      title = "Hardware (STM32 & RPLiDAR)"
      command = ${SETUP} && ros2 launch help_me_carry hardware.launch.py; exec bash
    [[[terminal_motion]]]
      type = Terminal
      parent = paned_top
      order = 1
      title = "Motion (SLAM, Nav2, Follower & RViz)"
      command = ${SETUP} && sleep 2 && ros2 launch help_me_carry motion.launch.py; exec bash
    [[[terminal_cam]]]
      type = Terminal
      parent = paned_bottom
      order = 0
      title = "Camera (ZED Vision & IMU)"
      command = ${SETUP} && sleep 3 && ros2 launch help_me_carry camera.launch.py; exec bash
    [[[terminal_hri]]]
      type = Terminal
      parent = paned_bottom
      order = 1
      title = "HRI (Speech & Web Dashboard)"
      command = ${SETUP} && sleep 4 && ros2 launch help_me_carry hri.launch.py; exec bash
[plugins]
EOF
    terminator -u -g "$CONFIG_FILE" -l help_me_carry &
    echo "Terminator layout 2x2 berhasil diluncurkan!"
else
    echo "[WARN] Terminator tidak ditemukan, menggunakan GNOME Terminal..."
    gnome-terminal --tab --working-directory="${WS_DIR}" --title="Hardware" -- bash -c "${SETUP} && ros2 launch help_me_carry hardware.launch.py; exec bash"
    sleep 1
    gnome-terminal --tab --working-directory="${WS_DIR}" --title="Motion" -- bash -c "${SETUP} && sleep 2 && ros2 launch help_me_carry motion.launch.py; exec bash"
    sleep 1
    gnome-terminal --tab --working-directory="${WS_DIR}" --title="Camera" -- bash -c "${SETUP} && sleep 3 && ros2 launch help_me_carry camera.launch.py; exec bash"
    sleep 1
    gnome-terminal --tab --working-directory="${WS_DIR}" --title="HRI" -- bash -c "${SETUP} && sleep 4 && ros2 launch help_me_carry hri.launch.py; exec bash"
fi
