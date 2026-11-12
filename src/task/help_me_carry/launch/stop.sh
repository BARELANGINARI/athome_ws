#!/usr/bin/env bash

echo "Menghentikan semua proses Help Me Carry & Arm..."

pkill -f "ros2 launch help_me_carry" 2>/dev/null || true
pkill -f "ros2 launch athome_arm" 2>/dev/null || true
pkill -f "help_me_carry_node" 2>/dev/null || true
pkill -f "help_me_carry_manager" 2>/dev/null || true
pkill -f "tracking_controller" 2>/dev/null || true
pkill -f "return_to_origin" 2>/dev/null || true
pkill -f "arm_motion_server" 2>/dev/null || true
pkill -f "athome_vision" 2>/dev/null || true
pkill -f "person_det" 2>/dev/null || true
pkill -f "stt_engine_node" 2>/dev/null || true
pkill -f "tts_engine_node" 2>/dev/null || true
pkill -f "rplidar_node" 2>/dev/null || true
pkill -f "nucleo" 2>/dev/null || true
pkill -f "rviz2" 2>/dev/null || true

echo "Semua proses Help Me Carry & Arm berhasil dihentikan."
