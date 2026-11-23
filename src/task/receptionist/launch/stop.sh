#!/usr/bin/env bash

echo "Menghentikan semua proses Receptionist & Arm..."

pkill -f "ros2 launch receptionist" 2>/dev/null || true
pkill -f "ros2 launch athome_arm" 2>/dev/null || true
pkill -f "ros2 launch athome_face" 2>/dev/null || true
pkill -f "receptionist_node" 2>/dev/null || true
pkill -f "receptionist_manager" 2>/dev/null || true
pkill -f "arm_motion_server" 2>/dev/null || true
pkill -f "athome_vision" 2>/dev/null || true
pkill -f "person_det" 2>/dev/null || true
pkill -f "stt_engine_node" 2>/dev/null || true
pkill -f "tts_engine_node" 2>/dev/null || true
pkill -f "rplidar_node" 2>/dev/null || true
pkill -f "nucleo" 2>/dev/null || true
pkill -f "rviz2" 2>/dev/null || true

echo "Semua proses Receptionist & Arm berhasil dihentikan."
