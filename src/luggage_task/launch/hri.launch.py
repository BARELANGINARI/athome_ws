"""Launch file HRI untuk Carry My Luggage.

Menjalankan backend HRI (Faster-Whisper, Piper TTS, Web Server, FSM)
bersama ekosistem luggage_task:
  Terminal 1: ros2 launch luggage_task hardware.launch.py
  Terminal 2: ros2 launch luggage_task camera.launch.py
  Terminal 3: ros2 launch luggage_task motion.launch.py
  Terminal 4: ros2 launch luggage_task hri.launch.py
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
  return LaunchDescription([
      Node(
          package="hri",
          executable="barelang_hri",
          name="barelang_hri",
          output="screen",
          emulate_tty=True,
      )
  ])

