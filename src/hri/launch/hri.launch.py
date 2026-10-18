"""Launch file HRI untuk Barelang Inari.
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

