from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
  return LaunchDescription([
      Node(
          package="athome_hri",
          executable="barelang_hri",
          name="barelang_hri",
          output="screen",
          emulate_tty=True,
      )
  ])

