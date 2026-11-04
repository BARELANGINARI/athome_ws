from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_index',
            default_value='2',
            description='V4L2 camera device index (e.g. 0, 2)',
        ),
        DeclareLaunchArgument(
            'conf_threshold',
            default_value='0.6',
            description='Person detection confidence threshold',
        ),
        DeclareLaunchArgument(
            'debug_window',
            default_value='true',
            description='Show OpenCV preview window with bounding boxes & attributes',
        ),
        Node(
            package='person_det',
            executable='person_det_node',
            name='person_det_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'camera_index': LaunchConfiguration('camera_index'),
                'conf_threshold': LaunchConfiguration('conf_threshold'),
                'debug_window': LaunchConfiguration('debug_window'),
            }],
        ),
    ])

