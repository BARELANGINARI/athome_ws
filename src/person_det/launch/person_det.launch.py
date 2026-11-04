from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_type',
            default_value='auto',
            description='Camera driver type: "auto" (try ZED first, fallback to USB), "zed", or "usb"',
        ),
        DeclareLaunchArgument(
            'publish_yaw',
            default_value='true',
            description='Publish IMU yaw topic from ZED camera to /zed/yaw_deg',
        ),
        DeclareLaunchArgument(
            'yaw_topic',
            default_value='/zed/yaw_deg',
            description='Topic name for published yaw Float64 angle',
        ),
        DeclareLaunchArgument(
            'camera_fps',
            default_value='30.0',
            description='Camera frame rate (FPS)',
        ),
        DeclareLaunchArgument(
            'camera_index',
            default_value='0',
            description='V4L2 camera device index (0 for ZED/default webcam, or 2 for external USB webcam)',
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
                'camera_type': LaunchConfiguration('camera_type'),
                'publish_yaw': LaunchConfiguration('publish_yaw'),
                'yaw_topic': LaunchConfiguration('yaw_topic'),
                'camera_fps': LaunchConfiguration('camera_fps'),
                'camera_index': LaunchConfiguration('camera_index'),
                'conf_threshold': LaunchConfiguration('conf_threshold'),
                'debug_window': LaunchConfiguration('debug_window'),
            }],
        ),
    ])

