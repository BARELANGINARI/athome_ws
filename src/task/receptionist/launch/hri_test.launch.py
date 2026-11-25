import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    launch_dir = os.path.dirname(os.path.realpath(__file__))
    hri_launch_path = os.path.join(launch_dir, 'hri.launch.py')

    return LaunchDescription([
        DeclareLaunchArgument(
            'stt_backend',
            default_value='cpp',
            description='STT backend to use: "cpp" (athome_hri_cpp) or "python" (athome_hri).',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(hri_launch_path),
            launch_arguments={
                'manager_node': 'true',
                'stt_backend': LaunchConfiguration('stt_backend'),
            }.items(),
        ),
        Node(
            package='receptionist',
            executable='mock_hardware.py',
            name='receptionist_mock_hardware',
            output='screen',
            emulate_tty=True,
        ),
    ])
