from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch Dynamixel robotic arm motion server."""
    default_config = PathJoinSubstitution(
        [FindPackageShare('athome_arm'), 'config', 'arm.yaml']
    )
    default_motions = PathJoinSubstitution(
        [FindPackageShare('athome_arm'), 'motions']
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'config',
            default_value=default_config,
            description='Path to arm YAML configuration file',
        ),
        DeclareLaunchArgument(
            'port',
            default_value='/dev/ttyUSB0',
            description='Dynamixel USB serial device port',
        ),
        DeclareLaunchArgument(
            'baudrate',
            default_value='1000000',
            description='Dynamixel communication baud rate',
        ),
        DeclareLaunchArgument(
            'motions_dir',
            default_value=default_motions,
            description='Path to arm motion files directory',
        ),
        Node(
            package='athome_arm',
            executable='arm_motion_server',
            name='arm_motion_server',
            output='screen',
            parameters=[
                LaunchConfiguration('config'),
                {
                    'port': LaunchConfiguration('port'),
                    'baudrate': LaunchConfiguration('baudrate'),
                    'motions_dir': LaunchConfiguration('motions_dir'),
                },
            ],
        ),
    ])
