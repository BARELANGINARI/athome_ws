from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
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
        SetEnvironmentVariable('RCUTILS_CONSOLE_OUTPUT_FORMAT', '[{severity}] {message}'),
        SetEnvironmentVariable('RCUTILS_COLORIZED_OUTPUT', '1'),
        DeclareLaunchArgument(
            'task',
            default_value='all',
            description='Task name (help_me_carry, receptionist, restaurant)',
        ),
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
                    'task': LaunchConfiguration('task'),
                    'port': LaunchConfiguration('port'),
                    'baudrate': LaunchConfiguration('baudrate'),
                    'motions_dir': LaunchConfiguration('motions_dir'),
                },
            ],
        ),
    ])
