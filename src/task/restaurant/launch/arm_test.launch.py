from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'launch_arm',
            default_value='true',
            description='Launch athome_arm arm_motion_server along with mock_arm_flow.',
        ),
        DeclareLaunchArgument(
            'port',
            default_value='/dev/ttyUSB0',
            description='Dynamixel USB serial device port.',
        ),
        DeclareLaunchArgument(
            'baudrate',
            default_value='1000000',
            description='Dynamixel communication baud rate.',
        ),
        DeclareLaunchArgument(
            'mode',
            default_value='auto',
            description='Execution mode: "auto" or "step" (press Enter per step).',
        ),
        DeclareLaunchArgument(
            'start_stage',
            default_value='grasp',
            description='Starting stage: "grasp" (full flow) or "delivery" (release flow only).',
        ),
        DeclareLaunchArgument(
            'order1_name',
            default_value='coke',
            description='Order 1 name for simulated TTS prompt.',
        ),
        DeclareLaunchArgument(
            'target_point',
            default_value='0.6,0.1,0.7',
            description='Target point string published to /arm/target_point before /arm/ambil_order.',
        ),
        DeclareLaunchArgument(
            'startup_delay_sec',
            default_value='3.0',
            description='Initial delay before running first arm service.',
        ),
        DeclareLaunchArgument(
            'between_grasp_and_delivery_sec',
            default_value='2.0',
            description='Simulated delay between grasp and delivery stages.',
        ),
        DeclareLaunchArgument(
            'countdown_sec',
            default_value='3.0',
            description='Simulated duration of "one, two, three" speech before release_gripper.',
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('athome_arm'),
                    'launch',
                    'arm.launch.py',
                ])
            ),
            launch_arguments={
                'task': 'restaurant',
                'port': LaunchConfiguration('port'),
                'baudrate': LaunchConfiguration('baudrate'),
            }.items(),
            condition=IfCondition(LaunchConfiguration('launch_arm')),
        ),

        Node(
            package='restaurant',
            executable='mock_arm_flow.py',
            name='restaurant_mock_arm_flow',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'mode': LaunchConfiguration('mode'),
                'start_stage': LaunchConfiguration('start_stage'),
                'order1_name': LaunchConfiguration('order1_name'),
                'target_point': LaunchConfiguration('target_point'),
                'startup_delay_sec': ParameterValue(
                    LaunchConfiguration('startup_delay_sec'), value_type=float
                ),
                'between_grasp_and_delivery_sec': ParameterValue(
                    LaunchConfiguration('between_grasp_and_delivery_sec'), value_type=float
                ),
                'countdown_sec': ParameterValue(
                    LaunchConfiguration('countdown_sec'), value_type=float
                ),
            }],
        ),
    ])
