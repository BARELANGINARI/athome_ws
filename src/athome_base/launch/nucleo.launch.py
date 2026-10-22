from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    stm_ip_arg = DeclareLaunchArgument(
        'stm_ip',
        default_value='192.168.0.4',
        description='STM32 Nucleo IP address',
    )
    stm_port_arg = DeclareLaunchArgument(
        'stm_port',
        default_value='2002',
        description='STM32 UDP port',
    )
    local_port_arg = DeclareLaunchArgument(
        'local_port',
        default_value='2023',
        description='Local UDP bind port',
    )
    local_ip_arg = DeclareLaunchArgument(
        'local_ip',
        default_value='',
        description='Local bind IP; empty means ANY interface',
    )
    # Control loop and motor limits
    loop_hz_arg = DeclareLaunchArgument(
        'loop_hz',
        default_value='33.33',
        description='Control loop frequency (Hz)',
    )
    rpm_max_arg = DeclareLaunchArgument(
        'rpm_max',
        default_value='428.0',
        description='Maximum motor RPM for PWM scaling',
    )
    pwm_max_arg = DeclareLaunchArgument(
        'pwm_max',
        default_value='225.0',
        description='Maximum PWM sent to STM32',
    )
    pwm_scale_arg = DeclareLaunchArgument(
        'pwm_scale',
        default_value='0.932',
        description='PWM calibration factor',
    )
    odom_frame_arg = DeclareLaunchArgument(
        'odom_frame',
        default_value='odom',
        description='Odometry frame ID',
    )
    base_frame_arg = DeclareLaunchArgument(
        'base_frame',
        default_value='base_footprint',
        description='Robot base frame ID (REP-105: odom -> base_footprint -> base_link)',
    )
    yaw_offset_arg = DeclareLaunchArgument(
        'yaw_offset_deg',
        default_value='0.0',
        description='IMU yaw offset in degrees',
    )
    use_yaw_topic_arg = DeclareLaunchArgument(
        'use_yaw_topic',
        default_value='true',
        description='Use /zed/yaw_deg when true; otherwise use STM32 yaw',
    )

    yaw_topic_arg = DeclareLaunchArgument(
        'yaw_topic',
        default_value='/zed/yaw_deg',
        description='External yaw topic in degrees',
    )
    yaw_timeout_arg = DeclareLaunchArgument(
        'yaw_topic_timeout',
        default_value='0.5',
        description='External yaw timeout (seconds)',
    )
    cmd_vel_timeout_arg = DeclareLaunchArgument(
        'cmd_vel_timeout',
        default_value='0.5',
        description='cmd_vel timeout (seconds)',
    )
    joy_topic_arg = DeclareLaunchArgument(
        'joy_topic',
        default_value='joy_vel',
        description='Topik override joystick (kosongkan/disable bila liar)',
    )
    test_topic_arg = DeclareLaunchArgument(
        'test_topic',
        default_value='test_vel',
        description='Topik override uji manual (kosongkan/disable bila liar)',
    )
    smoothing_enable_arg = DeclareLaunchArgument(
        'smoothing_enable',
        default_value='true',
        description='Enable slew-rate smoothing',
    )
    linear_accel_arg = DeclareLaunchArgument(
        'linear_accel_limit',
        default_value='2.5',
        description='Linear acceleration limit (m/s^2)',
    )
    angular_accel_arg = DeclareLaunchArgument(
        'angular_accel_limit',
        default_value='7.0',
        description='Angular acceleration limit (rad/s^2)',
    )
    stop_decel_arg = DeclareLaunchArgument(
        'stop_decel_scale',
        default_value='2.0',
        description='Deceleration multiplier on cmd_vel timeout',
    )
    pulses_per_rev_arg = DeclareLaunchArgument(
        'pulses_per_rev',
        default_value='7500.0',
        description='Pulses encoder per putaran roda penuh (kalibrasi odom)',
    )
    wheel_radius_arg = DeclareLaunchArgument(
        'wheel_radius_cm',
        default_value='5.0',
        description='Jari-jari roda dalam CM (bukan diameter)',
    )
    base_length_arg = DeclareLaunchArgument(
        'base_length_cm',
        default_value='20.8',
        description='Jarak roda belakang ke poros roda depan (cm)',
    )
    # Nucleo driver node
    nucleo_node = Node(
        package='athome_base',
        executable='nucleo',
        output='screen',
        parameters=[{
            'stm_ip': LaunchConfiguration('stm_ip'),
            'stm_port': LaunchConfiguration('stm_port'),
            'local_port': LaunchConfiguration('local_port'),
            'local_ip': LaunchConfiguration('local_ip'),
            'loop_hz': LaunchConfiguration('loop_hz'),
            'rpm_max': LaunchConfiguration('rpm_max'),
            'pwm_max': LaunchConfiguration('pwm_max'),
            'pwm_scale': LaunchConfiguration('pwm_scale'),
            'odom_frame': LaunchConfiguration('odom_frame'),
            'base_frame': LaunchConfiguration('base_frame'),
            'yaw_offset_deg': LaunchConfiguration('yaw_offset_deg'),
            'use_yaw_topic': LaunchConfiguration('use_yaw_topic'),
            'yaw_topic': LaunchConfiguration('yaw_topic'),
            'yaw_topic_timeout': LaunchConfiguration('yaw_topic_timeout'),
            'cmd_vel_timeout': LaunchConfiguration('cmd_vel_timeout'),
            'joy_topic': LaunchConfiguration('joy_topic'),
            'test_topic': LaunchConfiguration('test_topic'),
            'smoothing_enable': LaunchConfiguration('smoothing_enable'),
            'linear_accel_limit': LaunchConfiguration('linear_accel_limit'),
            'angular_accel_limit': LaunchConfiguration('angular_accel_limit'),
            'stop_decel_scale': LaunchConfiguration('stop_decel_scale'),
            'pulses_per_rev': LaunchConfiguration('pulses_per_rev'),
            'wheel_radius_cm': LaunchConfiguration('wheel_radius_cm'),
            'base_length_cm': LaunchConfiguration('base_length_cm'),
        }],
    )

    return LaunchDescription([
        stm_ip_arg,
        stm_port_arg,
        local_port_arg,
        local_ip_arg,
        loop_hz_arg,
        rpm_max_arg,
        pwm_max_arg,
        pwm_scale_arg,
        odom_frame_arg,
        base_frame_arg,
        yaw_offset_arg,
        use_yaw_topic_arg,
        yaw_topic_arg,
        yaw_timeout_arg,
        cmd_vel_timeout_arg,
        joy_topic_arg,
        test_topic_arg,
        smoothing_enable_arg,
        linear_accel_arg,
        angular_accel_arg,
        stop_decel_arg,
        pulses_per_rev_arg,
        wheel_radius_arg,
        base_length_arg,
        nucleo_node,
    ])