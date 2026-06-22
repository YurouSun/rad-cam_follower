from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    device_name_arg = DeclareLaunchArgument('device_name', default_value='/dev/ttyACM0')
    cmd_vel_topic_arg = DeclareLaunchArgument('cmd_vel_topic', default_value='/tracker/cmd_vel')
    odom_topic_arg = DeclareLaunchArgument('odom_topic', default_value='/wheel_odom')
    joint_state_topic_arg = DeclareLaunchArgument('joint_state_topic', default_value='/joint_states')
    log_path_arg = DeclareLaunchArgument('log_path', default_value='/tmp/arc_avoidance_real.csv')
    virtual_obstacle_x_arg = DeclareLaunchArgument('virtual_obstacle_x', default_value='1.0')
    virtual_obstacle_y_arg = DeclareLaunchArgument('virtual_obstacle_y', default_value='0.0')
    enable_disturbance_arg = DeclareLaunchArgument('enable_disturbance_injection', default_value='false')
    disturbance_profile_arg = DeclareLaunchArgument('disturbance_profile', default_value='execution')
    disturbance_vx_amp_arg = DeclareLaunchArgument('disturbance_vx_amp', default_value='0.02')
    disturbance_vy_amp_arg = DeclareLaunchArgument('disturbance_vy_amp', default_value='0.08')
    disturbance_freq_arg = DeclareLaunchArgument('disturbance_freq_hz', default_value='0.35')
    disturbance_start_arg = DeclareLaunchArgument('disturbance_start_sec', default_value='1.5')
    disturbance_duration_arg = DeclareLaunchArgument('disturbance_duration_sec', default_value='4.0')

    avoid_node = Node(
        package='mmwave_follower',
        executable='arc_avoidance_observer_node',
        name='arc_avoidance_observer_node',
        output='screen',
        parameters=[{
            'target_id': -1,
            'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
            'odom_topic': LaunchConfiguration('odom_topic'),
            'enable_virtual_obstacle': True,
            'virtual_obstacle_x': LaunchConfiguration('virtual_obstacle_x'),
            'virtual_obstacle_y': LaunchConfiguration('virtual_obstacle_y'),
            'force_virtual_arc': True,
            'stop_after_virtual_arc': True,
            'enable_disturbance_injection': LaunchConfiguration('enable_disturbance_injection'),
            'disturbance_profile': LaunchConfiguration('disturbance_profile'),
            'disturbance_vx_amp': LaunchConfiguration('disturbance_vx_amp'),
            'disturbance_vy_amp': LaunchConfiguration('disturbance_vy_amp'),
            'disturbance_freq_hz': LaunchConfiguration('disturbance_freq_hz'),
            'disturbance_start_sec': LaunchConfiguration('disturbance_start_sec'),
            'disturbance_duration_sec': LaunchConfiguration('disturbance_duration_sec'),
            'exec_gain_x_amp': 0.10,
            'exec_gain_y_amp': 0.18,
            'exec_bias_x_amp': 0.012,
            'exec_bias_y_amp': 0.030,
            'exec_lateral_lag_sec': 0.35,
            'exec_noise_std': 0.006,
            'safe_radius': 1.00,
            'arc_radius': 0.95,
            'arc_clearance': 0.0,
            'arc_sweep_deg': 180.0,
            'arc_tangent_speed': 0.14,
            'arc_track_gain': 1.8,
            'max_vx': 0.22,
            'max_vy': 0.20,
            'max_ax': 0.45,
            'max_ay': 0.45,
            'min_forward_speed': 0.03,
            'cruise_speed': 0.06,
            'avoid_forward_speed': 0.06,
            'close_pass_forward_speed': 0.05,
            'pass_side': 1,
            'reverse_cmd_y': True,
            'enable_csv_log': True,
            'log_path': LaunchConfiguration('log_path'),
        }]
    )

    robot_controller_node = Node(
        package='ros_robot_controller',
        executable='ros_robot_controller',
        name='ros_robot_controller',
        output='screen',
        parameters=[{
            'device_name': LaunchConfiguration('device_name'),
            'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
            'odom_topic': LaunchConfiguration('odom_topic'),
            'force_zero_wz': True,
            'publish_odom': False,
            'odom_publish_rate': 20.0,
            'vel_scale': 1.0,
            'min_v': 0.0,
            'min_vy': 0.0,
            'cmd_deadband_y': 0.0,
            'cmd_deadband_z': 0.0,
            'max_v': 0.6,
        }]
    )

    wheel_odom_node = Node(
        package='mmwave_follower',
        executable='wheel_odom_from_joint_states',
        name='wheel_odom_from_joint_states',
        output='screen',
        parameters=[{
            'joint_state_topic': LaunchConfiguration('joint_state_topic'),
            'odom_topic': LaunchConfiguration('odom_topic'),
            'base_frame': 'base_link',
            'odom_frame': 'odom',
        }]
    )

    return LaunchDescription([
        device_name_arg,
        cmd_vel_topic_arg,
        odom_topic_arg,
        joint_state_topic_arg,
        log_path_arg,
        virtual_obstacle_x_arg,
        virtual_obstacle_y_arg,
        enable_disturbance_arg,
        disturbance_profile_arg,
        disturbance_vx_amp_arg,
        disturbance_vy_amp_arg,
        disturbance_freq_arg,
        disturbance_start_arg,
        disturbance_duration_arg,
        wheel_odom_node,
        avoid_node,
        robot_controller_node,
    ])
