from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    try:
        mobile_tracker_pkg_share = get_package_share_directory('mobile_tracker')
        default_config_path = os.path.join(mobile_tracker_pkg_share, 'cfg', 'Mobile_Tracker_car.cfg')
    except Exception:
        default_config_path = '/home/ubuntu/ros2_ws/src/mobile_tracker/cfg/Mobile_Tracker_car.cfg'

    device_name_arg = DeclareLaunchArgument('device_name', default_value='/dev/ttyACM0')
    cli_port_arg = DeclareLaunchArgument('cli_port', default_value='/dev/ttyUSB0')
    data_port_arg = DeclareLaunchArgument('data_port', default_value='/dev/ttyUSB1')
    target_id_arg = DeclareLaunchArgument('target_id', default_value='1')
    log_path_arg = DeclareLaunchArgument('log_path', default_value='/tmp/arc_avoidance_log.csv')
    cmd_vel_topic_arg = DeclareLaunchArgument('cmd_vel_topic', default_value='/tracker/cmd_vel')
    odom_topic_arg = DeclareLaunchArgument('odom_topic', default_value='/wheel_odom')
    joint_state_topic_arg = DeclareLaunchArgument('joint_state_topic', default_value='/joint_states')
    imu_topic_arg = DeclareLaunchArgument('imu_topic', default_value='/ros_robot_controller/imu_raw')
    enable_virtual_obstacle_arg = DeclareLaunchArgument('enable_virtual_obstacle', default_value='false')
    virtual_obstacle_x_arg = DeclareLaunchArgument('virtual_obstacle_x', default_value='1.0')
    virtual_obstacle_y_arg = DeclareLaunchArgument('virtual_obstacle_y', default_value='0.0')

    mmwave_driver_node = Node(
        package='mobile_tracker',
        executable='mmwave_6843_driver',
        name='mmwave_6843_driver',
        output='screen',
        parameters=[{
            'config_file': default_config_path,
            'cli_port': LaunchConfiguration('cli_port'),
            'data_port': LaunchConfiguration('data_port'),
        }]
    )

    avoid_node = Node(
        package='mmwave_follower',
        executable='arc_avoidance_observer_node',
        name='arc_avoidance_observer_node',
        output='screen',
        parameters=[{
            'target_id': LaunchConfiguration('target_id'),
            'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
            'odom_topic': LaunchConfiguration('odom_topic'),
            'imu_topic': LaunchConfiguration('imu_topic'),
            'enable_virtual_obstacle': LaunchConfiguration('enable_virtual_obstacle'),
            'virtual_obstacle_x': LaunchConfiguration('virtual_obstacle_x'),
            'virtual_obstacle_y': LaunchConfiguration('virtual_obstacle_y'),
            'lost_hold_sec': 0.8,
            'use_radar_fallback': False,
            'fusion_swap_xy': False,
            'fusion_x_sign': 1.0,
            'fusion_y_sign': 1.0,
            'radar_swap_xy': False,
            'radar_x_sign': 1.0,
            'radar_y_sign': 1.0,
            'keep_distance': 0.45,
            'safe_radius': 0.95,
            'front_half_width': 0.45,
            'track_kp_x': 1.20,
            'track_kp_y': 1.10,
            'max_vx': 0.45,
            'max_vy': 0.30,
            'max_ax': 1.00,
            'max_ay': 1.40,
            'min_forward_speed': 0.10,
            'cruise_speed': 0.18,
            'retreat_speed': 0.12,
            'arc_tangent_speed': 0.22,
            'avoid_forward_speed': 0.26,
            'avoid_forward_decay_radius': 0.30,
            'near_lateral_boost': 1.35,
            'hard_retreat_radius': 0.36,
            'retreat_center_band': 0.16,
            'arc_exit_lateral': 0.24,
            'close_pass_forward_speed': 0.18,
            'arc_finish_hold_sec': 0.9,
            'arc_finish_vx': 0.16,
            'arc_finish_vy': 0.24,
            'arc_finish_extra_sec': 1.2,
            'arc_finish_trigger_x': 0.55,
            'arc_finish_trigger_y': 0.18,
            'arc_path_topic': '/arc_avoidance/reference_path',
            'arc_clearance': 0.18,
            'arc_sweep_deg': 110.0,
            'arc_track_gain': 1.25,
            'pass_side': 0,
            'reverse_cmd_x': False,
            'reverse_cmd_y': False,
            'log_path': LaunchConfiguration('log_path'),
            'enable_csv_log': True,
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

    robot_controller_node = Node(
        package='ros_robot_controller',
        executable='ros_robot_controller',
        name='ros_robot_controller',
        output='screen',
        parameters=[{
            'device_name': LaunchConfiguration('device_name'),
            'imu_frame': 'imu_link',
            'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
            'force_zero_wz': True,
            'vel_scale': 1.0,
            'min_v': 0.0,
            'min_vy': 0.10,
            'cmd_deadband_y': 0.0,
            'cmd_deadband_z': 0.0,
            'max_v': 0.8,
        }]
    )

    return LaunchDescription([
        device_name_arg,
        cli_port_arg,
        data_port_arg,
        target_id_arg,
        log_path_arg,
        cmd_vel_topic_arg,
        odom_topic_arg,
        joint_state_topic_arg,
        imu_topic_arg,
        enable_virtual_obstacle_arg,
        virtual_obstacle_x_arg,
        virtual_obstacle_y_arg,
        mmwave_driver_node,
        wheel_odom_node,
        avoid_node,
        robot_controller_node,
    ])
