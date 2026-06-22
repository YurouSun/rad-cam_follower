#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    try:
        mobile_tracker_pkg_share = get_package_share_directory('mobile_tracker')
        default_config_path = os.path.join(
            mobile_tracker_pkg_share,
            'cfg',
            'Mobile_Tracker_car.cfg'
        )
    except Exception:
        default_config_path = '/home/ubuntu/ros2_ws/src/mobile_tracker/cfg/Mobile_Tracker_car.cfg'

    device_name_arg = DeclareLaunchArgument('device_name', default_value='/dev/ttyACM0')
    cli_port_arg = DeclareLaunchArgument('cli_port', default_value='/dev/ttyUSB0')
    data_port_arg = DeclareLaunchArgument('data_port', default_value='/dev/ttyUSB1')

    # 1. 雷达驱动
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

    # 2. 你自己的 Car2 跟随节点副本
    # 注意：
    # 原节点订阅 /ti_mmwave/radar_scan_pcl
    # 但你实际 topic 是 /radar/pointcloud
    # 所以这里做 remap
    car2_safe_follower_node = ExecuteProcess(
        cmd=[
            'bash', '-lc',
            'python3 /home/ubuntu/ros2_ws/src/mmwave_follower/mmwave_follower/car2_line_follower_safe_node.py '
            '--ros-args '
            '-r /ti_mmwave/radar_scan_pcl:=/radar/pointcloud '
            '-p formation_x_distance:=0.50 '
            '-p formation_y_distance:=0.00 '
            '-p kp_x:=1.4 '
            '-p kp_y:=1.2 '
            '-p max_vx:=0.55 '
            '-p max_vy:=0.25 '
            '-p min_vx:=0.18 '
            '-p min_vy:=0.10 '
            '-p arrival_x_tolerance:=0.07 '
            '-p arrival_y_tolerance:=0.08 '
            '-p max_jump_dist_m:=1.50 -p pcl_range_max_m:=2.50 -p pcl_cluster_eps_m:=0.35 -p pcl_cluster_min_pts:=1 '
            '-p velocity_smooth_tau:=0.20 '
            '-p max_acc_x:=0.8 '
            '-p max_acc_y:=0.6 '
            '-p reverse_x_cmd:=false '
            '-p reverse_y_cmd:=true '
            '-p reverse_wz_cmd:=false'
        ],
        output='screen'
    )

    # 3. Car2 底层控制器
    robot_controller_node = Node(
        package='ros_robot_controller',
        executable='ros_robot_controller',
        name='ros_robot_controller',
        output='screen',
        parameters=[{
            'device_name': LaunchConfiguration('device_name'),

            'enable_cmd_vel': True,
            'cmd_vel_topic': '/car2/cmd_vel',

            # 关键：直接透传，便于实验解释
            'cmd_passthrough_mode': True,
            'cmd_trace_log': True,

            'vel_scale': 1.0,
            'max_v': 1.0,
            'max_w': 3.0,

            # 关键：不要让底层自己保持旧命令
            'ignore_zero_cmd': False,
            'zero_cmd_hold_sec': 0.0,
            'zero_cmd_stop_timeout': 0.0,
            'hold_last_cmd_on_timeout': False,
            'cmd_timeout_sec': 0.30,
            'cmd_resend_period_sec': 0.05,

            'publish_odom': True,
            'odom_topic': '/car2/odom',
            'odom_frame': 'odom',
            'base_frame': 'base_link',
            'odom_publish_rate': 20.0,

            'sign_vx': 1.0,
            'sign_vy': 1.0,
            'sign_wz': -1.0,
        }]
    )

    return LaunchDescription([
        device_name_arg,
        cli_port_arg,
        data_port_arg,
        mmwave_driver_node,
        car2_safe_follower_node,
        robot_controller_node,
    ])
