from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    """
    针对 20Hz 控制频率同步优化的启动文件。
    """
    
    # --- 1. 配置路径 ---
    try:
        mobile_tracker_pkg_share = get_package_share_directory('mobile_tracker')
        default_config_path = os.path.join(mobile_tracker_pkg_share, 'cfg', 'Mobile_Tracker_car.cfg')
    except Exception:
        default_config_path = '/home/ubuntu/ros2_ws/src/mobile_tracker/cfg/Mobile_Tracker_car.cfg'

    # --- 2. 声明启动参数 ---
    device_name_arg = DeclareLaunchArgument('device_name', default_value='/dev/ttyACM0')
    cli_port_arg = DeclareLaunchArgument('cli_port', default_value='/dev/ttyUSB0')
    data_port_arg = DeclareLaunchArgument('data_port', default_value='/dev/ttyUSB1')

    # --- 3. 定义节点 ---

    # (A) 雷达驱动
    mmwave_driver_node = Node(
        package='mobile_tracker',
        executable='mmwave_6843_driver',
        name='mmwave_6843_driver',
        parameters=[{
            'config_file': default_config_path,
            'cli_port': LaunchConfiguration('cli_port'),  
            'data_port': LaunchConfiguration('data_port')  
        }]
    )

    # (B) Car 2 融合编队节点 (匹配 20Hz 逻辑)
    car2_formation_node = Node(
        package='mmwave_follower',
        executable='car2_formation_node',
        name='car2_formation_node',
        output='screen',
        parameters=[{
            'formation_x_distance': 0.50,
            'formation_y_distance': 0.50,
            'kp_x': 1.6,
            'kp_y': 1.8,
            'max_vx': 0.45,
            'max_vy': 0.35,
            'min_vx': 0.15,
            'min_vy': 0.20,
            'accel_lim_x': 1.2,        # 限制加速度，防止打滑
            'accel_lim_y': 1.2,
            'vel_smooth_factor': 0.35,  # 平滑权重
            'arrival_x_tolerance': 0.05,
            'arrival_y_tolerance': 0.06,
            'max_jump_dist_m': 0.40
        }]
    )

    # (C) 机器人底盘控制器
    robot_controller_node = Node(
        package='ros_robot_controller',
        executable='ros_robot_controller',
        name='ros_robot_controller',
        parameters=[{
            'device_name': LaunchConfiguration('device_name'),
            'cmd_vel_topic': '/car2/cmd_vel', 
            'vel_scale': 1.0,
            'max_v': 0.60
        }],
        remappings=[
            ('/cmd_vel', '/car2/cmd_vel'),
            ('/tracker/cmd_vel', '/car2/cmd_vel')
        ]
    )

    return LaunchDescription([
        device_name_arg,
        cli_port_arg,
        data_port_arg,
        mmwave_driver_node,
        car2_formation_node,
        robot_controller_node
    ])