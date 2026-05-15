from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    """
    Launch file to start the mmWave driver, the tracker node (V6.1), and the robot controller.
    启动毫米波雷达驱动、跟踪节点(V6.1)以及机器人底盘控制器的组合启动文件。
    """
    
    # --- 1. 配置路径 ---
    # 获取 mobile_tracker 包的路径，用于查找雷达配置文件
    try:
        mobile_tracker_pkg_share = get_package_share_directory('mobile_tracker')
        default_config_path = os.path.join(mobile_tracker_pkg_share, 'cfg', 'Mobile_Tracker_car.cfg')
    except Exception:
        print("Warning: 'mobile_tracker' package not found. Using default path.")
        default_config_path = '/home/ubuntu/ros2_ws/src/mobile_tracker/cfg/Mobile_Tracker_car.cfg'

    # 让 ROS 日志目录提前存在，避免某些环境下 ros2 命令初始化日志失败
    log_dir_setup = ExecuteProcess(
        cmd=['bash', '-lc', 'mkdir -p /home/ubuntu/.ros/log'],
        output='screen'
    )

    # --- 2. 声明启动参数 (Launch Arguments) ---

    # 参数1: 底盘串口设备号
    device_name_arg = DeclareLaunchArgument(
        'device_name',
        default_value='/dev/ttyACM0',
        description='Serial port for robot controller'
    )

    # 参数2: 跟踪目标 ID
    # [修正] 对应 tracker_node.py 中的 'target_id' 参数
    # 默认为 -1 (自动选择最近目标)
    target_id_arg = DeclareLaunchArgument(
        'target_id',
        default_value='-1',
        description='Target ID to follow (from fusion node, -1 for auto)'
    )

    # --- 3. 定义节点 (Nodes) ---

    # (A) 毫米波雷达驱动节点
    # 负责启动雷达硬件，发布 /ti_mmwave/radar_scan_pcl
    mmwave_driver_node = Node(
        package='mobile_tracker',
        executable='mmwave_6843_driver',
        name='mmwave_6843_driver',
        output='screen',
        parameters=[{
            'config_file': default_config_path,
            'cli_port': '/dev/ttyUSB0',  # 雷达命令串口
            'data_port': '/dev/ttyUSB1'  # 雷达数据串口
        }]
    )

    # (B) 跟踪节点 (MmwaveTrackerNode V6.1)
    # 负责订阅融合数据并控制底盘运动
    tracker_node = Node(
        package='mmwave_follower',
        executable='tracker_node',
        name='tracker_node',
        output='screen',
        parameters=[{
            # [关键] 将 Launch 参数映射到节点参数
            'target_id': LaunchConfiguration('target_id'),
            'cmd_vel_topic': '/car3/cmd_vel',
            
            # [关键修正] 方向控制：取消反向，约定 x 向前为正
            'reverse_cmd_x': False, 
            'reverse_cmd_y': False,

            # 运动控制参数 (可以在这里直接修改，无需重新编译)
            'keep_distance': 0.5,  # 保持距离 0.5米
            'kp_linear': 0.8,      # 前后跟随灵敏度
            'kp_lateral': 1.0,     # 左右横移灵敏度
            'kp_angular': 0.0,     # 转向灵敏度（禁用）
            'lat_deadband_m': 0.05, # 侧向死带 5cm
            
            # 安全限速
            'max_vx': 0.4,
            'max_vy': 0.3,         # 侧向速度限制到 0.3 m/s
            'max_wz': 0.2          # 允许微小角速用于阻尼
            ,
            'yaw_damping_gain': 1.2,
            'yaw_damping_limit': 0.2,
            'filter_alpha_cmd': 0.6,
            'filter_alpha_coord': 0.5
        }]
    )

    # (C) 雷达-相机融合节点
    # 负责订阅 /ti_mmwave/radar_scan_pcl 和相机检测结果，发布 /tracked_objects_3d
    fusion_node = ExecuteProcess(
        cmd=['bash', '-lc', 'source /home/ubuntu/ros2_ws/install/setup.bash && python3 /home/ubuntu/ros2_ws/src/radar_camera_fusion.py'],
        output='screen'
    )

    # (D) 机器人底盘控制器
    # 负责接收 cmd_vel 并驱动电机
    robot_controller_node = Node(
        package='ros_robot_controller',
        executable='ros_robot_controller',
        name='ros_robot_controller',
        namespace='car3',
        output='screen',
        parameters=[{
            'device_name': LaunchConfiguration('device_name'),
            'cmd_vel_topic': '/car3/cmd_vel',
            'x_only_mode': False,
            'cmd_passthrough_mode': True,
            'sdk_debug': True,
            'imu_frame': 'imu_link',
            'vel_scale': 1.0, # 速度比例，视底盘具体情况调整
            'max_v': 1.0
        }]
    )

    # --- 4. 返回描述 ---
    return LaunchDescription([
        log_dir_setup,
        device_name_arg,
        target_id_arg,
        mmwave_driver_node,
        fusion_node,
        tracker_node,
        robot_controller_node
    ])