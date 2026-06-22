import glob
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.actions import ExecuteProcess
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    """
    Launch file to start the mmWave driver, the tracker node, the new fusion node, and the robot controller.
    启动毫米波雷达驱动、跟踪节点、新版融合节点以及机器人底盘控制器的组合启动文件。
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
        default_value=EnvironmentVariable('ROBOT_CONTROLLER_DEVICE', default_value='/dev/ttyACM1'),
        description='Serial port for robot controller; prefer /dev/serial/by-id/...'
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

    # (B) 跟踪节点
    # 负责订阅 /tracked_objects_3d 融合结果并控制底盘运动
    tracker_node = Node(
        package='mmwave_follower',
        executable='tracker_node',
        name='tracker_node',
        output='screen',
        parameters=[{
            # [关键] 将 Launch 参数映射到节点参数
            'target_id': LaunchConfiguration('target_id'),
            'tracked_objects_topic': '/tracked_objects_3d',
            'cmd_vel_topic': '/car2/cmd_vel_raw',
            
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
    # 切换到新的 fusion.py，负责发布 /tracked_objects_3d
    fusion_node = ExecuteProcess(
        cmd=[
            'bash',
            '-lc',
            'source /home/ubuntu/ros2_ws/install/setup.bash && '
            'python3 /home/ubuntu/ros2_ws/src/fusion.py '
            '--ros-args '
            '-p publish_topic:=/tracked_objects_3d '
            '-p det_timeout_sec:=0.8'
        ],
        output='screen'
    )


    # (SETM) 事件触发命令门控节点
    # 订阅 tracker_node 的 /car2/cmd_vel_raw，
    # 根据 /tracked_objects_3d 中的误差 z 判断是否允许更新 /car2/cmd_vel。
    setm_gate_node = ExecuteProcess(
        cmd=[
            'bash',
            '-lc',
            'source /opt/ros/humble/setup.bash && '
            'source /home/ubuntu/ros2_ws/install/setup.bash && '
            'python3 /home/ubuntu/ros2_ws/src/mmwave_follower/mmwave_follower/car2_setm_cmd_gate_node.py '
            '--ros-args '
            '-p target_id:=1 '
            '-p tracked_objects_topic:=/tracked_objects_3d '
            '-p raw_cmd_topic:=/car2/cmd_vel_raw '
            '-p cmd_topic:=/car2/cmd_vel '
            '-p keep_distance:=0.50 '
            '-p delta_far:=0.05 '
            '-p delta_near:=0.25 '
            '-p epsilon_z:=0.12 '
            '-p rho:=0.0004 '
            '-p max_hold_sec:=0.35 '
            '-p enable_csv_log:=true '
            '-p csv_path:=/tmp/car2_setm_gate_log.csv'
        ],
        output='screen'
    )

    def launch_robot_controller(context, *args, **kwargs):
        raw_device_name = LaunchConfiguration('device_name').perform(context).strip()
        if not raw_device_name:
            raise RuntimeError('device_name is empty; set it to a valid /dev/serial/by-id/... or /dev/ttyACM*/ttyUSB* path')

        if not os.path.exists(raw_device_name):
            available_serial_devices = sorted(glob.glob('/dev/serial/by-id/*')) + sorted(glob.glob('/dev/ttyACM*')) + sorted(glob.glob('/dev/ttyUSB*'))
            raise RuntimeError(
                f'Robot controller device {raw_device_name} does not exist. '
                f'Available serial devices: {available_serial_devices}'
            )

        selected_device_name = raw_device_name
        if not raw_device_name.startswith('/dev/serial/by-id/'):
            resolved_device = os.path.realpath(raw_device_name)
            by_id_candidates = [path for path in sorted(glob.glob('/dev/serial/by-id/*')) if os.path.realpath(path) == resolved_device]
            if by_id_candidates:
                selected_device_name = by_id_candidates[0]

        robot_controller_node = Node(
            package='ros_robot_controller',
            executable='ros_robot_controller',
            name='ros_robot_controller',
            namespace='car2',
            output='screen',
            parameters=[{
                'device_name': selected_device_name,
                'cmd_vel_topic': '/car2/cmd_vel',
                'x_only_mode': False,
                'cmd_passthrough_mode': True,
                'sdk_debug': True,
                'imu_frame': 'imu_link',
                'vel_scale': 1.0, # 速度比例，视底盘具体情况调整
                'max_v': 1.0
            }]
        )

        return [robot_controller_node]

    # --- 4. 返回描述 ---
    return LaunchDescription([
        log_dir_setup,
        device_name_arg,
        target_id_arg,
        mmwave_driver_node,
        fusion_node,
        tracker_node,
        setm_gate_node,
        OpaqueFunction(function=launch_robot_controller)
    ])
