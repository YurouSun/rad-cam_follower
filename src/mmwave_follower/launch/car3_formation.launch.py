import glob
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """
    Car 3 编队启动文件。
    启动毫米波雷达驱动、Car 3 逆雅可比编队节点以及底盘控制器。
    """

    try:
        mobile_tracker_pkg_share = get_package_share_directory('mobile_tracker')
        default_config_path = os.path.join(mobile_tracker_pkg_share, 'cfg', 'Mobile_Tracker_car.cfg')
    except Exception:
        default_config_path = '/home/ubuntu/ros2_ws/src/mobile_tracker/cfg/Mobile_Tracker_car.cfg'

    device_name_arg = DeclareLaunchArgument(
        'device_name',
        default_value=EnvironmentVariable('ROBOT_CONTROLLER_DEVICE', default_value='/dev/ttyACM1'),
        description='Serial port for robot controller; prefer /dev/serial/by-id/...'
    )
    cli_port_arg = DeclareLaunchArgument('cli_port', default_value='/dev/ttyUSB0')
    data_port_arg = DeclareLaunchArgument('data_port', default_value='/dev/ttyUSB1')

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

    car3_bearing_formation_node = Node(
        package='mmwave_follower',
        executable='car3_bearing_formation_node',
        name='car3_bearing_formation_node',
        output='screen',
        parameters=[{
            'target_1_id': 1,
            'target_2_id': 2,
            'target_2_angle_deg': 25.0,
            'kp': 1.2,
            'kv': 0.2,
            'reverse_x_cmd': False,
            'reverse_y_cmd': False,
            'max_vx': 0.80,
            'max_vy': 0.80,
            'min_vx': 0.25,
            'min_vy': 0.30,
            'arrival_theta_tolerance': 0.06,
            'max_acc_x': 1.5,
            'max_acc_y': 1.5,
            'derivative_filter_alpha': 0.15,
            'velocity_smooth_tau': 0.25,
            'control_period_sec': 0.05,
        }]
    )

    def launch_robot_controller(context, *args, **kwargs):
        raw_device_name = LaunchConfiguration('device_name').perform(context).strip()
        candidate_devices = sorted(glob.glob('/dev/serial/by-id/*')) + sorted(glob.glob('/dev/ttyACM*')) + sorted(glob.glob('/dev/ttyUSB*'))

        selected_device_name = raw_device_name
        if raw_device_name and os.path.exists(raw_device_name):
            if not raw_device_name.startswith('/dev/serial/by-id/'):
                resolved_device = os.path.realpath(raw_device_name)
                by_id_candidates = [path for path in sorted(glob.glob('/dev/serial/by-id/*')) if os.path.realpath(path) == resolved_device]
                if by_id_candidates:
                    selected_device_name = by_id_candidates[0]
        else:
            if candidate_devices:
                selected_device_name = candidate_devices[0]
                print(f"Warning: robot controller device '{raw_device_name}' not found, fallback to '{selected_device_name}'")
            else:
                raise RuntimeError(
                    f"Robot controller device '{raw_device_name}' does not exist and no serial devices were found. "
                    f"Available serial devices: {candidate_devices}"
                )

        robot_controller_node = Node(
            package='ros_robot_controller',
            executable='ros_robot_controller',
            name='ros_robot_controller',
            namespace='car3',
            output='screen',
            parameters=[{
                'device_name': selected_device_name,
                'cmd_vel_topic': '/car3/cmd_vel',
                'x_only_mode': False,
                'cmd_passthrough_mode': True,
                'sdk_debug': True,
                'imu_frame': 'imu_link',
                'vel_scale': 1.0,
                'max_v': 1.0,
            }]
        )

        return [robot_controller_node]

    return LaunchDescription([
        device_name_arg,
        cli_port_arg,
        data_port_arg,
        mmwave_driver_node,
        car3_bearing_formation_node,
        OpaqueFunction(function=launch_robot_controller),
    ])
