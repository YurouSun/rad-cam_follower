import glob
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


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

    log_dir_setup = ExecuteProcess(
        cmd=['bash', '-lc', 'mkdir -p /home/ubuntu/.ros/log'],
        output='screen'
    )

    device_name_arg = DeclareLaunchArgument(
        'device_name',
        default_value=EnvironmentVariable(
            'ROBOT_CONTROLLER_DEVICE',
            default_value='/dev/ttyACM0'
        ),
        description='Serial port for robot controller'
    )

    target_id_arg = DeclareLaunchArgument(
        'target_id',
        default_value='1',
        description='Target ID to follow'
    )

    fusion_output_topic = '/car3/tracked_objects_3d'
    vision_track_topic = '/car3/tracked_objects'
    radar_pointcloud_topic = '/radar/pointcloud'
    radar_tracks_topic = '/radar/tracked_targets'

    mmwave_driver_node = Node(
        package='mobile_tracker',
        executable='mmwave_6843_driver',
        name='mmwave_6843_driver',
        output='screen',
        parameters=[{
            'config_file': default_config_path,
            'cli_port': '/dev/ttyUSB0',
            'data_port': '/dev/ttyUSB1',
        }]
    )

    tracker_node = Node(
        package='mmwave_follower',
        executable='tracker_node',
        name='tracker_node',
        output='screen',
        parameters=[{
            'target_id': LaunchConfiguration('target_id'),
            'tracked_objects_topic': fusion_output_topic,
            'radar_tracks_topic': radar_tracks_topic,
            'cmd_vel_topic': '/car3/cmd_vel_raw',

            'reverse_cmd_x': False,
            'reverse_cmd_y': False,

            'keep_distance': 0.5,
            'kp_linear': 0.8,
            'kp_lateral': 1.0,
            'kp_angular': 0.0,
            'lat_deadband_m': 0.05,

            'max_vx': 0.4,
            'max_vy': 0.3,
            'max_wz': 0.2,
            'yaw_damping_gain': 1.2,
            'yaw_damping_limit': 0.2,
            'filter_alpha_cmd': 0.6,
            'filter_alpha_coord': 0.5
        }]
    )

    def launch_perception_chain(context, *args, **kwargs):
        target_id = LaunchConfiguration('target_id').perform(context).strip()

        fusion_node = ExecuteProcess(
            cmd=[
                'bash',
                '-lc',
                'source /home/ubuntu/ros2_ws/install/setup.bash && '
                'python3 /home/ubuntu/ros2_ws/src/fusion.py '
                '--ros-args '
                '-r __node:=car3_mot_tracker_fusion_node '
                '-p rgb_topic:=/car3/ascamera/camera_publisher/rgb0/image '
                f'-p det_topic:={vision_track_topic} '
                f'-p radar_topic:={radar_pointcloud_topic} '
                f'-p publish_topic:={fusion_output_topic} '
                '-p det_timeout_sec:=0.8'
            ],
            output='screen'
        )

        setm_gate_node = ExecuteProcess(
            cmd=[
                'bash',
                '-lc',
                'source /opt/ros/humble/setup.bash && '
                'source /home/ubuntu/ros2_ws/install/setup.bash && '
                'python3 /home/ubuntu/ros2_ws/src/mmwave_follower/mmwave_follower/car2_setm_cmd_gate_node.py '
                '--ros-args '
                f'-p target_id:={target_id} '
                f'-p tracked_objects_topic:={fusion_output_topic} '
                '-p raw_cmd_topic:=/car3/cmd_vel_raw '
                '-p cmd_topic:=/car3/cmd_vel '
                '-p keep_distance:=0.50 '
                '-p delta_far:=0.05 '
                '-p delta_near:=0.25 '
                '-p epsilon_z:=0.12 '
                '-p rho:=0.0004 '
                '-p max_hold_sec:=0.35 '
                '-p enable_csv_log:=true '
                '-p csv_path:=/tmp/car3_setm_gate_log.csv'
            ],
            output='screen'
        )

        return [fusion_node, setm_gate_node]

    def launch_robot_controller(context, *args, **kwargs):
        raw_device_name = LaunchConfiguration('device_name').perform(context).strip()

        if not raw_device_name:
            raise RuntimeError('device_name is empty')

        if not os.path.exists(raw_device_name):
            available_serial_devices = (
                sorted(glob.glob('/dev/serial/by-id/*')) +
                sorted(glob.glob('/dev/ttyACM*')) +
                sorted(glob.glob('/dev/ttyUSB*'))
            )
            raise RuntimeError(
                f'Robot controller device {raw_device_name} does not exist. '
                f'Available serial devices: {available_serial_devices}'
            )

        selected_device_name = raw_device_name

        if not raw_device_name.startswith('/dev/serial/by-id/'):
            resolved_device = os.path.realpath(raw_device_name)
            by_id_candidates = [
                path for path in sorted(glob.glob('/dev/serial/by-id/*'))
                if os.path.realpath(path) == resolved_device
            ]
            if by_id_candidates:
                selected_device_name = by_id_candidates[0]

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
                'max_v': 1.0
            }]
        )

        return [robot_controller_node]

    return LaunchDescription([
        log_dir_setup,
        device_name_arg,
        target_id_arg,
        mmwave_driver_node,
        tracker_node,
        OpaqueFunction(function=launch_perception_chain),
        OpaqueFunction(function=launch_robot_controller),
    ])