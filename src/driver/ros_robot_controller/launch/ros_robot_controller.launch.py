from launch_ros.actions import Node
from launch import LaunchDescription, LaunchService
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    imu_frame = LaunchConfiguration('imu_frame', default='imu_link')
    imu_frame_arg = DeclareLaunchArgument('imu_frame', default_value=imu_frame)
    enable_cmd_vel = LaunchConfiguration('enable_cmd_vel', default='true')
    enable_cmd_vel_arg = DeclareLaunchArgument('enable_cmd_vel', default_value=enable_cmd_vel)
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic', default='/tracker/cmd_vel')
    cmd_vel_topic_arg = DeclareLaunchArgument('cmd_vel_topic', default_value=cmd_vel_topic)

    ros_robot_controller_node = Node(
        package='ros_robot_controller',
        executable='ros_robot_controller',
        output='screen',
        parameters=[{
            'imu_frame': imu_frame,
            'zero_cmd_hold_sec': 0.25,
            'zero_cmd_epsilon': 1e-4,
            'enable_cmd_vel': enable_cmd_vel,
            'cmd_vel_topic': cmd_vel_topic,
        }]
    )

    return LaunchDescription([
        imu_frame_arg,
        enable_cmd_vel_arg,
        cmd_vel_topic_arg,
        ros_robot_controller_node
    ])

if __name__ == '__main__':
    # 创建一个LaunchDescription对象(create a LaunchDescription object)
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
