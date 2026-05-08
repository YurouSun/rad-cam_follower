import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 获取yolov5_ros2功能包的路径
    yolov5_ros2_share_dir = get_package_share_directory('yolov5_ros2')

    return LaunchDescription([
        # 声明模型参数，允许从命令行覆盖
        DeclareLaunchArgument(
            'model',
            default_value='best.pt',
            description='Name of the YOLOv5 model file (e.g., best.pt)'
        ),

        # 启动yolov5_ros2节点
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(yolov5_ros2_share_dir, 'launch', 'yolov5_ros2.launch.py')
            ),
            launch_arguments={
                'model': LaunchConfiguration('model')
            }.items()
        ),

        # 启动你的新物体定位节点
        Node(
            package='object_localizer',
            executable='object_localizer_node',
            name='object_localizer_node',
            output='screen'
        ),
    ])