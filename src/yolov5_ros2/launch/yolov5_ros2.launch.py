# yolo_v5_ros2_launch.py

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

def launch_setup(context):
    # 获取您的包的 config 目录路径，以便定位文件
    package_share_directory = get_package_share_directory('yolov5_ros2')
    config_dir = os.path.join(package_share_directory, 'config')

    # 从启动参数中获取模型文件名
    model_file_name = LaunchConfiguration('model').perform(context)
    model_path = os.path.join(config_dir, model_file_name)
    
    backend_val = LaunchConfiguration('backend').perform(context)
    robot_name_val = LaunchConfiguration('robot_name').perform(context)
    image_topic_val = LaunchConfiguration('image_topic').perform(context)

    # 定义类别文件的路径
    class_names_file_path = os.path.join(config_dir, 'classes.txt')

    yolov5_ros2_node = Node(
        package='yolov5_ros2',
        executable='yolo_detect', # 注意：这里的 'yolo_detect' 是您编译后的可执行文件名
        namespace=robot_name_val,
        output='screen',
        emulate_tty=True, # 增加这个可以改善日志输出格式
        parameters=[
            {"device": "cpu"},
            {"model": model_path}, # 传入模型的完整路径
            {"backend": backend_val},
            {"image_topic": image_topic_val},
            {"show_result": False}, # 保持False，通过ROS话题查看
            

            {"pub_result_img": True}, # <--- 修改为 True，发布结果图像以便RQT查看
            {"class_names_file": class_names_file_path} # <--- 新增此行，告诉节点类别文件的位置
        ]
    )

    return [
        yolov5_ros2_node,
    ]

def generate_launch_description():
    return LaunchDescription([
        # 声明一个名为 'model' 的启动参数，默认值为 'best_cleaned.onnx'
        DeclareLaunchArgument(
            'model', 
            default_value='best_cleaned.onnx', 
            description='Model file name (.pt, .tflite, .onnx) under config dir'
        ),
        DeclareLaunchArgument(
            'backend', 
            default_value='auto', 
            description='Inference backend: auto, onnx, ncnn, tflite'
        ),
        DeclareLaunchArgument(
            'robot_name',
            default_value=os.environ.get('HOST', ''),
            description='Robot namespace (for multi-robot topic isolation)'
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='ascamera/camera_publisher/rgb0/image',
            description='Image topic for detector (relative topic recommended)'
        ),
        OpaqueFunction(function=launch_setup)
    ])

if __name__ == '__main__':
    pass