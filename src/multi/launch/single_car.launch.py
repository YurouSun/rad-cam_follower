import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node, PushRosNamespace

def generate_launch_description():
    vehicle_id_arg = DeclareLaunchArgument('vehicle_id', default_value='car2')
    use_external_perception_arg = DeclareLaunchArgument('use_external_perception', default_value='false')
    tracked_topic_arg = DeclareLaunchArgument('tracked_topic', default_value='/tracked_objects_3d')
    
    vehicle_id = LaunchConfiguration('vehicle_id')
    use_external_perception = LaunchConfiguration('use_external_perception')
    tracked_topic = LaunchConfiguration('tracked_topic')

    # --- Car 1 (Header) ---
    car1_group = GroupAction(
        condition=IfCondition(PythonExpression(["'", vehicle_id, "' == 'car1'"])),
        actions=[
            PushRosNamespace('car1'),
            Node(package='multi', executable='platoon_node', name='header_controller',
                 parameters=[{'role': 'header', 'my_id': 'car1'}], output='screen')
        ]
    )

    # --- Car 2 (Middle Follower) ---
    car2_group = GroupAction(
        condition=IfCondition(PythonExpression(["'", vehicle_id, "' == 'car2'"])),
        actions=[
            PushRosNamespace('car2'),
            Node(package='multi', executable='platoon_node', name='follower_calculator',
                 parameters=[{'role': 'middle', 'my_id': 'car2'}], output='screen')
        ]
    )

    # --- Car 3 (Last Follower) ---
    car3_group = GroupAction(
        condition=IfCondition(PythonExpression(["'", vehicle_id, "' == 'car3'"])),
        actions=[
            PushRosNamespace('car3'),
            Node(package='multi', executable='platoon_node', name='follower_observer',
                 parameters=[
                     {'role': 'last', 'my_id': 'car3', 
                      'use_external_perception': use_external_perception, 
                      'tracked_topic': tracked_topic}
                 ], output='screen')
        ]
    )

    return LaunchDescription([
        vehicle_id_arg, use_external_perception_arg, tracked_topic_arg,
        car1_group, car2_group, car3_group
    ])