#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from tf2_ros import TransformBroadcaster
import math
import time

class TrajectoryVisualizerNode(Node):
    def __init__(self):
        super().__init__('trajectory_visualizer_node')
        
        # === 1. 通用参数声明 (完美支持命名空间隔离) ===
        # 要监听的速度指令话题
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        # 发布的轨迹路径话题
        self.declare_parameter('path_topic', 'trajectory')
        
        # 坐标系隔离设置
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        
        # 辅助参数
        self.declare_parameter('publish_rate', 20.0)      # 积分与发布频率 (Hz)
        self.declare_parameter('max_path_length', 1500)   # 轨迹保留的点数长度
        self.declare_parameter('cmd_timeout', 0.5)        # 安全超时(秒): 超过此时间没收到速度，认为已停车
        
        # === 2. 获取参数 ===
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.path_topic = self.get_parameter('path_topic').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.max_path_length = self.get_parameter('max_path_length').value
        self.cmd_timeout = self.get_parameter('cmd_timeout').value
        
        # === 3. 状态变量初始化 ===
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.yaw = 0.0
        
        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_wz = 0.0
        
        self.last_cmd_time = time.time()
        self.last_update_time = time.time()
        
        # === 4. 发布器与订阅器 ===
        self.tf_broadcaster = TransformBroadcaster(self)
        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        self.odom_pub = self.create_publisher(Odometry, 'odom_visual', 10) # 附加发布odom
        
        self.sub_cmd = self.create_subscription(Twist, self.cmd_vel_topic, self.cmd_callback, 10)
        
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.odom_frame
        
        # 创建固定频率的定时器用于微积分运算
        self.create_timer(1.0 / self.publish_rate, self.update_and_publish)
        
        self.get_logger().info("=========================================")
        self.get_logger().info("通用轨迹可视化节点已启动")
        self.get_logger().info(f"监听指令: {self.cmd_vel_topic}")
        self.get_logger().info(f"发布路径: {self.path_topic}")
        self.get_logger().info(f"坐标隔离: {self.odom_frame} -> {self.base_frame}")
        self.get_logger().info("=========================================")

    def cmd_callback(self, msg: Twist):
        """ 接收上游任意追踪器发出的控制速度 """
        self.current_vx = msg.linear.x
        self.current_vy = msg.linear.y
        self.current_wz = msg.angular.z
        self.last_cmd_time = time.time()

    def update_and_publish(self):
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now
        
        # 核心防撞保护：如果控制节点挂了，或者很久没发数据，强制速度归零，防止轨迹乱飘
        if now - self.last_cmd_time > self.cmd_timeout:
            self.current_vx = 0.0
            self.current_vy = 0.0
            self.current_wz = 0.0
            
        # --- 运动学积分 ---
        self.yaw += self.current_wz * dt
        self.pos_x += (self.current_vx * math.cos(self.yaw) - self.current_vy * math.sin(self.yaw)) * dt
        self.pos_y += (self.current_vx * math.sin(self.yaw) + self.current_vy * math.cos(self.yaw)) * dt
        
        current_time_msg = self.get_clock().now().to_msg()
        qx, qy, qz, qw = 0.0, 0.0, math.sin(self.yaw/2.0), math.cos(self.yaw/2.0)
        
        # --- 1. 广播 TF 树 ---
        t = TransformStamped()
        t.header.stamp = current_time_msg
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = self.pos_x
        t.transform.translation.y = self.pos_y
        t.transform.translation.z = 0.07  # 底盘高度假设
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)
        
        # --- 2. 发布 Path 轨迹 ---
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = current_time_msg
        pose_stamped.header.frame_id = self.odom_frame
        pose_stamped.pose.position.x = self.pos_x
        pose_stamped.pose.position.y = self.pos_y
        pose_stamped.pose.orientation.x = qx
        pose_stamped.pose.orientation.y = qy
        pose_stamped.pose.orientation.z = qz
        pose_stamped.pose.orientation.w = qw
        
        self.path_msg.header.stamp = current_time_msg
        self.path_msg.poses.append(pose_stamped)
        
        # 控制数组长度
        if len(self.path_msg.poses) > self.max_path_length:
            self.path_msg.poses.pop(0)
            
        self.path_pub.publish(self.path_msg)

        # --- 3. 补充发布 Odometry ---
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time_msg
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame
        odom_msg.pose.pose = pose_stamped.pose
        odom_msg.twist.twist.linear.x = self.current_vx
        odom_msg.twist.twist.linear.y = self.current_vy
        odom_msg.twist.twist.angular.z = self.current_wz
        self.odom_pub.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()