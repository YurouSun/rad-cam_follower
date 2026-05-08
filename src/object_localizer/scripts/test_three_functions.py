#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：监听对象定位器的三个核心功能输出
1. 物体名称
2. 物体位置  
3. 物体速度
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, TwistStamped
from std_msgs.msg import String
import json
import math

class ObjectLocalizerTester(Node):
    def __init__(self):
        super().__init__('object_localizer_tester')
        
        # 订阅三个核心功能的话题
        self.info_sub = self.create_subscription(
            String, '/detected_objects_info', self.info_callback, 10)
        self.position_sub = self.create_subscription(
            PointStamped, '/detected_objects_3d', self.position_callback, 10)
        self.velocity_sub = self.create_subscription(
            TwistStamped, '/detected_objects_velocity', self.velocity_callback, 10)
        
        self.get_logger().info(" 对象定位器测试节点启动 - 监听三个核心功能:")
        self.get_logger().info("  1. 物体名称信息: /detected_objects_info")
        self.get_logger().info("  2. 物体位置信息: /detected_objects_3d") 
        self.get_logger().info("  3. 物体速度信息: /detected_objects_velocity")
        self.get_logger().info("-" * 60)

    def info_callback(self, msg):
        """功能1: 显示检测到的物体名称和综合信息"""
        try:
            data = json.loads(msg.data)
            object_name = data.get('object_name', 'unknown')
            position = data.get('position', {})
            
            self.get_logger().info(f" 功能1 - 检测物体: {object_name}")
            self.get_logger().info(f"   位置: ({position.get('x', 0):.3f}, {position.get('y', 0):.3f}, {position.get('z', 0):.3f})")
            self.get_logger().info(f"   坐标系: {position.get('frame_id', 'unknown')}")
            
            if 'note' in data:
                self.get_logger().info(f"   注意: {data['note']}")
            
        except json.JSONDecodeError:
            self.get_logger().warn("无法解析物体信息JSON数据")

    def position_callback(self, msg):
        """功能2: 显示物体位置"""
        x, y, z = msg.point.x, msg.point.y, msg.point.z
        frame = msg.header.frame_id
        distance = math.sqrt(x**2 + y**2 + z**2)
        
        self.get_logger().info(f" 功能2 - 物体位置:")
        self.get_logger().info(f"   坐标: x={x:.3f}, y={y:.3f}, z={z:.3f}")
        self.get_logger().info(f"   距离: {distance:.3f}m, 坐标系: {frame}")

    def velocity_callback(self, msg):
        """功能3: 显示物体速度"""
        vx, vy = msg.twist.linear.x, msg.twist.linear.y
        speed = math.sqrt(vx**2 + vy**2)
        frame = msg.header.frame_id
        
        self.get_logger().info(f" 功能3 - 物体速度:")
        self.get_logger().info(f"   速度: vx={vx:.3f}, vy={vy:.3f} m/s")
        self.get_logger().info(f"   速率: {speed:.3f} m/s, 坐标系: {frame}")
        
        if speed > 0.1:
            self.get_logger().info(f"   状态: 物体正在移动")
        else:
            self.get_logger().info(f"   状态: 物体静止或缓慢移动")

def main(args=None):
    rclpy.init(args=args)
    
    tester = ObjectLocalizerTester()
    
    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()