#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math
import re
import time
import socket

try:
    from interfaces.msg import ObjectsInfo
except ImportError:
    ObjectsInfo = None

class PlatoonNode(Node):
    def __init__(self):
        super().__init__('platoon_node')
        
        self.declare_parameter('role', 'middle')
        self.declare_parameter('my_id', 'car2')   
        self.declare_parameter('tracked_topic', '/tracked_objects_3d')
        self.declare_parameter('relative_pose_topic', 'platoon/relative_pose')
        
        self.role = self.get_parameter('role').value
        self.my_id = self.get_parameter('my_id').value
        self.cb_group = ReentrantCallbackGroup()

        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.cache_pose1 = None
        self.cache_pose2 = None
        self.last_seen_time1 = 0.0
        self.last_seen_time2 = 0.0
        
        self.pose_r1 = None  
        self.pose_r2 = None  
        self.last_r1_recv_time = 0.0
        self.last_r2_recv_time = 0.0
        self.last_net_time = 0.0
        
        self.send_count = 0
        
        # 【修改】：按照你的思路，放宽过期时间回到 1.5s。
        # 允许短暂跟丢时，利用 Tracker 的预测坐标继续计算并发前方位的趋势。
        # 即使预测错了，Car 2 的雷达向量内积也能把它拦下来。
        self.expire_limit = 1.5 

        local_ip = "Unknown"
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except: pass

        self.get_logger().info(f'[{self.my_id}] 通信节点启动 | 纯方位角(Bearing)传输 | IP: {local_ip}')

        if self.role == 'last':
            self.setup_observer()
        else:
            self.setup_calculator()

    def setup_observer(self):
        self.pub_r1 = self.create_publisher(PoseStamped, '/car3/perception/robot1', self.qos)
        self.pub_r2 = self.create_publisher(PoseStamped, '/car3/perception/robot2', self.qos)
        
        if ObjectsInfo is not None:
            self.create_subscription(ObjectsInfo, self.get_parameter('tracked_topic').value, 
                                     self.cb_sensor, 10, callback_group=self.cb_group)
        
        self.create_timer(0.05, self.timer_publish_logic, callback_group=self.cb_group)

    def cb_sensor(self, msg):
        now_ros_time = self.get_clock().now().to_msg()
        p1 = self._extract_pose(msg, 1, ["RED", "car1", "ID:1"], now_ros_time)
        p2 = self._extract_pose(msg, 2, ["BLUE", "car2", "ID:2"], now_ros_time)

        if p1:
            self.cache_pose1 = p1
            self.last_seen_time1 = time.time()
        if p2:
            self.cache_pose2 = p2
            self.last_seen_time2 = time.time()

    def timer_publish_logic(self):
        now_time = time.time()
        self.send_count += 1
        token = f"s:{self.send_count}"

        if self.cache_pose1 and (now_time - self.last_seen_time1 < self.expire_limit):
            self.cache_pose1.header.frame_id = token
            self.pub_r1.publish(self.cache_pose1)
        else:
            hb1 = PoseStamped()
            hb1.header.frame_id = f"heartbeat_r1:{self.send_count}"
            self.pub_r1.publish(hb1)
            
        if self.cache_pose2 and (now_time - self.last_seen_time2 < self.expire_limit):
            self.cache_pose2.header.frame_id = token
            self.pub_r2.publish(self.cache_pose2)
        else:
            hb2 = PoseStamped()
            hb2.header.frame_id = f"heartbeat_r2:{self.send_count}"
            self.pub_r2.publish(hb2)

    def _extract_pose(self, msg, target_id, keywords, now_msg):
        for obj in msg.objects:
            try:
                is_match = False
                if getattr(obj, 'id', -1) == target_id: is_match = True
                
                # fallback for string matching if strictly needed, though id is preferred
                if not is_match:
                    name_upper = obj.class_name.upper()
                    for kw in keywords:
                        if kw.upper() in name_upper:
                            is_match = True
                            break
                            
                if is_match and getattr(obj, 'position', None) is not None:
                    # In radar_camera_fusion, it's already X-forward, Y-left
                    val_x, val_y = float(obj.position.x), float(obj.position.y)
                    p = PoseStamped()
                    p.header.stamp = now_msg
                    p.pose.position.x = val_x
                    p.pose.position.y = val_y
                    return p
            except: pass
        return None

    def setup_calculator(self):
        self.create_subscription(PoseStamped, '/car3/perception/robot1', self.cb_r1, self.qos, callback_group=self.cb_group)
        self.create_subscription(PoseStamped, '/car3/perception/robot2', self.cb_r2, self.qos, callback_group=self.cb_group)
        rel_topic = self.get_parameter('relative_pose_topic').value
        self.pub_relative = self.create_publisher(PoseStamped, rel_topic, 10)
        self.create_timer(0.05, self.calculate_logic, callback_group=self.cb_group)

    def cb_r1(self, msg): 
        if "heartbeat" not in msg.header.frame_id:
            self.pose_r1 = msg
            self.last_r1_recv_time = time.time()
        self.last_net_time = time.time()

    def cb_r2(self, msg): 
        if "heartbeat" not in msg.header.frame_id:
            self.pose_r2 = msg
            self.last_r2_recv_time = time.time()
        self.last_net_time = time.time()

    def calculate_logic(self):
        now = time.time()
        # 通信整体静默超过3秒才停止（比如网络完全断开）
        if self.last_net_time == 0 or now - self.last_net_time > 3.0:
            return

        if self.pose_r1 is None or self.pose_r2 is None:
            return

        # 【修改】：移除了之前的严格新鲜度掐断逻辑！
        # 现在，只要 pose_r1 和 pose_r2 没有超时被清空，就会计算它们的方位角发给下游。
        # 如果它们是视觉追踪器发来的短时间预测值，只要方向没错得太离谱，对 Car 2 就是有用的。

        rel_x = self.pose_r1.pose.position.x - self.pose_r2.pose.position.x
        rel_y = self.pose_r1.pose.position.y - self.pose_r2.pose.position.y
        dist = math.sqrt(rel_x**2 + rel_y**2)

        if dist > 0.05:
            phi_x = rel_x / dist
            phi_y = rel_y / dist
        else:
            phi_x, phi_y = 1.0, 0.0

        rel_msg = PoseStamped()
        rel_msg.header.stamp = self.get_clock().now().to_msg()
        rel_msg.header.frame_id = 'platoon_bearing_only'
        rel_msg.pose.position.x = phi_x
        rel_msg.pose.position.y = phi_y
        rel_msg.pose.position.z = 1.0 
        self.pub_relative.publish(rel_msg)

        self.get_logger().info(f"[{self.my_id}] 发送方位角 -> Phi:[{phi_x:.2f}, {phi_y:.2f}]", throttle_duration_sec=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = PlatoonNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try: executor.spin()
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()