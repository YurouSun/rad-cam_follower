#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import random
from typing import Dict, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu

import sys

# 强制把底层 SDK 所在的绝对路径加入系统查找目录
sdk_dir = "/home/ubuntu/ros2_ws/src/driver/ros_robot_controller/ros_robot_controller"
if sdk_dir not in sys.path:
    sys.path.insert(0, sdk_dir)

try:
    from ros_robot_controller_sdk import Board
    HAS_SDK = True
except ImportError as e:
    HAS_SDK = False
    print(f"\n【警告】 未找到底层的 SDK！详细错误: {e}\n")

Segment = Tuple[float, float, float, float]

def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def build_routes(vx_forward: float, vy_lateral: float) -> Dict[str, List[Segment]]:
    vx, vy = float(vx_forward), float(vy_lateral)
    
    # 严格遵循：X正为前，X负为后；Y正为右，Y负为左
    routes = {
        # --- 前向与横向 ---
        "straight": [(4.0, vx, 0.0, 0.0)],
        "lateral_right": [(3.5, 0.0, +vy, 0.0)],   # 向右平移 (+vy)
        "lateral_left": [(3.5, 0.0, -vy, 0.0)],    # 向左平移 (-vy)
        "diagonal_right_front": [(3.5, 0.8 * vx, +0.8 * vy, 0.0)], # 右前
        "diagonal_left_front": [(3.5, 0.8 * vx, -0.8 * vy, 0.0)],  # 左前
        
        # --- 新增：后退控制 ---
        "backward_straight": [(4.0, -vx, 0.0, 0.0)],               # 向后直行
        "backward_diagonal_right": [(3.5, -0.8 * vx, +0.8 * vy, 0.0)], # 右后斜退
        "backward_diagonal_left": [(3.5, -0.8 * vx, -0.8 * vy, 0.0)],  # 左后斜退
        
        "stop_pause": [(1.5, 0.0, 0.0, 0.0)], 
        
        # 矩形路线 (前进 -> 右移 -> 后退 -> 左移)
        "rectangle": [
            (4.0, +vx, 0.0, 0.0),
            (4.0, 0.0, +vy, 0.0),
            (4.0, -vx, 0.0, 0.0),
            (4.0, 0.0, -vy, 0.0),
        ],
    }
    # 恒定前向动力 S 型平移 (带【左侧非对称补偿】)
    # 因为很多底盘在左前向滑移时摩擦力极大，导致动力相互抵消而卡死，所以针对负的 vy 给更大的驱动系数
    routes["sine_swerve"] = [
        (0.1, 
         0.5 * vx,  # 💡秘诀：把向前的速度减半，给横向平移留出更多展现空间，S 形的兜圈弧度就会变得非常深和明显
         # 当计算出需要向左走(sin<0)时，给予 2.8倍 的动力补偿；向右走时保持 1.5倍
         vy * (1.5 if math.sin(i * 0.1 * (math.pi / 4.0)) >= 0 else 2.8) * math.sin(i * 0.1 * (math.pi / 4.0)), 
         0.0) 
        for i in range(80) # 80 * 0.1 = 8秒刚好是一个完整的大 S 形周期
    ]
    # 环绕扫射平移 (大圆弧)
    routes["circle_sweep"] = [(0.1, vx * 2.5 * math.sin((i * 0.1) * (math.pi / 10.0)), vy * 1.5 * math.cos((i * 0.1) * (math.pi / 10.0)), 0.0) for i in range(100)]
    return routes

class LeaderMotionNode(Node):
    def __init__(self) -> None:
        super().__init__("leader_motion_node")

        if HAS_SDK:
            try:
                self.board = Board(device="/dev/ttyACM0", baudrate=115200, debug=False)
                self.board.enable_reception() 
                self.get_logger().info("✅ 成功连接到底盘串口 SDK！")
            except Exception as e:
                self.board = None
                self.get_logger().error(f"❌ 串口连接失败: {e}")
        else:
            self.board = None

        self.declare_parameter("cmd_topic", "cmd_vel")
        self.declare_parameter("imu_topic", "imu")
        self.declare_parameter("route_name", "random_mix") 
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("vx_forward", 0.3)   # 稍微提高默认速度，让动作更清晰
        self.declare_parameter("vy_lateral", 0.3)
        self.declare_parameter("loop", True) 
        self.declare_parameter("max_record_time_sec", 300.0) 
        self.declare_parameter("enable_imu_correction", True)
        self.declare_parameter("kp_yaw", 1.5)
        self.declare_parameter("max_wz", 0.5)

        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.route_name = str(self.get_parameter("route_name").value)
        self.rate_hz = max(1.0, float(self.get_parameter("rate_hz").value))
        self.vx_forward = float(self.get_parameter("vx_forward").value)
        self.vy_lateral = float(self.get_parameter("vy_lateral").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.max_record_time_sec = float(self.get_parameter("max_record_time_sec").value)
        
        self.enable_imu_correction = bool(self.get_parameter("enable_imu_correction").value)
        self.kp_yaw = float(self.get_parameter("kp_yaw").value)
        self.max_wz = float(self.get_parameter("max_wz").value)

        self.current_yaw, self.target_yaw = 0.0, 0.0
        self.is_yaw_locked, self.has_imu_data = False, False
        self.last_logged_segment = -1
        
        # 将所有新加入的后退动作注册到随机抽取池中
        self.primitives = [
            "straight", "lateral_left", "lateral_right", 
            "diagonal_left_front", "diagonal_right_front", 
            "backward_straight", "backward_diagonal_right", "backward_diagonal_left",
            "sine_swerve", "circle_sweep", "stop_pause"
        ]
        
        self.pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.sub_imu = self.create_subscription(Imu, self.imu_topic, self.imu_callback, qos_profile_sensor_data)
        self.routes = build_routes(self.vx_forward, self.vy_lateral)
        
        if self.route_name == "random_mix":
            self.current_route_name = random.choice(self.primitives)
        else:
            self.current_route_name = self.route_name if self.route_name in self.routes else "straight"
            
        self.segments = self.routes[self.current_route_name]
        self.segment_index = 0
        self.state = "delay"
        
        self.global_start_time = time.monotonic() 
        self.state_start_time = time.monotonic()
        self.timer = self.create_timer(1.0 / self.rate_hz, self.on_timer)
        self.get_logger().info(f"纯运动控制节点")

    def imu_callback(self, msg: Imu):
        self.current_yaw = quaternion_to_yaw(msg.orientation)
        self.has_imu_data = True
        if self.enable_imu_correction and not self.is_yaw_locked:
            self.target_yaw = self.current_yaw
            self.is_yaw_locked = True

    def calculate_wz_correction(self) -> float:
        if not self.enable_imu_correction or not self.has_imu_data: return 0.0
        angle = self.target_yaw - self.current_yaw
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        
        # 提示：如果发现开启 IMU 纠偏后车子反而打转，请把下一行的 kp_yaw 加上负号 (-self.kp_yaw)
        return max(-self.max_wz, min(self.max_wz, self.kp_yaw * angle))

    def publish_cmd(self, x: float, y: float) -> None:
        wz = self.calculate_wz_correction() if (abs(x) > 0.001 or abs(y) > 0.001) else 0.0
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(x), float(y), float(wz)
        self.pub.publish(msg)

        if self.board is not None:
            self.board.set_motor_speed([[0, wz], [1, x], [2, y]])

    def on_timer(self) -> None:
        now = time.monotonic()

        if (now - self.global_start_time) > self.max_record_time_sec and self.state != "stop":
            self.get_logger().info("⏱️ 达到设定的采集时长，准备自动停车并退出。")
            self.state, self.state_start_time = "stop", now

        if self.state == "delay":
            self.publish_cmd(0.0, 0.0)
            if (now - self.state_start_time) >= 1.0: 
                self.state, self.state_start_time = "run", now
            return

        if self.state == "run":
            if self.segment_index >= len(self.segments):
                if self.route_name == "random_mix":
                    self.current_route_name = random.choice(self.primitives)
                    self.segments = self.routes[self.current_route_name]
                    self.segment_index, self.state_start_time = 0, now
                    self.last_logged_segment = -1
                elif self.loop:
                    self.segment_index, self.state_start_time = 0, now
                    self.last_logged_segment = -1
                else:
                    self.state, self.state_start_time = "stop", now
                return

            duration, vx, vy, _ = self.segments[self.segment_index]
            
            # 添加段落打印，方便观察状态
            if self.segment_index != self.last_logged_segment:
                self.get_logger().info(
                    f"🏃 路线 {self.current_route_name} | 段落 {self.segment_index + 1}/{len(self.segments)} | "
                    f"vx={vx:.2f}, vy={vy:.2f}"
                )
                self.last_logged_segment = self.segment_index
                
            self.publish_cmd(vx, vy) 

            if (now - self.state_start_time) >= duration:
                self.segment_index += 1
                self.state_start_time = now
            return

        if self.state == "stop":
            self.publish_cmd(0.0, 0.0)
            if (now - self.state_start_time) >= 1.5: 
                self.timer.cancel()
                rclpy.shutdown()

def main(args=None) -> None:
    rclpy.init(args=args)
    node = LeaderMotionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.board is not None:
            node.board.set_motor_speed([[0, 0.0], [1, 0.0], [2, 0.0]])
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == "__main__":
    main()