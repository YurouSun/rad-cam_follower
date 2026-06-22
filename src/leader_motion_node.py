#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import random
from typing import Dict, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import ParameterDescriptor
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


ROUTE_PROFILES = {
    "default": {"vx_forward": 0.30, "vy_lateral": 0.30, "loop": True, "max_record_time_sec": 300.0},
    "straight": {"vx_forward": 0.28, "vy_lateral": 0.22, "loop": True, "max_record_time_sec": 8.0},
    "lateral_right": {"vx_forward": 0.28, "vy_lateral": 0.22, "loop": True, "max_record_time_sec": 15.0},
    "lateral_left": {"vx_forward": 0.28, "vy_lateral": 0.22, "loop": True, "max_record_time_sec": 15.0},
    "diagonal_right_front": {"vx_forward": 0.28, "vy_lateral": 0.22, "loop": True, "max_record_time_sec": 15.0},
    "diagonal_left_front": {"vx_forward": 0.28, "vy_lateral": 0.22, "loop": True, "max_record_time_sec": 15.0},
    "backward_straight": {"vx_forward": 0.24, "vy_lateral": 0.20, "loop": True, "max_record_time_sec": 10.0},
    "backward_diagonal_right": {"vx_forward": 0.24, "vy_lateral": 0.20, "loop": True, "max_record_time_sec": 12.0},
    "backward_diagonal_left": {"vx_forward": 0.24, "vy_lateral": 0.20, "loop": True, "max_record_time_sec": 12.0},
    "stop_pause": {"vx_forward": 0.25, "vy_lateral": 0.20, "loop": True, "max_record_time_sec": 6.0},
    "rectangle": {"vx_forward": 0.28, "vy_lateral": 0.22, "loop": True, "max_record_time_sec": 22.0},
    "sine_swerve": {"vx_forward": 0.28, "vy_lateral": 0.22, "loop": True, "max_record_time_sec": 22.0},
    "circle_sweep": {"vx_forward": 0.28, "vy_lateral": 0.22, "loop": True, "max_record_time_sec": 24.0},
    "record_easy": {"vx_forward": 0.28, "vy_lateral": 0.22, "loop": False, "max_record_time_sec": 35.0},
    "record_zigzag": {"vx_forward": 0.28, "vy_lateral": 0.22, "loop": False, "max_record_time_sec": 45.0},
    "record_recovery": {"vx_forward": 0.28, "vy_lateral": 0.22, "loop": False, "max_record_time_sec": 32.0},
    "record_curve": {"vx_forward": 0.28, "vy_lateral": 0.22, "loop": False, "max_record_time_sec": 50.0},
    "record_full": {"vx_forward": 0.28, "vy_lateral": 0.22, "loop": False, "max_record_time_sec": 110.0},
    "record_long": {"vx_forward": 0.28, "vy_lateral": 0.22, "loop": False, "max_record_time_sec": 180.0},
    "random_mix": {"vx_forward": 0.30, "vy_lateral": 0.28, "loop": True, "max_record_time_sec": 300.0},
    "record_random": {"vx_forward": 0.28, "vy_lateral": 0.22, "loop": True, "max_record_time_sec": 300.0},
}

def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def make_segment(duration: float, vx: float, vy: float) -> Segment:
    return (float(duration), float(vx), float(vy), 0.0)


def clone_route(
    routes: Dict[str, List[Segment]],
    route_name: str,
    duration_scale: float = 1.0,
    vx_scale: float = 1.0,
    vy_scale: float = 1.0,
) -> List[Segment]:
    return [
        (duration * duration_scale, cmd_vx * vx_scale, cmd_vy * vy_scale, wz)
        for duration, cmd_vx, cmd_vy, wz in routes[route_name]
    ]


def append_route(
    dst: List[Segment],
    routes: Dict[str, List[Segment]],
    route_name: str,
    pause_sec: float = 0.7,
    duration_scale: float = 1.0,
    vx_scale: float = 1.0,
    vy_scale: float = 1.0,
) -> None:
    dst.extend(clone_route(routes, route_name, duration_scale, vx_scale, vy_scale))
    if pause_sec > 0.0:
        dst.append(make_segment(pause_sec, 0.0, 0.0))

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
    # 为了更容易越过 follower 的运动死区，这里把前向速度再压低一点，
    # 同时明显放大横向摆动幅度，并把周期拉长，形成更大的实际弧线位移。
    routes["sine_swerve"] = [
        (
            0.1,
            0.22 * vx,
            # tanh 会把峰值“压平”，让车在左右两侧停留更久，不会刚检测到就回摆。
            vy
            * (2.8 if math.sin(i * 0.1 * (math.pi / 7.0)) >= 0 else 4.4)
            * math.tanh(1.8 * math.sin(i * 0.1 * (math.pi / 7.0))),
            0.0,
        )
        for i in range(140)  # 14 秒完整大 S，横向拉开更慢更深
    ]
    # 环绕扫射平移 (大圆弧)
    # 拉长周期并增大横向半径，让 follower 能更早感知到偏移。
    routes["circle_sweep"] = [
        (
            0.1,
            vx * (0.28 + 0.16 * math.sin((i * 0.1) * (math.pi / 8.0))),
            vy * 2.8 * math.tanh(1.6 * math.cos((i * 0.1) * (math.pi / 8.0))),
            0.0,
        )
        for i in range(160)
    ]

    # ---- 下面这些是“可以直接录制”的完整路线，不需要再靠基础动作手动拼 ----
    routes["record_easy"] = [
        make_segment(2.0, +0.65 * vx, 0.0),
        make_segment(0.8, 0.0, 0.0),
        make_segment(3.8, +0.45 * vx, -1.10 * vy),
        make_segment(1.2, 0.0, -0.95 * vy),
        make_segment(0.7, 0.0, 0.0),
        make_segment(4.2, 0.0, -1.30 * vy),
        make_segment(1.0, 0.0, 0.0),
        make_segment(3.8, +0.45 * vx, +1.10 * vy),
        make_segment(1.2, 0.0, +0.95 * vy),
        make_segment(0.7, 0.0, 0.0),
        make_segment(4.2, 0.0, +1.30 * vy),
        make_segment(0.8, 0.0, 0.0),
        make_segment(2.0, +0.65 * vx, 0.0),
        make_segment(1.2, 0.0, 0.0),
    ]

    routes["record_zigzag"] = [
        make_segment(4.0, +0.40 * vx, -1.20 * vy),
        make_segment(1.0, 0.0, -1.00 * vy),
        make_segment(0.7, 0.0, 0.0),
        make_segment(4.0, +0.40 * vx, +1.20 * vy),
        make_segment(1.0, 0.0, +1.00 * vy),
        make_segment(0.7, 0.0, 0.0),
        make_segment(4.0, +0.40 * vx, -1.20 * vy),
        make_segment(1.0, 0.0, -1.00 * vy),
        make_segment(0.7, 0.0, 0.0),
        make_segment(4.0, +0.40 * vx, +1.20 * vy),
        make_segment(1.0, 0.0, +1.00 * vy),
        make_segment(0.8, 0.0, 0.0),
        make_segment(3.6, 0.0, -1.30 * vy),
        make_segment(0.8, 0.0, 0.0),
        make_segment(3.6, 0.0, +1.30 * vy),
        make_segment(1.2, 0.0, 0.0),
    ]

    routes["record_recovery"] = [
        make_segment(1.8, +0.65 * vx, 0.0),
        make_segment(0.8, 0.0, 0.0),
        make_segment(3.4, -0.42 * vx, -1.10 * vy),
        make_segment(1.0, 0.0, -0.95 * vy),
        make_segment(0.8, 0.0, 0.0),
        make_segment(2.0, +0.55 * vx, 0.0),
        make_segment(0.8, 0.0, 0.0),
        make_segment(3.4, -0.42 * vx, +1.10 * vy),
        make_segment(1.0, 0.0, +0.95 * vy),
        make_segment(0.8, 0.0, 0.0),
        make_segment(2.0, +0.55 * vx, 0.0),
        make_segment(0.6, 0.0, 0.0),
        make_segment(1.6, -0.70 * vx, 0.0),
        make_segment(1.2, 0.0, 0.0),
    ]

    record_curve = []
    append_route(record_curve, routes, "sine_swerve", pause_sec=1.0, duration_scale=1.0, vx_scale=1.0, vy_scale=1.0)
    append_route(record_curve, routes, "circle_sweep", pause_sec=1.0, duration_scale=1.0, vx_scale=0.95, vy_scale=1.0)
    record_curve.extend(
        [
            make_segment(3.8, +0.35 * vx, -1.20 * vy),
            make_segment(1.0, 0.0, -1.00 * vy),
            make_segment(0.7, 0.0, 0.0),
            make_segment(3.8, +0.35 * vx, +1.20 * vy),
            make_segment(1.0, 0.0, +1.00 * vy),
            make_segment(1.2, 0.0, 0.0),
        ]
    )
    routes["record_curve"] = record_curve

    record_full = []
    append_route(record_full, routes, "record_easy", pause_sec=1.0)
    append_route(record_full, routes, "record_zigzag", pause_sec=1.0)
    append_route(record_full, routes, "record_recovery", pause_sec=1.0)
    append_route(record_full, routes, "rectangle", pause_sec=0.8, duration_scale=0.8, vx_scale=0.55, vy_scale=1.20)
    append_route(record_full, routes, "sine_swerve", pause_sec=1.0, duration_scale=1.0, vx_scale=1.0, vy_scale=1.05)
    record_full.extend(
        [
            make_segment(1.8, +0.60 * vx, 0.0),
            make_segment(0.7, 0.0, 0.0),
            make_segment(4.0, 0.0, -1.30 * vy),
            make_segment(1.0, 0.0, -1.05 * vy),
            make_segment(0.7, 0.0, 0.0),
            make_segment(4.0, 0.0, +1.30 * vy),
            make_segment(1.0, 0.0, +1.05 * vy),
            make_segment(1.2, 0.0, 0.0),
        ]
    )
    routes["record_full"] = record_full

    record_long = []
    append_route(record_long, routes, "record_full", pause_sec=1.2)
    append_route(record_long, routes, "record_curve", pause_sec=1.2)
    append_route(record_long, routes, "circle_sweep", pause_sec=1.0, duration_scale=1.0, vx_scale=0.95, vy_scale=1.0)
    record_long.extend(
        [
            make_segment(1.8, -0.55 * vx, 0.0),
            make_segment(0.8, 0.0, 0.0),
            make_segment(4.2, +0.35 * vx, -1.25 * vy),
            make_segment(1.0, 0.0, -1.05 * vy),
            make_segment(0.7, 0.0, 0.0),
            make_segment(4.2, +0.35 * vx, +1.25 * vy),
            make_segment(1.0, 0.0, +1.05 * vy),
            make_segment(1.5, 0.0, 0.0),
        ]
    )
    routes["record_long"] = record_long

    return routes


def resolve_route_profile(route_name: str) -> Dict[str, float]:
    profile = dict(ROUTE_PROFILES["default"])
    profile.update(ROUTE_PROFILES.get(route_name, {}))
    return profile

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
        self.declare_parameter("max_record_time_sec", 300.0, ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("enable_imu_correction", True)
        self.declare_parameter("kp_yaw", 1.5)
        self.declare_parameter("max_wz", 0.5)

        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.route_name = str(self.get_parameter("route_name").value)
        self.rate_hz = max(1.0, float(self.get_parameter("rate_hz").value))
        route_profile = resolve_route_profile(self.route_name)
        parameter_overrides = getattr(self, "_parameter_overrides", {})
        self.vx_forward = float(parameter_overrides["vx_forward"].value) if "vx_forward" in parameter_overrides else float(route_profile["vx_forward"])
        self.vy_lateral = float(parameter_overrides["vy_lateral"].value) if "vy_lateral" in parameter_overrides else float(route_profile["vy_lateral"])
        self.loop = bool(parameter_overrides["loop"].value) if "loop" in parameter_overrides else bool(route_profile["loop"])
        self.max_record_time_sec = float(parameter_overrides["max_record_time_sec"].value) if "max_record_time_sec" in parameter_overrides else float(route_profile["max_record_time_sec"])
        
        self.enable_imu_correction = bool(self.get_parameter("enable_imu_correction").value)
        self.kp_yaw = float(self.get_parameter("kp_yaw").value)
        self.max_wz = float(self.get_parameter("max_wz").value)

        self.current_yaw, self.target_yaw = 0.0, 0.0
        self.is_yaw_locked, self.has_imu_data = False, False
        self.last_logged_segment = -1
        
        # 基础动作随机池：适合做简单的 random_mix
        self.primitives = [
            "straight", "lateral_left", "lateral_right", 
            "diagonal_left_front", "diagonal_right_front", 
            "backward_straight", "backward_diagonal_right", "backward_diagonal_left",
            "sine_swerve", "circle_sweep", "stop_pause"
        ]
        # 完整录制路线：适合直接指定 route_name 开录
        self.record_courses = [
            "record_easy",
            "record_zigzag",
            "record_recovery",
            "record_curve",
            "record_full",
            "record_long",
        ]
        
        self.pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.sub_imu = self.create_subscription(Imu, self.imu_topic, self.imu_callback, qos_profile_sensor_data)
        self.routes = build_routes(self.vx_forward, self.vy_lateral)
        self.get_logger().info(
            f"路线参数: route={self.route_name}, vx_forward={self.vx_forward:.2f}, "
            f"vy_lateral={self.vy_lateral:.2f}, loop={self.loop}, max_record_time_sec={self.max_record_time_sec:.1f}"
        )
        
        if self.route_name == "random_mix":
            self.current_route_name = random.choice(self.primitives)
        elif self.route_name == "record_random":
            self.current_route_name = random.choice(self.record_courses)
        else:
            if self.route_name not in self.routes:
                available_routes = ", ".join(sorted(self.routes.keys()) + ["record_random"])
                self.get_logger().warn(f"未知 route_name={self.route_name}，自动切回 straight。可选路线: {available_routes}")
                self.current_route_name = "straight"
            else:
                self.current_route_name = self.route_name
            
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
                elif self.route_name == "record_random":
                    self.current_route_name = random.choice(self.record_courses)
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
