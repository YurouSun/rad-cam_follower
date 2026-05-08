#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from interfaces.msg import ObjectsInfo
import math
import time
import numpy as np

class Car3BearingFormationNode(Node):
    def __init__(self):
        super().__init__('car3_bearing_formation_node')
        self.get_logger().info('🚙 Car 3 节点启动 [纯平移全向版：逆雅可比矩阵直接解算 (破除鞍点死锁)]')

        # === 1. 目标 ID 与几何形态参数 ===
        self.declare_parameter('target_1_id', 1) # 1车 ID (主领航者)
        self.declare_parameter('target_2_id', 2) # 2车 ID (侧向锚点)
        
        # 【核心改进：适配窄FOV相机】
        # 将原先硬编码的 45 度改为可配置参数。
        # 角度越小，3车为了满足该视角，就会自动退得越远，从而让1、2车都挤进窄相机的视野中心！
        self.declare_parameter('target_2_angle_deg', 25.0) 

        # === 2. 雅可比解算核心增益参数 ===
        self.declare_parameter('kp', 1.2)  
        self.declare_parameter('kv', 0.2) 

        # === 3. 物理底盘修正参数 ===
        self.declare_parameter('reverse_x_cmd', False)
        self.declare_parameter('reverse_y_cmd', False) 
        
        self.declare_parameter('max_vx', 0.80)
        self.declare_parameter('max_vy', 0.80)
        
        # 静摩擦击穿电压，保证平移时绝对不会卡顿
        self.declare_parameter('min_vx', 0.25) 
        self.declare_parameter('min_vy', 0.3) 
        
        # 角度到达容忍度 (rad)，约 3.5 度。进入此误差内允许平滑停车
        self.declare_parameter('arrival_theta_tolerance', 0.06)
        
        # === 4. 速度平滑与频率匹配参数 ===
        self.declare_parameter('max_acc_x', 1.5) 
        self.declare_parameter('max_acc_y', 1.5) 
        
        self.declare_parameter('derivative_filter_alpha', 0.15) 
        self.declare_parameter('velocity_smooth_tau', 0.25) 

        # === 5. 状态变量 ===
        self.theta_1 = 0.0
        self.theta_2 = 0.0
        self.t_target_1 = 0.0
        self.t_target_2 = 0.0
        
        self.last_theta_1 = None
        self.last_theta_2 = None
        self.theta_1_dot = 0.0
        self.theta_2_dot = 0.0
        
        self.cmd_v = np.zeros(2) 
        self.last_ctrl_time = time.time()

        # === 6. ROS 通信 ===
        self.cb_group = ReentrantCallbackGroup()
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        
        self.sub_fusion = self.create_subscription(
            ObjectsInfo, 
            '/tracked_objects_3d', 
            self.fusion_callback, 
            qos,
            callback_group=self.cb_group
        )
        self.pub_cmd_vel = self.create_publisher(Twist, '/car3/cmd_vel', 10)

        # 20Hz 独立控制循环
        self.declare_parameter('control_period_sec', 0.05) 
        self.create_timer(float(self.get_parameter('control_period_sec').value), self.control_loop, callback_group=self.cb_group)

    def fusion_callback(self, msg: ObjectsInfo):
        """ 仅负责提取最新角度状态 """
        id_1 = self.get_parameter('target_1_id').value
        id_2 = self.get_parameter('target_2_id').value
        
        now = time.time()
        for obj in msg.objects:
            obj_id = int(getattr(obj, 'id', -1))
            pos = getattr(obj, 'position', None)
            
            tx, ty = None, None
            if hasattr(pos, 'x') and hasattr(pos, 'y'):
                tx, ty = pos.x, pos.y
            elif isinstance(pos, (list, tuple, np.ndarray)) and len(pos) >= 2:
                tx, ty = pos[0], pos[1]
                
            if tx is not None and ty is not None:
                measured_angle = math.atan2(ty, tx) 
                
                if obj_id == id_1:
                    self.theta_1 = measured_angle
                    self.t_target_1 = now
                elif obj_id == id_2:
                    self.theta_2 = measured_angle
                    self.t_target_2 = now

    def control_loop(self):
        """ 严格 20Hz 平滑控制循环 """
        now = time.time()
        dt = max(0.01, min(0.1, now - self.last_ctrl_time))
        self.last_ctrl_time = now

        seen_1 = (now - self.t_target_1) < 0.6
        seen_2 = (now - self.t_target_2) < 0.6

        kp = float(self.get_parameter('kp').value)
        kv = float(self.get_parameter('kv').value)
        alpha_d = float(self.get_parameter('derivative_filter_alpha').value)
        target_2_angle_rad = math.radians(float(self.get_parameter('target_2_angle_deg').value))

        # 模式 1：全丢模式 -> 软刹车
        if not seen_1 and not seen_2:
            self.get_logger().warn("🛑 目标1和2全丢，执行安全软刹车", throttle_duration_sec=2.0)
            self.stop_robot()
            return

        # 模式 2：降级模式 (只看到领航者1车) -> 纯惯性滑行衰减
        if seen_1 and not seen_2:
            self.get_logger().info("⚠️ 丢失侧向锚点(2车)，进入降级滑行模式", throttle_duration_sec=1.0)
            self.apply_smoothing_and_publish(0.0, 0.0, dt, is_arrived=False, target_2_angle_rad=target_2_angle_rad)
            return

        # =========================================================
        # 模式 3：正常模式 -> 逆雅可比矩阵解算 (Inverse Jacobian)
        # =========================================================

        if self.last_theta_1 is None:
            self.last_theta_1 = self.theta_1
            self.last_theta_2 = self.theta_2

        # 1. 角度误差 (最短路径限制在 -pi 到 pi)
        e1 = (self.theta_1 - 0.0 + math.pi) % (2 * math.pi) - math.pi
        e2 = (self.theta_2 - target_2_angle_rad + math.pi) % (2 * math.pi) - math.pi

        # 2. 角度微分与低通滤波
        diff_1 = (self.theta_1 - self.last_theta_1 + math.pi) % (2 * math.pi) - math.pi
        diff_2 = (self.theta_2 - self.last_theta_2 + math.pi) % (2 * math.pi) - math.pi

        raw_dot_1 = np.clip(diff_1 / dt, -3.0, 3.0)
        raw_dot_2 = np.clip(diff_2 / dt, -3.0, 3.0)

        self.theta_1_dot = alpha_d * raw_dot_1 + (1.0 - alpha_d) * self.theta_1_dot
        self.theta_2_dot = alpha_d * raw_dot_2 + (1.0 - alpha_d) * self.theta_2_dot

        self.last_theta_1 = self.theta_1
        self.last_theta_2 = self.theta_2

        # 3. 雅可比行列式分母保护 (防止矩阵奇异除以0)
        D = math.sin(self.theta_2 - self.theta_1)
        if abs(D) < 0.15:
            D = math.copysign(0.15, D)

        cos1, sin1 = math.cos(self.theta_1), math.sin(self.theta_1)
        cos2, sin2 = math.cos(self.theta_2), math.sin(self.theta_2)

        # 4. 核心解算：逆雅可比乘法 (完美解耦，无论怎么站位都不会力互相抵消！)
        target_vx = kp * (e1 * cos2 - e2 * cos1) / D
        target_vy = kp * (e1 * sin2 - e2 * sin1) / D

        # 增加微分阻尼，防过冲
        target_vx += kv * (self.theta_1_dot * cos2 - self.theta_2_dot * cos1) / D
        target_vy += kv * (self.theta_1_dot * sin2 - self.theta_2_dot * sin1) / D

        # 误差到达判断
        tol_theta = float(self.get_parameter('arrival_theta_tolerance').value)
        is_arrived = (abs(e1) < tol_theta) and (abs(e2) < tol_theta)

        self.apply_smoothing_and_publish(target_vx, target_vy, dt, is_arrived, target_2_angle_rad)

    def apply_smoothing_and_publish(self, target_vx, target_vy, dt, is_arrived, target_2_angle_rad):
        """ 速度低通平滑 + 爬升率限制 + 静摩擦击穿 """
        tau_v = float(self.get_parameter('velocity_smooth_tau').value)
        max_ax = float(self.get_parameter('max_acc_x').value)
        max_ay = float(self.get_parameter('max_acc_y').value)
        
        # 1. 动态自适应一阶低通滤波 (极其顺滑地跟踪目标速度)
        alpha_v = dt / (tau_v + dt)
        filtered_vx = (1.0 - alpha_v) * self.cmd_v[0] + alpha_v * target_vx
        filtered_vy = (1.0 - alpha_v) * self.cmd_v[1] + alpha_v * target_vy

        # 2. 爬升率限制 (物理防冲)
        max_step_x = max_ax * dt
        max_step_y = max_ay * dt

        self.cmd_v[0] += np.clip(filtered_vx - self.cmd_v[0], -max_step_x, max_step_x)
        self.cmd_v[1] += np.clip(filtered_vy - self.cmd_v[1], -max_step_y, max_step_y)

        # 3. 速度截断
        max_vx = float(self.get_parameter('max_vx').value)
        max_vy = float(self.get_parameter('max_vy').value)
        vx_calc = np.clip(self.cmd_v[0], -max_vx, max_vx)
        vy_calc = np.clip(self.cmd_v[1], -max_vy, max_vy)

        # =========================================================
        # 4. 静摩擦力物理击穿 (纯平移版本)
        # =========================================================
        min_vx = float(self.get_parameter('min_vx').value)
        min_vy = float(self.get_parameter('min_vy').value)

        if not is_arrived:
            vx_pub = math.copysign(max(abs(vx_calc), min_vx), vx_calc) if abs(vx_calc) > 0.01 else 0.0
            vy_pub = math.copysign(max(abs(vy_calc), min_vy), vy_calc) if abs(vy_calc) > 0.01 else 0.0
        else:
            # 已到达，撤销击穿，允许丝滑归零
            vx_pub, vy_pub = vx_calc, vy_calc

        # 5. 反转映射与发布
        final_vx = -vx_pub if self.get_parameter('reverse_x_cmd').value else vx_pub
        final_vy = -vy_pub if self.get_parameter('reverse_y_cmd').value else vy_pub

        msg = Twist()
        msg.linear.x = float(final_vx)
        msg.linear.y = float(final_vy)
        msg.angular.z = 0.0  # 【彻底砍死 Wz，强制为 0】
        
        try:
            if rclpy.ok(): 
                self.pub_cmd_vel.publish(msg)
        except: pass

        self.get_logger().info(
            f"🎯 [雅可比纯平移] 1车:{math.degrees(self.theta_1):.1f}°(距0°) | 2车:{math.degrees(self.theta_2):.1f}°(距{math.degrees(target_2_angle_rad):.0f}°) | 发送 Vx:{final_vx:.2f}, Vy:{final_vy:.2f}",
            throttle_duration_sec=0.5
        )

    def stop_robot(self):
        # 丝滑软刹车
        self.cmd_v *= 0.8
        
        if np.linalg.norm(self.cmd_v) < 0.01:
            self.cmd_v = np.zeros(2)
            
        try:
            if rclpy.ok(): 
                msg = Twist()
                msg.linear.x = float(self.cmd_v[0])
                msg.linear.y = float(self.cmd_v[1])
                msg.angular.z = 0.0
                self.pub_cmd_vel.publish(msg)
        except: pass

def main(args=None):
    rclpy.init(args=args)
    node = Car3BearingFormationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()