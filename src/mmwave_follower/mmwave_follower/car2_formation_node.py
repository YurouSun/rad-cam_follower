#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
import math
import time
import numpy as np
from collections import deque

try:
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
except Exception:
    PointCloud2 = None
    point_cloud2 = None

class Car2FormationNode(Node):
    def __init__(self):
        super().__init__('car2_formation_node')
        self.get_logger().info('🚗 Car 2 节点 [Y轴正反馈修复版：打断死亡螺旋 + 强化防撞墙 + 20Hz动态速度平滑]')

        # === 1. 物理底盘修正参数 ===
        # 【致敬真理】X轴是正常的(负数后退)，但Y轴必须反转！(负数误差需要向左移动)
        self.declare_parameter('reverse_x_cmd', False)
        self.declare_parameter('reverse_y_cmd', True)  # <-- 核心修复：终结向右乱跑的正反馈！
        self.declare_parameter('reverse_wz_cmd', False) 

        # === 2. 编队目标参数 ===
        # 目标：前车在自身前方 0.50m，右边 0.50m (机器人向右为Y正)
        self.declare_parameter('formation_x_distance', 0.50)  
        self.declare_parameter('formation_y_distance', 0.50)  
        
        # === 3. 核心增益参数 ===
        self.declare_parameter('kp_x', 1.8)   
        self.declare_parameter('kp_y', 2.0)   
        
        # === 4. 机器人运动学限制 ===
        self.declare_parameter('max_vx', 0.80) 
        self.declare_parameter('max_vy', 0.80)
        
        # 静摩擦击穿电压：保证麦轮横移绝对不会转不动
        self.declare_parameter('min_vx', 0.3) 
        self.declare_parameter('min_vy', 0.3) 
        
        self.declare_parameter('arrival_x_tolerance', 0.05) 
        self.declare_parameter('arrival_y_tolerance', 0.05) 
        
        self.declare_parameter('pcl_range_min_m', 0.20) 
        self.declare_parameter('pcl_range_max_m', 1.50)  # [核心修改] 将最大侦测距离从 3.0m 缩短到 1.5m，无视后方背景杂物
        self.declare_parameter('pcl_cluster_eps_m', 0.16)
        self.declare_parameter('pcl_cluster_min_pts', 4)

        # 上一帧与当前帧的最大允许跳变距离（类似 IoU 阈值）
        self.declare_parameter('max_jump_dist_m', 0.60) # [适度放宽] 把单帧跳变容忍从0.45m放宽到0.60m，防止移动稍快就丢失历史
        
        self.declare_parameter('imu_topic', '/ros_robot_controller/imu_raw')
        
        # === 5. 速度平滑与频率匹配参数 (新增) ===
        self.declare_parameter('velocity_smooth_tau', 0.2) # 速度一阶低通时间常数(秒)
        self.declare_parameter('max_acc_x', 0.8)           # X方向最大加速度(m/s^2)
        self.declare_parameter('max_acc_y', 0.8)           # Y方向最大加速度(m/s^2)

        self.current_gyro_z = 0.0 
        self.yaw_integral = 0.0 
        
        self.z_pcl = np.array([0.0, 0.0])
        self.t_pcl = 0.0
        
        self.cmd_v = np.array([0.0, 0.0]) 
        self.last_ctrl_time = time.time()
        
        # === 6. ROS 通信设置 ===
        self.cb_group = ReentrantCallbackGroup()
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        if PointCloud2 is not None and point_cloud2 is not None:
            self.sub_pcl = self.create_subscription(PointCloud2, '/ti_mmwave/radar_scan_pcl', self.cb_radar_pcl, qos, callback_group=self.cb_group)
            
        self.sub_imu = self.create_subscription(Imu, self.get_parameter('imu_topic').value, self.cb_imu, 10, callback_group=self.cb_group)
        self.pub_cmd_vel = self.create_publisher(Twist, '/car2/cmd_vel', 10)

        self.declare_parameter('control_period_sec', 0.05) # 20Hz 控制周期
        self.create_timer(float(self.get_parameter('control_period_sec').value), self.control_loop, callback_group=self.cb_group)

    def cb_imu(self, msg):
        self.current_gyro_z = msg.angular_velocity.z

    @staticmethod
    def _euclidean_clusters(points_xy, eps, min_pts):
        n = len(points_xy)
        if n == 0: return []
        visited = [False] * n
        clusters = []
        for i in range(n):
            if visited[i]: continue
            visited[i] = True
            queue = deque([i])
            cluster_idx = [i]
            while queue:
                cur = queue.popleft()
                px, py = points_xy[cur]
                for j in range(n):
                    if visited[j]: continue
                    qx, qy = points_xy[j]
                    if math.hypot(px - qx, py - qy) <= eps:
                        visited[j] = True
                        queue.append(j)
                        cluster_idx.append(j)
            if len(cluster_idx) >= min_pts:
                clusters.append(cluster_idx)
        return clusters

    def cb_radar_pcl(self, msg):
        if point_cloud2 is None: return
        valid_points = []
        range_min = float(self.get_parameter('pcl_range_min_m').value)
        range_max = float(self.get_parameter('pcl_range_max_m').value)
        cluster_eps = float(self.get_parameter('pcl_cluster_eps_m').value)
        cluster_min_pts = int(self.get_parameter('pcl_cluster_min_pts').value)
        max_jump = float(self.get_parameter('max_jump_dist_m').value)

        try:
            for p in point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
                raw_x, raw_y = float(p[0]), float(p[1]) 
                d = math.hypot(raw_x, raw_y)
                if d < range_min or d > range_max: continue
                if raw_y <= 0.0: continue
                # [核心修改] 将横向视野从 ±1.5m(全宽3m) 严苛收缩到 ±0.8m(全宽1.6m)，避免错跟平行的人或墙面
                if abs(raw_x) > 0.8: continue 
                valid_points.append((d, raw_x, raw_y))
        except Exception: return
            
        if len(valid_points) < cluster_min_pts: return

        points_xy = [(p[1], p[2]) for p in valid_points]
        clusters = self._euclidean_clusters(points_xy, cluster_eps, cluster_min_pts)
        if not clusters: return

        has_history = (self.t_pcl > 0 and (time.time() - self.t_pcl) < 1.0)

        best_score = float('inf')
        best_candidate = None

        for idx_list in clusters:
            cx = sum(points_xy[i][0] for i in idx_list) / len(idx_list) # 雷达X (向右)
            cy = sum(points_xy[i][1] for i in idx_list) / len(idx_list) # 雷达Y (向前)
            cdist = math.hypot(cx, cy)
            
            # 正交映射：X前向，Y向右。绝不加负号！
            candidate = np.array([cy, cx])

            if has_history:
                # 【时序波门关联】
                dist_to_last = np.linalg.norm(candidate - self.z_pcl)
                if dist_to_last > max_jump:
                    # 只要超过0.45m跳变，直接丢弃，不被后方杂波诱骗
                    continue
                score = dist_to_last * 2.0 - len(idx_list) * 0.01 
            else:
                score = cdist - len(idx_list) * 0.05 
            
            if score < best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is not None:
            now_time = time.time()
            if has_history:
                # 【修改点1：雷达传感器动态滤波适配频率】使用真实雷达时间间隔 dt_pcl 来动态调整权重
                dt_pcl = max(0.01, min(0.2, now_time - self.t_pcl))
                tau_pcl = 0.15 
                alpha_pcl = dt_pcl / (tau_pcl + dt_pcl)
                self.z_pcl = (1.0 - alpha_pcl) * self.z_pcl + alpha_pcl * best_candidate
            else:
                self.z_pcl = best_candidate
            self.t_pcl = now_time

    def get_local_sensor_target(self):
        now = time.time()
        if (now - self.t_pcl) < 0.8: 
            return self.z_pcl
        return None

    def control_loop(self):
        now = time.time()
        # 计算距离上一帧的确切时间步长 dt
        dt = max(0.01, min(0.1, now - self.last_ctrl_time))
        self.last_ctrl_time = now
        
        z_current = self.get_local_sensor_target()
        
        if z_current is None:
            self.get_logger().warn("🛑 目标丢失或跳变异常，主动刹车等待", throttle_duration_sec=1.5)
            self.stop_robot()
            return

        x_star = float(self.get_parameter('formation_x_distance').value)
        y_star = float(self.get_parameter('formation_y_distance').value)
        
        # 误差计算：当前位置 - 目标位置。正数代表离得不够远(需要后退/往左)。
        err_x = z_current[0] - x_star
        err_y = z_current[1] - y_star
        
        tol_x = float(self.get_parameter('arrival_x_tolerance').value)
        tol_y = float(self.get_parameter('arrival_y_tolerance').value)
        
        if abs(err_x) < tol_x and abs(err_y) < tol_y:
            self.stop_robot()
            self.get_logger().info("✅ 完美落位阵位，待命中", throttle_duration_sec=1.5)
            return

        kp_x = float(self.get_parameter('kp_x').value)
        kp_y = float(self.get_parameter('kp_y').value)
        
        # 1. 计算未经平滑的目标速度
        target_vx = kp_x * err_x
        target_vy = kp_y * err_y

        # 【修改点2：速度综合平滑升级 - 自适应20Hz定时器】
        tau_v = float(self.get_parameter('velocity_smooth_tau').value)
        max_acc_x = float(self.get_parameter('max_acc_x').value)
        max_acc_y = float(self.get_parameter('max_acc_y').value)

        # 1. 动态自适应一阶低通滤波 (时间常数 tau_v，解决定时器抖动问题)
        alpha_v = dt / (tau_v + dt)
        filtered_vx = (1.0 - alpha_v) * self.cmd_v[0] + alpha_v * target_vx
        filtered_vy = (1.0 - alpha_v) * self.cmd_v[1] + alpha_v * target_vy

        # 2. 爬升率限制 (Slew Rate Limit)，严格限制本帧(本 dt 时间)最大可改变的速度差，防加速冲击
        max_step_x = max_acc_x * dt
        max_step_y = max_acc_y * dt
        
        self.cmd_v[0] += np.clip(filtered_vx - self.cmd_v[0], -max_step_x, max_step_x)
        self.cmd_v[1] += np.clip(filtered_vy - self.cmd_v[1], -max_step_y, max_step_y)
        # =========================================================
        
        max_vx = float(self.get_parameter('max_vx').value)
        max_vy = float(self.get_parameter('max_vy').value)
        min_vx = float(self.get_parameter('min_vx').value)
        min_vy = float(self.get_parameter('min_vy').value)

        vx_calc = np.clip(self.cmd_v[0], -max_vx, max_vx)
        vy_calc = np.clip(self.cmd_v[1], -max_vy, max_vy)

        # 静摩擦击穿
        vx_pub = math.copysign(max(abs(vx_calc), min_vx), vx_calc) if abs(err_x) > tol_x else 0.0
        vy_pub = math.copysign(max(abs(vy_calc), min_vy), vy_calc) if abs(err_y) > tol_y else 0.0

        # 【升级版紧急防撞】：距离低于 0.35m，无条件全速退避，杜绝麦轮打滑漂移前冲！
        if z_current[0] < 0.35:
            self.get_logger().warn(f"⚠️ 触发雷达极度防撞边界({z_current[0]:.2f}m)！强制后退避险！", throttle_duration_sec=0.5)
            vx_pub = -0.35

        # 航向防偏积分器：死死锁住车头
        if abs(vx_pub) > 0 or abs(vy_pub) > 0:
            self.yaw_integral += self.current_gyro_z * dt
            wz_pub = 2.0 * self.yaw_integral + 0.5 * self.current_gyro_z
            wz_pub = np.clip(wz_pub, -0.6, 0.6)
        else:
            wz_pub = 0.0
            self.yaw_integral *= 0.9 

        # 【真理时刻】应用翻转。Y轴反转后，向右的错误驱动将被硬生生掰成正确的向左滑行！
        final_vx = -vx_pub if self.get_parameter('reverse_x_cmd').value else vx_pub
        final_vy = -vy_pub if self.get_parameter('reverse_y_cmd').value else vy_pub
        final_wz = -wz_pub if self.get_parameter('reverse_wz_cmd').value else wz_pub

        msg = Twist()
        msg.linear.x = float(final_vx)
        msg.linear.y = float(final_vy)
        msg.angular.z = float(final_wz) 
        self.pub_cmd_vel.publish(msg)
        
        # 增加日志直观显示动作方向
        action_x = "后退" if final_vx < 0 else "前进" if final_vx > 0 else "静止"
        action_y = "左移" if final_vy > 0 else "右移" if final_vy < 0 else "静止"
        
        self.get_logger().info(
            f"🚀 {action_x}{action_y} | 前距:{z_current[0]:.2f}m(差{err_x:.2f}) | 右距:{z_current[1]:.2f}m(差{err_y:.2f}) | 发送Vx:{final_vx:.2f}, Vy:{final_vy:.2f}",
            throttle_duration_sec=0.5
        )

    def stop_robot(self):
        self.cmd_v = np.array([0.0, 0.0])
        self.yaw_integral = 0.0 
        try:
            if rclpy.ok(): self.pub_cmd_vel.publish(Twist())
        except: pass

def main(args=None):
    rclpy.init(args=args)
    node = Car2FormationNode()
    rclpy.spin(node)
    node.stop_robot()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()