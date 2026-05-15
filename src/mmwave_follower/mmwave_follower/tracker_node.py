import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker, MarkerArray
from interfaces.msg import ObjectsInfo
import math
import struct
import time
import numpy as np
#修改单独跟踪  稳定  需要启动控制节点


class MmwaveTrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_node')
        self.get_logger().info('毫米波雷达跟踪节点已启动 (V9.9 低速限制版)')

        # --- 1. 参数声明 ---
        self.declare_parameter('target_id', -1)
        self.declare_parameter('keep_distance', 0.5) 
        self.declare_parameter('chase_threshold', 1.2) 
        self.declare_parameter('cmd_vel_topic', '/car3/cmd_vel')

        # 调试开关
        self.declare_parameter('debug_cmd_monitor', True)

        # [方向设置] - 全部关闭反向，依赖 Controller 层进行标准化
        self.declare_parameter('reverse_cmd_x', False)  
        self.declare_parameter('reverse_cmd_y', False)  
        self.declare_parameter('reverse_cmd_z', False)

        # [新增] 直行漂移补偿
        self.declare_parameter('straight_bias_y', 0.0) 

        # PID 参数 - 恢复默认
        self.declare_parameter('kp_linear', 2.0)
        self.declare_parameter('ki_linear', 0.25)
        self.declare_parameter('kd_linear', 0.08)
        self.declare_parameter('kp_lateral', 2.5) 
        self.declare_parameter('ki_lateral', 0.20)
        self.declare_parameter('kd_lateral', 0.06)
        self.declare_parameter('kp_angular', 3.0)
        self.declare_parameter('linear_boost_x', 1.2)
        self.declare_parameter('min_effective_vx', 0.30)
        self.declare_parameter('linear_boost_y', 1.0)
        self.declare_parameter('min_effective_vy', 0.30)
        self.declare_parameter('min_effective_wz', 0.20)
        self.declare_parameter('angular_boost_z', 1.6)
        self.declare_parameter('integral_limit_x', 1.0)
        self.declare_parameter('integral_limit_y', 1.0)
        self.declare_parameter('angular_sign', -1.0)
        self.declare_parameter('fixed_turn_wz', 0.4)
        self.declare_parameter('heading_stop_deg', 20.0)
        self.declare_parameter('heading_resume_deg', 8.0)

        # 控制门限：减小“角度轻微抖动导致持续自转”
        self.declare_parameter('lat_deadband_m', 0.06)
        self.declare_parameter('yaw_enable_rad', 0.22)

        # 速度上限 - 恢复默认
        self.declare_parameter('max_normal_vx', 0.3) 
        self.declare_parameter('max_fast_vx', 0.45)
        self.declare_parameter('max_vy', 0.3)
        self.declare_parameter('max_wz', 0.4)

        # 最小启动速度 (死区门槛) - 恢复默认
        self.declare_parameter('min_start_vx', 0.15) 
        self.declare_parameter('min_start_vy', 0.15)
        self.declare_parameter('min_start_wz', 0.15)
        
        # 丢失搜索
        self.declare_parameter('enable_lost_search', False)
        self.declare_parameter('search_speed', 0.3)

        # 关闭直连底盘，必须通过 /controller/cmd_vel 通信
        self.declare_parameter('use_direct_board', False)

        #计数器
        self.miss_frame_count = 0
        self.max_miss_tolerance = 10

        # 滤波系数
        self.filter_alpha_coord = 0.25 #  0.15 ~ 0.3 之间，越小越平滑但会有延迟
        self.filter_alpha_cmd = 0.3

        # --- 2. 订阅与发布 ---
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub_fusion = self.create_subscription(ObjectsInfo, '/tracked_objects_3d', self.fusion_callback, qos)
        self.sub_pcl = self.create_subscription(PointCloud2, '/ti_mmwave/radar_scan_pcl', self.pcl_callback, qos)
        
        cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.pub_cmd_vel = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/tracker/debug_markers', 10)
        self.get_logger().info(f'cmd_vel publish topic: {cmd_vel_topic}')

        # --- 3. 核心状态 ---
        self.last_track_time = time.time()
        self.track_timeout = 2.0 
        self.is_lost = False

        self.target_x_rob = 0.0 
        self.target_y_rob = 0.0 

        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.last_ctrl_time = time.time()
        self.err_int_x = 0.0
        self.err_int_y = 0.0
        self.prev_err_x = 0.0
        self.prev_err_y = 0.0
        self.heading_align_mode = False

        self.locked_id = -1 
        
        self.create_timer(0.05, self.control_loop_callback)
        # 看门狗
        self.create_timer(0.05, self.watchdog_timer_callback)

    def control_loop_callback(self):
        """保留周期回调，当前逻辑由回调驱动，无需额外处理。"""
        pass

    def pcl_callback(self, msg):
        pass 

    def watchdog_timer_callback(self):
        """独立检测上游是否断流"""
        now = time.time()
        if now - self.last_track_time > self.track_timeout:
            if not self.is_lost:
                self.get_logger().warn(f"通信中断 (上游静默 > {self.track_timeout}s)，停车。", throttle_duration_sec=2.0)
                self.is_lost = True
            self.stop_robot()

    def fusion_callback(self, msg: ObjectsInfo):
        self.last_track_time = time.time()
        target_id_param = self.get_parameter('target_id').value

        candidates = []

        for obj in msg.objects:
            try:
                oid = int(getattr(obj, 'id', 0))
                if getattr(obj, 'position', None) is not None:
                    tx = float(obj.position.x)
                    ty = float(obj.position.y)
                    if tx > 0:
                        candidates.append({'id': oid, 'x': tx, 'y': ty})
            except Exception as e:
                self.get_logger().debug(f"fusion callback parse error: {e}")
                continue

        if not candidates and msg.objects:
            self.stop_robot()
            return

        target_obj = None
        if target_id_param != -1:
            for cand in candidates:
                if cand['id'] == target_id_param:
                    target_obj = cand
                    break
            if target_obj is None and candidates:
                self.stop_robot()
                return
        else:
            found_locked = False
            if self.locked_id != -1:
                for cand in candidates:
                    if cand['id'] == self.locked_id:
                        target_obj = cand
                        found_locked = True
                        break
            if not found_locked and candidates:
                candidates.sort(key=lambda x: x['x'])
                target_obj = candidates[0]
                self.locked_id = target_obj['id']


        if target_obj is None:
            self.miss_frame_count += 1
            if self.miss_frame_count > self.max_miss_tolerance:
                # 真正丢失超过 0.5 秒，才执行刹车
                if not self.is_lost:
                    self.get_logger().warn("目标丢失超过容忍时间，执行停车！")
                    self.is_lost = True
                self.stop_robot()
            else:
                # 只是短暂丢帧，让小车依靠惯性滑行 (速度缓慢衰减)
                self.cmd_vx *= 0.85
                self.cmd_vy *= 0.85
                self.cmd_wz *= 0.85
                
                # 组装滑行 Twist 消息并发布
                msg = Twist()
                rev_x = self.get_parameter('reverse_cmd_x').value
                rev_y = self.get_parameter('reverse_cmd_y').value
                rev_z = self.get_parameter('reverse_cmd_z').value
                msg.linear.x = -float(self.cmd_vx) if rev_x else float(self.cmd_vx)
                msg.linear.y = -float(self.cmd_vy) if rev_y else float(self.cmd_vy)
                msg.angular.z = -float(self.cmd_wz) if rev_z else float(self.cmd_wz)
                self.pub_cmd_vel.publish(msg)
            return

        # 如果找到了目标，重置丢失计数器
        self.miss_frame_count = 0
        self.is_lost = False
        # ----------------------------------------------------

        self.is_lost = False
        
        # 显式坐标约定: x向前为正, y向左为正
        raw_x_rob = target_obj['x']        
        raw_y_rob = target_obj['y'] 

        alpha = self.filter_alpha_coord
        self.target_x_rob = alpha * raw_x_rob + (1 - alpha) * self.target_x_rob
        self.target_y_rob = alpha * raw_y_rob + (1 - alpha) * self.target_y_rob

        self.calculate_velocity(self.target_x_rob, self.target_y_rob, target_obj['id'])

    def calculate_velocity(self, tx, ty, oid):
        keep_dist = self.get_parameter('keep_distance').value
        chase_thresh = self.get_parameter('chase_threshold').value
        lat_deadband = float(self.get_parameter('lat_deadband_m').value)
        yaw_enable_rad = float(self.get_parameter('yaw_enable_rad').value)
        linear_boost_x = float(self.get_parameter('linear_boost_x').value)
        min_effective_vx = float(self.get_parameter('min_effective_vx').value)
        linear_boost_y = float(self.get_parameter('linear_boost_y').value)
        min_effective_vy = float(self.get_parameter('min_effective_vy').value)
        min_effective_wz = float(self.get_parameter('min_effective_wz').value)
        angular_boost_z = float(self.get_parameter('angular_boost_z').value)
        fixed_turn_wz = float(self.get_parameter('fixed_turn_wz').value)
        heading_stop_deg = float(self.get_parameter('heading_stop_deg').value)
        heading_resume_deg = float(self.get_parameter('heading_resume_deg').value)
        kp_linear = float(self.get_parameter('kp_linear').value)
        ki_linear = float(self.get_parameter('ki_linear').value)
        kd_linear = float(self.get_parameter('kd_linear').value)
        kp_lateral = float(self.get_parameter('kp_lateral').value)
        ki_lateral = float(self.get_parameter('ki_lateral').value)
        kd_lateral = float(self.get_parameter('kd_lateral').value)
        int_lim_x = float(self.get_parameter('integral_limit_x').value)
        int_lim_y = float(self.get_parameter('integral_limit_y').value)
        angular_sign = float(self.get_parameter('angular_sign').value)
        
        if tx < 0.25:
            self.stop_robot()
            return

        now = time.time()
        dt = max(0.02, min(0.2, now - self.last_ctrl_time))
        self.last_ctrl_time = now

        # 1. 角度误差
        angle_err_rad = math.atan2(ty, tx)
        self.angle_err_deg = math.degrees(angle_err_rad)
        turn_suppression = max(0.25, 1.0 - abs(self.angle_err_deg) / 70.0)
        abs_angle_deg = abs(self.angle_err_deg)

        # 超过角度阈值先原地转头；对正后再恢复 x/y 跟随（带迟滞，避免抖动）
        if self.heading_align_mode:
            if abs_angle_deg <= heading_resume_deg:
                self.heading_align_mode = False
        else:
            if abs_angle_deg >= heading_stop_deg:
                self.heading_align_mode = True

        # 2. 前后 (Linear X)
        tgt_vx = 0.0
        err_dist = tx - keep_dist
        if (not self.heading_align_mode) and abs(err_dist) >= 0.12:
            self.err_int_x += err_dist * dt
            self.err_int_x = max(-int_lim_x, min(int_lim_x, self.err_int_x))
            derr_x = (err_dist - self.prev_err_x) / dt
            self.prev_err_x = err_dist

            tgt_vx = kp_linear * err_dist + ki_linear * self.err_int_x + kd_linear * derr_x
            tgt_vx *= linear_boost_x
            max_vx = self.get_parameter('max_fast_vx').value if tx > chase_thresh else self.get_parameter('max_normal_vx').value
            tgt_vx = max(-max_vx, min(max_vx, tgt_vx))
            if abs(tgt_vx) > 0.001 and abs(tgt_vx) < min_effective_vx:
                tgt_vx = math.copysign(min_effective_vx, tgt_vx)
        else:
            self.err_int_x *= 0.8

        # 3. 横向/转向协同：避免同时强烈 vy + wz 造成“打架”
        tgt_vy = 0.0
        tgt_wz = 0.0
        err_lat = ty

        # 侧偏很小时横向清零
        if self.heading_align_mode:
            tgt_vy = 0.0
            self.err_int_y *= 0.8
        elif abs(err_lat) < lat_deadband:
            tgt_vy = 0.0
            self.err_int_y *= 0.8
        else:
            self.err_int_y += err_lat * dt
            self.err_int_y = max(-int_lim_y, min(int_lim_y, self.err_int_y))
            derr_y = (err_lat - self.prev_err_y) / dt
            self.prev_err_y = err_lat

            tgt_vy = kp_lateral * err_lat + ki_lateral * self.err_int_y + kd_lateral * derr_y
            tgt_vy *= linear_boost_y
            tgt_vy *= turn_suppression
            max_vy = float(self.get_parameter('max_vy').value)
            tgt_vy = max(-max_vy, min(max_vy, tgt_vy))
            if abs(tgt_vy) > 0.001 and abs(tgt_vy) < min_effective_vy:
                tgt_vy = math.copysign(min_effective_vy, tgt_vy)

        # 先对正再跟随：仅在对正模式下执行转头
        if self.heading_align_mode:
            turn_dir = 1.0 if angle_err_rad >= 0.0 else -1.0
            tgt_wz = fixed_turn_wz * angular_sign * turn_dir
            # 对正模式强制停止平移
            tgt_vx = 0.0
            tgt_vy = 0.0
        else:
            tgt_wz = 0.0

        # 直行时允许少量横向补偿
        if abs(tgt_vx) > 0.05 and abs(tgt_vy) < 0.01:
            bias_y = self.get_parameter('straight_bias_y').value
            tgt_vy = max(-self.get_parameter('max_vy').value, min(self.get_parameter('max_vy').value, tgt_vy + bias_y))

        # 横移较大时轻微抑制自转，减少“边移边甩”
        if abs(tgt_vy) > 0.25:
            tgt_wz *= 0.5

        # 4. 限制每个周期的变化量，抑制抖动
        if self.heading_align_mode:
            # 对正阶段直接给固定角速度，不经过缓升和滤波
            self.cmd_vx = 0.0
            self.cmd_vy = 0.0
            self.cmd_wz = tgt_wz
        else:
            max_accel_x = 1.5 
            max_accel_y = 1.5
            max_accel_w = 2.0
            max_dvx = max_accel_x * dt
            max_dvy = max_accel_y * dt
            max_dwz = max_accel_w * dt
            tgt_vx = self.cmd_vx + max(-max_dvx, min(max_dvx, tgt_vx - self.cmd_vx))
            tgt_vy = self.cmd_vy + max(-max_dvy, min(max_dvy, tgt_vy - self.cmd_vy))
            tgt_wz = self.cmd_wz + max(-max_dwz, min(max_dwz, tgt_wz - self.cmd_wz))

            # 5. 滤波与发布
            alpha = self.filter_alpha_cmd
            self.cmd_vx = alpha * tgt_vx + (1 - alpha) * self.cmd_vx
            self.cmd_vy = alpha * tgt_vy + (1 - alpha) * self.cmd_vy
            self.cmd_wz = alpha * tgt_wz + (1 - alpha) * self.cmd_wz

        # 最终硬限幅，避免滤波和历史状态把速度带出目标区间
        max_vx_limit = float(self.get_parameter('max_fast_vx').value if tx > chase_thresh else self.get_parameter('max_normal_vx').value)
        max_vy_limit = float(self.get_parameter('max_vy').value)
        max_wz_limit = float(self.get_parameter('max_wz').value)
        self.cmd_vx = max(-max_vx_limit, min(max_vx_limit, self.cmd_vx))
        self.cmd_vy = max(-max_vy_limit, min(max_vy_limit, self.cmd_vy))
        self.cmd_wz = max(-max_wz_limit, min(max_wz_limit, self.cmd_wz))

        # 清掉极小残差，避免下游最小启动速度逻辑把“尾巴”放大
        if abs(self.cmd_vx) < 0.01:
            self.cmd_vx = 0.0
        if abs(self.cmd_vy) < 0.01:
            self.cmd_vy = 0.0
        if abs(self.cmd_wz) < 0.01:
            self.cmd_wz = 0.0

        msg = Twist()
        rev_x = self.get_parameter('reverse_cmd_x').value
        rev_y = self.get_parameter('reverse_cmd_y').value
        rev_z = self.get_parameter('reverse_cmd_z').value

        # 标准输出，不反转
        msg.linear.x = -float(self.cmd_vx) if rev_x else float(self.cmd_vx)
        msg.linear.y = -float(self.cmd_vy) if rev_y else float(self.cmd_vy)
        msg.angular.z = -float(self.cmd_wz) if rev_z else float(self.cmd_wz)
        
        self.pub_cmd_vel.publish(msg)

        if self.get_parameter('debug_cmd_monitor').value:
            self.get_logger().info(
                f"ID:{oid} | 距{self.target_x_rob:.2f}m 侧偏{self.target_y_rob:.2f}m | 发送x:{msg.linear.x:.2f}, y:{msg.linear.y:.2f}, z:{msg.angular.z:.2f}",
                throttle_duration_sec=0.2
            )

    def stop_robot(self):
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.pub_cmd_vel.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = MmwaveTrackerNode()
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