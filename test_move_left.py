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

class MmwaveTrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_node')
        self.get_logger().info('毫米波雷达跟踪节点已启动 (V8.0 物理校准 & 漂移补偿版)')

        # --- 1. 参数声明 ---
        self.declare_parameter('target_id', -1)
        self.declare_parameter('keep_distance', 0.5) 
        self.declare_parameter('chase_threshold', 1.2) 

        # [关键校准]
        # 基于你的实际反馈：发送负数x时车才前进，发送负数y时车才左移
        self.declare_parameter('reverse_cmd_x', True)  # 必须为True
        self.declare_parameter('reverse_cmd_y', True)  # 必须为True
        self.declare_parameter('reverse_cmd_z', False)

        # [新增] 直行漂移补偿 (Straight Bias)
        # 如果车直行时总是不由自主向左偏，设为 -0.05 (向右补)
        # 如果向右偏，设为 0.05 (向左补)
        self.declare_parameter('straight_bias_y', 0.0) 

        # PID 参数 (针对 5-6Hz 视觉优化)
        # 视觉帧率较低，P值不宜过大，否则容易在两帧之间画龙
        self.declare_parameter('kp_linear', 0.6)   
        self.declare_parameter('kp_lateral', 1.0)  
        self.declare_parameter('kp_angular', 1.2)  

        # [速度匹配]
        # 视觉 5-6Hz => 约0.18秒一帧。
        # 建议最高速控制在 0.35m/s 左右，这样帧间隔内移动约 6cm，雷达能跟上修正
        self.declare_parameter('max_normal_vx', 0.35)
        self.declare_parameter('max_fast_vx', 0.5)
        self.declare_parameter('max_vy', 0.25)
        self.declare_parameter('max_wz', 0.6)

        # 最小启动速度 (克服静摩擦)
        self.declare_parameter('min_start_vx', 0.12) 
        self.declare_parameter('min_start_vy', 0.12)
        self.declare_parameter('min_start_wz', 0.20)
        
        # 丢失搜索
        self.declare_parameter('enable_lost_search', False)
        self.declare_parameter('search_speed', 0.2)

        # 滤波系数
        self.filter_alpha_coord = 0.5 
        self.filter_alpha_cmd = 0.3  

        # --- 2. 订阅与发布 ---
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub_fusion = self.create_subscription(ObjectsInfo, '/tracked_objects_3d', self.fusion_callback, qos)
        self.sub_pcl = self.create_subscription(PointCloud2, '/ti_mmwave/radar_scan_pcl', self.pcl_callback, qos)
        
        self.pub_cmd_vel = self.create_publisher(Twist, '/controller/cmd_vel', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/tracker/debug_markers', 10)

        # --- 3. 核心状态 ---
        self.last_track_time = time.time()
        self.track_timeout = 2.0 
        self.is_lost = False

        self.target_x_rob = 0.0 
        self.target_y_rob = 0.0 

        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0

        self.locked_id = -1 
        
        # 看门狗
        self.create_timer(0.1, self.watchdog_timer_callback)

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

    def parse_object_data(self, class_name):
        """兼容多种数据格式的解析器"""
        rx, ry = 0.0, 0.0
        valid = False

        try:
            # 格式 A
            if '_X' in class_name and '_Y' in class_name:
                parts = class_name.split('_')
                for p in parts:
                    if p.startswith('X'): rx = float(p[1:])
                    if p.startswith('Y'): ry = float(p[1:])
                valid = True

            # 格式 B
            elif 'x=' in class_name and 'y=' in class_name:
                start_x = class_name.find('x=') + 2
                end_x = class_name.find(',', start_x)
                if end_x == -1: end_x = class_name.find(' ', start_x) 
                if end_x == -1: end_x = len(class_name) 
                rx_str = class_name[start_x:end_x]
                
                start_y = class_name.find('y=') + 2
                end_y = class_name.find(',', start_y)
                if end_y == -1: end_y = class_name.find(' ', start_y)
                if end_y == -1: end_y = len(class_name)
                ry_str = class_name[start_y:end_y]

                rx = float(rx_str)
                ry = float(ry_str)
                valid = True
                
        except Exception:
            valid = False
        
        return valid, rx, ry

    def fusion_callback(self, msg: ObjectsInfo):
        self.last_track_time = time.time()
        target_id_param = self.get_parameter('target_id').value

        # 1. 解析数据
        candidates = []
        raw_names = [] 

        for obj in msg.objects:
            raw_names.append(obj.class_name)
            try:
                oid = int(obj.score)
                valid_parse, rx, ry = self.parse_object_data(obj.class_name)
                
                if valid_parse and ry > 0: 
                    candidates.append({'id': oid, 'rx': rx, 'ry': ry})
            except:
                continue

        # [诊断] 收到数据但解析为空
        if not candidates and msg.objects:
            if 'x=' not in raw_names[0] and '_X' not in raw_names[0]:
                self.get_logger().warn(f"等待有效数据... ({raw_names[0]})", throttle_duration_sec=2.0)
            self.stop_robot()
            return

        # 2. 目标选择逻辑
        target_obj = None
        
        if target_id_param != -1:
            for cand in candidates:
                if cand['id'] == target_id_param:
                    target_obj = cand
                    break
            
            if target_obj is None and candidates:
                current_ids = [c['id'] for c in candidates]
                if time.time() % 2.0 < 0.2:
                    self.get_logger().info(f"寻找ID {target_id_param}... 可见: {current_ids}")
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
                candidates.sort(key=lambda x: x['ry'])
                target_obj = candidates[0]
                self.locked_id = target_obj['id']

        if target_obj is None:
            if not self.is_lost:
                self.is_lost = True
            self.stop_robot()
            return

        # 3. 正常跟踪
        self.is_lost = False
        
        # 坐标转换
        raw_x_rob = target_obj['ry']        
        raw_y_rob = -1.0 * target_obj['rx'] 

        # 滤波
        alpha = self.filter_alpha_coord
        self.target_x_rob = alpha * raw_x_rob + (1 - alpha) * self.target_x_rob
        self.target_y_rob = alpha * raw_y_rob + (1 - alpha) * self.target_y_rob

        self.calculate_velocity(self.target_x_rob, self.target_y_rob, target_obj['id'])

    def calculate_velocity(self, tx, ty, oid):
        keep_dist = self.get_parameter('keep_distance').value
        chase_thresh = self.get_parameter('chase_threshold').value
        
        min_start_vx = self.get_parameter('min_start_vx').value
        min_start_vy = self.get_parameter('min_start_vy').value
        min_start_wz = self.get_parameter('min_start_wz').value
        
        # 安全停车
        if tx < 0.25:
            self.stop_robot()
            self.get_logger().warn(f"距离过近 ({tx:.2f}m)，刹车！", throttle_duration_sec=1.0)
            return

        # 1. 角度
        angle_err_rad = math.atan2(ty, tx)
        self.angle_err_deg = math.degrees(angle_err_rad)
        turn_suppression = max(0.0, 1.0 - abs(self.angle_err_deg) / 45.0) 
        
        # 2. 前后 (Linear X)
        err_dist = tx - keep_dist
        if abs(err_dist) < 0.05: # 死区
            tgt_vx = 0.0
        else:
            tgt_vx = err_dist * self.get_parameter('kp_linear').value
            limit_vx = self.get_parameter('max_fast_vx').value if tx > chase_thresh else self.get_parameter('max_normal_vx').value
            tgt_vx *= turn_suppression
            
            if abs(tgt_vx) > 0.001 and abs(tgt_vx) < min_start_vx:
                tgt_vx = math.copysign(min_start_vx, tgt_vx)
            tgt_vx = max(-limit_vx, min(limit_vx, tgt_vx))

        # 3. 横向 (Linear Y)
        err_lat = ty 
        if abs(err_lat) < 0.12: # 死区
            tgt_vy = 0.0
        else:
            tgt_vy = err_lat * self.get_parameter('kp_lateral').value
            tgt_vy *= turn_suppression
            if abs(tgt_vy) > 0.001 and abs(tgt_vy) < min_start_vy:
                tgt_vy = math.copysign(min_start_vy, tgt_vy)
            tgt_vy = max(-self.get_parameter('max_vy').value, min(self.get_parameter('max_vy').value, tgt_vy))

        # [漂移补偿] 当车在直行时，添加反向微调力
        # 如果车子总是向左滑，straight_bias_y 设为 -0.0x
        if abs(tgt_vx) > 0.05 and abs(tgt_vy) < 0.01:
            bias_y = self.get_parameter('straight_bias_y').value
            tgt_vy += bias_y

        # 4. 转向 (Angular Z)
        if abs(angle_err_rad) < 0.15: 
            tgt_wz = 0.0
        else:
            tgt_wz = angle_err_rad * self.get_parameter('kp_angular').value
            if abs(tgt_wz) > 0.001 and abs(tgt_wz) < min_start_wz:
                tgt_wz = math.copysign(min_start_wz, tgt_wz)
            tgt_wz = max(-self.get_parameter('max_wz').value, min(self.get_parameter('max_wz').value, tgt_wz))

        # 5. 滤波与发布
        alpha = self.filter_alpha_cmd
        self.cmd_vx = alpha * tgt_vx + (1 - alpha) * self.cmd_vx
        self.cmd_vy = alpha * tgt_vy + (1 - alpha) * self.cmd_vy
        self.cmd_wz = alpha * tgt_wz + (1 - alpha) * self.cmd_wz

        msg = Twist()
        rev_x = self.get_parameter('reverse_cmd_x').value
        rev_y = self.get_parameter('reverse_cmd_y').value
        rev_z = self.get_parameter('reverse_cmd_z').value

        # 反转逻辑
        msg.linear.x = -float(self.cmd_vx) if rev_x else float(self.cmd_vx)
        msg.linear.y = -float(self.cmd_vy) if rev_y else float(self.cmd_vy)
        msg.angular.z = -float(self.cmd_wz) if rev_z else float(self.cmd_wz)
        
        self.final_msg_x = msg.linear.x
        self.pub_cmd_vel.publish(msg)

        # 诊断
        intent_x = "前进" if self.cmd_vx > 0.01 else ("后退" if self.cmd_vx < -0.01 else "停")
        intent_y = "左移" if self.cmd_vy > 0.01 else ("右移" if self.cmd_vy < -0.01 else "直")
        intent_z = "左转" if self.cmd_wz > 0.01 else ("右转" if self.cmd_wz < -0.01 else "正")

        self.get_logger().info(
            f"ID:{oid} | 距{self.target_x_rob:.2f}m | "
            f"意图:[{intent_x}] -> 发送x:{msg.linear.x:.2f} (修正版)",
            throttle_duration_sec=0.5
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