import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
from ti_mmwave_rospkg_msgs.msg import RadarTrackArray
from interfaces.msg import ObjectsInfo
import math
import time

class MmwaveTrackerNode2(Node):
    def __init__(self):
        super().__init__('tracker_node2')
        self.get_logger().info('毫米波雷达跟踪节点已启动 (V12.0 平滑抗顿挫版)')

        # --- 1. 参数声明 ---
        self.declare_parameter('target_id', -1)
        self.declare_parameter('keep_distance', 0.5) 
        self.declare_parameter('chase_threshold', 1.2) 
        self.declare_parameter('cmd_vel_topic', '/car3/cmd_vel')
        self.declare_parameter('debug_cmd_monitor', True)
        
        # 方向设置
        self.declare_parameter('reverse_cmd_x', False)  
        self.declare_parameter('reverse_cmd_y', False)  

        # PID 参数 (移除所有角速度参数)
        self.declare_parameter('kp_linear', 2.0)
        self.declare_parameter('ki_linear', 0.25)
        self.declare_parameter('kd_linear', 0.08)
        self.declare_parameter('kp_lateral', 2.5) 
        self.declare_parameter('ki_lateral', 0.20)
        self.declare_parameter('kd_lateral', 0.06)
        
        self.declare_parameter('linear_boost_x', 1.2)
        self.declare_parameter('min_effective_vx', 0.30)
        self.declare_parameter('linear_boost_y', 1.0)
        self.declare_parameter('min_effective_vy', 0.30)
        
        # 速度上限
        self.declare_parameter('max_normal_vx', 0.9) 
        self.declare_parameter('max_fast_vx', 1.35)
        self.declare_parameter('max_vy', 0.9)
        
        self.declare_parameter('lat_deadband_m', 0.06)
        self.declare_parameter('integral_limit_x', 1.0)
        self.declare_parameter('integral_limit_y', 1.0)

        # 计数器与滤波
        self.miss_frame_count = 0
        self.max_miss_tolerance = 15  # 放大容忍度到约 0.75 秒
        self.filter_alpha_coord = 0.15 # [修改] 降低坐标平滑系数，增加滤波程度
        self.filter_alpha_cmd = 0.15   # [修改] 降低速度控制平滑系数，减缓速度剧烈跳变

        # --- 2. 订阅与发布 ---
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub_fusion = self.create_subscription(ObjectsInfo, '/tracked_objects_3d', self.fusion_callback, qos)
        self.sub_radar_track = self.create_subscription(RadarTrackArray, '/ti_mmwave/radar_track_array', self.radar_track_callback, 10)
        
        cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.pub_cmd_vel = self.create_publisher(Twist, cmd_vel_topic, 10)

        # --- 3. 核心状态 ---
        self.last_track_time = time.time()
        self.track_timeout = 2.0 
        self.is_lost = False

        self.target_x_rob = 0.0 
        self.target_y_rob = 0.0 
        
        # 冗余融合专属状态
        self.latest_radar_tracks = []
        self.last_known_valid_x = 0.0
        self.last_known_valid_y = 0.0

        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.last_ctrl_time = time.time()
        self.err_int_x = 0.0
        self.err_int_y = 0.0
        self.prev_err_x = 0.0
        self.prev_err_y = 0.0
        self.locked_id = -1 
        
        self.create_timer(0.05, self.watchdog_timer_callback)

    def watchdog_timer_callback(self):
        now = time.time()
        if now - self.last_track_time > self.track_timeout:
            if not self.is_lost:
                self.get_logger().warn(f"所有感知均中断 > {self.track_timeout}s，停车保护。")
                self.is_lost = True
            self.stop_robot()

    def radar_track_callback(self, msg: RadarTrackArray):
        tracks = []
        if msg.track:
            for t in msg.track:
                rx = t.posy
                ry = -t.posx
                if rx > 0:
                    tracks.append({'x': rx, 'y': ry})
        self.latest_radar_tracks = tracks

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
            except Exception:
                continue

        target_obj = None
        if target_id_param != -1:
            for cand in candidates:
                if cand['id'] == target_id_param:
                    target_obj = cand
                    break
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

        # ====================================================
        # 【强力冗余降级逻辑】
        # ====================================================
        if target_obj is None:
            closest_radar_dist = float('inf')
            best_radar_track = None
            
            for rt in self.latest_radar_tracks:
                dist_to_last_known = math.hypot(rt['x'] - self.last_known_valid_x, rt['y'] - self.last_known_valid_y)
                
                # 放宽搜索半径至 2.5 米，应对突发的大幅人工干扰推移
                if dist_to_last_known < 2.5 and dist_to_last_known < closest_radar_dist:
                    closest_radar_dist = dist_to_last_known
                    best_radar_track = rt
            
            if best_radar_track is not None:
                target_obj = best_radar_track
                self.filter_alpha_coord = 0.05 # [修改] 雷达噪点大，坐标平滑系数降到极低，防止车体发抖
                self.get_logger().info("视觉丢失！纯雷达盲跟接管中！", throttle_duration_sec=1.0)
            else:
                self.miss_frame_count += 1
                if self.miss_frame_count > self.max_miss_tolerance:
                    if not self.is_lost:
                        self.get_logger().warn("目标彻底丢失，执行停车！")
                        self.is_lost = True
                    self.stop_robot()
                else:
                    # [修改] 缓冲期：平移惯性衰减变得更加平缓 (0.85 -> 0.92)
                    self.cmd_vx *= 0.92
                    self.cmd_vy *= 0.92
                    self._publish_current_cmd()
                return
        else:
            self.filter_alpha_coord = 0.15 # [修改] 视觉恢复正常权重（调低）

        # ====================================================

        self.miss_frame_count = 0
        self.is_lost = False
        
        raw_x_rob = target_obj['x']        
        raw_y_rob = target_obj['y'] 

        # 【核心修复】：无论数据来自视觉还是雷达，永远更新 last_known_valid
        # 这样搜索圆环才会跟着目标一起移动，不会造成二次丢失！
        self.last_known_valid_x = raw_x_rob
        self.last_known_valid_y = raw_y_rob

        alpha = self.filter_alpha_coord
        self.target_x_rob = alpha * raw_x_rob + (1 - alpha) * self.target_x_rob
        self.target_y_rob = alpha * raw_y_rob + (1 - alpha) * self.target_y_rob

        self.calculate_velocity(self.target_x_rob, self.target_y_rob, target_obj.get('id', 'RADAR'))

    def _publish_current_cmd(self):
        msg = Twist()
        rev_x = self.get_parameter('reverse_cmd_x').value
        rev_y = self.get_parameter('reverse_cmd_y').value
        
        msg.linear.x = -float(self.cmd_vx) if rev_x else float(self.cmd_vx)
        msg.linear.y = -float(self.cmd_vy) if rev_y else float(self.cmd_vy)
        
        # 严格切除 Z 轴角速度输出，纯平移控制
        msg.angular.z = 0.0 
        
        self.pub_cmd_vel.publish(msg)

    def calculate_velocity(self, tx, ty, oid):
        keep_dist = self.get_parameter('keep_distance').value
        chase_thresh = self.get_parameter('chase_threshold').value
        lat_deadband = float(self.get_parameter('lat_deadband_m').value)
        linear_boost_x = float(self.get_parameter('linear_boost_x').value)
        min_effective_vx = float(self.get_parameter('min_effective_vx').value)
        linear_boost_y = float(self.get_parameter('linear_boost_y').value)
        min_effective_vy = float(self.get_parameter('min_effective_vy').value)
        
        kp_linear = float(self.get_parameter('kp_linear').value)
        ki_linear = float(self.get_parameter('ki_linear').value)
        kd_linear = float(self.get_parameter('kd_linear').value)
        kp_lateral = float(self.get_parameter('kp_lateral').value)
        ki_lateral = float(self.get_parameter('ki_lateral').value)
        kd_lateral = float(self.get_parameter('kd_lateral').value)
        int_lim_x = float(self.get_parameter('integral_limit_x').value)
        int_lim_y = float(self.get_parameter('integral_limit_y').value)
        
        if tx < 0.25:
            self.stop_robot()
            return

        now = time.time()
        dt = max(0.02, min(0.2, now - self.last_ctrl_time))
        self.last_ctrl_time = now

        # 1. 前后控制 (Linear X)
        tgt_vx = 0.0
        err_dist = tx - keep_dist
        if abs(err_dist) >= 0.12:
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

        # 2. 横向控制 (Linear Y)
        tgt_vy = 0.0
        err_lat = ty

        if abs(err_lat) < lat_deadband:
            tgt_vy = 0.0
            self.err_int_y *= 0.8
        else:
            self.err_int_y += err_lat * dt
            self.err_int_y = max(-int_lim_y, min(int_lim_y, self.err_int_y))
            derr_y = (err_lat - self.prev_err_y) / dt
            self.prev_err_y = err_lat

            tgt_vy = kp_lateral * err_lat + ki_lateral * self.err_int_y + kd_lateral * derr_y
            tgt_vy *= linear_boost_y
            max_vy = float(self.get_parameter('max_vy').value)
            tgt_vy = max(-max_vy, min(max_vy, tgt_vy))
            if abs(tgt_vy) > 0.001 and abs(tgt_vy) < min_effective_vy:
                tgt_vy = math.copysign(min_effective_vy, tgt_vy)

        # 3. 物理加速度限幅与滤波 (纯平移)
        max_accel_x = 3.0 
        max_accel_y = 3.0
        max_dvx = max_accel_x * dt
        max_dvy = max_accel_y * dt
        
        tgt_vx = self.cmd_vx + max(-max_dvx, min(max_dvx, tgt_vx - self.cmd_vx))
        tgt_vy = self.cmd_vy + max(-max_dvy, min(max_dvy, tgt_vy - self.cmd_vy))

        alpha = self.filter_alpha_cmd
        self.cmd_vx = alpha * tgt_vx + (1 - alpha) * self.cmd_vx
        self.cmd_vy = alpha * tgt_vy + (1 - alpha) * self.cmd_vy

        # 最终限幅与死区清零
        max_vx_limit = float(self.get_parameter('max_fast_vx').value if tx > chase_thresh else self.get_parameter('max_normal_vx').value)
        max_vy_limit = float(self.get_parameter('max_vy').value)
        self.cmd_vx = max(-max_vx_limit, min(max_vx_limit, self.cmd_vx))
        self.cmd_vy = max(-max_vy_limit, min(max_vy_limit, self.cmd_vy))

        if abs(self.cmd_vx) < 0.01: self.cmd_vx = 0.0
        if abs(self.cmd_vy) < 0.01: self.cmd_vy = 0.0

        self._publish_current_cmd()

        if self.get_parameter('debug_cmd_monitor').value:
            self.get_logger().info(
                f"ID:{oid} | 距{self.target_x_rob:.2f}m 侧偏{self.target_y_rob:.2f}m | 发送x:{self.cmd_vx:.2f}, y:{self.cmd_vy:.2f}, z:0.00",
                throttle_duration_sec=0.2
            )

    def stop_robot(self):
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.pub_cmd_vel.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = MmwaveTrackerNode2()
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