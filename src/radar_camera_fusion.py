#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多目标融合跟踪节点 (Vision + Radar Fusion) - V2.0 智能回退版
策略: 优先匹配雷达; 若雷达无匹配且视觉深度可靠(<3.0m)，则回退使用视觉深度。
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from interfaces.msg import ObjectsInfo, ObjectInfo
from cv_bridge import CvBridge
import numpy as np
import time
import json
import struct
import threading
import cv2

# --- 标定文件路径配置 ---
CALIB_CONFIG = {
    "INTRINSICS_PATH": "/home/ubuntu/ros2_ws/src/config/camera_intrinsics.json",
    "EXTRINSICS_PATH": "/home/ubuntu/ros2_ws/src/config/extrinsics.json"
}

class MOTTrackerFusionNode(Node):
    def __init__(self):
        super().__init__('mot_tracker_fusion_node')
        self.bridge = CvBridge()

        # --- 1. 参数声明 ---
        self.declare_parameter('rgb_topic', '/ascamera/camera_publisher/rgb0/image')
        self.declare_parameter('radar_topic', '/ti_mmwave/radar_scan_pcl')
        self.declare_parameter('det_topic', '/tracked_objects')
        self.declare_parameter('publish_topic', '/tracked_objects_3d')
        
        # 融合匹配阈值 (像素)
        self.declare_parameter('fusion_match_pixel_thresh', 50.0)
        self.fusion_match_thresh = self.get_parameter('fusion_match_pixel_thresh').value

        # [新增] 视觉深度信任阈值 (米)
        # 当雷达没匹配上，且视觉深度小于此值时，信任视觉数据
        self.declare_parameter('visual_trust_dist', 3.0) 

        # --- 2. 加载标定参数 ---
        self.K = None
        self.D = None
        self.rvec = None
        self.tvec = None
        self.load_calibration()

        # --- 3. 订阅与发布 ---
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        
        self.sub_rgb = self.create_subscription(Image, self.get_parameter('rgb_topic').value, self.on_rgb, qos)
        self.sub_det = self.create_subscription(ObjectsInfo, self.get_parameter('det_topic').value, self.on_detection, qos_profile_sensor_data)
        self.sub_radar = self.create_subscription(PointCloud2, self.get_parameter('radar_topic').value, self.on_radar, qos)

        self.pub_tracks = self.create_publisher(ObjectsInfo, self.get_parameter('publish_topic').value, 10)
        self.pub_vis = self.create_publisher(Image, '/tracked_objects/image_fusion', 10)

        # --- 4. 状态变量 ---
        self.last_rgb = None
        self.last_radar_points = None 
        self.last_radar_img_pts = None 
        self.tracks = [] 
        self.det_lock = threading.Lock()
        
        self.last_det_time = 0.0
        self.det_timeout = 0.2 

        self.create_timer(0.05, self.on_timer)
        self.get_logger().info("Fusion Tracker V2.0 Started (Radar Dominant + Visual Fallback).")

    def load_calibration(self):
        try:
            with open(CALIB_CONFIG["INTRINSICS_PATH"], 'r') as f:
                d = json.load(f)
                self.K = np.array(d["camera_matrix"], dtype=np.float32)
                self.D = np.array(d["distortion_coefficients"], dtype=np.float32)
            
            with open(CALIB_CONFIG["EXTRINSICS_PATH"], 'r') as f:
                d = json.load(f)
                self.rvec = np.array(d["rvec"], dtype=np.float32)
                self.tvec = np.array(d["tvec"], dtype=np.float32)
            
            self.get_logger().info("标定文件加载成功")
        except Exception as e:
            self.get_logger().error(f"标定文件加载失败: {e}")

    def on_rgb(self, msg):
        try:
            self.last_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            pass

    def on_radar(self, msg):
        points = []
        try:
            offset_x, offset_y, offset_z = -1, -1, -1
            for f in msg.fields:
                if f.name == 'x': offset_x = f.offset
                if f.name == 'y': offset_y = f.offset
                if f.name == 'z': offset_z = f.offset
            
            if offset_x == -1: return

            data = msg.data
            step = msg.point_step
            n_points = msg.width * msg.height
            
            for i in range(n_points):
                start = i * step
                x = struct.unpack_from('<f', data, start + offset_x)[0]
                y = struct.unpack_from('<f', data, start + offset_y)[0]
                z = struct.unpack_from('<f', data, start + offset_z)[0]
                points.append([x, y, z])
            
            self.last_radar_points = np.array(points, dtype=np.float32)
            
        except Exception as e:
            self.get_logger().warn_once(f"Radar parse error: {e}")

    def on_detection(self, msg: ObjectsInfo):
        with self.det_lock:
            self.last_det_time = time.time()
            dedup_by_id = {}
            for obj in msg.objects:
                if len(obj.box) < 4: continue
                x1, y1, x2, y2 = obj.box[:4]
                cls_name = obj.class_name or 'obj'

                # 解析 ID
                track_id = None
                if '_' in cls_name:
                    try:
                        base, tail = cls_name.rsplit('_', 1)
                        if tail.isdigit():
                            track_id = int(tail)
                            cls_name = base # 去掉ID后缀，保留名字
                    except:
                        pass
                
                if track_id is None:
                    track_id = int(obj.score) # 某些节点把ID放在score里

                track_id_out = obj.id if getattr(obj, 'id', -1) != -1 else int(obj.score)
                item = {
                    'id': track_id_out,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'score': float(obj.score),
                    'raw_class': str(obj.class_name), # 保留原始字符串用于解析视觉深度
                    'clean_class': str(cls_name),
                    'position': obj.position, # 保存上游传来的三维坐标对象
                    'radar_data': None,
                    'vis_data': None
                }

                prev = dedup_by_id.get(track_id_out)
                if prev is None:
                    dedup_by_id[track_id_out] = item
                else:
                    # 同一ID多条输入时保留置信度更高的一条，避免一帧双框
                    if item['score'] >= prev['score']:
                        dedup_by_id[track_id_out] = item

            self.tracks = list(dedup_by_id.values())

    def fuse_radar_to_tracks(self):
        # 1. 尝试雷达投影与匹配
        has_radar = (self.last_radar_points is not None and self.rvec is not None)
        
        if has_radar:
            img_pts, _ = cv2.projectPoints(self.last_radar_points, self.rvec, self.tvec, self.K, self.D)
            img_pts = img_pts.reshape(-1, 2)
            self.last_radar_img_pts = img_pts
        else:
            self.last_radar_img_pts = None

        visual_trust_dist = self.get_parameter('visual_trust_dist').value

        for tr in self.tracks:
            # === Step A: 优先尝试雷达匹配 ===
            matched_radar = False
            if has_radar:
                x1, y1, x2, y2 = tr['bbox']
                # 简单的包围盒内点检查
                mask_x = (img_pts[:, 0] >= x1) & (img_pts[:, 0] <= x2)
                mask_y = (img_pts[:, 1] >= y1) & (img_pts[:, 1] <= y2)
                matched_indices = np.where(mask_x & mask_y)[0]
                
                if len(matched_indices) > 0:
                    matched_3d = self.last_radar_points[matched_indices]
                    dists_y = matched_3d[:, 1] 
                    valid_mask = dists_y > 0.2
                    
                    if np.any(valid_mask):
                        valid_dists = dists_y[valid_mask]
                        valid_pts = matched_3d[valid_mask]
                        
                        # 【修改】获取视觉深度作为参考
                        has_vis = False
                        vx, vy, vz = 0.0, 0.0, 0.0
                        if 'position' in tr and getattr(tr['position'], 'y', 0.0) != 0.0:
                            has_vis = True
                            vx = tr['position'].x
                            vy = tr['position'].y
                            vz = tr['position'].z
                        
                        best_pt = None
                        if has_vis and vy > 0.1:
                            # 找出当前能找到的最接近匹配
                            diffs = np.abs(valid_dists - vy)
                            idx_best = np.argmin(diffs)
                            
                            # **关键修复：如果最近偏差小于1.5m，则认为有效；否则直接丢弃全部雷达点（保留最佳值为None），以防匹配到背景墙！**
                            if diffs[idx_best] < 1.5:
                                best_pt = valid_pts[idx_best]
                        
                        if best_pt is None:
                            # 如果实在无法匹配物理视觉深度（或者没视觉），使用原始方法：盲取最靠近摄影机的雷达点
                            if not has_vis:
                                idx_min = np.argsort(valid_dists)[0]
                                best_pt = valid_pts[idx_min]
                            else:
                                # 如果有视觉，且雷达点偏离太原，说明雷达点根本打偏到墙上了，强制抛弃，交给视觉降级
                                pass
                                
                        if best_pt is not None:
                            tr['radar_data'] = {
                                'x': float(best_pt[0]),
                                'y': float(best_pt[1]),
                                'z': float(best_pt[2]),
                                'count': len(matched_indices)
                            }
                            matched_radar = True

            # === Step B: 雷达未匹配，尝试视觉回退 ===
            if not matched_radar:
                has_vis = False
                vx, vy, vz = 0.0, 0.0, 0.0
                if 'position' in tr and getattr(tr['position'], 'y', 0.0) != 0.0:
                    has_vis = True
                    vx = float(tr['position'].x)
                    vy = float(tr['position'].y)
                    vz = float(tr['position'].z)
                
                # 如果解析成功，且距离在信任范围内 (例如 < 3.0m)
                if has_vis and 0.1 < vy < visual_trust_dist:
                    tr['vis_data'] = {
                        'x': float(vx), # 视觉坐标系通常也是 右x 前y
                        'y': float(vy),
                        'z': float(vz)
                    }

    def on_timer(self):
        if time.time() - self.last_det_time > self.det_timeout:
            with self.det_lock:
                if self.tracks: self.tracks = []
        
        with self.det_lock:
            self.fuse_radar_to_tracks()
            
            # 发布逻辑
            out_msg = ObjectsInfo()
            for tr in self.tracks:
                obj = ObjectInfo()
                obj.score = float(tr['id']) # ID传下去
                obj.box = tr['bbox']
                
                # 构建下游能识别的 class_name 格式
                # 标准格式: Name_X{}_Y{}_Z{}
                
                base_name = tr['clean_class']
                # 去掉可能存在的原始 xy 后缀，保持干净
                if ' ' in base_name: base_name = base_name.split(' ')[0]

                final_x, final_y, final_z = 0, 0, 0
                source_tag = ""

                if tr['radar_data']:
                    # 优先使用雷达
                    final_x = tr['radar_data']['x']
                    final_y = tr['radar_data']['y']
                    final_z = tr['radar_data']['z']
                    source_tag = "_Rad"
                elif tr['vis_data']:
                    # 降级使用视觉
                    final_x = tr['vis_data']['x']
                    final_y = tr['vis_data']['y']
                    final_z = tr['vis_data']['z']
                    source_tag = "_Vis"
                
                if source_tag:
                    # 按照新版统一规范，写入 position，不再硬编码到 class_name 中
                    obj.position.x = float(final_y) # Y前
                    obj.position.y = -float(final_x) # X右 取反为左
                    obj.position.z = float(final_z)
                    obj.id = int(tr['id'])
                    obj.class_name = f"{base_name}{source_tag}"
                else:
                    # 既无雷达也无视觉深度，保持原样（TrackerNode会识别为丢失）
                    obj.id = int(tr['id'])
                    obj.class_name = base_name

                out_msg.objects.append(obj)
                
                # 打印日志 (仅打印有数据的)
                if source_tag:
                    print(f"ID:{tr['id']} | Src:{source_tag} | pos:[{obj.position.x:.2f},{obj.position.y:.2f},{obj.position.z:.2f}]")

            self.pub_tracks.publish(out_msg)

            # 可视化：在 RGB 上叠加目标框、ID与融合来源，并叠加雷达投影点
            if self.last_rgb is not None:
                vis = self.last_rgb.copy()

                if self.last_radar_img_pts is not None:
                    h, w = vis.shape[:2]
                    for p in self.last_radar_img_pts:
                        px, py = int(p[0]), int(p[1])
                        if 0 <= px < w and 0 <= py < h:
                            cv2.circle(vis, (px, py), 2, (0, 255, 255), -1)

                for tr in self.tracks:
                    x1, y1, x2, y2 = tr['bbox']
                    src = 'None'
                    color = (120, 120, 120)
                    if tr['radar_data']:
                        src = 'Rad'
                        color = (0, 255, 0)
                    elif tr['vis_data']:
                        src = 'Vis'
                        color = (0, 165, 255)

                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{tr['id']} {tr['clean_class']} {src}"
                    cv2.putText(vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                try:
                    vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
                    self.pub_vis.publish(vis_msg)
                except Exception as e:
                    self.get_logger().warn_once(f"Publish vis image failed: {e}")

def main():
    rclpy.init()
    node = MOTTrackerFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()