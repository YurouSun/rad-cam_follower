#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
advanced_object_localizer.py
一个功能完备的ROS2节点，用于高级3D物体定位与速度估计。

集成功能:
1.  **鲁棒的3D定位**: 融合深度图直接测量与基于像素尺寸的估算。
2.  **动态智能融合**: 根据深度数据质量和一致性，自动选择最优融合策略。
3.  **稳定的相对速度估计**: 计算目标相对于机器人本体的平滑速度，并以TwistStamped格式发布。
4.  **机器人中心坐标**: 所有输出均通过TF2转换到'base_link'坐标系，便于机器人控制。
5.  **状态化跟踪**: 跟踪检测到的物体，以实现时间滤波和速度计算。
"""

# 导入必要的库
import math
import time
from typing import Dict, Tuple, Optional
import json

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PointStamped, TwistStamped, Vector3, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import String
import message_filters
from cv_bridge import CvBridge, CvBridgeError

import tf2_ros
from tf2_ros import TransformException
import tf2_geometry_msgs
import rclpy.duration

# ---------------------------- 工具函数 (Helper Functions) ----------------------------

def depth_units_scale(ros_image: Image, cv_img: np.ndarray) -> float:
    if cv_img.dtype == np.uint16: return 1.0 / 1000.0
    if cv_img.dtype in [np.float32, np.float64]: return 1.0
    v = float(np.nanmedian(cv_img)); return 1.0 / 1000.0 if v > 10.0 else 1.0

def robust_roi_depth(depth_patch: np.ndarray, min_valid_pts: int = 30) -> Tuple[bool, float]:
    if depth_patch.size == 0: return False, float("nan")
    arr = depth_patch.astype(np.float32); arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size < min_valid_pts: return False, float("nan")
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75); iqr = max(q3 - q1, 1e-6)
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    arr = arr[(arr >= lo) & (arr <= hi)]
    if arr.size == 0: return False, float("nan")
    return True, float(np.median(arr))

def pixel_depth_by_size(fx: float, fy: float, bbox_w: float, bbox_h: float,
                        real_w: Optional[float], real_h: Optional[float]) -> Optional[float]:
    if real_w and bbox_w > 2.0: return fx * real_w / bbox_w
    if real_h and bbox_h > 2.0: return fy * real_h / bbox_h
    return None

def ema(prev: Optional[float], now: float, alpha: float) -> float:
    return now if prev is None else (alpha * now + (1.0 - alpha) * prev)

def assess_depth_quality(patch: np.ndarray, scale_to_m: float, min_valid_pts: int) -> Tuple[str, float]:
    arr = patch.astype(np.float32)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size < min_valid_pts:
        return 'LOW', float('inf')
    arr_m = arr * scale_to_m
    q1, q3 = np.percentile(arr_m, 25), np.percentile(arr_m, 75)
    iqr_m = q3 - q1
    if iqr_m < 0.02: return 'HIGH', iqr_m
    elif iqr_m < 0.08: return 'MEDIUM', iqr_m
    else: return 'LOW', iqr_m

# ---------------------------- 主节点 (Main Node Class) ----------------------------

class AdvancedObjectLocalizer(Node):
    def __init__(self):
        super().__init__('advanced_object_localizer_node')
        self.bridge = CvBridge()
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # 参数声明
        self.declare_parameter('camera_info_topic', '/ascamera/camera_publisher/rgb0/camera_info')
        self.declare_parameter('depth_topic', '/ascamera/camera_publisher/depth0/image_raw')
        self.declare_parameter('detection_topic', '/yolo_result')
        self.declare_parameter('reliable_depth_min', 0.3)
        self.declare_parameter('reliable_depth_max', 3.0)
        self.declare_parameter('roi_scale', 0.25)
        self.declare_parameter('roi_min', 5)
        self.declare_parameter('roi_max', 40)
        self.declare_parameter('min_valid_depth_points', 30)
        self.declare_parameter('use_temporal_filter', True)
        self.declare_parameter('temporal_alpha', 0.45)
        self.declare_parameter('jump_reject_threshold', 1.0)
        self.declare_parameter('fusion_bias_to_depth', 0.7)
        self.declare_parameter('velocity_alpha', 0.3)
        self.declare_parameter('log_level', 1)
        self.declare_parameter('object_real_sizes', '{"car":0.165}')
        self.declare_parameter('object_real_heights', '{"car":0.1}')
        
        # 加载参数
        self.reliable_min = float(self.get_parameter('reliable_depth_min').value)
        self.reliable_max = float(self.get_parameter('reliable_depth_max').value)
        self.roi_scale = float(self.get_parameter('roi_scale').value)
        self.roi_min = int(self.get_parameter('roi_min').value)
        self.roi_max = int(self.get_parameter('roi_max').value)
        self.min_valid_pts = int(self.get_parameter('min_valid_depth_points').value)
        self.use_temporal = bool(self.get_parameter('use_temporal_filter').value)
        self.temporal_alpha = float(self.get_parameter('temporal_alpha').value)
        self.jump_thresh = float(self.get_parameter('jump_reject_threshold').value)
        self.fusion_bias = float(self.get_parameter('fusion_bias_to_depth').value)
        self.velocity_alpha = float(self.get_parameter('velocity_alpha').value)
        self.log_level = int(self.get_parameter('log_level').value)
        try:
            self.real_sizes = json.loads(self.get_parameter('object_real_sizes').value)
            self.real_heights = json.loads(self.get_parameter('object_real_heights').value)
        except Exception as e:
            self.get_logger().error(f"解析 'object_real_sizes' 参数失败: {e}")
            self.real_sizes, self.real_heights = {}, {}
            
        # 订阅/发布
        self.camera_intrinsics = None
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value, self.camera_info_callback, qos_profile=qos_profile_sensor_data
        )
        det_sub = message_filters.Subscriber(self, Detection2DArray, self.get_parameter('detection_topic').value, qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(self, Image, self.get_parameter('depth_topic').value, qos_profile=qos_profile_sensor_data)
        self.ts = message_filters.ApproximateTimeSynchronizer([det_sub, depth_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.cb_sync)
        # 发布器定义 - 你需要的三个核心功能
        self.point_pub = self.create_publisher(PointStamped, '/detected_objects_3d', 10)        # 功能2: 物体位置
        self.marker_pub = self.create_publisher(Marker, '/object_marker', 10)
        self.velocity_pub = self.create_publisher(TwistStamped, '/detected_objects_velocity', 10)  # 功能3: 物体速度
        self.object_info_pub = self.create_publisher(String, '/detected_objects_info', 10)     # 功能1: 物体名称及综合信息
        
        # 状态管理字典
        self.prev_z_by_key: Dict[Tuple[int, int], float] = {}
        self.object_states: Dict[Tuple[int, int], Dict] = {}
        self._last_info_ts = 0.0
        self.get_logger().info(f"高级3D定位与速度估计节点已启动...")

    def camera_info_callback(self, msg: CameraInfo): 
        if self.camera_intrinsics is None:
            self.camera_intrinsics = msg
            self.get_logger().info(f"获取相机内参: fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}, cx={msg.k[2]:.2f}, cy={msg.k[5]:.2f}")
            self.destroy_subscription(self.camera_info_sub)

    def cb_sync(self, det_msg: Detection2DArray, depth_msg: Image):
        if self.camera_intrinsics is None: return
        try:
            depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        except CvBridgeError: return

        # 清理旧的物体状态
        current_time_sec = self.get_clock().now().nanoseconds / 1e9
        keys_to_delete = [k for k, v in self.object_states.items() if current_time_sec - rclpy.time.Time.from_msg(v['timestamp']).nanoseconds / 1e9 > 2.0]
        for key in keys_to_delete: del self.object_states[key]

        scale_to_m = depth_units_scale(depth_msg, depth_cv)
        h, w = depth_cv.shape[:2]
        fx, fy, cx, cy = self.camera_intrinsics.k[0], self.camera_intrinsics.k[4], self.camera_intrinsics.k[2], self.camera_intrinsics.k[5]

        for det in det_msg.detections:
            u, v = int(det.bbox.center.position.x), int(det.bbox.center.position.y)
            bw, bh = float(det.bbox.size_x), float(det.bbox.size_y)
            if not (0 <= u < w and 0 <= v < h): continue
            
            # 步骤1: 获取两种独立的深度估计
            half = int(max(self.roi_min, min(self.roi_max, min(bw, bh) * self.roi_scale)))
            u0, u1, v0, v1 = max(0, u-half), min(w, u+half), max(0, v-half), min(h, v+half)
            patch = depth_cv[v0:v1, u0:u1]
            valid_depth, depth_raw = robust_roi_depth(patch, min_valid_pts=self.min_valid_pts)
            z_depth_m = depth_raw * scale_to_m if valid_depth else None
            
            class_id = det.results[0].hypothesis.class_id if det.results else None
            real_w, real_h = (self.real_sizes.get(class_id), self.real_heights.get(class_id)) if class_id else (None, None)
            z_px_m = pixel_depth_by_size(fx, fy, bw, bh, real_w, real_h)
            
            # 步骤2: 动态融合决策树
            z_fused, key = None, (u // 20, v // 20)
            AGREEMENT_THRESHOLD = 0.2 

            if z_depth_m is not None:
                depth_quality, iqr = assess_depth_quality(patch, scale_to_m, self.min_valid_pts)
                discrepancy = abs(z_depth_m - z_px_m) if z_px_m is not None else float('inf')
            else:
                depth_quality, iqr, discrepancy = 'LOW', float('inf'), float('inf')
            
            if depth_quality == 'HIGH':
                if discrepancy < AGREEMENT_THRESHOLD:
                    z_fused = z_depth_m
                else:
                    z_fused = self.prev_z_by_key.get(key)
            elif depth_quality == 'MEDIUM':
                conf_depth = self._confidence_depth(z_depth_m, patch, scale_to_m)
                conf_px = self._confidence_pixel(z_px_m, bw, bh) if z_px_m is not None else 0.0
                if conf_depth > 0 and conf_px > 0:
                    w_d, w_p = self.fusion_bias * conf_depth, (1.0 - self.fusion_bias) * conf_px
                    z_fused = (w_d * z_depth_m + w_p * z_px_m) / max(w_d + w_p, 1e-6)
                else:
                    z_fused = z_depth_m if conf_depth > 0 else z_px_m
            else:
                z_fused = z_px_m
            
            if z_fused is None: continue

            # 步骤3: 时间滤波与坐标转换
            prev_z = self.prev_z_by_key.get(key)
            if prev_z is not None and abs(z_fused - prev_z) > self.jump_thresh: z_fused = prev_z
            if self.use_temporal: z_fused = ema(prev_z, z_fused, self.temporal_alpha)
            self.prev_z_by_key[key] = z_fused
            
            x_cam, y_cam, z_cam = (u - cx) * z_fused / fx, (v - cy) * z_fused / fy, z_fused
            
            # 步骤4: TF2转换和发布
            source_frame, target_frame = depth_msg.header.frame_id, 'base_link'
            try:
                # 尝试获取最新可用的变换时间，避免时间外推错误
                try:
                    # 首先尝试获取最新可用的变换
                    latest_transform_time = self.tf_buffer.get_latest_common_time(source_frame, target_frame)
                    use_time = latest_transform_time
                except Exception:
                    # 如果获取最新时间失败，使用当前时间
                    use_time = rclpy.time.Time()
                
                point_in_camera = PointStamped()
                point_in_camera.header.stamp = use_time.to_msg()
                point_in_camera.header.frame_id = source_frame
                point_in_camera.point.x, point_in_camera.point.y, point_in_camera.point.z = x_cam, y_cam, z_cam
                
                # 使用合适的超时时间进行转换
                transformed_point = self.tf_buffer.transform(point_in_camera, target_frame, timeout=rclpy.duration.Duration(seconds=0.1))
                x_base, y_base, z_base = transformed_point.point.x, transformed_point.point.y, transformed_point.point.z
                # 发布位置 - 功能2: 物体位置
                now_msg_time = self.get_clock().now().to_msg()
                pt_base = PointStamped()
                pt_base.header.stamp = now_msg_time
                pt_base.header.frame_id = target_frame
                pt_base.point.x, pt_base.point.y, pt_base.point.z = x_base, y_base, z_base
                self.point_pub.publish(pt_base)
                
                # 功能1: 发布检测物体的名称和综合信息
                object_name = class_id if class_id else "unknown"
                object_info = {
                    "timestamp": now_msg_time.sec + now_msg_time.nanosec * 1e-9,
                    "object_name": object_name,
                    "position": {
                        "x": float(x_base),
                        "y": float(y_base), 
                        "z": float(z_base),
                        "frame_id": target_frame
                    },
                    "bbox": {
                        "center_x": float(u),
                        "center_y": float(v),
                        "width": float(bw),
                        "height": float(bh)
                    },
                    "depth_method": "depth_sensor" if valid_depth else "pixel_estimation",
                    "depth_quality": depth_quality if z_depth_m is not None else "unavailable"
                }
                
                info_msg = String()
                info_msg.data = json.dumps(object_info)
                self.object_info_pub.publish(info_msg)
                
                
                mk = Marker(ns='detected_objects', id=0, type=Marker.SPHERE, action=Marker.ADD)
                mk.header.stamp = now_msg_time
                mk.header.frame_id = target_frame
                mk.pose.position = pt_base.point
                mk.pose.orientation.w = 1.0
                mk.scale = Vector3(x=0.1, y=0.1, z=0.1)
                mk.color.a, mk.color.r = 0.8, 1.0
                self.marker_pub.publish(mk)

                # 步骤5: 速度计算与发布
                current_time = self.get_clock().now()
                last_state = self.object_states.get(key)
                
                # 控制台输出 - 便于调试和查看
                if self._should_log():
                    self.get_logger().info(f" 检测到物体: {object_name}")
                    self.get_logger().info(f" 位置: x={x_base:.3f}, y={y_base:.3f}, z={z_base:.3f} ({target_frame})")
                    if last_state and 'velocity' in last_state:
                        vel = last_state['velocity']
                        speed = math.sqrt(vel.x**2 + vel.y**2)
                        self.get_logger().info(f" 速度: vx={vel.x:.3f}, vy={vel.y:.3f}, 速率={speed:.3f} m/s")
                current_velocity = Vector3()

                if last_state:
                    dt = (current_time - rclpy.time.Time.from_msg(last_state['timestamp'])).nanoseconds / 1e9
                    if dt > 1e-3:
                        dx = pt_base.point.x - last_state['position'].x
                        dy = pt_base.point.y - last_state['position'].y
                        vx_raw, vy_raw = dx / dt, dy / dt
                        
                        last_vel = last_state['velocity']
                        current_velocity.x = ema(last_vel.x, vx_raw, self.velocity_alpha)
                        current_velocity.y = ema(last_vel.y, vy_raw, self.velocity_alpha)
                        
                        # 功能3: 发布物体速度信息
                        vel_msg = TwistStamped()
                        vel_msg.header.stamp = now_msg_time
                        vel_msg.header.frame_id = target_frame
                        vel_msg.twist.linear = current_velocity
                        self.velocity_pub.publish(vel_msg)
                        
                        # 发布包含物体名称的速度综合信息
                        speed_magnitude = math.sqrt(current_velocity.x**2 + current_velocity.y**2)
                        velocity_info = {
                            "timestamp": now_msg_time.sec + now_msg_time.nanosec * 1e-9,
                            "object_name": object_name,
                            "velocity": {
                                "vx": float(current_velocity.x),
                                "vy": float(current_velocity.y),
                                "speed": float(speed_magnitude),
                                "frame_id": target_frame
                            },
                            "tracking_dt": float(dt)
                        }
                        
                        velocity_info_msg = String()
                        velocity_info_msg.data = json.dumps(velocity_info)
                        # 可以创建专门的速度信息话题，或者合并到物体信息中
                
                # 步骤6: 更新状态
                self.object_states[key] = {
                    'position': pt_base.point, 
                    'timestamp': now_msg_time,
                    'velocity': current_velocity if last_state else Vector3(),
                    'object_name': object_name,
                    'class_id': class_id
                }

            except TransformException as ex:
                if self._should_log(): 
                    self.get_logger().warn(f'坐标转换失败，使用相机坐标系发布: {ex}')
                
                # 当TF转换失败时，直接在相机坐标系下发布点位置作为后备方案
                try:
                    now_msg_time = self.get_clock().now().to_msg()
                    pt_camera = PointStamped()
                    pt_camera.header.stamp = now_msg_time
                    pt_camera.header.frame_id = source_frame  # 使用相机坐标系
                    pt_camera.point.x, pt_camera.point.y, pt_camera.point.z = x_cam, y_cam, z_cam
                    self.point_pub.publish(pt_camera)
                    
                    # 在后备方案中也发布物体信息
                    object_name = class_id if class_id else "unknown"
                    fallback_info = {
                        "timestamp": now_msg_time.sec + now_msg_time.nanosec * 1e-9,
                        "object_name": object_name,
                        "position": {
                            "x": float(x_cam),
                            "y": float(y_cam), 
                            "z": float(z_cam),
                            "frame_id": source_frame
                        },
                        "note": "使用相机坐标系 - TF转换失败"
                    }
                    
                    info_msg = String()
                    info_msg.data = json.dumps(fallback_info)
                    self.object_info_pub.publish(info_msg)
                    
                    # 也在相机坐标系下发布标记
                    mk = Marker(ns='detected_objects', id=0, type=Marker.SPHERE, action=Marker.ADD)
                    mk.header.stamp = now_msg_time
                    mk.header.frame_id = source_frame
                    mk.pose.position = pt_camera.point
                    mk.pose.orientation.w = 1.0
                    mk.scale = Vector3(x=0.1, y=0.1, z=0.1)
                    mk.color.a, mk.color.r = 0.8, 1.0
                    mk.color.g = 0.5  # 使用不同颜色表示这是相机坐标系的点
                    self.marker_pub.publish(mk)
                    
                    if self._should_log():
                        self.get_logger().info(f"在相机坐标系发布点: x={x_cam:.3f}, y={y_cam:.3f}, z={z_cam:.3f}")
                except Exception as fallback_ex:
                    if self._should_log():
                        self.get_logger().error(f'后备发布方案也失败: {fallback_ex}')
    
    def _confidence_depth(self, z_m, patch, scale_to_m):
        c_range = 1.0 if (self.reliable_min <= z_m <= self.reliable_max) else max(0.2, 1.0 - 0.25 * max(0.0, z_m - self.reliable_max))
        arr = patch.astype(np.float32); arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size < self.min_valid_pts: return 0.0
        arr_m = arr * scale_to_m; q1, q3 = np.percentile(arr_m, 25), np.percentile(arr_m, 75)
        iqr = max(q3 - q1, 1e-6); c_noise = 1.0 / (1.0 + iqr)
        return float(np.clip(0.6 * c_range + 0.4 * c_noise, 0.0, 1.0))

    def _confidence_pixel(self, z_m, bw, bh):
        size_score = np.clip(min(bw, bh) / 100.0, 0.0, 1.0)
        far_penalty = 1.0 / (1.0 + max(0.0, z_m - self.reliable_max))
        return float(np.clip(0.7 * size_score + 0.3 * far_penalty, 0.0, 1.0))

    def _should_log(self):
        if self.log_level <= 0: return False
        if self.log_level >= 2: return True
        now = time.time()
        if now - self._last_info_ts > 0.5: self._last_info_ts = now; return True
        return False

def main(args=None):
    rclpy.init(args=args)
    node = AdvancedObjectLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()