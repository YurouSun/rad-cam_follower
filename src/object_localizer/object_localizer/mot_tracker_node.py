#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from interfaces.msg import ObjectsInfo, ObjectInfo
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import time
import cv2
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# ==========================================
# --- 用户配置区域 (在此处添加新颜色) ---
# ==========================================
COLOR_CONFIG = {
    'RED': {
        'id': 1,
        'ranges': [
            (np.array([0, 100, 80]), np.array([10, 255, 255])),
            (np.array([170, 100, 80]), np.array([180, 255, 255]))
        ]
    },
    'BLUE': {
        'id': 2,
        'ranges': [
            (np.array([100, 120, 70]), np.array([130, 255, 255]))
        ]
    },
    'YELLOW': {
        'id': 3,
        'ranges': [
            (np.array([20, 100, 100]), np.array([35, 255, 255]))
        ]
    },
}

# --- 工具函数 ---

def iou_batch(bboxes1, bboxes2):
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    return wh / (area1 + area2 - wh + 1e-6)

def extract_spatial_feat(image, bbox):
    # 为节省树莓派算力，对于尺寸太小的目标跳过特征提取
    if image is None: return None
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
    if x2 - x1 < 16 or y2 - y1 < 16: return None # 忽略过小目标
    
    crop = image[y1:y2, x1:x2]
    # 极速降采样，降低直方图计算量
    crop = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_c, w_c = hsv.shape[:2]
    h_half, w_half = h_c // 2, w_c // 2
    segments = [hsv[:h_half, :w_half], hsv[:h_half, w_half:], hsv[h_half:, :w_half], hsv[h_half:, w_half:]]
    features = []
    bins = [8, 4, 4]; ranges = [0, 180, 0, 256, 0, 256]
    for seg in segments:
        hist = cv2.calcHist([seg], [0, 1, 2], None, bins, ranges)
        cv2.normalize(hist, hist)
        features.append(hist.flatten())
    full_feature = np.concatenate(features)
    cv2.normalize(full_feature, full_feature)
    return full_feature.flatten()

# 移除 Tracker 中的 detect_color_tag 以节省树莓派 CPU，统一依赖 YOLO 节点的识别结果

def get_depth_dist(depth_img, bbox):
    if depth_img is None: return None
    x1, y1, x2, y2 = map(int, bbox)
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    h, w = depth_img.shape[:2]
    if cx < 0 or cx >= w or cy < 0 or cy >= h: return None
    roi_size = 5
    x_min = max(0, cx - roi_size); x_max = min(w, cx + roi_size)
    y_min = max(0, cy - roi_size); y_max = min(h, cy + roi_size)
    roi = depth_img[y_min:y_max, x_min:x_max]
    valid_depths = roi[(roi > 0) & (roi < 10000)]
    if len(valid_depths) == 0: return None
    dist_mm = np.median(valid_depths)
    return dist_mm / 1000.0 

class GMC:
    def __init__(self, method='sparse_optflow', downscale=4): 
        # [优化] downscale 由 2 改为 4，成倍降低树莓派上的光流计算压力
        self.method = method; self.downscale = downscale
        self.prev_frame = None; self.prev_kpts = None
        self.feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=5, blockSize=3)
        
    def apply(self, raw_frame, detections_to_mask=[]):
        if raw_frame is None: return np.eye(2, 3, dtype=np.float32), False
        height, width = raw_frame.shape[:2]
        scale = 1.0 / self.downscale
        target_w = int(width * scale); target_h = int(height * scale)
        curr_frame_small = cv2.resize(raw_frame, (target_w, target_h))
        curr_gray = cv2.cvtColor(curr_frame_small, cv2.COLOR_BGR2GRAY)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        is_moving = False

        if self.prev_frame is not None:
            mask = np.full_like(self.prev_frame, 255)
            for box in detections_to_mask:
                x1, y1, x2, y2 = [int(b * scale) for b in box]
                pad = 5
                cv2.rectangle(mask, (max(0, x1-pad), max(0, y1-pad)), (x2+pad, y2+pad), 0, -1)
            if self.prev_kpts is None or len(self.prev_kpts) < 50:
                self.prev_kpts = cv2.goodFeaturesToTrack(self.prev_frame, mask=mask, **self.feature_params)
            if self.prev_kpts is not None and len(self.prev_kpts) > 0:
                curr_kpts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, curr_gray, self.prev_kpts, None)
                if status is not None:
                    matched_prev = self.prev_kpts[status == 1]
                    matched_curr = curr_kpts[status == 1]
                    if len(matched_prev) > 10:
                        m, inliers = cv2.estimateAffinePartial2D(matched_prev, matched_curr, method=cv2.RANSAC, ransacReprojThreshold=4)
                        if m is not None:
                            warp_matrix = m
                            warp_matrix[0, 2] /= scale; warp_matrix[1, 2] /= scale
                            translation = np.sqrt(m[0,2]**2 + m[1,2]**2)
                            if translation > 0.5: is_moving = True
                    self.prev_kpts = matched_curr.reshape(-1, 1, 2)
            else: self.prev_kpts = None
        self.prev_frame = curr_gray
        return warp_matrix, is_moving

class KalmanBoxTracker(object):
    count = 0
    def __init__(self, bbox, cls_name, score, feature=None, forced_id=None):
        self.kf = self._init_kf(bbox)
        self.time_since_update = 0
        
        if forced_id is not None:
            self.id = forced_id
            if forced_id >= KalmanBoxTracker.count:
                KalmanBoxTracker.count = forced_id + 1
        else:
            self.id = KalmanBoxTracker.count
            KalmanBoxTracker.count += 1
            
        self.hits = 0; self.hit_streak = 0; self.age = 0
        self.state = 0 
        self.cls_name = cls_name
        self.score = score
        self.last_observation = bbox 
        self.feature = feature
        self.alpha = 0.85
        self.velocity = np.array([0., 0.]) 
        self.depth_dist = None 
        self.pos_3d = None 
        self.velocity_3d = np.array([0.0, 0.0])

    def _init_kf(self, bbox):
        from filterpy.kalman import KalmanFilter
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([[1,0,0,0,1,0,0], [0,1,0,0,0,1,0], [0,0,1,0,0,0,1], [0,0,0,1,0,0,0],
                         [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]])
        kf.H = np.array([[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]])
        kf.P[4:, 4:] *= 1000.0; kf.P *= 10.0
        kf.Q[-1,-1] *= 0.01; kf.Q[4:,4:] *= 0.01
        kf.x[:4] = self.convert_bbox_to_z(bbox)
        return kf

    def convert_bbox_to_z(self, bbox):
        w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.; y = bbox[1] + h/2.
        s = w * h; r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    def convert_x_to_bbox(self, x):
        cx, cy, s, r = x[0], x[1], x[2], x[3]
        w = np.sqrt(s * r); h = s / w
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2]).reshape((1,4))

    def apply_affine_correction(self, warp_matrix):
        R = warp_matrix[:2, :2]; T = warp_matrix[:2, 2]
        self.kf.x[0] = self.kf.x[0] * R[0,0] + self.kf.x[1] * R[0,1] + T[0]
        self.kf.x[1] = self.kf.x[0] * R[1,0] + self.kf.x[1] * R[1,1] + T[1]
        vx = self.kf.x[4]; vy = self.kf.x[5]
        self.kf.x[4] = vx * R[0,0] + vy * R[0,1]
        self.kf.x[5] = vx * R[1,0] + vy * R[1,1]
        box = self.last_observation
        pts = np.array([[box[0], box[1], 1], [box[2], box[3], 1]]).T 
        new_pts = warp_matrix @ pts 
        self.last_observation = np.array([new_pts[0,0], new_pts[1,0], new_pts[0,1], new_pts[1,1]])

    def update(self, bbox, cls_name, score, feature=None, depth_dist=None, pos_3d=None, dt=None):
        if self.time_since_update > 0:
            delta = self.time_since_update + 1
            z_now = self.convert_bbox_to_z(bbox)
            z_prev = self.convert_bbox_to_z(self.last_observation)
            self.velocity = np.array([(z_now[0] - z_prev[0]) / delta, (z_now[1] - z_prev[1]) / delta]).flatten()
            
        self.last_observation = bbox
        self.time_since_update = 0
        self.hits += 1; self.hit_streak += 1
        
        # 【修改点】：增加名字抗抖动机制
        is_colored_update = any(c in cls_name for c in COLOR_CONFIG.keys())
        is_self_colored = any(c in self.cls_name for c in COLOR_CONFIG.keys())

        if is_colored_update:
            # 如果新来的是带颜色的，且自己之前没颜色，或者自己已经Lost了很久，接受新名字
            if not is_self_colored or self.state == 2:
                self.cls_name = cls_name
            # 如果自己已经是蓝色，新来的一帧YOLO突然说是红色，拒绝单帧突变，保持原有名字！
        elif not is_self_colored:
            # 双方都没颜色（例如转到侧面），只更新普通信息
            self.cls_name = cls_name
        
        self.score = score
        if depth_dist is not None: self.depth_dist = depth_dist
        
        if pos_3d is not None and dt is not None and dt > 0.001:
            if self.pos_3d is not None:
                vx = (pos_3d[0] - self.pos_3d[0]) / dt
                vy = (pos_3d[1] - self.pos_3d[1]) / dt
                alpha = 0.6
                self.velocity_3d[0] = alpha * self.velocity_3d[0] + (1-alpha) * vx
                self.velocity_3d[1] = alpha * self.velocity_3d[1] + (1-alpha) * vy
            self.pos_3d = pos_3d
        elif pos_3d is not None:
            self.pos_3d = pos_3d

        if feature is not None:
            if self.feature is not None:
                self.feature = self.alpha * self.feature + (1 - self.alpha) * feature
                self.feature /= np.linalg.norm(self.feature) 
            else: self.feature = feature
            
        self.kf.update(self.convert_bbox_to_z(bbox))
        
        if self.state == 0 and self.hits >= 3: self.state = 1
        elif self.state == 2: self.state = 1

    def predict(self, is_camera_moving=False):
        if((self.kf.x[6]+self.kf.x[2])<=0): self.kf.x[6] *= 0.0
        std_q_pos = 1.0; std_q_vel = 1.0
        if is_camera_moving: std_q_pos *= 1.2; std_q_vel *= 2.0
        if self.time_since_update > 0:
            self.kf.P[0:2, 0:2] += 50.0 * self.time_since_update 
            std_q_vel *= 3.0 
        self.kf.Q[:4, :4] *= std_q_pos; self.kf.Q[4:, 4:] *= std_q_vel
        self.kf.predict()
        self.kf.Q[:4, :4] /= std_q_pos; self.kf.Q[4:, 4:] /= std_q_vel
        self.age += 1
        if(self.time_since_update > 0): self.hit_streak = 0
        self.time_since_update += 1
        return self.convert_x_to_bbox(self.kf.x)[0]
    
    def get_state(self): return self.convert_x_to_bbox(self.kf.x)[0]

class OCSortTracker:
    def __init__(self, max_age=100, min_hits=3, iou_threshold=0.2, max_track_count=0):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_track_count = max_track_count 
        self.tracks: List[KalmanBoxTracker] = []
        self.frame_count = 0
        # 降采样改为4，极大减少光流法在树莓派上的开销
        self.gmc = GMC(method='sparse_optflow', downscale=4)

    def update(self, dets_high: List[dict], dets_low: List[dict], curr_frame=None, dt=0.1):
        self.frame_count += 1
        warp_matrix = np.eye(2, 3); is_moving = False
        
        if curr_frame is not None:
            active_boxes = [t.get_state() for t in self.tracks if t.time_since_update < 5]
            warp_matrix, is_moving = self.gmc.apply(curr_frame, detections_to_mask=active_boxes)
            if is_moving:
                for t in self.tracks: t.apply_affine_correction(warp_matrix)

        bboxes_high = np.array([d['bbox'] for d in dets_high]) if len(dets_high) > 0 else np.empty((0, 4))
        feats_high = np.array([d['feat'] for d in dets_high]) if len(dets_high) > 0 else np.empty((0, 0))
        
        trks_pool = []; trks_preds = []
        for t in self.tracks:
            pred = t.predict(is_camera_moving=is_moving)
            trks_pool.append(t); trks_preds.append(pred)
        trks_preds = np.array(trks_preds) if len(trks_preds) > 0 else np.empty((0, 4))
        
        # ---------------------------------------------------------
        #   阶段 1: 绝对颜色锁定 + 空间校验 (防止ID瞬移跳跃)
        # ---------------------------------------------------------
        matched_det_indices = set() 
        matched_trk_indices = set() 
        
        for d_idx, d in enumerate(dets_high):
            detected_color = None
            for color_name in COLOR_CONFIG.keys():
                if color_name in d['class']:
                    detected_color = color_name
                    break
            
            if detected_color:
                target_id = COLOR_CONFIG[detected_color]['id']
                
                found_track_idx = -1
                for t_idx, t in enumerate(trks_pool):
                    if t.id == target_id:
                        found_track_idx = t_idx
                        break
                
                if found_track_idx != -1:
                    # 【核心改进】：IOU校验！如果物理位置对不上，说明是颜色误识别，拒绝强行转移ID
                    t = trks_pool[found_track_idx]
                    pred_box = trks_preds[found_track_idx]
                    iou_matrix = iou_batch(np.array([d['bbox']]), np.array([pred_box]))
                    iou_val = iou_matrix[0][0]
                    
                    # 判定：如果轨迹一直存在(未丢失过久)，且当前新框和预测框完全不重合
                    if t.time_since_update < 10 and iou_val < 0.05:
                        found_track_idx = -1 # 拒绝匹配，让它降级去参加阶段2匹配
                    else:
                        if found_track_idx not in matched_trk_indices:
                            t.update(d['bbox'], d['class'], d['score'], d['feat'], d.get('depth', None), d.get('pos_3d', None), dt=dt)
                            matched_trk_indices.add(found_track_idx)
                            matched_det_indices.add(d_idx)
                
                if found_track_idx == -1 and d_idx not in matched_det_indices:
                    # 避免同一强制ID并存多条轨迹（Active/Lost各一条）
                    is_id_exists = any(t.id == target_id for t in self.tracks)
                    if not is_id_exists:
                        new_trk = KalmanBoxTracker(d['bbox'], d['class'], d['score'], d['feat'], forced_id=target_id)
                        new_trk.depth_dist = d.get('depth', None)
                        new_trk.pos_3d = d.get('pos_3d', None)
                        self.tracks.append(new_trk)
                        matched_det_indices.add(d_idx)
                    
        # ---------------------------------------------------------
        #   阶段 2: 常规匈牙利匹配 (处理被拒绝的和无颜色的)
        # ---------------------------------------------------------
        remain_det_indices = [i for i in range(len(dets_high)) if i not in matched_det_indices]
        remain_trk_indices = [i for i, t in enumerate(trks_pool) if i not in matched_trk_indices and (t.state==1 or t.state==2)]
        
        rem_boxes_trk = trks_preds[remain_trk_indices] if len(remain_trk_indices)>0 else np.empty((0,4))
        rem_boxes_det = bboxes_high[remain_det_indices] if len(remain_det_indices)>0 else np.empty((0,4))
        rem_feats_det = feats_high[remain_det_indices] if len(remain_det_indices)>0 else np.empty((0,0))
        
        matches_a, unmatched_trks_rel, unmatched_dets_rel = self._associate_hybrid(
            trks_pool, remain_trk_indices, rem_boxes_trk, rem_boxes_det, rem_feats_det,
            self.iou_threshold, is_moving, feature_weight=0.3
        )
        
        for t_rel_idx, d_rel_idx in matches_a:
            real_t_idx = remain_trk_indices[t_rel_idx]
            real_d_idx = remain_det_indices[d_rel_idx]
            t = trks_pool[real_t_idx]
            d = dets_high[real_d_idx]
            t.update(d['bbox'], d['class'], d['score'], d['feat'], d.get('depth', None), d.get('pos_3d', None), dt=dt)
            
        # ---------------------------------------------------------
        #   阶段 3: 二次匹配 (BYTETrack核心：处理低置信度框，防止频繁 LOST)
        # ---------------------------------------------------------
        unmatched_trks_from_p2 = [remain_trk_indices[i] for i in unmatched_trks_rel]
        # 只取当前依然活跃 (state==1) 的轨迹去匹配低置信度框
        second_stage_trk_indices = [idx for idx in unmatched_trks_from_p2 if trks_pool[idx].state == 1]
        
        if len(second_stage_trk_indices) > 0 and len(dets_low) > 0:
            rem_boxes_trk2 = trks_preds[second_stage_trk_indices]
            rem_boxes_det2 = np.array([d['bbox'] for d in dets_low])
            
            # 使用简单的 IOU 匹配
            matches_b, unmatch_trk2_rel, _ = self._associate_simple(
                rem_boxes_trk2, rem_boxes_det2, thresh=0.4 # 放宽 IOU 阈值
            )
            
            for t_rel_idx, d_idx in matches_b:
                real_t_idx = second_stage_trk_indices[t_rel_idx]
                t = trks_pool[real_t_idx]
                d = dets_low[d_idx]
                # 用低置信度框更新位置，特征传 None (信任卡尔曼滤波的状态稳定器)
                t.update(d['bbox'], d['class'], d['score'], None, d.get('depth', None), d.get('pos_3d', None), dt=dt)
                unmatched_trks_from_p2.remove(real_t_idx) # 匹配成功，从丢失名单中移除

        # ---------------------------------------------------------
        #   阶段 4: 严格清理
        # ---------------------------------------------------------
        for idx in unmatched_trks_from_p2:
            trks_pool[idx].state = 2 # 彻底找不到才标记为 Lost

        ret_tracks = []
        vip_ids = [cfg['id'] for cfg in COLOR_CONFIG.values()]
        
        for t in self.tracks:
            if t not in trks_pool: 
                ret_tracks.append(t)
                continue
                
            if t.state == 3: continue 
            
            time_limit = self.max_age
            if t.id in vip_ids: time_limit = 999999 
            
            if t.time_since_update <= time_limit:
                ret_tracks.append(t)
                
        self.tracks = ret_tracks

        # 同ID去重：优先保留 Active，其次保留更新时间更近的轨迹
        def _track_priority(t):
            state_rank = 2 if t.state == 1 else (1 if t.state == 0 else 0)
            return (state_rank, -t.time_since_update, t.hits)

        uniq = {}
        for t in self.tracks:
            if t.id not in uniq or _track_priority(t) > _track_priority(uniq[t.id]):
                uniq[t.id] = t
        self.tracks = list(uniq.values())

        is_locked = len([t for t in self.tracks if t.id in vip_ids]) > 0
        return self.tracks, is_moving, is_locked

    def _associate_hybrid(self, tracks_pool, track_indices, boxes1, boxes2, feats2, thresh, is_camera_moving, use_iou_only=False, feature_weight=0.3):
        if len(track_indices) == 0 or len(boxes2) == 0: return [], list(range(len(track_indices))), list(range(len(boxes2)))
        if len(boxes1.shape) == 1: boxes1 = np.expand_dims(boxes1, 0)
        cost_matrix = 1.0 - iou_batch(boxes1, boxes2)
        trk_feats = np.array([tracks_pool[i].feature for i in track_indices])
        valid_feat_mask = np.array([f is not None for f in trk_feats])
        dets_valid = False
        if feats2 is not None and len(feats2) > 0 and np.any(np.not_equal(feats2, None)): dets_valid = True
        if not use_iou_only and np.any(valid_feat_mask) and dets_valid:
            app_cost = np.ones_like(cost_matrix)
            valid_indices = np.where(valid_feat_mask)[0]
            valid_det_indices = np.where(np.not_equal(feats2, None))[0]
            if len(valid_indices) > 0 and len(valid_det_indices) > 0:
                valid_trk_feats = np.stack(trk_feats[valid_indices])
                valid_det_feats = np.stack(feats2[valid_det_indices])
                small_dists = cdist(valid_trk_feats, valid_det_feats, metric='cosine')
                for i, r in enumerate(valid_indices):
                    for j, c in enumerate(valid_det_indices):
                        app_cost[r, c] = small_dists[i, j]
            for r in range(app_cost.shape[0]):
                if np.sum(app_cost[r, :] < 0.25) > 1: app_cost[r, :] = 1.0 
            cost_matrix = (1 - feature_weight) * cost_matrix + feature_weight * app_cost
        
        w1 = boxes1[:, 2] - boxes1[:, 0]; h1 = boxes1[:, 3] - boxes1[:, 1]
        w2 = boxes2[:, 2] - boxes2[:, 0]; h2 = boxes2[:, 3] - boxes2[:, 1]
        ar1 = w1 / (h1 + 1e-6); ar2 = w2 / (h2 + 1e-6)
        ar1_exp = np.expand_dims(ar1, 1); ar2_exp = np.expand_dims(ar2, 0)
        ar_diff = np.maximum(ar1_exp / (ar2_exp + 1e-6), ar2_exp / (ar1_exp + 1e-6))
        cost_matrix[ar_diff > 2.0] += 0.5
        
        if is_camera_moving: 
            centers_trk = (boxes1[:, :2] + boxes1[:, 2:]) / 2.0
            centers_det = (boxes2[:, :2] + boxes2[:, 2:]) / 2.0
            for i, trk_idx in enumerate(track_indices):
                t = tracks_pool[trk_idx]; speed = np.linalg.norm(t.velocity)
                if speed > 5.0: 
                    to_det_vecs = centers_det - centers_trk[i]
                    norms = np.linalg.norm(to_det_vecs, axis=1) + 1e-6
                    cos_sim = np.sum(t.velocity * to_det_vecs, axis=1) / (speed * norms)
                    cost_matrix[i, cos_sim < -0.3] = 1.0
                    
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        unmatched_t = list(range(len(track_indices))); unmatched_d = list(range(len(boxes2)))
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < (1.0 - thresh):
                matches.append((r, c)); unmatched_t.remove(r); unmatched_d.remove(c)
        return matches, unmatched_t, unmatched_d

    def _associate_simple(self, boxes1, boxes2, thresh):
        if len(boxes1) == 0 or len(boxes2) == 0: return [], list(range(len(boxes1))), list(range(len(boxes2)))
        cost_matrix = 1.0 - iou_batch(boxes1, boxes2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        unmatched_t = list(range(len(boxes1))); unmatched_d = list(range(len(boxes2)))
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < (1.0 - thresh):
                matches.append((r, c)); unmatched_t.remove(r); unmatched_d.remove(c)
        return matches, unmatched_t, unmatched_d

class MOTTrackerNode(Node):
    def __init__(self):
        super().__init__('mot_tracker_node')
        self.declare_parameter('det_topic', 'yolov5_ros2/object_detect')
        self.declare_parameter('image_topic', '/ascamera/camera_publisher/rgb0/image') 
        self.declare_parameter('depth_topic', '/ascamera/camera_publisher/depth0/image_raw')
        self.declare_parameter('publish_topic', '/tracked_objects')
        self.declare_parameter('camera_info_topic', '/ascamera/camera_publisher/rgb0/camera_info')
        self.declare_parameter('max_track_count', 2)
        
        self.cv_bridge = CvBridge()
        self.latest_image = None
        self.latest_depth = None 
        self.last_img_header = None
        self.last_img_time = 0
        self.last_det_time = 0
        self.frame_id_counter = 0 
        self.intrinsics = None
        
        max_tracks = self.get_parameter('max_track_count').value
        self.tracker = OCSortTracker(max_age=100, max_track_count=max_tracks) 
        
        det_topic = self.get_parameter('det_topic').value
        self.img_topic_name = self.get_parameter('image_topic').value 
        self.depth_topic_name = self.get_parameter('depth_topic').value
        self.cam_info_topic = self.get_parameter('camera_info_topic').value
        pub_topic = self.get_parameter('publish_topic').value

        self.det_sub = self.create_subscription(ObjectsInfo, det_topic, self.on_detection, qos_profile_sensor_data)
        self.img_sub = self.create_subscription(Image, self.img_topic_name, self.on_image, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(Image, self.depth_topic_name, self.on_depth, qos_profile_sensor_data)
        self.cam_info_sub = self.create_subscription(CameraInfo, self.cam_info_topic, self.on_camera_info, 10)
        
        self.pub = self.create_publisher(ObjectsInfo, pub_topic, 10)
        img_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.img_pub = self.create_publisher(Image, '/object_tracking/image_result', img_qos)
        self.get_logger().info(f'Tracker Started. Locked Target Count: {max_tracks if max_tracks > 0 else "Unlimited"}')

    def on_camera_info(self, msg):
        if self.intrinsics is None:
            fx = msg.k[0]; cx = msg.k[2]; fy = msg.k[4]; cy = msg.k[5]
            self.intrinsics = (fx, fy, cx, cy)
            self.get_logger().info(f"Camera Intrinsics Loaded: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    def on_image(self, msg):
        try:
            self.latest_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.last_img_header = msg.header
            self.last_img_time = time.time()
            self.frame_id_counter += 1
            if self.frame_id_counter % 100 == 0: self.get_logger().info(f'Image Stream OK. Count: {self.frame_id_counter}')
        except CvBridgeError as e: self.get_logger().error(f"CV Bridge Error: {e}")

    def on_depth(self, msg):
        try:
            self.latest_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e: self.get_logger().error(f"Depth Bridge Error: {e}")

    def on_detection(self, msg: ObjectsInfo):
        curr_time = time.time()
        dt = curr_time - self.last_det_time if self.last_det_time > 0 else 0.1
        self.last_det_time = curr_time
        
        curr_img = None; curr_depth = None; use_image = False
        if self.latest_image is not None:
            time_diff = curr_time - self.last_img_time
            if time_diff < 1.0:
                curr_img = self.latest_image.copy()
                if self.latest_depth is not None: curr_depth = self.latest_depth.copy()
                use_image = True
            elif self.frame_id_counter % 50 == 0: self.get_logger().warn(f'Image lag ({time_diff:.1f}s). Fallback.')
        elif self.frame_id_counter % 50 == 0: self.get_logger().warn(f'Waiting for image topic "{self.img_topic_name}"...')

        dets_high = []; dets_low = []
        for obj in msg.objects:
            if len(obj.box) != 4: continue
            score = float(obj.score)
            bbox = np.array([float(v) for v in obj.box])
            feat = None; depth_val = None; pos_3d_val = None
            
            if use_image and score > 0.3: 
                feat = extract_spatial_feat(curr_img, bbox)
                # 【修改点】：直接移除了这里的 detect_color_tag 冗余计算
                if curr_depth is not None:
                    depth_val = get_depth_dist(curr_depth, bbox)
                    if depth_val is not None and self.intrinsics is not None:
                        fx, fy, cx, cy = self.intrinsics
                        u = (bbox[0] + bbox[2]) / 2.0; v = (bbox[1] + bbox[3]) / 2.0
                        z = depth_val; x = (u - cx) * z / fx; y = (v - cy) * z / fy
                        pos_3d_val = (x, z, y)

            # YOLO 节点已经包含了 _RED 或 _BLUE，这里直接用就行
            class_name = obj.class_name

            det = {'bbox': bbox, 'score': score, 'class': class_name, 'feat': feat, 'depth': depth_val, 'pos_3d': pos_3d_val}
            if score >= 0.5: dets_high.append(det)
            elif score >= 0.1: dets_low.append(det)
        
        tracked_objects, is_moving, closed_mode = self.tracker.update(dets_high, dets_low, curr_frame=curr_img, dt=dt)

        out_msg = ObjectsInfo()
        debug_log = []
        vis_img = curr_img.copy() if curr_img is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        active_cnt = 0; lost_cnt = 0

        for tr in tracked_objects:
            vip_ids = [cfg['id'] for cfg in COLOR_CONFIG.values()]
            if tr.id not in vip_ids: continue 

            obj = ObjectInfo()
            status_suffix = "_LOST" if tr.state == 2 else ""
            
            obj.class_name = f"{tr.cls_name}{status_suffix}" 
            obj.id = int(tr.id)
            
            if tr.pos_3d is not None:
                obj.position.x = float(tr.pos_3d[0])
                obj.position.y = float(tr.pos_3d[1])
                obj.position.z = float(tr.pos_3d[2])
                
            box = tr.get_state()
            obj.box = [int(max(0, box[0])), int(max(0, box[1])), int(max(0, box[2])), int(max(0, box[3]))]
            obj.score = float(tr.score)
            out_msg.objects.append(obj)
            
            if tr.state == 1: active_cnt += 1
            else: lost_cnt += 1

            log_tag = f"ID:{tr.id}({tr.cls_name})"
            if tr.pos_3d is not None:
                log_tag += f" pos:[{tr.pos_3d[0]:.2f},{tr.pos_3d[1]:.2f},{tr.pos_3d[2]:.2f}]"
            if tr.state == 2: log_tag += "(LOST)"
            debug_log.append(log_tag)
            
            color = (0, 255, 0)
            if "RED" in tr.cls_name: color = (0, 0, 255)
            elif "BLUE" in tr.cls_name: color = (255, 0, 0)
            elif "YELLOW" in tr.cls_name: color = (0, 255, 255)
            
            if tr.state == 2: thickness = 1; color = (color[0]//2, color[1]//2, color[2]//2)
            else: thickness = 2
            
            cv2.rectangle(vis_img, (obj.box[0], obj.box[1]), (obj.box[2], obj.box[3]), color, thickness)
            
            label_line1 = f"ID:{tr.id} {tr.cls_name}"
            if tr.state == 2: label_line1 += " LOST"
            cv2.putText(vis_img, label_line1, (obj.box[0], max(20, obj.box[1]-25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if tr.pos_3d is not None:
                label_line2 = f"X:{tr.pos_3d[0]:.2f} Y:{tr.pos_3d[1]:.2f} Z:{tr.pos_3d[2]:.2f}m"
                cv2.putText(vis_img, label_line2, (obj.box[0], max(20, obj.box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        mode_str = "[LOCKED]" if closed_mode else "[OPEN]"
        gmc_str = "Mov" if is_moving else "Stat"
        status_text = f"{mode_str} {gmc_str} Act:{active_cnt} Lost:{lost_cnt} FPS:{1.0/dt:.1f}"
        cv2.putText(vis_img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        self.pub.publish(out_msg)
        try:
            vis_msg = self.cv_bridge.cv2_to_imgmsg(vis_img, encoding="bgr8")
            if self.last_img_header is not None: vis_msg.header = self.last_img_header
            else:
                vis_msg.header.stamp = self.get_clock().now().to_msg()
                vis_msg.header.frame_id = "camera_frame"
            self.img_pub.publish(vis_msg)
        except Exception as e:
            if self.frame_id_counter % 50 == 0: self.get_logger().error(f"Img Pub Fail: {e}")
        
        if len(debug_log) > 0: self.get_logger().info(f"{status_text} | " + " ".join(debug_log))
        elif len(msg.objects) > 0: self.get_logger().info(f"Detect {len(msg.objects)} but Tracking 0")

def main():
    rclpy.init()
    node = MOTTrackerNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()