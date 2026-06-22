#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多目标融合跟踪节点 (Vision + Radar Fusion)

策略概述:
1. 视觉只提供 2D bbox，负责圈定目标所在图像区域。
2. 雷达先经过全局 ROI 筛选，再投影到图像，只保留落在 bbox 内的候选点。
3. 对每个 bbox 内的小集合做距离/横向/高度门控，然后用小规模 DBSCAN 聚类。
4. 不再取最近单点，而是选择得分最高的主簇，并用鲁棒形心估计目标位置。
5. 将形心送入 Alpha-Beta 轻量时序滤波器，雷达失配时先用预测撑帧，再回退视觉。
"""

import json
import struct
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from interfaces.msg import ObjectInfo, ObjectsInfo
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image, PointCloud2


CALIB_CONFIG = {
    "INTRINSICS_PATH": "/home/ubuntu/ros2_ws/src/config/camera_intrinsics.json",
    "EXTRINSICS_PATH": "/home/ubuntu/ros2_ws/src/config/extrinsics.json",
}


def as_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def simple_dbscan(points_xy, eps, min_samples):
    if len(points_xy) == 0:
        return np.empty((0,), dtype=np.int32)

    labels = np.full(len(points_xy), -1, dtype=np.int32)
    visited = np.zeros(len(points_xy), dtype=bool)
    cluster_id = 0

    def region_query(idx):
        dists = np.linalg.norm(points_xy - points_xy[idx], axis=1)
        return np.where(dists <= eps)[0]

    for idx in range(len(points_xy)):
        if visited[idx]:
            continue

        visited[idx] = True
        neighbors = region_query(idx)
        if len(neighbors) < min_samples:
            continue

        labels[idx] = cluster_id
        seeds = list(neighbors.tolist())
        seed_pos = 0

        while seed_pos < len(seeds):
            neighbor_idx = seeds[seed_pos]
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                neighbor_neighbors = region_query(neighbor_idx)
                if len(neighbor_neighbors) >= min_samples:
                    for candidate in neighbor_neighbors.tolist():
                        if candidate not in seeds:
                            seeds.append(candidate)

            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id

            seed_pos += 1

        cluster_id += 1

    return labels


class MOTTrackerFusionNode(Node):
    def __init__(self):
        super().__init__("mot_tracker_fusion_node")
        self.bridge = CvBridge()

        self.declare_parameter("rgb_topic", "/car3/ascamera/camera_publisher/rgb0/image")
        self.declare_parameter("radar_topic", "/radar/pointcloud")
        self.declare_parameter("det_topic", "/car3/tracked_objects")
        self.declare_parameter("publish_topic", "/car3/tracked_objects_3d")
        self.declare_parameter("fusion_match_pixel_thresh", 50.0)
        self.declare_parameter("visual_trust_dist", 3.0)
        self.declare_parameter("det_timeout_sec", 0.8)

        self.declare_parameter("radar_forward_min", 0.2)
        self.declare_parameter("radar_forward_max", 6.0)
        self.declare_parameter("radar_lateral_abs_max", 2.0)
        self.declare_parameter("radar_height_min", -0.5)
        self.declare_parameter("radar_height_max", 0.8)

        self.declare_parameter("radar_depth_gate", 1.0)
        self.declare_parameter("radar_lateral_gate", 0.7)
        self.declare_parameter("radar_height_gate", 0.35)

        self.declare_parameter("cluster_eps", 0.16)
        self.declare_parameter("cluster_min_samples", 2)
        self.declare_parameter("cluster_pred_scale", 0.35)
        self.declare_parameter("cluster_vis_scale", 0.60)
        self.declare_parameter("cluster_spread_scale", 0.08)

        self.declare_parameter("filter_alpha", 0.45)
        self.declare_parameter("filter_beta", 0.12)
        self.declare_parameter("filter_z_alpha", 0.35)
        self.declare_parameter("predict_hold_frames", 3)
        self.declare_parameter("visual_fallback_delay_frames", 3)
        self.declare_parameter("track_state_ttl_sec", 0.8)
        self.declare_parameter("track_max_missed_frames", 5)
        self.declare_parameter("debug_log_enabled", True)

        self.K = None
        self.D = None
        self.rvec = None
        self.tvec = None
        self.load_calibration()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub_rgb = self.create_subscription(
            Image, self.get_parameter("rgb_topic").value, self.on_rgb, qos
        )
        self.sub_det = self.create_subscription(
            ObjectsInfo,
            self.get_parameter("det_topic").value,
            self.on_detection,
            qos_profile_sensor_data,
        )
        self.sub_radar = self.create_subscription(
            PointCloud2, self.get_parameter("radar_topic").value, self.on_radar, qos
        )

        self.pub_tracks = self.create_publisher(
            ObjectsInfo, self.get_parameter("publish_topic").value, 10
        )
        self.pub_vis = self.create_publisher(Image, "/tracked_objects/image_fusion", 10)

        self.last_rgb = None
        self.last_radar_points = None
        self.last_radar_img_pts = None
        self.tracks = []
        self.track_states = {}
        self.det_lock = threading.Lock()
        self.last_det_time = 0.0
        self.det_timeout = float(self.get_parameter("det_timeout_sec").value)

        self.create_timer(0.05, self.on_timer)
        self.get_logger().info(
            "Fusion Tracker Started (Radar ROI + Small DBSCAN + Robust Centroid + Alpha-Beta)."
        )

    def load_calibration(self):
        try:
            with open(CALIB_CONFIG["INTRINSICS_PATH"], "r") as f:
                data = json.load(f)
                self.K = np.array(data["camera_matrix"], dtype=np.float32)
                self.D = np.array(data["distortion_coefficients"], dtype=np.float32)

            with open(CALIB_CONFIG["EXTRINSICS_PATH"], "r") as f:
                data = json.load(f)
                self.rvec = np.array(data["rvec"], dtype=np.float32)
                self.tvec = np.array(data["tvec"], dtype=np.float32)

            self.get_logger().info("标定文件加载成功")
        except Exception as exc:
            self.get_logger().error(f"标定文件加载失败: {exc}")

    def on_rgb(self, msg):
        try:
            self.last_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def on_radar(self, msg):
        points = []
        try:
            offset_x, offset_y, offset_z = -1, -1, -1
            for field in msg.fields:
                if field.name == "x":
                    offset_x = field.offset
                elif field.name == "y":
                    offset_y = field.offset
                elif field.name == "z":
                    offset_z = field.offset

            if offset_x == -1 or offset_y == -1 or offset_z == -1:
                return

            data = msg.data
            step = msg.point_step
            n_points = msg.width * msg.height

            for i in range(n_points):
                start = i * step
                x = struct.unpack_from("<f", data, start + offset_x)[0]
                y = struct.unpack_from("<f", data, start + offset_y)[0]
                z = struct.unpack_from("<f", data, start + offset_z)[0]
                points.append([x, y, z])

            if points:
                self.last_radar_points = np.array(points, dtype=np.float32)
            else:
                self.last_radar_points = np.empty((0, 3), dtype=np.float32)
        except Exception as exc:
            self.get_logger().warn(f"Radar parse error: {exc}")

    def on_detection(self, msg: ObjectsInfo):
        with self.det_lock:
            self.last_det_time = time.time()
            dedup_by_id = {}

            for obj in msg.objects:
                if len(obj.box) < 4:
                    continue

                x1, y1, x2, y2 = obj.box[:4]
                cls_name = obj.class_name or "obj"
                track_id_out = obj.id if getattr(obj, "id", -1) != -1 else int(obj.score)

                if "_" in cls_name:
                    try:
                        base, tail = cls_name.rsplit("_", 1)
                        if tail.isdigit():
                            cls_name = base
                    except Exception:
                        pass

                item = {
                    "id": track_id_out,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "score": float(obj.score),
                    "raw_class": str(obj.class_name),
                    "clean_class": str(cls_name),
                    "position": obj.position,
                    "radar_data": None,
                    "vis_data": None,
                    "fusion_output": None,
                    "debug": {},
                }

                prev = dedup_by_id.get(track_id_out)
                if prev is None or item["score"] >= prev["score"]:
                    dedup_by_id[track_id_out] = item

            self.tracks = list(dedup_by_id.values())

    def _get_param(self, name):
        return self.get_parameter(name).value

    def _extract_visual_measurement(self, tr):
        pos = tr.get("position", None)
        if pos is None:
            return None

        vx = as_float(getattr(pos, "x", 0.0))
        vy = as_float(getattr(pos, "y", 0.0))
        vz = as_float(getattr(pos, "z", 0.0))
        if vy <= 0.0:
            return None
        return np.array([vx, vy, vz], dtype=np.float32)

    def _prepare_radar_points(self):
        if self.last_radar_points is None or self.rvec is None or self.K is None:
            self.last_radar_img_pts = None
            return None, None

        if len(self.last_radar_points) == 0:
            self.last_radar_img_pts = np.empty((0, 2), dtype=np.float32)
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        pts = self.last_radar_points
        finite_mask = np.isfinite(pts).all(axis=1)

        fwd_min = self._get_param("radar_forward_min")
        fwd_max = self._get_param("radar_forward_max")
        lat_abs_max = self._get_param("radar_lateral_abs_max")
        z_min = self._get_param("radar_height_min")
        z_max = self._get_param("radar_height_max")

        roi_mask = (
            finite_mask
            & (pts[:, 1] >= fwd_min)
            & (pts[:, 1] <= fwd_max)
            & (np.abs(pts[:, 0]) <= lat_abs_max)
            & (pts[:, 2] >= z_min)
            & (pts[:, 2] <= z_max)
        )

        roi_points = pts[roi_mask]
        if len(roi_points) == 0:
            self.last_radar_img_pts = np.empty((0, 2), dtype=np.float32)
            return roi_points, np.empty((0, 2), dtype=np.float32)

        img_pts, _ = cv2.projectPoints(roi_points, self.rvec, self.tvec, self.K, self.D)
        img_pts = img_pts.reshape(-1, 2).astype(np.float32)
        self.last_radar_img_pts = img_pts
        return roi_points, img_pts

    def _predict_track_state(self, track_id, now):
        state = self.track_states.get(track_id)
        if state is None:
            return None

        dt = max(1e-2, min(now - state["timestamp"], 0.3))
        return {
            "x": state["x"] + state["vx"] * dt,
            "y": state["y"] + state["vy"] * dt,
            "z": state["z"],
            "vx": state["vx"],
            "vy": state["vy"],
            "dt": dt,
            "missed_frames": state["missed_frames"],
        }

    def _state_to_output(self, state):
        return {
            "x": float(state["x"]),
            "y": float(state["y"]),
            "z": float(state["z"]),
            "vx": float(state["vx"]),
            "vy": float(state["vy"]),
            "missed_frames": int(state["missed_frames"]),
        }

    def _update_track_state(self, track_id, measurement, now):
        alpha = self._get_param("filter_alpha")
        beta = self._get_param("filter_beta")
        z_alpha = self._get_param("filter_z_alpha")

        state = self.track_states.get(track_id)
        if state is None:
            state = {
                "x": float(measurement[0]),
                "y": float(measurement[1]),
                "z": float(measurement[2]),
                "vx": 0.0,
                "vy": 0.0,
                "timestamp": now,
                "missed_frames": 0,
                "last_seen": now,
            }
            self.track_states[track_id] = state
            return self._state_to_output(state)

        dt = max(1e-2, min(now - state["timestamp"], 0.3))
        pred_x = state["x"] + state["vx"] * dt
        pred_y = state["y"] + state["vy"] * dt

        rx = float(measurement[0]) - pred_x
        ry = float(measurement[1]) - pred_y

        state["x"] = pred_x + alpha * rx
        state["y"] = pred_y + alpha * ry
        state["vx"] = state["vx"] + beta * rx / dt
        state["vy"] = state["vy"] + beta * ry / dt
        state["z"] = state["z"] + z_alpha * (float(measurement[2]) - state["z"])
        state["timestamp"] = now
        state["last_seen"] = now
        state["missed_frames"] = 0
        return self._state_to_output(state)

    def _mark_track_missed(self, track_id, now):
        state = self.track_states.get(track_id)
        if state is None:
            return None

        dt = max(1e-2, min(now - state["timestamp"], 0.3))
        state["x"] = state["x"] + state["vx"] * dt
        state["y"] = state["y"] + state["vy"] * dt
        state["timestamp"] = now
        state["last_seen"] = now
        state["missed_frames"] += 1
        return self._state_to_output(state)

    def _prune_track_states(self, now, active_track_ids):
        ttl = self._get_param("track_state_ttl_sec")
        max_missed = self._get_param("track_max_missed_frames")

        stale_ids = []
        for track_id, state in self.track_states.items():
            if (now - state["last_seen"]) > ttl and track_id not in active_track_ids:
                stale_ids.append(track_id)
            elif state["missed_frames"] > max_missed and track_id not in active_track_ids:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            self.track_states.pop(track_id, None)

    def _gate_points_for_track(self, tr, radar_points, img_pts, visual_meas, pred_state):
        x1, y1, x2, y2 = tr["bbox"]
        bbox_mask = (
            (img_pts[:, 0] >= x1)
            & (img_pts[:, 0] <= x2)
            & (img_pts[:, 1] >= y1)
            & (img_pts[:, 1] <= y2)
        )

        raw_points = radar_points[bbox_mask]
        raw_img_pts = img_pts[bbox_mask]
        if len(raw_points) == 0:
            return raw_points, raw_points, raw_img_pts, raw_img_pts

        depth_gate = self._get_param("radar_depth_gate")
        lateral_gate = self._get_param("radar_lateral_gate")
        height_gate = self._get_param("radar_height_gate")

        gate_mask = np.ones(len(raw_points), dtype=bool)

        ref_x = None
        ref_y = None
        ref_z = None

        if pred_state is not None:
            ref_x = pred_state["x"]
            ref_y = pred_state["y"]
            ref_z = pred_state["z"]

        if visual_meas is not None:
            if ref_x is None:
                ref_x = float(visual_meas[0])
            if ref_y is None:
                ref_y = float(visual_meas[1])
            if ref_z is None:
                ref_z = float(visual_meas[2])

        if ref_y is not None:
            gate_mask &= np.abs(raw_points[:, 1] - ref_y) <= depth_gate
        if ref_x is not None:
            gate_mask &= np.abs(raw_points[:, 0] - ref_x) <= lateral_gate
        if ref_z is not None:
            gate_mask &= np.abs(raw_points[:, 2] - ref_z) <= height_gate

        gated_points = raw_points[gate_mask]
        gated_img_pts = raw_img_pts[gate_mask]
        return raw_points, gated_points, raw_img_pts, gated_img_pts

    def _cluster_points(self, points):
        min_samples = int(self._get_param("cluster_min_samples"))
        if len(points) < min_samples:
            return []

        labels = simple_dbscan(
            points[:, :2],
            eps=float(self._get_param("cluster_eps")),
            min_samples=min_samples,
        )

        clusters = []
        for label in np.unique(labels):
            if label < 0:
                continue
            mask = labels == label
            cluster_points = points[mask]
            if len(cluster_points) >= min_samples:
                clusters.append(cluster_points)
        return clusters

    def _robust_centroid(self, points):
        median = np.median(points, axis=0)
        if len(points) <= 2:
            return median.astype(np.float32), "median"

        dists = np.linalg.norm(points[:, :2] - median[:2], axis=1)
        keep_count = max(2, len(points) - 1)
        keep_idx = np.argsort(dists)[:keep_count]
        trimmed = points[keep_idx]

        if len(trimmed) >= 4:
            centroid = np.mean(trimmed, axis=0)
            method = "drop_farthest_mean"
        else:
            centroid = np.median(trimmed, axis=0)
            method = "median"

        return centroid.astype(np.float32), method

    def _score_cluster(self, cluster_points, centroid, visual_meas, pred_state):
        count_score = 2.0 * len(cluster_points)

        spread = float(
            np.mean(np.linalg.norm(cluster_points[:, :2] - centroid[:2], axis=1))
        )
        spread_scale = max(1e-3, float(self._get_param("cluster_spread_scale")))
        compact_score = 1.5 * np.exp(-spread / spread_scale)

        pred_score = 0.0
        pred_dist = None
        if pred_state is not None:
            pred_xy = np.array([pred_state["x"], pred_state["y"]], dtype=np.float32)
            pred_dist = float(np.linalg.norm(centroid[:2] - pred_xy))
            pred_scale = max(1e-3, float(self._get_param("cluster_pred_scale")))
            pred_score = 3.0 * np.exp(-pred_dist / pred_scale)

        vis_score = 0.0
        vis_depth_diff = None
        if visual_meas is not None:
            vis_depth_diff = abs(float(centroid[1]) - float(visual_meas[1]))
            vis_scale = max(1e-3, float(self._get_param("cluster_vis_scale")))
            vis_score = 2.0 * np.exp(-vis_depth_diff / vis_scale)

        return {
            "score": float(count_score + compact_score + pred_score + vis_score),
            "spread": spread,
            "pred_dist": pred_dist,
            "vis_depth_diff": vis_depth_diff,
        }

    def _select_best_cluster(self, clusters, visual_meas, pred_state):
        best = None
        summaries = []

        for cluster_points in clusters:
            centroid, centroid_method = self._robust_centroid(cluster_points)
            metrics = self._score_cluster(cluster_points, centroid, visual_meas, pred_state)
            summary = {
                "size": int(len(cluster_points)),
                "score": round(metrics["score"], 3),
                "spread": round(metrics["spread"], 3),
                "centroid": np.round(centroid, 3).tolist(),
                "centroid_method": centroid_method,
                "pred_dist": None
                if metrics["pred_dist"] is None
                else round(metrics["pred_dist"], 3),
                "vis_depth_diff": None
                if metrics["vis_depth_diff"] is None
                else round(metrics["vis_depth_diff"], 3),
                "points": np.round(cluster_points, 3).tolist(),
            }
            summaries.append(summary)

            if best is None or metrics["score"] > best["metrics"]["score"]:
                best = {
                    "points": cluster_points,
                    "centroid": centroid,
                    "centroid_method": centroid_method,
                    "metrics": metrics,
                    "summary": summary,
                }

        return best, summaries

    def _match_radar_for_track(self, tr, radar_points, img_pts, visual_meas, pred_state):
        raw_points, gated_points, raw_img_pts, gated_img_pts = self._gate_points_for_track(
            tr, radar_points, img_pts, visual_meas, pred_state
        )

        debug = {
            "raw_points": np.round(raw_points, 3).tolist(),
            "gated_points": np.round(gated_points, 3).tolist(),
            "cluster_candidates": [],
            "cluster": None,
            "centroid": None,
            "centroid_method": None,
            "filtered_state": None,
            "source": "none",
            "raw_count": int(len(raw_points)),
            "gated_count": int(len(gated_points)),
            "raw_img_points": np.round(raw_img_pts, 1).tolist(),
            "cluster_img_points": [],
        }

        clusters = self._cluster_points(gated_points)
        if not clusters:
            return None, debug

        best_cluster, summaries = self._select_best_cluster(clusters, visual_meas, pred_state)
        debug["cluster_candidates"] = summaries

        if best_cluster is None:
            return None, debug

        best_points = best_cluster["points"]
        centroid = best_cluster["centroid"]

        cluster_img_points = []
        if len(gated_points) > 0:
            centroid_like = np.isclose(gated_points[:, None, :], best_points[None, :, :], atol=1e-6)
            cluster_mask = np.any(np.all(centroid_like, axis=2), axis=1)
            cluster_img_points = np.round(gated_img_pts[cluster_mask], 1).tolist()

        debug["cluster"] = best_cluster["summary"]
        debug["centroid"] = np.round(centroid, 3).tolist()
        debug["centroid_method"] = best_cluster["centroid_method"]
        debug["cluster_img_points"] = cluster_img_points
        return centroid, debug

    def fuse_radar_to_tracks(self):
        now = time.time()
        radar_points, radar_img_pts = self._prepare_radar_points()
        has_radar = radar_points is not None and radar_img_pts is not None

        visual_trust_dist = self._get_param("visual_trust_dist")
        hold_predict_frames = int(self._get_param("predict_hold_frames"))
        visual_delay_frames = int(self._get_param("visual_fallback_delay_frames"))

        active_track_ids = {tr["id"] for tr in self.tracks}
        self._prune_track_states(now, active_track_ids)

        for tr in self.tracks:
            track_id = tr["id"]
            tr["radar_data"] = None
            tr["vis_data"] = None
            tr["fusion_output"] = None

            visual_meas = self._extract_visual_measurement(tr)
            pred_state = self._predict_track_state(track_id, now)
            debug = {
                "raw_points": [],
                "gated_points": [],
                "cluster_candidates": [],
                "cluster": None,
                "centroid": None,
                "centroid_method": None,
                "filtered_state": None,
                "source": "none",
                "pred_state": None if pred_state is None else self._state_to_output(pred_state),
                "visual_measurement": None
                if visual_meas is None
                else np.round(visual_meas, 3).tolist(),
            }

            if visual_meas is not None and 0.1 < float(visual_meas[1]) < visual_trust_dist:
                tr["vis_data"] = {
                    "x": float(visual_meas[0]),
                    "y": float(visual_meas[1]),
                    "z": float(visual_meas[2]),
                }

            if has_radar and len(radar_points) > 0:
                centroid, radar_debug = self._match_radar_for_track(
                    tr, radar_points, radar_img_pts, visual_meas, pred_state
                )
                debug.update(radar_debug)

                if centroid is not None:
                    filtered_state = self._update_track_state(track_id, centroid, now)
                    tr["radar_data"] = {
                        "x": float(filtered_state["x"]),
                        "y": float(filtered_state["y"]),
                        "z": float(filtered_state["z"]),
                        "count": int(debug["cluster"]["size"]),
                    }
                    tr["fusion_output"] = {
                        "x": float(filtered_state["x"]),
                        "y": float(filtered_state["y"]),
                        "z": float(filtered_state["z"]),
                        "source": "radar",
                    }
                    debug["filtered_state"] = filtered_state
                    debug["source"] = "radar"
                    tr["debug"] = debug
                    continue

            predicted_state = self._mark_track_missed(track_id, now)
            if predicted_state is not None and predicted_state["missed_frames"] <= hold_predict_frames:
                tr["fusion_output"] = {
                    "x": float(predicted_state["x"]),
                    "y": float(predicted_state["y"]),
                    "z": float(predicted_state["z"]),
                    "source": "predict",
                }
                debug["filtered_state"] = predicted_state
                debug["source"] = "predict"
            elif (
                tr["vis_data"] is not None
                and (
                    predicted_state is None
                    or predicted_state["missed_frames"] > visual_delay_frames
                )
            ):
                tr["fusion_output"] = {
                    "x": float(tr["vis_data"]["x"]),
                    "y": float(tr["vis_data"]["y"]),
                    "z": float(tr["vis_data"]["z"]),
                    "source": "visual",
                }
                debug["filtered_state"] = {
                    "x": float(tr["vis_data"]["x"]),
                    "y": float(tr["vis_data"]["y"]),
                    "z": float(tr["vis_data"]["z"]),
                    "vx": 0.0,
                    "vy": 0.0,
                    "missed_frames": 0 if predicted_state is None else predicted_state["missed_frames"],
                }
                debug["source"] = "visual"
            elif predicted_state is not None:
                debug["filtered_state"] = predicted_state
                debug["source"] = "missed"

            tr["debug"] = debug

    def _source_to_suffix(self, source):
        if source == "radar":
            return "_Rad"
        if source == "predict":
            return "_Pred"
        if source == "visual":
            return "_Vis"
        return ""

    def _build_debug_payload(self, tr):
        dbg = tr.get("debug", {})
        return {
            "id": int(tr["id"]),
            "raw_points": dbg.get("raw_points", []),
            "cluster": dbg.get("cluster", None),
            "centroid": dbg.get("centroid", None),
            "filtered_state": dbg.get("filtered_state", None),
            "source": dbg.get("source", "none"),
        }

    def on_timer(self):
        if time.time() - self.last_det_time > self.det_timeout:
            with self.det_lock:
                if self.tracks:
                    self.tracks = []

        with self.det_lock:
            self.fuse_radar_to_tracks()

            out_msg = ObjectsInfo()
            debug_lines = []

            for tr in self.tracks:
                obj = ObjectInfo()
                obj.score = float(tr["id"])
                obj.box = tr["bbox"]

                base_name = tr["clean_class"]
                if " " in base_name:
                    base_name = base_name.split(" ")[0]

                output = tr.get("fusion_output", None)
                if output is not None:
                    obj.position.x = float(output["y"])
                    obj.position.y = -float(output["x"])
                    obj.position.z = float(output["z"])
                    obj.id = int(tr["id"])
                    obj.class_name = f"{base_name}{self._source_to_suffix(output['source'])}"
                else:
                    obj.id = int(tr["id"])
                    obj.class_name = base_name

                out_msg.objects.append(obj)

                if output is not None and self._get_param("debug_log_enabled"):
                    debug_payload = self._build_debug_payload(tr)
                    debug_lines.append(json.dumps(debug_payload, ensure_ascii=False))

            self.pub_tracks.publish(out_msg)

            if self.last_rgb is not None:
                vis = self.last_rgb.copy()
                h, w = vis.shape[:2]

                if self.last_radar_img_pts is not None:
                    for p in self.last_radar_img_pts:
                        px, py = int(p[0]), int(p[1])
                        if 0 <= px < w and 0 <= py < h:
                            cv2.circle(vis, (px, py), 2, (0, 255, 255), -1)

                for tr in self.tracks:
                    x1, y1, x2, y2 = tr["bbox"]
                    source = tr.get("fusion_output", {}).get("source", "none")
                    color = (120, 120, 120)
                    if source == "radar":
                        color = (0, 255, 0)
                    elif source == "predict":
                        color = (255, 255, 0)
                    elif source == "visual":
                        color = (0, 165, 255)

                    for p in tr.get("debug", {}).get("cluster_img_points", []):
                        px, py = int(p[0]), int(p[1])
                        if 0 <= px < w and 0 <= py < h:
                            cv2.circle(vis, (px, py), 4, (0, 255, 0), -1)

                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{tr['id']} {tr['clean_class']} {source}"
                    cv2.putText(
                        vis,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

                try:
                    vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
                    self.pub_vis.publish(vis_msg)
                except Exception as exc:
                    self.get_logger().warn(f"Publish vis image failed: {exc}")

            if debug_lines and self._get_param("debug_log_enabled"):
                self.get_logger().info(" || ".join(debug_lines))


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


if __name__ == "__main__":
    main()
