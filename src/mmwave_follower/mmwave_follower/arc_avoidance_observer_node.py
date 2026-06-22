#!/usr/bin/env python3
"""ROS2 node that treats the specified target id as the obstacle to avoid."""

from __future__ import annotations

import csv
import math
import os
import time
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from interfaces.msg import ObjectsInfo
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from ti_mmwave_rospkg_msgs.msg import RadarTrackArray


class ArcAvoidanceObserverNode(Node):
    def __init__(self) -> None:
        super().__init__("arc_avoidance_observer_node")

        self.declare_parameter("target_id", 1)
        self.declare_parameter("tracked_topic", "/tracked_objects_3d")
        self.declare_parameter("radar_track_topic", "/ti_mmwave/radar_track_array")
        self.declare_parameter("cmd_vel_topic", "/tracker/cmd_vel")
        self.declare_parameter("odom_topic", "/wheel_odom")
        self.declare_parameter("enable_virtual_obstacle", False)
        self.declare_parameter("virtual_obstacle_x", 1.0)
        self.declare_parameter("virtual_obstacle_y", 0.0)
        self.declare_parameter("force_virtual_arc", True)
        self.declare_parameter("stop_after_virtual_arc", True)
        self.declare_parameter("enable_disturbance_injection", False)
        self.declare_parameter("disturbance_vx_amp", 0.02)
        self.declare_parameter("disturbance_vy_amp", 0.08)
        self.declare_parameter("disturbance_freq_hz", 0.35)
        self.declare_parameter("disturbance_start_sec", 1.5)
        self.declare_parameter("disturbance_duration_sec", 4.0)
        self.declare_parameter("disturbance_profile", "execution")
        self.declare_parameter("exec_gain_x_amp", 0.10)
        self.declare_parameter("exec_gain_y_amp", 0.18)
        self.declare_parameter("exec_bias_x_amp", 0.012)
        self.declare_parameter("exec_bias_y_amp", 0.030)
        self.declare_parameter("exec_lateral_lag_sec", 0.35)
        self.declare_parameter("exec_noise_std", 0.006)
        self.declare_parameter("observer_alpha", 0.85)
        self.declare_parameter("control_period", 0.05)
        self.declare_parameter("target_timeout", 0.6)
        self.declare_parameter("lost_hold_sec", 0.8)
        self.declare_parameter("use_radar_fallback", False)
        self.declare_parameter("fusion_swap_xy", False)
        self.declare_parameter("fusion_x_sign", 1.0)
        self.declare_parameter("fusion_y_sign", 1.0)
        self.declare_parameter("radar_swap_xy", False)
        self.declare_parameter("radar_x_sign", 1.0)
        self.declare_parameter("radar_y_sign", 1.0)
        self.declare_parameter("keep_distance", 0.60)
        self.declare_parameter("safe_radius", 0.95)
        self.declare_parameter("front_half_width", 0.45)
        self.declare_parameter("track_kp_x", 1.20)
        self.declare_parameter("track_kp_y", 1.10)
        self.declare_parameter("max_vx", 0.45)
        self.declare_parameter("max_vy", 0.25)
        self.declare_parameter("max_ax", 1.00)
        self.declare_parameter("max_ay", 0.90)
        self.declare_parameter("min_forward_speed", 0.10)
        self.declare_parameter("arc_tangent_speed", 0.28)
        self.declare_parameter("avoid_forward_speed", 0.16)
        self.declare_parameter("avoid_forward_decay_radius", 0.30)
        self.declare_parameter("near_lateral_boost", 1.20)
        self.declare_parameter("hard_retreat_radius", 0.50)
        self.declare_parameter("retreat_center_band", 0.16)
        self.declare_parameter("arc_exit_lateral", 0.24)
        self.declare_parameter("close_pass_forward_speed", 0.18)
        self.declare_parameter("arc_finish_hold_sec", 0.9)
        self.declare_parameter("arc_finish_vx", 0.16)
        self.declare_parameter("arc_finish_vy", 0.24)
        self.declare_parameter("arc_finish_extra_sec", 1.2)
        self.declare_parameter("arc_finish_trigger_x", 0.55)
        self.declare_parameter("arc_finish_trigger_y", 0.18)
        self.declare_parameter("arc_path_topic", "/arc_avoidance/reference_path")
        self.declare_parameter("arc_clearance", 0.18)
        self.declare_parameter("arc_sweep_deg", 110.0)
        self.declare_parameter("arc_track_gain", 1.25)
        self.declare_parameter("enable_csv_log", True)
        self.declare_parameter("log_path", os.path.join(os.getcwd(), "arc_avoidance_log.csv"))
        self.declare_parameter("pass_side", 0)
        self.declare_parameter("retreat_speed", 0.12)
        self.declare_parameter("cruise_speed", 0.18)
        self.declare_parameter("reverse_cmd_x", False)
        self.declare_parameter("reverse_cmd_y", False)

        # Keep old launch parameters declared for compatibility.
        self.declare_parameter("radar_topic", "/ti_mmwave/radar_scan_pcl")
        self.declare_parameter("imu_topic", "/ros_robot_controller/imu_raw")
        self.declare_parameter("prefer_radar_track_target", True)
        self.declare_parameter("enable_pcl_target_fallback", False)
        self.declare_parameter("pcl_target_min_x", 0.45)
        self.declare_parameter("enable_virtual_goal_fallback", False)
        self.declare_parameter("virtual_goal_x", 0.0)
        self.declare_parameter("virtual_goal_y", 0.0)
        self.declare_parameter("front_range_min", 0.20)
        self.declare_parameter("front_range_max", 4.50)
        self.declare_parameter("cluster_eps", 0.22)
        self.declare_parameter("cluster_min_pts", 3)
        self.declare_parameter("target_exclusion_radius", 0.18)
        self.declare_parameter("accel_k", 1.40)
        self.declare_parameter("vel_damping", 0.22)
        self.declare_parameter("observer_comp_gain", 0.80)
        self.declare_parameter("arc_radius", 0.95)
        self.declare_parameter("arc_radial_gain", 0.95)
        self.declare_parameter("arc_goal_gain", 0.45)
        self.declare_parameter("corridor_margin", 0.22)

        self.target_id = int(self.get_parameter("target_id").value)
        self.latest_fusion_target: Optional[np.ndarray] = None
        self.latest_radar_target: Optional[np.ndarray] = None
        self.latest_fusion_raw: Optional[np.ndarray] = None
        self.latest_radar_raw: Optional[np.ndarray] = None
        self.last_target_rel: Optional[np.ndarray] = None
        self.last_target_source = "none"
        self.avoid_side_lock = 0.0
        self.arc_active = False
        self.arc_theta = 0.0
        self.arc_theta_goal = 0.0
        self.arc_radius = 0.0
        self.arc_last_obstacle_rel: Optional[np.ndarray] = None
        self.virtual_obstacle_rel: Optional[np.ndarray] = None
        self.virtual_obstacle_world: Optional[np.ndarray] = None
        self.virtual_arc_started = False
        self.virtual_arc_completed = False
        self.arc_finish_until = 0.0
        self.last_fusion_stamp = 0.0
        self.last_radar_stamp = 0.0
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.actual_vx = 0.0
        self.actual_vy = 0.0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.disturbance_hat = np.zeros(2, dtype=float)
        self.exec_cmd_state = np.zeros(2, dtype=float)
        self.start_time = time.time()
        self.last_tick = time.time()
        self.last_odom_msg_time = 0.0
        self.odom_topic = str(self.get_parameter("odom_topic").value)

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(ObjectsInfo, str(self.get_parameter("tracked_topic").value), self.on_tracks, qos)
        self.create_subscription(RadarTrackArray, str(self.get_parameter("radar_track_topic").value), self.on_radar_tracks, 10)
        self.create_subscription(Odometry, self.odom_topic, self.on_odom, 10)
        self.cmd_pub = self.create_publisher(Twist, str(self.get_parameter("cmd_vel_topic").value), 10)
        self.path_pub = self.create_publisher(Path, str(self.get_parameter("arc_path_topic").value), 10)

        self.logger_csv = None
        self.logger_fp = None
        if bool(self.get_parameter("enable_csv_log").value):
            self._open_log()

        self.create_timer(float(self.get_parameter("control_period").value), self.on_timer)
        self._warn_if_synthetic_odom_source()
        self.get_logger().info("Simple avoidance node started: target_id is treated as obstacle")

    def destroy_node(self) -> bool:
        self.stop_robot()
        if self.logger_fp is not None:
            self.logger_fp.close()
        return super().destroy_node()

    def _open_log(self) -> None:
        log_path = str(self.get_parameter("log_path").value)
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.logger_fp = open(log_path, "w", newline="", encoding="utf-8")
        self.logger_csv = csv.writer(self.logger_fp)
        self.logger_csv.writerow(
            [
                "time",
                "mode",
                "source",
                "target_x",
                "target_y",
                "obstacle_x",
                "obstacle_y",
                "ref_x",
                "ref_y",
                "pose_x",
                "pose_y",
                "cmd_vx",
                "cmd_vy",
                "published_vx",
                "published_vy",
                "real_vx",
                "real_vy",
                "execution_mismatch_x",
                "execution_mismatch_y",
                "measured_residual_x",
                "measured_residual_y",
                "residual_hat_x",
                "residual_hat_y",
                "arc_theta",
                "arc_active",
            ]
        )

    def _virtual_obstacle_target(self) -> Optional[np.ndarray]:
        if not bool(self.get_parameter("enable_virtual_obstacle").value):
            self.virtual_obstacle_rel = None
            self.virtual_obstacle_world = None
            self.virtual_arc_started = False
            self.virtual_arc_completed = False
            return None
        if self.virtual_obstacle_world is None:
            self.virtual_obstacle_world = np.array(
                [
                    self.pose_x + float(self.get_parameter("virtual_obstacle_x").value),
                    self.pose_y + float(self.get_parameter("virtual_obstacle_y").value),
                ],
                dtype=float,
            )
        self.virtual_obstacle_rel = self.virtual_obstacle_world - np.array([self.pose_x, self.pose_y], dtype=float)
        return self.virtual_obstacle_rel.copy()

    def on_odom(self, msg: Odometry) -> None:
        self.last_odom_msg_time = time.time()
        self.pose_x = float(msg.pose.pose.position.x)
        self.pose_y = float(msg.pose.pose.position.y)
        self.actual_vx = float(msg.twist.twist.linear.x)
        self.actual_vy = float(msg.twist.twist.linear.y)

    def _warn_if_synthetic_odom_source(self) -> None:
        synthetic_topics = {"/car3/odom", "/odom_raw"}
        if self.odom_topic in synthetic_topics:
            self.get_logger().warn(
                f"odom_topic={self.odom_topic} is command-integrated pseudo odometry in this workspace, "
                "not encoder-ground-truth. Prefer a real fused/localization topic such as /odom.",
                throttle_duration_sec=5.0,
            )

    @staticmethod
    def _extract_position_xy(position) -> Optional[np.ndarray]:
        if position is None:
            return None
        if hasattr(position, "x") and hasattr(position, "y"):
            return np.array([float(position.x), float(position.y)], dtype=float)
        if isinstance(position, (list, tuple, np.ndarray)) and len(position) >= 2:
            return np.array([float(position[0]), float(position[1])], dtype=float)
        return None

    def _transform_rel(self, rel: np.ndarray, prefix: str) -> np.ndarray:
        x = float(rel[0])
        y = float(rel[1])
        if bool(self.get_parameter(f"{prefix}_swap_xy").value):
            x, y = y, x
        x *= float(self.get_parameter(f"{prefix}_x_sign").value)
        y *= float(self.get_parameter(f"{prefix}_y_sign").value)
        return np.array([x, y], dtype=float)

    def _publish_reference_arc(self, theta_start: float, theta_goal: float, radius: float) -> Optional[np.ndarray]:
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "base_link"

        if math.isclose(theta_start, theta_goal, rel_tol=0.0, abs_tol=1e-6):
            theta_samples = [theta_goal]
        else:
            theta_samples = np.linspace(theta_start, theta_goal, 30)

        ref_point = None
        for theta in theta_samples:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = -radius * math.cos(float(theta))
            pose.pose.position.y = -radius * math.sin(float(theta))
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
            if ref_point is None:
                ref_point = np.array([pose.pose.position.x, pose.pose.position.y], dtype=float)

        self.path_pub.publish(path)
        return ref_point

    def _begin_arc_plan(self, obstacle_rel: np.ndarray, now: float, pass_side: float) -> None:
        radius = max(
            float(self.get_parameter("safe_radius").value) + float(self.get_parameter("arc_clearance").value),
            float(self.get_parameter("arc_radius").value),
        )
        theta_start = math.atan2(-float(obstacle_rel[1]), -float(obstacle_rel[0]))
        sweep = math.radians(float(self.get_parameter("arc_sweep_deg").value))
        side_sign = 1.0 if pass_side > 0.0 else -1.0

        self.arc_active = True
        self.arc_radius = radius
        self.arc_theta = theta_start
        self.arc_theta_goal = theta_start + side_sign * sweep
        self.arc_last_obstacle_rel = obstacle_rel.copy()
        self.arc_finish_until = now + float(self.get_parameter("arc_finish_hold_sec").value)

    def _advance_arc_theta(self, dt: float) -> None:
        if not self.arc_active:
            return
        side_sign = 1.0 if self.arc_theta_goal >= self.arc_theta else -1.0
        omega = float(self.get_parameter("arc_tangent_speed").value) / max(self.arc_radius, 1e-3)
        step = omega * dt * side_sign
        if side_sign > 0.0:
            self.arc_theta = min(self.arc_theta + step, self.arc_theta_goal)
        else:
            self.arc_theta = max(self.arc_theta + step, self.arc_theta_goal)

    def _arc_reference(self) -> np.ndarray:
        return np.array(
            [-self.arc_radius * math.cos(self.arc_theta), -self.arc_radius * math.sin(self.arc_theta)],
            dtype=float,
        )

    def _arc_reference_velocity(self) -> np.ndarray:
        side_sign = 1.0 if self.arc_theta_goal >= self.arc_theta else -1.0
        omega = float(self.get_parameter("arc_tangent_speed").value) / max(self.arc_radius, 1e-3)
        theta_dot = omega * side_sign
        return np.array(
            [
                self.arc_radius * math.sin(self.arc_theta) * theta_dot,
                -self.arc_radius * math.cos(self.arc_theta) * theta_dot,
            ],
            dtype=float,
        )

    def on_tracks(self, msg: ObjectsInfo) -> None:
        if bool(self.get_parameter("enable_virtual_obstacle").value):
            return
        selected = None
        selected_raw = None
        for obj in msg.objects:
            rel = self._extract_position_xy(getattr(obj, "position", None))
            if rel is None:
                continue
            raw_rel = rel.copy()
            rel = self._transform_rel(rel, "fusion")
            oid = int(getattr(obj, "id", -1))
            if self.target_id == -1:
                if rel[0] <= 0.0:
                    continue
            if self.target_id == -1 or oid == self.target_id:
                selected = rel
                selected_raw = raw_rel
                break

        if selected is not None:
            self.latest_fusion_target = selected
            self.latest_fusion_raw = selected_raw
            self.last_target_rel = selected.copy()
            self.last_target_source = "fusion"
            self.last_fusion_stamp = time.time()
            self.get_logger().info(
                f"fusion_raw=({selected_raw[0]:.2f},{selected_raw[1]:.2f}) "
                f"control_obstacle=({selected[0]:.2f},{selected[1]:.2f})",
                throttle_duration_sec=0.5,
            )

    def on_radar_tracks(self, msg: RadarTrackArray) -> None:
        if bool(self.get_parameter("enable_virtual_obstacle").value):
            return
        selected = None
        selected_raw = None
        for track in msg.track:
            rel = np.array([float(track.posy), float(-track.posx)], dtype=float)
            raw_rel = rel.copy()
            rel = self._transform_rel(rel, "radar")
            tid = int(getattr(track, "tid", -1))
            if self.target_id == -1:
                if rel[0] <= 0.0:
                    continue
            if self.target_id == -1 or tid == self.target_id:
                selected = rel
                selected_raw = raw_rel
                break

        if selected is not None:
            self.latest_radar_target = selected
            self.latest_radar_raw = selected_raw
            self.last_target_rel = selected.copy()
            self.last_target_source = "radar_track"
            self.last_radar_stamp = time.time()

    def _current_target(self) -> tuple[Optional[np.ndarray], str]:
        now = time.time()
        timeout = float(self.get_parameter("target_timeout").value)
        virtual_target = self._virtual_obstacle_target()
        if virtual_target is not None:
            self.latest_fusion_target = None
            self.latest_radar_target = None
            self.latest_fusion_raw = None
            self.latest_radar_raw = None
            self.last_target_rel = virtual_target.copy()
            self.last_target_source = "virtual_obstacle"
            return virtual_target, "virtual_obstacle"
        if self.latest_fusion_target is not None and (now - self.last_fusion_stamp) < timeout:
            return self.latest_fusion_target.copy(), "fusion"
        if (
            bool(self.get_parameter("use_radar_fallback").value)
            and self.latest_radar_target is not None
            and (now - self.last_radar_stamp) < timeout
        ):
            return self.latest_radar_target.copy(), "radar_track"
        if self.avoid_side_lock != 0.0 and self.last_target_rel is not None and now < self.arc_finish_until:
            return self.last_target_rel.copy(), f"{self.last_target_source}_avoid_hold"
        if self.last_target_rel is not None and (
            min(now - self.last_fusion_stamp if self.last_fusion_stamp > 0.0 else 1e9,
                now - self.last_radar_stamp if self.last_radar_stamp > 0.0 else 1e9)
            < float(self.get_parameter("lost_hold_sec").value)
        ):
            return self.last_target_rel.copy(), f"{self.last_target_source}_hold"
        return None, "none"

    def _limit_step(self, current: float, target: float, max_step: float) -> float:
        return current + max(-max_step, min(max_step, target - current))

    def _apply_velocity_limits(self, target_vx: float, target_vy: float, dt: float) -> tuple[float, float]:
        max_vx = float(self.get_parameter("max_vx").value)
        max_vy = float(self.get_parameter("max_vy").value)
        max_ax = float(self.get_parameter("max_ax").value)
        max_ay = float(self.get_parameter("max_ay").value)

        limited_vx = self._limit_step(self.cmd_vx, target_vx, max_ax * dt)
        limited_vy = self._limit_step(self.cmd_vy, target_vy, max_ay * dt)

        self.cmd_vx = max(-max_vx, min(max_vx, limited_vx))
        self.cmd_vy = max(-max_vy, min(max_vy, limited_vy))
        return self.cmd_vx, self.cmd_vy

    def stop_robot(self) -> None:
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.avoid_side_lock = 0.0
        self.arc_finish_until = 0.0
        self.arc_active = False
        self.virtual_arc_started = False
        self.virtual_arc_completed = False
        self.cmd_pub.publish(Twist())

    def _apply_execution_mismatch(self, nominal_cmd: np.ndarray, dt: float, now: float) -> tuple[np.ndarray, np.ndarray]:
        if not bool(self.get_parameter("enable_disturbance_injection").value):
            self.exec_cmd_state = np.array(nominal_cmd, dtype=float)
            return self.exec_cmd_state.copy(), np.zeros(2, dtype=float)

        t_rel = now - self.start_time
        t0 = float(self.get_parameter("disturbance_start_sec").value)
        duration = float(self.get_parameter("disturbance_duration_sec").value)
        if not (t0 <= t_rel < t0 + duration):
            self.exec_cmd_state = np.array(nominal_cmd, dtype=float)
            return self.exec_cmd_state.copy(), np.zeros(2, dtype=float)

        tau = (t_rel - t0) / max(duration, 1e-6)
        envelope = math.sin(math.pi * tau) ** 2
        freq = float(self.get_parameter("disturbance_freq_hz").value)
        omega = 2.0 * math.pi * freq * (t_rel - t0)
        profile = str(self.get_parameter("disturbance_profile").value)

        gain_x_amp = float(self.get_parameter("exec_gain_x_amp").value)
        gain_y_amp = float(self.get_parameter("exec_gain_y_amp").value)
        bias_x_amp = float(self.get_parameter("exec_bias_x_amp").value)
        bias_y_amp = float(self.get_parameter("exec_bias_y_amp").value)
        lag_nominal = max(float(self.get_parameter("exec_lateral_lag_sec").value), 1e-3)
        noise_std = float(self.get_parameter("exec_noise_std").value)
        if profile == "challenging":
            gain_x_amp *= 1.7
            gain_y_amp *= 1.8
            bias_x_amp *= 1.8
            bias_y_amp *= 2.0
            lag_nominal *= 1.6
            noise_std *= 2.2

        gain_x = 1.0 + envelope * gain_x_amp * math.sin(omega)
        gain_y = 1.0 + envelope * gain_y_amp * math.cos(1.3 * omega - 0.5)
        bias = envelope * np.array(
            [
                bias_x_amp * math.sin(2.1 * omega + 0.3),
                bias_y_amp * math.cos(1.7 * omega - 0.8),
            ],
            dtype=float,
        )
        if profile == "challenging":
            bias += envelope * np.array(
                [
                    0.5 * bias_x_amp * math.sin(3.4 * omega - 0.2),
                    0.55 * bias_y_amp * math.cos(2.8 * omega + 0.4),
                ],
                dtype=float,
            )

        pre_lag = np.array([gain_x * nominal_cmd[0], gain_y * nominal_cmd[1]], dtype=float) + bias

        lag_y = lag_nominal * (1.0 + 0.45 * envelope * (1.0 + math.sin(0.6 * omega)))
        alpha_y = min(max(dt / lag_y, 0.0), 1.0)
        self.exec_cmd_state[0] = pre_lag[0]
        self.exec_cmd_state[1] = self.exec_cmd_state[1] + alpha_y * (pre_lag[1] - self.exec_cmd_state[1])

        if noise_std > 0.0:
            self.exec_cmd_state += envelope * noise_std * np.random.standard_normal(2)

        mismatch = self.exec_cmd_state - nominal_cmd
        return self.exec_cmd_state.copy(), mismatch

    def on_timer(self) -> None:
        now = time.time()
        dt = max(0.02, min(0.2, now - self.last_tick))
        self.last_tick = now

        if self.last_odom_msg_time <= 0.0:
            self.get_logger().warn(
                f"Waiting for odometry on {self.odom_topic}; virtual arc tracking needs a real pose/velocity source.",
                throttle_duration_sec=2.0,
            )
        elif now - self.last_odom_msg_time > 0.5:
            self.get_logger().warn(
                f"Odometry on {self.odom_topic} is stale by {now - self.last_odom_msg_time:.2f}s.",
                throttle_duration_sec=2.0,
            )

        target_rel, source = self._current_target()
        if (
            source == "virtual_obstacle"
            and bool(self.get_parameter("stop_after_virtual_arc").value)
            and self.virtual_arc_completed
            and not self.arc_active
        ):
            self.stop_robot()
            return
        arc_finish_hold_sec = float(self.get_parameter("arc_finish_hold_sec").value)
        arc_finish_vx = float(self.get_parameter("arc_finish_vx").value)
        arc_finish_vy = float(self.get_parameter("arc_finish_vy").value)
        arc_finish_extra_sec = float(self.get_parameter("arc_finish_extra_sec").value)
        arc_finish_trigger_x = float(self.get_parameter("arc_finish_trigger_x").value)
        arc_finish_trigger_y = float(self.get_parameter("arc_finish_trigger_y").value)
        arc_track_gain = float(self.get_parameter("arc_track_gain").value)

        if target_rel is None:
            if self.arc_active and self.arc_last_obstacle_rel is not None:
                target_rel = self.arc_last_obstacle_rel.copy()
                source = f"{source}_arc_hold"
            elif self.avoid_side_lock != 0.0 and now < self.arc_finish_until:
                cmd_vx, cmd_vy = self._apply_velocity_limits(
                    arc_finish_vx,
                    self.avoid_side_lock * arc_finish_vy,
                    dt,
                )
                twist = Twist()
                reverse_cmd_x = bool(self.get_parameter("reverse_cmd_x").value)
                reverse_cmd_y = bool(self.get_parameter("reverse_cmd_y").value)
                twist.linear.x = -cmd_vx if reverse_cmd_x else cmd_vx
                twist.linear.y = -cmd_vy if reverse_cmd_y else cmd_vy
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                self.get_logger().info(
                    f"mode=arc_finish source=memory pass={'left' if self.avoid_side_lock > 0 else 'right'} "
                    f"pub_cmd=({twist.linear.x:.2f},{twist.linear.y:.2f})",
                    throttle_duration_sec=0.25,
                )
                published_cmd = np.array([cmd_vx, cmd_vy], dtype=float)
                exec_mismatch = published_cmd - np.array([cmd_vx, cmd_vy], dtype=float)
                measured_residual = np.array([self.actual_vx - cmd_vx, self.actual_vy - cmd_vy], dtype=float)
                self.disturbance_hat = (
                    float(self.get_parameter("observer_alpha").value) * self.disturbance_hat
                    + (1.0 - float(self.get_parameter("observer_alpha").value))
                    * measured_residual
                )
                if self.logger_csv is not None:
                    self.logger_csv.writerow(
                        [
                            f"{now:.6f}",
                            "arc_finish",
                            "memory",
                            f"{math.nan:.6f}",
                            f"{math.nan:.6f}",
                            f"{math.nan:.6f}",
                            f"{math.nan:.6f}",
                            f"{math.nan:.6f}",
                            f"{math.nan:.6f}",
                            f"{self.pose_x:.6f}",
                            f"{self.pose_y:.6f}",
                            f"{cmd_vx:.6f}",
                            f"{cmd_vy:.6f}",
                            f"{published_cmd[0]:.6f}",
                            f"{published_cmd[1]:.6f}",
                            f"{self.actual_vx:.6f}",
                            f"{self.actual_vy:.6f}",
                            f"{exec_mismatch[0]:.6f}",
                            f"{exec_mismatch[1]:.6f}",
                            f"{measured_residual[0]:.6f}",
                            f"{measured_residual[1]:.6f}",
                            f"{self.disturbance_hat[0]:.6f}",
                            f"{self.disturbance_hat[1]:.6f}",
                            f"{self.arc_theta:.6f}",
                            "0",
                        ]
                    )
                    self.logger_fp.flush()
                return
            self.stop_robot()
            return

        obstacle_x = float(target_rel[0])
        obstacle_y = float(target_rel[1])
        keep_distance = float(self.get_parameter("keep_distance").value)
        safe_radius = float(self.get_parameter("safe_radius").value)
        front_half_width = float(self.get_parameter("front_half_width").value)
        cruise_speed = float(self.get_parameter("cruise_speed").value)
        retreat_speed = float(self.get_parameter("retreat_speed").value)
        avoid_lateral_speed = float(self.get_parameter("arc_tangent_speed").value)
        avoid_forward_speed = float(self.get_parameter("avoid_forward_speed").value)
        avoid_forward_decay_radius = float(self.get_parameter("avoid_forward_decay_radius").value)
        near_lateral_boost = float(self.get_parameter("near_lateral_boost").value)
        hard_retreat_radius = float(self.get_parameter("hard_retreat_radius").value)
        retreat_center_band = float(self.get_parameter("retreat_center_band").value)
        arc_exit_lateral = float(self.get_parameter("arc_exit_lateral").value)
        close_pass_forward_speed = float(self.get_parameter("close_pass_forward_speed").value)
        min_forward_speed = float(self.get_parameter("min_forward_speed").value)
        forced_pass_side = int(self.get_parameter("pass_side").value)
        force_virtual_arc = bool(self.get_parameter("force_virtual_arc").value)

        is_blocking = obstacle_x < safe_radius and abs(obstacle_y) < front_half_width
        if source == "virtual_obstacle" and force_virtual_arc and not self.virtual_arc_started:
            is_blocking = True
        if is_blocking:
            self.arc_finish_until = now + arc_finish_hold_sec
            if self.avoid_side_lock == 0.0:
                if forced_pass_side != 0:
                    self.avoid_side_lock = float(1 if forced_pass_side > 0 else -1)
                elif abs(obstacle_y) > 0.03:
                    self.avoid_side_lock = -1.0 if obstacle_y > 0.0 else 1.0
                else:
                    self.avoid_side_lock = -1.0
            pass_side = self.avoid_side_lock
            if not self.arc_active:
                self._begin_arc_plan(target_rel, now, pass_side)
                if source == "virtual_obstacle":
                    self.virtual_arc_started = True
                    self.virtual_arc_completed = False
        else:
            if not self.arc_active:
                self.avoid_side_lock = 0.0
            if forced_pass_side != 0:
                pass_side = float(1 if forced_pass_side > 0 else -1)
            else:
                pass_side = 0.0

        mode = "cruise"
        target_vx = max(min_forward_speed, cruise_speed)
        target_vy = 0.0
        ref_point = None
        arc_ref = None

        if self.arc_active:
            if source.endswith("_arc_hold") and self.arc_last_obstacle_rel is not None:
                target_rel = self.arc_last_obstacle_rel - np.array([self.cmd_vx, self.cmd_vy], dtype=float) * dt
            self.arc_last_obstacle_rel = target_rel.copy()
            self._advance_arc_theta(dt)
            ref_point = self._publish_reference_arc(self.arc_theta, self.arc_theta_goal, self.arc_radius)

            arc_ref = self._arc_reference()
            arc_ref_dot = self._arc_reference_velocity()
            arc_error = target_rel - arc_ref
            mode = "arc_follow"
            target_vx = -arc_ref_dot[0] + arc_track_gain * arc_error[0]
            target_vy = -arc_ref_dot[1] + arc_track_gain * arc_error[1]
            target_vx = max(min_forward_speed, target_vx)

            side_sign = 1.0 if self.arc_theta_goal >= self.arc_theta else -1.0
            finished = (side_sign > 0.0 and self.arc_theta >= self.arc_theta_goal) or (
                side_sign < 0.0 and self.arc_theta <= self.arc_theta_goal
            )
            if finished:
                self.arc_active = False
                self.arc_finish_until = now + arc_finish_hold_sec + arc_finish_extra_sec
                if source == "virtual_obstacle":
                    self.virtual_arc_completed = True
                mode = "arc_finish"

        if is_blocking:
            if not self.arc_active:
                mode = "avoid"
                closeness = 1.0 - max(0.0, min(1.0, (obstacle_x - keep_distance) / max(safe_radius - keep_distance, 1e-3)))
                lateral_scale = 1.0 + closeness * max(0.0, near_lateral_boost - 1.0)
                if abs(obstacle_y) >= arc_exit_lateral:
                    target_vx = max(close_pass_forward_speed, avoid_forward_speed)
                    self.arc_finish_until = now + arc_finish_extra_sec
                else:
                    target_vx = max(min_forward_speed, avoid_forward_speed * max(0.75, 1.0 - 0.25 * closeness))
                target_vy = pass_side * min(
                    float(self.get_parameter("max_vy").value),
                    avoid_lateral_speed * lateral_scale,
                )

                if obstacle_x <= arc_finish_trigger_x and abs(obstacle_y) >= arc_finish_trigger_y:
                    self.arc_finish_until = now + arc_finish_extra_sec

                if obstacle_x <= hard_retreat_radius and abs(obstacle_y) <= retreat_center_band:
                    mode = "retreat_avoid"
                    retreat_ratio = 1.0 - max(
                        0.0,
                        min(1.0, (obstacle_x - keep_distance) / max(avoid_forward_decay_radius, 1e-3)),
                    )
                    target_vx = avoid_forward_speed * (1.0 - retreat_ratio) - retreat_speed * retreat_ratio
                    target_vy = pass_side * min(
                        float(self.get_parameter("max_vy").value),
                        avoid_lateral_speed * max(1.15, near_lateral_boost),
                    )

        cmd_vx, cmd_vy = self._apply_velocity_limits(target_vx, target_vy, dt)
        nominal_cmd = np.array([cmd_vx, cmd_vy], dtype=float)

        if abs(cmd_vx) < 0.01:
            cmd_vx = 0.0
        if abs(cmd_vy) < 0.01:
            cmd_vy = 0.0
        self.cmd_vx = cmd_vx
        self.cmd_vy = cmd_vy

        reverse_cmd_x = bool(self.get_parameter("reverse_cmd_x").value)
        reverse_cmd_y = bool(self.get_parameter("reverse_cmd_y").value)
        published_cmd, exec_mismatch = self._apply_execution_mismatch(nominal_cmd, dt, now)
        measured_residual = np.array([self.actual_vx, self.actual_vy], dtype=float) - nominal_cmd
        self.disturbance_hat = (
            float(self.get_parameter("observer_alpha").value) * self.disturbance_hat
            + (1.0 - float(self.get_parameter("observer_alpha").value))
            * measured_residual
        )

        obstacle_world = None
        if source == "virtual_obstacle" and self.virtual_obstacle_world is not None:
            obstacle_world = self.virtual_obstacle_world.copy()
        else:
            obstacle_world = np.array([self.pose_x + obstacle_x, self.pose_y + obstacle_y], dtype=float)
        ref_world = None if arc_ref is None else obstacle_world - arc_ref

        twist = Twist()
        twist.linear.x = -float(published_cmd[0]) if reverse_cmd_x else float(published_cmd[0])
        twist.linear.y = -float(published_cmd[1]) if reverse_cmd_y else float(published_cmd[1])
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

        side_text = "left" if obstacle_y > 0.0 else "right"
        pass_text = "left" if pass_side > 0.0 else "right" if pass_side < 0.0 else "none"
        raw_text = ""
        if source.startswith("fusion") and self.latest_fusion_raw is not None:
            raw_text = f"fusion_raw=({self.latest_fusion_raw[0]:.2f},{self.latest_fusion_raw[1]:.2f}) "
        elif source.startswith("radar_track") and self.latest_radar_raw is not None:
            raw_text = f"radar_raw=({self.latest_radar_raw[0]:.2f},{self.latest_radar_raw[1]:.2f}) "
        self.get_logger().info(
            f"mode={mode} source={source} {raw_text}control_obstacle=({obstacle_x:.2f},{obstacle_y:.2f}) "
            f"ref=({(ref_point[0] if ref_point is not None else 0.0):.2f},{(ref_point[1] if ref_point is not None else 0.0):.2f}) "
            f"side={side_text} pass={pass_text} blocking={is_blocking} raw_cmd=({cmd_vx:.2f},{cmd_vy:.2f}) "
            f"exec_mismatch=({exec_mismatch[0]:.2f},{exec_mismatch[1]:.2f}) res=({measured_residual[0]:.2f},{measured_residual[1]:.2f}) "
            f"pub_cmd=({twist.linear.x:.2f},{twist.linear.y:.2f})",
            throttle_duration_sec=0.25,
        )

        if self.logger_csv is not None:
            self.logger_csv.writerow(
                [
                    f"{now:.6f}",
                    mode,
                    source,
                    f"{obstacle_x:.6f}",
                    f"{obstacle_y:.6f}",
                    f"{obstacle_world[0]:.6f}",
                    f"{obstacle_world[1]:.6f}",
                    f"{(ref_world[0] if ref_world is not None else math.nan):.6f}",
                    f"{(ref_world[1] if ref_world is not None else math.nan):.6f}",
                    f"{self.pose_x:.6f}",
                    f"{self.pose_y:.6f}",
                    f"{cmd_vx:.6f}",
                    f"{cmd_vy:.6f}",
                    f"{published_cmd[0]:.6f}",
                    f"{published_cmd[1]:.6f}",
                    f"{self.actual_vx:.6f}",
                    f"{self.actual_vy:.6f}",
                    f"{exec_mismatch[0]:.6f}",
                    f"{exec_mismatch[1]:.6f}",
                    f"{measured_residual[0]:.6f}",
                    f"{measured_residual[1]:.6f}",
                    f"{self.disturbance_hat[0]:.6f}",
                    f"{self.disturbance_hat[1]:.6f}",
                    f"{self.arc_theta:.6f}",
                    f"{1.0 if self.arc_active else 0.0:.0f}",
                ]
            )
            self.logger_fp.flush()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArcAvoidanceObserverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
