#!/usr/bin/env python3
"""Reusable disturbance observer and arc-avoidance control core."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


def _safe_norm(vec: np.ndarray, eps: float = 1e-9) -> float:
    return max(float(np.linalg.norm(vec)), eps)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = _safe_norm(vec)
    return vec / norm


def _smooth_sign(vec: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    out = np.zeros_like(vec, dtype=float)
    out[vec > eps] = 1.0
    out[vec < -eps] = -1.0
    return out


@dataclass
class ObserverConfig:
    tau_o: float = 1.0
    mu_max: float = 80.0
    phi_bound: float = 6.0
    sigma1: float = 4.0
    sigma2: float = 6.0
    sigma3: float = 1.0
    switch_ratio: float = 0.85
    omega_o: float = 12.0


class PrescribedTimeObserver2D:
    """Port of the MATLAB prescribed-time disturbance observer."""

    def __init__(self, config: ObserverConfig):
        self.config = config
        self.beta1 = 3.0 * config.omega_o
        self.beta2 = 3.0 * config.omega_o ** 2
        self.beta3 = config.omega_o ** 3
        self.time = 0.0
        self.reset()

    def reset(self, p0: Optional[np.ndarray] = None) -> None:
        self.time = 0.0
        base = np.zeros(2, dtype=float) if p0 is None else np.array(p0, dtype=float)
        self.p_hat = base.copy()
        self.v_hat = np.zeros(2, dtype=float)
        self.phi_hat = np.zeros(2, dtype=float)

    def update(self, p_meas: np.ndarray, u_input: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
        p_meas = np.array(p_meas, dtype=float)
        u_input = np.array(u_input, dtype=float)
        dt = max(float(dt), 1e-4)
        self.time += dt

        error = p_meas - self.p_hat
        tau_o = self.config.tau_o
        t_sw = self.config.switch_ratio * tau_o

        if self.time < t_sw:
            denom = max(tau_o - self.time, dt)
            mu = min(tau_o / denom, self.config.mu_max)
            z = (mu ** 2) * error
            sign_z = _smooth_sign(z)
            abs_z = np.abs(z)
            h1 = self.config.sigma1 * (self.config.phi_bound ** (1.0 / 3.0)) * (abs_z ** (2.0 / 3.0)) * sign_z
            h2 = self.config.sigma2 * (self.config.phi_bound ** (2.0 / 3.0)) * (abs_z ** (1.0 / 3.0)) * sign_z
            h3 = self.config.sigma3 * self.config.phi_bound * sign_z
            q1 = (3.0 * mu / tau_o) * error + (1.0 / mu) * h1
            q2 = (mu ** 2 / tau_o ** 2) * error + (1.0 / tau_o) * h1 + h2
            q3 = mu * h3
        else:
            q1 = self.beta1 * error
            q2 = self.beta2 * error
            q3 = self.beta3 * error

        self.p_hat = self.p_hat + dt * (self.v_hat + q1)
        self.v_hat = self.v_hat + dt * (u_input + self.phi_hat + q2)
        self.phi_hat = self.phi_hat + dt * q3

        return {
            "p_hat": self.p_hat.copy(),
            "v_hat": self.v_hat.copy(),
            "phi_hat": self.phi_hat.copy(),
            "error": error.copy(),
        }


@dataclass
class ArcAvoidanceConfig:
    keep_distance: float = 0.8
    track_kp_x: float = 0.85
    track_kp_y: float = 1.00
    accel_k: float = 1.8
    vel_damping: float = 0.20
    observer_comp_gain: float = 0.8
    max_vx: float = 0.50
    max_vy: float = 0.45
    max_ax: float = 1.20
    max_ay: float = 1.20
    min_forward_speed: float = 0.08
    safe_radius: float = 0.70
    arc_radius: float = 0.95
    arc_tangent_speed: float = 0.35
    arc_radial_gain: float = 1.10
    arc_goal_gain: float = 0.55
    obstacle_influence: float = 1.10
    corridor_margin: float = 0.18
    memory_timeout: float = 2.0
    release_rear_x: float = -0.30
    cbf_c1: float = 2.0
    cbf_c2: float = 3.0
    use_cbf: bool = True


class DisturbanceAwareArcController:
    """Nominal tracking + obstacle memory + arc vector field + CBF filter."""

    def __init__(self, observer_cfg: Optional[ObserverConfig] = None, controller_cfg: Optional[ArcAvoidanceConfig] = None):
        self.observer = PrescribedTimeObserver2D(observer_cfg or ObserverConfig())
        self.cfg = controller_cfg or ArcAvoidanceConfig()
        self.reset()

    def reset(self) -> None:
        self.cmd_v = np.zeros(2, dtype=float)
        self.last_u = np.zeros(2, dtype=float)
        self.ego_pose = np.zeros(2, dtype=float)
        self.ego_yaw = 0.0
        self.obstacle_rel: Optional[np.ndarray] = None
        self.obstacle_world: Optional[np.ndarray] = None
        self.obstacle_age = 0.0
        self.avoidance_active = False
        self.pass_side = 1
        self.last_mode = "idle"

    def update_yaw(self, yaw: float) -> None:
        self.ego_yaw = float(yaw)

    def _body_to_world(self, vec: np.ndarray) -> np.ndarray:
        c = math.cos(self.ego_yaw)
        s = math.sin(self.ego_yaw)
        rot = np.array([[c, -s], [s, c]], dtype=float)
        return rot @ vec

    def _world_to_body(self, vec: np.ndarray) -> np.ndarray:
        c = math.cos(self.ego_yaw)
        s = math.sin(self.ego_yaw)
        rot = np.array([[c, s], [-s, c]], dtype=float)
        return rot @ vec

    def _choose_side(self, obstacle_rel: np.ndarray, target_rel: np.ndarray) -> int:
        if abs(obstacle_rel[1]) > 0.08:
            return -1 if obstacle_rel[1] > 0.0 else 1
        if abs(target_rel[1]) > 0.08:
            return 1 if target_rel[1] <= 0.0 else -1
        return self.pass_side

    def _is_blocking(self, target_rel: np.ndarray, obstacle_rel: np.ndarray) -> bool:
        if target_rel[0] <= 0.2 or obstacle_rel[0] <= 0.1:
            return False
        target_dist = _safe_norm(target_rel)
        goal_dir = target_rel / target_dist
        proj = float(np.dot(obstacle_rel, goal_dir))
        lateral = float(np.linalg.norm(obstacle_rel - proj * goal_dir))
        return 0.0 < proj < target_dist and lateral < (self.cfg.safe_radius + self.cfg.corridor_margin)

    def _update_obstacle_memory(
        self,
        obstacle_measurement: Optional[np.ndarray],
        dt: float,
        yaw_rate: float,
    ) -> None:
        if obstacle_measurement is not None:
            obs_rel = np.array(obstacle_measurement, dtype=float)
            self.obstacle_rel = obs_rel
            self.obstacle_world = self.ego_pose + self._body_to_world(obs_rel)
            self.obstacle_age = 0.0
            return

        self.obstacle_age += dt
        if self.obstacle_world is None or self.obstacle_age > self.cfg.memory_timeout:
            self.obstacle_rel = None
            self.obstacle_world = None
            return

        body_v = self.cmd_v.copy()
        if self.obstacle_rel is not None:
            obs = self.obstacle_rel
            obs_dot = np.array(
                [
                    -body_v[0] + yaw_rate * obs[1],
                    -body_v[1] - yaw_rate * obs[0],
                ],
                dtype=float,
            )
            self.obstacle_rel = obs + dt * obs_dot
            self.obstacle_world = self.ego_pose + self._body_to_world(self.obstacle_rel)

    def _tracking_velocity_ref(self, target_rel: np.ndarray) -> np.ndarray:
        error = target_rel - np.array([self.cfg.keep_distance, 0.0], dtype=float)
        ref_v = np.array(
            [
                self.cfg.track_kp_x * error[0],
                self.cfg.track_kp_y * error[1],
            ],
            dtype=float,
        )
        ref_v[0] = float(np.clip(ref_v[0], -self.cfg.max_vx, self.cfg.max_vx))
        ref_v[1] = float(np.clip(ref_v[1], -self.cfg.max_vy, self.cfg.max_vy))
        return ref_v

    def _arc_velocity_ref(self, target_rel: np.ndarray, obstacle_rel: np.ndarray) -> np.ndarray:
        radial = -np.array(obstacle_rel, dtype=float)
        dist = _safe_norm(radial)
        radial_unit = radial / dist
        tangent = np.array([-radial_unit[1], radial_unit[0]], dtype=float)
        if self.pass_side < 0:
            tangent = -tangent

        track_ref = self._tracking_velocity_ref(target_rel)
        arc_speed = self.cfg.arc_tangent_speed + 0.12 * np.clip(target_rel[0] - self.cfg.keep_distance, 0.0, 1.0)
        arc_speed = float(np.clip(arc_speed, 0.20, self.cfg.max_vx))

        radial_term = -self.cfg.arc_radial_gain * (dist - self.cfg.arc_radius) * radial_unit
        goal_dir = _normalize(target_rel - np.array([self.cfg.keep_distance, 0.0], dtype=float))
        goal_term = self.cfg.arc_goal_gain * goal_dir
        ref_v = arc_speed * tangent + radial_term + goal_term + 0.35 * track_ref

        if obstacle_rel[0] > 0.0 and ref_v[0] < self.cfg.min_forward_speed:
            ref_v[0] = self.cfg.min_forward_speed
        min_side_speed = 0.06
        if self.pass_side > 0:
            ref_v[1] = max(ref_v[1], min_side_speed)
        else:
            ref_v[1] = min(ref_v[1], -min_side_speed)

        ref_v[0] = float(np.clip(ref_v[0], -self.cfg.max_vx, self.cfg.max_vx))
        ref_v[1] = float(np.clip(ref_v[1], -self.cfg.max_vy, self.cfg.max_vy))
        return ref_v

    def _cbf_project(self, obstacle_rel: np.ndarray, u_nom: np.ndarray) -> Dict[str, np.ndarray]:
        q = -np.array(obstacle_rel, dtype=float)
        v = self.cmd_v.copy()
        h = float(np.dot(q, q) - self.cfg.safe_radius ** 2)
        hdot = float(2.0 * np.dot(q, v))
        psi1 = hdot + self.cfg.cbf_c1 * h
        b_vec = 2.0 * q
        c_val = 2.0 * float(np.dot(v, v)) + self.cfg.cbf_c1 * hdot + self.cfg.cbf_c2 * psi1
        required = -c_val
        score = float(np.dot(b_vec, u_nom))
        u_safe = u_nom.copy()
        if score < required:
            denom = float(np.dot(b_vec, b_vec)) + 1e-6
            u_safe = u_nom + ((required - score) / denom) * b_vec
        return {"u_safe": u_safe, "h": np.array([h]), "psi1": np.array([psi1])}

    def update(
        self,
        target_rel: Optional[np.ndarray],
        obstacle_measurement: Optional[np.ndarray],
        dt: float,
        yaw_rate: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        dt = max(float(dt), 1e-3)

        if target_rel is None:
            self._update_obstacle_memory(obstacle_measurement, dt, yaw_rate)
            self.cmd_v *= 0.85
            self.last_u = -self.cfg.vel_damping * self.cmd_v
            self.ego_pose = self.ego_pose + dt * self._body_to_world(self.cmd_v)
            self.last_mode = "search"
            return {
                "mode": np.array([0.0]),
                "cmd_v": self.cmd_v.copy(),
                "u_safe": self.last_u.copy(),
                "phi_hat": self.observer.phi_hat.copy(),
                "target_hat_v": self.observer.v_hat.copy(),
                "obstacle_rel": np.zeros(2, dtype=float) if self.obstacle_rel is None else self.obstacle_rel.copy(),
                "ego_pose": self.ego_pose.copy(),
                "barrier": np.array([math.nan]),
                "psi1": np.array([math.nan]),
                "pass_side": np.array([self.pass_side], dtype=float),
            }

        target_rel = np.array(target_rel, dtype=float)
        obs_state = self.observer.update(target_rel, -self.last_u, dt)
        phi_hat = obs_state["phi_hat"]
        target_hat_v = obs_state["v_hat"]

        self._update_obstacle_memory(obstacle_measurement, dt, yaw_rate)
        obstacle_rel = None if self.obstacle_rel is None else np.array(self.obstacle_rel, dtype=float)

        if obstacle_rel is not None:
            blocking = self._is_blocking(target_rel, obstacle_rel)
            if blocking and not self.avoidance_active:
                self.pass_side = self._choose_side(obstacle_rel, target_rel)
            self.avoidance_active = self.avoidance_active or blocking
            if obstacle_rel[0] < self.cfg.release_rear_x:
                self.avoidance_active = False
        else:
            self.avoidance_active = False

        if self.avoidance_active and obstacle_rel is not None:
            ref_v = self._arc_velocity_ref(target_rel, obstacle_rel)
            mode_name = "avoid_arc"
        else:
            ref_v = self._tracking_velocity_ref(target_rel)
            mode_name = "track"

        u_nom = self.cfg.accel_k * (ref_v - self.cmd_v) - self.cfg.vel_damping * self.cmd_v + self.cfg.observer_comp_gain * phi_hat
        u_nom[0] = float(np.clip(u_nom[0], -self.cfg.max_ax, self.cfg.max_ax))
        u_nom[1] = float(np.clip(u_nom[1], -self.cfg.max_ay, self.cfg.max_ay))

        barrier = np.array([math.nan])
        psi1 = np.array([math.nan])
        u_safe = u_nom.copy()
        if self.cfg.use_cbf and self.avoidance_active and obstacle_rel is not None:
            cbf_state = self._cbf_project(obstacle_rel, u_nom)
            u_safe = cbf_state["u_safe"]
            barrier = cbf_state["h"]
            psi1 = cbf_state["psi1"]

        u_safe[0] = float(np.clip(u_safe[0], -self.cfg.max_ax, self.cfg.max_ax))
        u_safe[1] = float(np.clip(u_safe[1], -self.cfg.max_ay, self.cfg.max_ay))
        self.cmd_v = self.cmd_v + dt * u_safe
        self.cmd_v[0] = float(np.clip(self.cmd_v[0], -self.cfg.max_vx, self.cfg.max_vx))
        self.cmd_v[1] = float(np.clip(self.cmd_v[1], -self.cfg.max_vy, self.cfg.max_vy))
        self.last_u = u_safe.copy()
        self.ego_pose = self.ego_pose + dt * self._body_to_world(self.cmd_v)
        self.last_mode = mode_name

        return {
            "mode": np.array([1.0 if mode_name == "avoid_arc" else 2.0]),
            "cmd_v": self.cmd_v.copy(),
            "u_safe": u_safe.copy(),
            "phi_hat": phi_hat.copy(),
            "target_hat_v": target_hat_v.copy(),
            "obstacle_rel": np.zeros(2, dtype=float) if obstacle_rel is None else obstacle_rel.copy(),
            "ego_pose": self.ego_pose.copy(),
            "barrier": barrier.copy(),
            "psi1": psi1.copy(),
            "pass_side": np.array([self.pass_side], dtype=float),
        }
