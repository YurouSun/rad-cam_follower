#!/usr/bin/env python3
"""Plot CSV logs exported by the virtual arc avoidance simulation or ROS node."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def _get_field(data: np.ndarray, name: str, default=None):
    return data[name] if name in data.dtype.names else default


def _build_traj_mask(data: np.ndarray) -> np.ndarray:
    n = len(data)
    mask = np.ones(n, dtype=bool)
    if "arc_active" in data.dtype.names:
        arc_active = np.asarray(data["arc_active"], dtype=float) > 0.5
        active_idx = np.flatnonzero(arc_active)
        if active_idx.size > 0:
            start_idx = int(active_idx[0])
            end_idx = int(active_idx[0])
            for idx in active_idx[1:]:
                if int(idx) == end_idx + 1:
                    end_idx = int(idx)
                else:
                    break
            tail = min(n - 1, end_idx + 3)
            trimmed = np.zeros(n, dtype=bool)
            trimmed[: tail + 1] = True
            return trimmed

    if "mode" in data.dtype.names:
        mode = np.asarray(data["mode"]).astype(str)
        active_idx = np.flatnonzero(mode == "arc_follow")
        if active_idx.size > 0:
            end_idx = int(active_idx[-1])
            tail = min(n - 1, end_idx + 3)
            trimmed = np.zeros(n, dtype=bool)
            trimmed[: tail + 1] = True
            return trimmed

    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot arc avoidance CSV log")
    parser.add_argument("csv_path", type=str, help="path to arc_avoidance_log.csv")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="directory for figures, defaults to CSV directory",
    )
    parser.add_argument(
        "--y-sign",
        type=float,
        default=-1.0,
        help="multiply all y-related plotted signals by this sign",
    )
    parser.add_argument(
        "--show-safety-circle",
        action="store_true",
        help="also draw the full safety circle around the obstacle",
    )
    args = parser.parse_args()

    data = np.genfromtxt(args.csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.csv_path))
    os.makedirs(out_dir, exist_ok=True)

    t = data["time"]
    t_rel = t - t[0]
    traj_mask = _build_traj_mask(data)
    ref_x = _get_field(data, "ref_x")
    ref_y = _get_field(data, "ref_y")
    pose_x = _get_field(data, "pose_x")
    pose_y = _get_field(data, "pose_y")
    pose_noisy_x = _get_field(data, "pose_noisy_x")
    pose_noisy_y = _get_field(data, "pose_noisy_y")
    cmd_vx = _get_field(data, "cmd_vx")
    cmd_vy = _get_field(data, "cmd_vy")
    published_vx = _get_field(data, "published_vx")
    published_vy = _get_field(data, "published_vy")
    real_vx = _get_field(data, "real_vx")
    real_vy = _get_field(data, "real_vy")
    execution_mismatch_x = _get_field(data, "execution_mismatch_x", _get_field(data, "injected_disturbance_x"))
    execution_mismatch_y = _get_field(data, "execution_mismatch_y", _get_field(data, "injected_disturbance_y"))
    measured_residual_x = _get_field(data, "measured_residual_x", _get_field(data, "disturbance_x"))
    measured_residual_y = _get_field(data, "measured_residual_y", _get_field(data, "disturbance_y"))
    residual_hat_x = _get_field(data, "residual_hat_x", _get_field(data, "disturbance_hat_x"))
    residual_hat_y = _get_field(data, "residual_hat_y", _get_field(data, "disturbance_hat_y"))
    obstacle_x = _get_field(data, "obstacle_x")
    obstacle_y = _get_field(data, "obstacle_y")
    arc_radius = _get_field(data, "arc_radius")
    target_x = _get_field(data, "target_x")
    target_y = _get_field(data, "target_y")

    if obstacle_x is None:
        obstacle_x = target_x
    if obstacle_y is None:
        obstacle_y = target_y

    y_sign = float(args.y_sign)
    if ref_y is not None:
        ref_y = y_sign * ref_y
    if pose_y is not None:
        pose_y = y_sign * pose_y
    if pose_noisy_y is not None:
        pose_noisy_y = y_sign * pose_noisy_y
    if obstacle_y is not None:
        obstacle_y = y_sign * obstacle_y
    if target_y is not None:
        target_y = y_sign * target_y
    if cmd_vy is not None:
        cmd_vy = y_sign * cmd_vy
    if published_vy is not None:
        published_vy = y_sign * published_vy
    if real_vy is not None:
        real_vy = y_sign * real_vy
    if execution_mismatch_y is not None:
        execution_mismatch_y = y_sign * execution_mismatch_y
    if measured_residual_y is not None:
        measured_residual_y = y_sign * measured_residual_y
    if residual_hat_y is not None:
        residual_hat_y = y_sign * residual_hat_y

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    if ref_x is not None and ref_y is not None:
        ax1.plot(ref_x[traj_mask], ref_y[traj_mask], "--", linewidth=2.2, label="reference arc")
    if pose_x is not None and pose_y is not None:
        ax1.plot(pose_x[traj_mask], pose_y[traj_mask], linewidth=2.2, label="disturbed trajectory")
    if obstacle_x is not None and obstacle_y is not None:
        ax1.plot(obstacle_x[0], obstacle_y[0], "ro", markersize=8, label="obstacle")
        if args.show_safety_circle and arc_radius is not None:
            theta = np.linspace(0.0, 2.0 * np.pi, 200)
            ax1.plot(
                obstacle_x[0] + float(arc_radius[0]) * np.cos(theta),
                obstacle_y[0] + float(arc_radius[0]) * np.sin(theta),
                "r:",
                linewidth=1.2,
                label="arc radius",
            )
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.grid(True, which="both", alpha=0.35)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "real_log_pose.png"), dpi=180)

    fig2, axes2 = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    axes2[0].plot(t_rel, cmd_vx, linewidth=1.6, label="cmd_vx")
    if published_vx is not None:
        axes2[0].plot(t_rel, published_vx, ":", linewidth=1.2, label="published_vx")
    if real_vx is not None:
        axes2[0].plot(t_rel, real_vx, "--", linewidth=1.6, label="real_vx")
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.35)

    axes2[1].plot(t_rel, cmd_vy, linewidth=1.6, label="cmd_vy")
    if published_vy is not None:
        axes2[1].plot(t_rel, published_vy, ":", linewidth=1.2, label="published_vy")
    if real_vy is not None:
        axes2[1].plot(t_rel, real_vy, "--", linewidth=1.6, label="real_vy")
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.35)

    if execution_mismatch_x is not None:
        axes2[2].plot(t_rel, execution_mismatch_x, ":", linewidth=1.1, label="execution_mismatch_x")
    if execution_mismatch_y is not None:
        axes2[2].plot(t_rel, execution_mismatch_y, ":", linewidth=1.1, label="execution_mismatch_y")
    if measured_residual_x is not None:
        axes2[2].plot(t_rel, measured_residual_x, linewidth=1.4, label="measured_residual_x")
    if measured_residual_y is not None:
        axes2[2].plot(t_rel, measured_residual_y, linewidth=1.4, label="measured_residual_y")
    if residual_hat_x is not None:
        axes2[2].plot(t_rel, residual_hat_x, "--", linewidth=1.3, label="residual_hat_x")
    if residual_hat_y is not None:
        axes2[2].plot(t_rel, residual_hat_y, "--", linewidth=1.3, label="residual_hat_y")
    axes2[2].legend()
    axes2[2].grid(True, alpha=0.35)
    axes2[2].set_xlabel("time since start (s)")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "real_log_observer.png"), dpi=180)

    print(f"figures saved to {out_dir}")


if __name__ == "__main__":
    main()
