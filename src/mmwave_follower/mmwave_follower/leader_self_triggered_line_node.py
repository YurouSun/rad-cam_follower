#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import csv
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu


def quaternion_to_yaw(q) -> float:
    """Convert IMU quaternion to yaw angle."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class LeaderSelfTriggeredLineNode(Node):
    """
    Leader vehicle controller for line-platoon hardware experiments.

    Main functions:
    1. Generate a repeatable low-speed leader motion profile.
    2. Hold heading using IMU yaw feedback.
    3. Apply sampled self-triggered command updates.
    4. Log trigger statistics for hardware-experiment analysis.

    Recommended experiment:
        Car1: this leader node
        Car2: follower node tracking Car1
    """

    def __init__(self):
        super().__init__("leader_self_triggered_line_node")

        # =========================================================
        # 1. Basic parameters
        # =========================================================
        self.declare_parameter("cmd_topic", "/tracker/cmd_vel")
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("rate_hz", 20.0)

        # =========================================================
        # 2. Leader velocity profile parameters
        # =========================================================
        # Profile:
        # [0, t_wait): stop
        # [t_wait, t_wait + t_ramp): ramp from 0 to vx_high
        # [t_wait + t_ramp, t_stage1): high-speed cruising
        # [t_stage1, t_stage2): linearly decelerate from vx_high to vx_low
        # [t_stage2, total_duration): low-speed cruising
        # [total_duration, end): stop
        self.declare_parameter("t_wait", 1.0)
        self.declare_parameter("t_ramp", 2.0)
        self.declare_parameter("t_stage1", 10.0)
        self.declare_parameter("t_stage2", 15.0)
        self.declare_parameter("total_duration", 20.0)

        self.declare_parameter("vx_high", 0.30)
        self.declare_parameter("vx_low", 0.20)
        self.declare_parameter("vy_ref", 0.0)
        # =========================================================
        # 2.1 Velocity constraint parameters
        # =========================================================
        self.declare_parameter("enable_velocity_constraint", True)
        self.declare_parameter("v_min_move", 0.20)
        self.declare_parameter("v_max", 0.30)
        self.declare_parameter("stop_epsilon", 1e-3)

        # =========================================================
        # 3. Yaw-hold parameters
        # =========================================================
        self.declare_parameter("enable_yaw_hold", True)
        self.declare_parameter("kp_yaw", 1.0)
        self.declare_parameter("max_wz", 0.25)
        # Fixed-time yaw-hold parameters
        self.declare_parameter("enable_fixed_time_yaw", False)
        self.declare_parameter("alpha_yaw", 0.7)
        self.declare_parameter("beta_yaw", 1.3)
        self.declare_parameter("k_yaw_low", 0.6)
        self.declare_parameter("k_yaw_high", 1.0)
        self.declare_parameter("sign_epsilon", 1e-3)

        # =========================================================
        # 4. Self-triggered parameters
        # =========================================================
        self.declare_parameter("enable_self_triggered", True)
        self.declare_parameter("trigger_threshold", 0.01)
        self.declare_parameter("max_hold_time", 0.20)

        # Optional command heartbeat:
        # If your chassis watchdog requires continuous cmd_vel,
        # set enable_cmd_heartbeat=True. Heartbeat publishes the
        # last command but does NOT count as a new control update.
        self.declare_parameter("enable_cmd_heartbeat", False)
        self.declare_parameter("heartbeat_period", 0.20)

        # =========================================================
        # 5. Comparison mode
        # =========================================================
        # For leader-follower experiments, it is recommended to set
        # enable_comparison_mode=False and run periodic/self-triggered
        # experiments separately with the same initial conditions.
        self.declare_parameter("enable_comparison_mode", False)
        self.declare_parameter("transition_pause", 2.0)

        # =========================================================
        # 6. Stop command repeat
        # =========================================================
        self.declare_parameter("stop_repeat_max", 10)

        # =========================================================
        # 7. CSV log parameters
        # =========================================================
        self.declare_parameter("enable_csv_log", True)
        self.declare_parameter(
            "csv_path",
            "/tmp/car1_leader_self_triggered_line_log.csv"
        )

        # =========================================================
        # Read parameters
        # =========================================================
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)

        self.t_wait = float(self.get_parameter("t_wait").value)
        self.t_ramp = float(self.get_parameter("t_ramp").value)
        self.t_stage1 = float(self.get_parameter("t_stage1").value)
        self.t_stage2 = float(self.get_parameter("t_stage2").value)
        self.total_duration = float(self.get_parameter("total_duration").value)

        self.vx_high = float(self.get_parameter("vx_high").value)
        self.vx_low = float(self.get_parameter("vx_low").value)
        self.vy_ref = float(self.get_parameter("vy_ref").value)
        self.enable_velocity_constraint = bool(self.get_parameter("enable_velocity_constraint").value)
        self.v_min_move = float(self.get_parameter("v_min_move").value)
        self.v_max = float(self.get_parameter("v_max").value)
        self.stop_epsilon = float(self.get_parameter("stop_epsilon").value)

        self.enable_yaw_hold = bool(self.get_parameter("enable_yaw_hold").value)
        self.kp_yaw = float(self.get_parameter("kp_yaw").value)
        self.max_wz = float(self.get_parameter("max_wz").value)
        self.enable_fixed_time_yaw = bool(self.get_parameter("enable_fixed_time_yaw").value)
        self.alpha_yaw = float(self.get_parameter("alpha_yaw").value)
        self.beta_yaw = float(self.get_parameter("beta_yaw").value)
        self.k_yaw_low = float(self.get_parameter("k_yaw_low").value)
        self.k_yaw_high = float(self.get_parameter("k_yaw_high").value)
        self.sign_epsilon = float(self.get_parameter("sign_epsilon").value)

        self.enable_self_triggered = bool(self.get_parameter("enable_self_triggered").value)
        self.trigger_threshold = float(self.get_parameter("trigger_threshold").value)
        self.max_hold_time = float(self.get_parameter("max_hold_time").value)

        self.enable_cmd_heartbeat = bool(self.get_parameter("enable_cmd_heartbeat").value)
        self.heartbeat_period = float(self.get_parameter("heartbeat_period").value)

        self.enable_comparison_mode = bool(self.get_parameter("enable_comparison_mode").value)
        self.transition_pause = float(self.get_parameter("transition_pause").value)

        self.stop_repeat_max = int(self.get_parameter("stop_repeat_max").value)

        self.enable_csv_log = bool(self.get_parameter("enable_csv_log").value)
        self.csv_path = str(self.get_parameter("csv_path").value)

        # =========================================================
        # State variables
        # =========================================================
        self.current_yaw = 0.0
        self.has_imu = False
        self.yaw_ref = 0.0
        self.yaw_locked = False

        self.finished = False
        self.in_transition = False
        self.transition_start_time = 0.0

        self.phase_index = 0
        self.phase_start_time = time.monotonic()

        self.last_cmd_sent = (0.0, 0.0, 0.0)
        self.last_trigger_time = self.phase_start_time
        self.last_publish_time = self.phase_start_time
        self.last_inter_trigger_interval = 0.0

        self.timer_count = 0
        self.motion_timer_count = 0

        self.trigger_count = 0
        self.motion_trigger_count = 0
        self.stop_trigger_count = 0
        self.heartbeat_count = 0

        self.stop_repeat_count = 0
        self.phase_results = []

        # =========================================================
        # Phase configuration
        # =========================================================
        if self.enable_comparison_mode:
            self.phases = [
                {"name": "periodic", "enable_self_triggered": False},
                {"name": "self_triggered", "enable_self_triggered": True},
            ]
        else:
            self.phases = [
                {
                    "name": "single_run_self_triggered"
                    if self.enable_self_triggered
                    else "single_run_periodic",
                    "enable_self_triggered": self.enable_self_triggered,
                }
            ]

        self.current_phase_name = self.phases[self.phase_index]["name"]
        self.current_enable_self_triggered = self.phases[self.phase_index]["enable_self_triggered"]

        # =========================================================
        # ROS communication
        # =========================================================
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

        self.imu_sub = self.create_subscription(
            Imu,
            self.imu_topic,
            self.imu_callback,
            qos_profile_sensor_data
        )

        # =========================================================
        # CSV log
        # =========================================================
        self.csv_file = None
        self.csv_writer = None
        if self.enable_csv_log:
            self._init_csv_log()

        # =========================================================
        # Timer
        # =========================================================
        self.timer = self.create_timer(1.0 / self.rate_hz, self.on_timer)

        self.get_logger().info("Leader self-triggered line node started.")
        self.get_logger().info(f"cmd_topic = {self.cmd_topic}")
        self.get_logger().info(f"imu_topic = {self.imu_topic}")
        self.get_logger().info(
            f"phase = {self.current_phase_name}, "
            f"enable_self_triggered = {self.current_enable_self_triggered}"
        )
        self.get_logger().info(
            f"profile: t_wait={self.t_wait}, t_ramp={self.t_ramp}, "
            f"t_stage1={self.t_stage1}, t_stage2={self.t_stage2}, "
            f"total_duration={self.total_duration}, "
            f"vx_high={self.vx_high}, vx_low={self.vx_low}"
        )

    # =============================================================
    # CSV log
    # =============================================================
    def _init_csv_log(self):
        path = Path(self.csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.csv_file = open(path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            "phase_name",
            "enable_self_triggered",
            "t",
            "vx_ref",
            "vy_ref",
            "yaw_ref",
            "yaw_meas",
            "yaw_error",
            "vx_cmd_new",
            "vy_cmd_new",
            "wz_cmd_new",
            "vx_cmd_sent",
            "vy_cmd_sent",
            "wz_cmd_sent",
            "trigger_flag",
            "heartbeat_flag",
            "trigger_reason",
            "cmd_diff",
            "hold_time",
            "inter_trigger_interval",
            "timer_count",
            "motion_timer_count",
            "trigger_count",
            "motion_trigger_count",
            "stop_trigger_count",
            "heartbeat_count",
        ])

        self.get_logger().info(f"CSV log path: {self.csv_path}")

    # =============================================================
    # IMU callback
    # =============================================================
    def imu_callback(self, msg: Imu):
        self.current_yaw = quaternion_to_yaw(msg.orientation)
        self.has_imu = True

        if self.enable_yaw_hold and not self.yaw_locked:
            self.yaw_ref = self.current_yaw
            self.yaw_locked = True
            self.get_logger().info(
                f"[{self.current_phase_name}] yaw reference locked: "
                f"{self.yaw_ref:.3f} rad"
            )

    # =============================================================
    # Reference profile
    # =============================================================
    def generate_reference(self, t: float):
        """
        Generate a smooth low-speed line-platoon leader profile.
        """
        # Defensive normalization of timing parameters
        t_wait = max(0.0, self.t_wait)
        t_ramp = max(0.0, self.t_ramp)

        t_ramp_end = t_wait + t_ramp
        t_stage1 = max(self.t_stage1, t_ramp_end)
        t_stage2 = max(self.t_stage2, t_stage1 + 1e-6)
        total_duration = max(self.total_duration, t_stage2 + 1e-6)

        if t < t_wait:
            vx_ref = 0.0

        elif t < t_ramp_end:
            if t_ramp <= 1e-6:
                vx_ref = self.vx_high
            else:
                ratio = (t - t_wait) / t_ramp
                ratio = clamp(ratio, 0.0, 1.0)
                vx_ref = self.vx_high * ratio

        elif t < t_stage1:
            vx_ref = self.vx_high

        elif t < t_stage2:
            ratio = (t - t_stage1) / (t_stage2 - t_stage1)
            ratio = clamp(ratio, 0.0, 1.0)
            vx_ref = self.vx_high + (self.vx_low - self.vx_high) * ratio

        elif t < total_duration:
            vx_ref = self.vx_low

        else:
            vx_ref = 0.0

        vy_ref = self.vy_ref
        yaw_ref = self.yaw_ref if self.yaw_locked else 0.0

        return vx_ref, vy_ref, yaw_ref

    # =============================================================
    # Control
    # =============================================================
    @staticmethod
    def wrap_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    


    def compute_control(self, vx_ref: float, vy_ref: float, yaw_ref: float):
        """
        Simple leader control:
        - longitudinal/lateral velocity directly follows reference;
        - angular velocity uses proportional yaw-hold feedback.
        """
        vx_cmd = vx_ref
        vy_cmd = vy_ref

        yaw_error = 0.0
        wz_cmd = 0.0

        if self.enable_yaw_hold and self.has_imu:
            yaw_error = self.wrap_angle(yaw_ref - self.current_yaw)
            wz_cmd = self.kp_yaw * yaw_error
            wz_cmd = clamp(wz_cmd, -self.max_wz, self.max_wz)

        return (vx_cmd, vy_cmd, wz_cmd), yaw_error

    def should_trigger(self, new_cmd, now: float):
        """
        Sampled self-triggered command update condition.

        Trigger if:
        1. command difference exceeds trigger_threshold;
        2. hold time exceeds max_hold_time;
        3. self-triggered mode is disabled, i.e., periodic mode.
        """
        vx_new, vy_new, wz_new = new_cmd
        vx_old, vy_old, wz_old = self.last_cmd_sent

        cmd_diff = (
            abs(vx_new - vx_old)
            + abs(vy_new - vy_old)
            + 0.8 * abs(wz_new - wz_old)
        )
        hold_time = now - self.last_trigger_time

        if not self.current_enable_self_triggered:
            return True, "periodic", cmd_diff, hold_time

        if cmd_diff >= self.trigger_threshold:
            return True, "cmd_diff", cmd_diff, hold_time

        if hold_time >= self.max_hold_time:
            return True, "max_hold", cmd_diff, hold_time

        return False, "hold", cmd_diff, hold_time

    # =============================================================
    # Publish commands
    # =============================================================
    def publish_cmd(self, cmd):
        vx, vy, wz = cmd

        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wz)

        self.cmd_pub.publish(msg)
        self.last_publish_time = time.monotonic()

    def publish_stop(self):
        self.publish_cmd((0.0, 0.0, 0.0))

    def maybe_publish_heartbeat(self, now: float):
        """
        Optional chassis keep-alive command publication.

        This publishes the last command but does not count as
        a new control update. Use only if the chassis requires
        continuous cmd_vel.
        """
        if not self.enable_cmd_heartbeat:
            return False

        if now - self.last_publish_time >= self.heartbeat_period:
            self.publish_cmd(self.last_cmd_sent)
            self.heartbeat_count += 1
            return True

        return False

    # =============================================================
    # Logging and summary
    # =============================================================
    def log_row(
        self,
        t,
        vx_ref,
        vy_ref,
        yaw_ref,
        yaw_error,
        new_cmd,
        sent_cmd,
        trigger_flag,
        heartbeat_flag,
        trigger_reason,
        cmd_diff,
        hold_time,
        inter_trigger_interval,
    ):
        if not self.enable_csv_log or self.csv_writer is None:
            return

        self.csv_writer.writerow([
            self.current_phase_name,
            int(self.current_enable_self_triggered),
            round(t, 4),
            round(vx_ref, 4),
            round(vy_ref, 4),
            round(yaw_ref, 4),
            round(self.current_yaw, 4),
            round(yaw_error, 4),
            round(new_cmd[0], 4),
            round(new_cmd[1], 4),
            round(new_cmd[2], 4),
            round(sent_cmd[0], 4),
            round(sent_cmd[1], 4),
            round(sent_cmd[2], 4),
            int(trigger_flag),
            int(heartbeat_flag),
            trigger_reason,
            round(cmd_diff, 6),
            round(hold_time, 6),
            round(inter_trigger_interval, 6),
            self.timer_count,
            self.motion_timer_count,
            self.trigger_count,
            self.motion_trigger_count,
            self.stop_trigger_count,
            self.heartbeat_count,
        ])

        self.csv_file.flush()

    def save_phase_result(self):
        motion_ratio = self.motion_trigger_count / max(1, self.motion_timer_count)
        total_ratio = self.trigger_count / max(1, self.timer_count)
        motion_reduction = 1.0 - motion_ratio
        total_reduction = 1.0 - total_ratio

        result = {
            "phase_name": self.current_phase_name,
            "enable_self_triggered": self.current_enable_self_triggered,
            "timer_count": self.timer_count,
            "motion_timer_count": self.motion_timer_count,
            "trigger_count": self.trigger_count,
            "motion_trigger_count": self.motion_trigger_count,
            "stop_trigger_count": self.stop_trigger_count,
            "heartbeat_count": self.heartbeat_count,
            "motion_trigger_ratio": motion_ratio,
            "total_trigger_ratio": total_ratio,
            "motion_reduction": motion_reduction,
            "total_reduction": total_reduction,
        }

        self.phase_results.append(result)

        self.get_logger().info(
            f"[{self.current_phase_name}] summary: "
            f"motion_trigger_count={self.motion_trigger_count}, "
            f"motion_timer_count={self.motion_timer_count}, "
            f"motion_trigger_ratio={motion_ratio:.3f}, "
            f"motion_reduction={100.0 * motion_reduction:.2f}%, "
            f"stop_trigger_count={self.stop_trigger_count}, "
            f"heartbeat_count={self.heartbeat_count}"
        )

    def reset_phase_state(self):
        self.phase_start_time = time.monotonic()

        self.last_cmd_sent = (0.0, 0.0, 0.0)
        self.last_trigger_time = self.phase_start_time
        self.last_publish_time = self.phase_start_time
        self.last_inter_trigger_interval = 0.0

        self.timer_count = 0
        self.motion_timer_count = 0

        self.trigger_count = 0
        self.motion_trigger_count = 0
        self.stop_trigger_count = 0
        self.heartbeat_count = 0

        self.stop_repeat_count = 0

        # Re-lock yaw at each phase
        self.yaw_locked = False

    def start_next_phase(self):
        self.phase_index += 1

        if self.phase_index >= len(self.phases):
            self.finished = True
            self.get_logger().info("All phases finished.")
            self.get_logger().info("========== FINAL COMPARISON SUMMARY ==========")

            for result in self.phase_results:
                self.get_logger().info(
                    f"Phase={result['phase_name']}, "
                    f"enable_self_triggered={result['enable_self_triggered']}, "
                    f"motion_trigger_count={result['motion_trigger_count']}, "
                    f"motion_timer_count={result['motion_timer_count']}, "
                    f"motion_trigger_ratio={result['motion_trigger_ratio']:.3f}, "
                    f"motion_reduction={100.0 * result['motion_reduction']:.2f}%, "
                    f"total_trigger_count={result['trigger_count']}, "
                    f"total_timer_count={result['timer_count']}, "
                    f"total_trigger_ratio={result['total_trigger_ratio']:.3f}"
                )
            return

        self.current_phase_name = self.phases[self.phase_index]["name"]
        self.current_enable_self_triggered = self.phases[self.phase_index]["enable_self_triggered"]

        self.reset_phase_state()
        self.in_transition = False

        self.get_logger().info("==============================================")
        self.get_logger().info(
            f"Start next phase: {self.current_phase_name}, "
            f"enable_self_triggered={self.current_enable_self_triggered}"
        )
        self.get_logger().info("==============================================")

    # =============================================================
    # Main timer
    # =============================================================
    def on_timer(self):
        if self.finished:
            return

        now = time.monotonic()

        # Transition between comparison phases
        if self.in_transition:
            self.publish_stop()
            if now - self.transition_start_time >= self.transition_pause:
                self.start_next_phase()
            return

        self.timer_count += 1

        t = now - self.phase_start_time

        vx_ref, vy_ref, yaw_ref = self.generate_reference(t)
        new_cmd, yaw_error = self.compute_control(vx_ref, vy_ref, yaw_ref)

        # ---------------------------------------------------------
        # Stop stage
        # ---------------------------------------------------------
        if t >= self.total_duration:
            stop_cmd = (0.0, 0.0, 0.0)

            self.publish_stop()
            self.last_cmd_sent = stop_cmd

            self.stop_repeat_count += 1
            self.stop_trigger_count += 1
            self.trigger_count += 1

            self.log_row(
                t=t,
                vx_ref=vx_ref,
                vy_ref=vy_ref,
                yaw_ref=yaw_ref,
                yaw_error=yaw_error,
                new_cmd=stop_cmd,
                sent_cmd=stop_cmd,
                trigger_flag=True,
                heartbeat_flag=False,
                trigger_reason="stop_repeat",
                cmd_diff=0.0,
                hold_time=now - self.last_trigger_time,
                inter_trigger_interval=0.0,
            )

            self.get_logger().info(
                f"[{self.current_phase_name}] stopping... "
                f"{self.stop_repeat_count}/{self.stop_repeat_max}",
                throttle_duration_sec=0.5
            )

            if self.stop_repeat_count >= self.stop_repeat_max:
                self.get_logger().info(
                    f"[{self.current_phase_name}] phase finished. Robot stopped."
                )
                self.save_phase_result()

                if self.phase_index < len(self.phases) - 1:
                    self.in_transition = True
                    self.transition_start_time = time.monotonic()
                    self.get_logger().info(
                        f"Transition pause {self.transition_pause:.2f}s before next phase."
                    )
                else:
                    self.start_next_phase()

            return

        # ---------------------------------------------------------
        # Motion stage
        # ---------------------------------------------------------
        self.motion_timer_count += 1

        trigger_flag, trigger_reason, cmd_diff, hold_time = self.should_trigger(new_cmd, now)

        heartbeat_flag = False
        inter_trigger_interval = 0.0

        if trigger_flag:
            inter_trigger_interval = now - self.last_trigger_time
            self.last_inter_trigger_interval = inter_trigger_interval

            self.publish_cmd(new_cmd)

            self.last_cmd_sent = new_cmd
            self.last_trigger_time = now

            self.trigger_count += 1
            self.motion_trigger_count += 1

        else:
            heartbeat_flag = self.maybe_publish_heartbeat(now)
            inter_trigger_interval = self.last_inter_trigger_interval

        self.get_logger().info(
            f"[{self.current_phase_name}] "
            f"t={t:.2f}, vx_ref={vx_ref:.3f}, "
            f"sent=({self.last_cmd_sent[0]:.3f}, {self.last_cmd_sent[1]:.3f}, {self.last_cmd_sent[2]:.3f}), "
            f"trigger={trigger_flag}, reason={trigger_reason}, "
            f"motion_trigger={self.motion_trigger_count}/{self.motion_timer_count}",
            throttle_duration_sec=1.0
        )

        self.log_row(
            t=t,
            vx_ref=vx_ref,
            vy_ref=vy_ref,
            yaw_ref=yaw_ref,
            yaw_error=yaw_error,
            new_cmd=new_cmd,
            sent_cmd=self.last_cmd_sent,
            trigger_flag=trigger_flag,
            heartbeat_flag=heartbeat_flag,
            trigger_reason=trigger_reason,
            cmd_diff=cmd_diff,
            hold_time=hold_time,
            inter_trigger_interval=inter_trigger_interval,
        )

    # =============================================================
    # Shutdown
    # =============================================================
    def destroy_node(self):
        try:
            self.publish_stop()
        except Exception:
            pass

        if self.csv_file is not None:
            self.csv_file.close()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LeaderSelfTriggeredLineNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()