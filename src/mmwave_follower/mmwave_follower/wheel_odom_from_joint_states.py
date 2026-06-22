#!/usr/bin/env python3
#发布weel odom

from __future__ import annotations

import math
import time
from typing import Dict, Optional

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState


class WheelOdomFromJointStates(Node):
    def __init__(self) -> None:
        super().__init__("wheel_odom_from_joint_states")

        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("odom_topic", "/wheel_odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("wheel_lf_joint", "wheel_lf_Joint")
        self.declare_parameter("wheel_rf_joint", "wheel_rf_Joint")
        self.declare_parameter("wheel_lb_joint", "wheel_lb_Joint")
        self.declare_parameter("wheel_rb_joint", "wheel_rb_Joint")
        self.declare_parameter("wheelbase", 0.1368)
        self.declare_parameter("track_width", 0.1410)
        self.declare_parameter("wheel_diameter", 0.065)
        self.declare_parameter("joint_sign_lf", -1.0)
        self.declare_parameter("joint_sign_lb", -1.0)
        self.declare_parameter("joint_sign_rf", 1.0)
        self.declare_parameter("joint_sign_rb", 1.0)

        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.wheelbase = float(self.get_parameter("wheelbase").value)
        self.track_width = float(self.get_parameter("track_width").value)
        self.wheel_radius = 0.5 * float(self.get_parameter("wheel_diameter").value)
        self.angular_span = 0.5 * (self.wheelbase + self.track_width)

        self.wheel_names = {
            "lf": str(self.get_parameter("wheel_lf_joint").value),
            "rf": str(self.get_parameter("wheel_rf_joint").value),
            "lb": str(self.get_parameter("wheel_lb_joint").value),
            "rb": str(self.get_parameter("wheel_rb_joint").value),
        }
        self.joint_sign = {
            "lf": float(self.get_parameter("joint_sign_lf").value),
            "lb": float(self.get_parameter("joint_sign_lb").value),
            "rf": float(self.get_parameter("joint_sign_rf").value),
            "rb": float(self.get_parameter("joint_sign_rb").value),
        }

        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.create_subscription(JointState, self.joint_state_topic, self.on_joint_state, 10)

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_stamp: Optional[float] = None
        self.last_msg_time = 0.0
        self.last_wheel_stamp: Optional[float] = None
        self.last_wheel_position: Dict[str, float] = {}

        self.get_logger().info(
            f"Publishing wheel odom on {self.odom_topic} from {self.joint_state_topic}"
        )

    def on_joint_state(self, msg: JointState) -> None:
        self.last_msg_time = time.time()
        if not msg.name:
            self.get_logger().warn(
                "JointState has no wheel names; cannot build wheel odometry.",
                throttle_duration_sec=2.0,
            )
            return

        stamp = self._stamp_to_sec(msg)
        if stamp is None:
            stamp = time.time()

        wheel_lin = self._extract_wheel_linear_speeds(msg, stamp)
        if wheel_lin is None:
            return

        vx, vy, wz = self._inverse_mecanum(wheel_lin)

        if self.last_stamp is None:
            dt = 0.0
        else:
            dt = max(0.0, min(0.2, stamp - self.last_stamp))
        self.last_stamp = stamp

        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        self.x += (vx * cos_yaw - vy * sin_yaw) * dt
        self.y += (vx * sin_yaw + vy * cos_yaw) * dt
        self.yaw += wz * dt

        odom = Odometry()
        odom.header = msg.header
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.z = math.sin(self.yaw * 0.5)
        odom.pose.pose.orientation.w = math.cos(self.yaw * 0.5)
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = wz
        self.odom_pub.publish(odom)

    def _extract_wheel_linear_speeds(self, msg: JointState, stamp: float) -> Optional[Dict[str, float]]:
        name_to_index = {name: idx for idx, name in enumerate(msg.name)}
        wheel_lin: Dict[str, float] = {}
        velocity_count = len(msg.velocity)
        position_count = len(msg.position)
        use_position_fallback = velocity_count == 0 or velocity_count < len(msg.name)
        position_dt = None if self.last_wheel_stamp is None else max(0.0, stamp - self.last_wheel_stamp)

        for key, joint_name in self.wheel_names.items():
            idx = name_to_index.get(joint_name)
            if idx is None:
                self.get_logger().warn(
                    f"Missing wheel joint {joint_name} on {self.joint_state_topic}.",
                    throttle_duration_sec=2.0,
                )
                return None
            if idx < velocity_count and math.isfinite(float(msg.velocity[idx])):
                wheel_lin[key] = self.joint_sign[key] * float(msg.velocity[idx]) * self.wheel_radius
                continue

            if idx < position_count and position_dt is not None and position_dt > 1e-4:
                current_pos = float(msg.position[idx])
                last_pos = self.last_wheel_position.get(key)
                if last_pos is not None and math.isfinite(current_pos):
                    angular_speed = (current_pos - last_pos) / position_dt
                    wheel_lin[key] = self.joint_sign[key] * angular_speed * self.wheel_radius
                    continue

            if use_position_fallback:
                self._update_position_cache(msg, name_to_index, stamp)
                self.get_logger().warn(
                    "JointState velocity is unavailable; waiting for enough wheel position samples to derive odometry.",
                    throttle_duration_sec=2.0,
                )
                return None

            self.get_logger().warn(
                f"Velocity for wheel joint {joint_name} is unavailable.",
                throttle_duration_sec=2.0,
            )
            return None

        self._update_position_cache(msg, name_to_index, stamp)
        return wheel_lin

    def _update_position_cache(self, msg: JointState, name_to_index: Dict[str, int], stamp: float) -> None:
        for key, joint_name in self.wheel_names.items():
            idx = name_to_index.get(joint_name)
            if idx is None or idx >= len(msg.position):
                continue
            position = float(msg.position[idx])
            if math.isfinite(position):
                self.last_wheel_position[key] = position
        self.last_wheel_stamp = stamp

    def _inverse_mecanum(self, wheel_lin: Dict[str, float]) -> tuple[float, float, float]:
        m1 = wheel_lin["lf"]
        m2 = wheel_lin["lb"]
        m3 = wheel_lin["rf"]
        m4 = wheel_lin["rb"]

        vx = 0.25 * (m1 + m2 + m3 + m4)
        vy = 0.25 * (-m1 + m2 + m3 - m4)
        wz = (m3 + m4 - m1 - m2) / (4.0 * self.angular_span)
        return vx, vy, wz

    @staticmethod
    def _stamp_to_sec(msg: JointState) -> Optional[float]:
        stamp = msg.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            return None
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def main() -> None:
    rclpy.init()
    node = WheelOdomFromJointStates()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
