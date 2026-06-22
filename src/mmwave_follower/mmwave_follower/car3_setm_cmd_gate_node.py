#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import os
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class Car3SetmCmdGateNode(Node):
    def __init__(self):
        super().__init__('car3_setm_cmd_gate_node')

        self.declare_parameter('raw_cmd_topic', '/car3/cmd_vel_raw')
        self.declare_parameter('cmd_topic', '/car3/cmd_vel')
        self.declare_parameter('csv_path', '/tmp/car3_setm_gate_log.csv')

        self.declare_parameter('control_period_sec', 0.05)
        self.declare_parameter('raw_cmd_timeout_sec', 0.8)
        self.declare_parameter('max_hold_sec', 0.35)

        self.declare_parameter('epsilon_z', 0.12)
        self.declare_parameter('delta_far', 0.05)
        self.declare_parameter('delta_near', 0.25)
        self.declare_parameter('rho', 0.0004)

        self.raw_cmd_topic = str(self.get_parameter('raw_cmd_topic').value)
        self.cmd_topic = str(self.get_parameter('cmd_topic').value)
        self.csv_path = str(self.get_parameter('csv_path').value)

        self.control_period_sec = float(self.get_parameter('control_period_sec').value)
        self.raw_cmd_timeout_sec = float(self.get_parameter('raw_cmd_timeout_sec').value)
        self.max_hold_sec = float(self.get_parameter('max_hold_sec').value)

        self.epsilon_z = float(self.get_parameter('epsilon_z').value)
        self.delta_far = float(self.get_parameter('delta_far').value)
        self.delta_near = float(self.get_parameter('delta_near').value)
        self.rho = float(self.get_parameter('rho').value)

        self.raw_cmd = Twist()
        self.held_cmd = Twist()

        self.last_raw_time = 0.0
        self.last_trigger_time = 0.0
        self.last_trigger_z = None

        self.timer_count = 0
        self.trigger_count = 0

        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            't',
            'z_x', 'z_y', 'z_norm',
            'raw_vx', 'raw_vy', 'raw_wz',
            'held_vx', 'held_vy', 'held_wz',
            'trigger', 'reason',
            'timer_count', 'trigger_count',
            'trigger_ratio', 'reduction_percent'
        ])
        self.csv_file.flush()

        self.sub_raw = self.create_subscription(
            Twist,
            self.raw_cmd_topic,
            self.raw_cmd_callback,
            10
        )

        self.pub_cmd = self.create_publisher(
            Twist,
            self.cmd_topic,
            10
        )

        self.create_timer(self.control_period_sec, self.timer_callback)

        self.get_logger().info('=======================================')
        self.get_logger().info('Car3 SETM command gate started')
        self.get_logger().info(f'raw_cmd_topic = {self.raw_cmd_topic}')
        self.get_logger().info(f'cmd_topic     = {self.cmd_topic}')
        self.get_logger().info(f'csv_path      = {self.csv_path}')
        self.get_logger().info('=======================================')

    def raw_cmd_callback(self, msg):
        self.raw_cmd = msg
        self.last_raw_time = time.time()

    @staticmethod
    def is_zero_cmd(cmd, eps=1e-5):
        return (
            abs(cmd.linear.x) < eps and
            abs(cmd.linear.y) < eps and
            abs(cmd.angular.z) < eps
        )

    @staticmethod
    def copy_twist(src):
        dst = Twist()
        dst.linear.x = float(src.linear.x)
        dst.linear.y = float(src.linear.y)
        dst.linear.z = float(src.linear.z)
        dst.angular.x = float(src.angular.x)
        dst.angular.y = float(src.angular.y)
        dst.angular.z = float(src.angular.z)
        return dst

    def publish_zero(self, reason, now):
        zero = Twist()
        self.held_cmd = zero
        self.pub_cmd.publish(zero)
        self.log_row(now, 0.0, 0.0, 0.0, trigger=0, reason=reason)

    def timer_callback(self):
        now = time.time()
        self.timer_count += 1

        raw_valid = (now - self.last_raw_time) <= self.raw_cmd_timeout_sec

        if not raw_valid:
            self.publish_zero('raw_timeout_stop', now)
            return

        z_x = float(self.raw_cmd.linear.x)
        z_y = float(self.raw_cmd.linear.y)
        z_norm = math.sqrt(z_x * z_x + z_y * z_y)
        z_now = (z_x, z_y)

        trigger = False
        reason = 'hold'

        if self.last_trigger_z is None:
            trigger = True
            reason = 'init'

        elif self.is_zero_cmd(self.raw_cmd) and not self.is_zero_cmd(self.held_cmd):
            trigger = True
            reason = 'stop_priority'

        elif now - self.last_trigger_time >= self.max_hold_sec:
            trigger = True
            reason = 'max_hold'

        else:
            dz_x = self.last_trigger_z[0] - z_now[0]
            dz_y = self.last_trigger_z[1] - z_now[1]
            lhs = dz_x * dz_x + dz_y * dz_y

            sigma = self.delta_far if z_norm >= self.epsilon_z else self.delta_near
            rhs = sigma * z_norm * z_norm + self.rho

            if lhs >= rhs:
                trigger = True
                reason = f'setm_lhs={lhs:.5f}_rhs={rhs:.5f}'

        if trigger:
            self.held_cmd = self.copy_twist(self.raw_cmd)
            self.last_trigger_z = z_now
            self.last_trigger_time = now
            self.trigger_count += 1

        self.pub_cmd.publish(self.held_cmd)
        self.log_row(now, z_x, z_y, z_norm, int(trigger), reason)

        if self.timer_count % 20 == 0:
            reduction = 100.0 * (1.0 - self.trigger_count / max(1, self.timer_count))
            self.get_logger().info(
                f'Car3 SETM trigger={trigger} reason={reason} | '
                f'raw=({self.raw_cmd.linear.x:.2f},{self.raw_cmd.linear.y:.2f},{self.raw_cmd.angular.z:.2f}) | '
                f'held=({self.held_cmd.linear.x:.2f},{self.held_cmd.linear.y:.2f},{self.held_cmd.angular.z:.2f}) | '
                f'count={self.trigger_count}/{self.timer_count}, reduction={reduction:.1f}%'
            )

    def log_row(self, now, z_x, z_y, z_norm, trigger, reason):
        trigger_ratio = self.trigger_count / max(1, self.timer_count)
        reduction_percent = 100.0 * (1.0 - trigger_ratio)

        self.csv_writer.writerow([
            now,
            z_x, z_y, z_norm,
            float(self.raw_cmd.linear.x), float(self.raw_cmd.linear.y), float(self.raw_cmd.angular.z),
            float(self.held_cmd.linear.x), float(self.held_cmd.linear.y), float(self.held_cmd.angular.z),
            int(trigger), reason,
            int(self.timer_count), int(self.trigger_count),
            trigger_ratio, reduction_percent
        ])
        self.csv_file.flush()

    def destroy_node(self):
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Car3SetmCmdGateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()