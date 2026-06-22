#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import os
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from interfaces.msg import ObjectsInfo


class Car2SETMCmdGateNode(Node):
    def __init__(self):
        super().__init__('car2_setm_cmd_gate_node')

        self.declare_parameter('target_id', 1)
        self.declare_parameter('tracked_objects_topic', '/tracked_objects_3d')
        self.declare_parameter('raw_cmd_topic', '/car2/cmd_vel_raw')
        self.declare_parameter('cmd_topic', '/car2/cmd_vel')

        self.declare_parameter('keep_distance', 0.50)

        # SETM parameters
        self.declare_parameter('delta_far', 0.05)
        self.declare_parameter('delta_near', 0.25)
        self.declare_parameter('epsilon_z', 0.12)
        self.declare_parameter('rho', 0.0004)
        self.declare_parameter('max_hold_sec', 0.35)

        self.declare_parameter('target_timeout_sec', 0.8)
        self.declare_parameter('raw_cmd_timeout_sec', 0.8)
        self.declare_parameter('control_period_sec', 0.05)

        self.declare_parameter('enable_csv_log', True)
        self.declare_parameter('csv_path', '/tmp/car2_setm_gate_log.csv')

        self.raw_cmd = Twist()
        self.raw_cmd_time = 0.0

        self.held_cmd = Twist()
        self.last_trigger_time = 0.0
        self.last_trigger_z = None

        self.current_z = None
        self.current_target_x = None
        self.current_target_y = None
        self.target_time = 0.0

        self.timer_count = 0
        self.trigger_count = 0

        self.csv_file = None
        self.csv_writer = None

        if bool(self.get_parameter('enable_csv_log').value):
            csv_path = str(self.get_parameter('csv_path').value)
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            self.csv_file = open(csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                't',
                'target_x',
                'target_y',
                'z_x',
                'z_y',
                'z_norm',
                'raw_vx',
                'raw_vy',
                'raw_wz',
                'held_vx',
                'held_vy',
                'held_wz',
                'trigger',
                'reason',
                'timer_count',
                'trigger_count',
                'trigger_ratio',
                'reduction_percent',
            ])
            self.get_logger().info(f'SETM CSV log path: {csv_path}')

        tracked_topic = str(self.get_parameter('tracked_objects_topic').value)
        raw_topic = str(self.get_parameter('raw_cmd_topic').value)
        cmd_topic = str(self.get_parameter('cmd_topic').value)

        self.sub_target = self.create_subscription(
            ObjectsInfo,
            tracked_topic,
            self.target_callback,
            10
        )

        self.sub_raw_cmd = self.create_subscription(
            Twist,
            raw_topic,
            self.raw_cmd_callback,
            10
        )

        self.pub_cmd = self.create_publisher(Twist, cmd_topic, 10)

        period = float(self.get_parameter('control_period_sec').value)
        self.timer = self.create_timer(period, self.control_loop)

        self.get_logger().info(
            f'Car2 SETM gate started: {raw_topic} -> {cmd_topic}, target topic={tracked_topic}'
        )

    def clone_twist(self, msg):
        out = Twist()
        out.linear.x = float(msg.linear.x)
        out.linear.y = float(msg.linear.y)
        out.linear.z = float(msg.linear.z)
        out.angular.x = float(msg.angular.x)
        out.angular.y = float(msg.angular.y)
        out.angular.z = float(msg.angular.z)
        return out

    def zero_twist(self):
        return Twist()

    def target_callback(self, msg):
        target_id = int(self.get_parameter('target_id').value)
        keep_distance = float(self.get_parameter('keep_distance').value)

        best = None

        for obj in msg.objects:
            try:
                oid = int(getattr(obj, 'id', 0))
                if target_id != -1 and oid != target_id:
                    continue

                pos = getattr(obj, 'position', None)
                if pos is None:
                    continue

                x = float(pos.x)
                y = float(pos.y)

                if x <= 0.0:
                    continue

                best = (x, y)
                break

            except Exception:
                continue

        if best is None:
            return

        x, y = best
        self.current_target_x = x
        self.current_target_y = y

        # z = [longitudinal spacing error, lateral error]
        self.current_z = (x - keep_distance, y)
        self.target_time = time.time()

    def raw_cmd_callback(self, msg):
        self.raw_cmd = self.clone_twist(msg)
        self.raw_cmd_time = time.time()

    def cmd_is_zero(self, msg):
        return (
            abs(msg.linear.x) < 1e-6 and
            abs(msg.linear.y) < 1e-6 and
            abs(msg.angular.z) < 1e-6
        )

    def control_loop(self):
        now = time.time()
        self.timer_count += 1

        target_timeout = float(self.get_parameter('target_timeout_sec').value)
        raw_timeout = float(self.get_parameter('raw_cmd_timeout_sec').value)

        target_valid = self.current_z is not None and (now - self.target_time) <= target_timeout
        raw_valid = (now - self.raw_cmd_time) <= raw_timeout

        reason = 'hold'
        trigger = False

        if not raw_valid:
            self.held_cmd = self.zero_twist()
            self.last_trigger_z = None
            self.pub_cmd.publish(self.held_cmd)
            self.write_log(now, False, 'raw_timeout_stop')
            self.get_logger().warn(
                'SETM raw command timeout, publish zero',
                throttle_duration_sec=1.0
            )
            return

        # 临时兜底：如果目标 z 没解析到，就用 raw command 作为事件触发状态。
        # 这样 SETM gate 先能工作，不会因为 /tracked_objects_3d 格式不匹配一直刹车。
        if not target_valid:
            self.current_z = (
                float(self.raw_cmd.linear.x),
                float(self.raw_cmd.linear.y)
            )
            self.current_target_x = 0.0
            self.current_target_y = 0.0
            target_valid = True

        z_x, z_y = self.current_z
        z_norm = math.sqrt(z_x * z_x + z_y * z_y)

        raw_is_zero = self.cmd_is_zero(self.raw_cmd)
        held_is_zero = self.cmd_is_zero(self.held_cmd)

        if self.last_trigger_z is None:
            trigger = True
            reason = 'init'
        elif raw_is_zero and not held_is_zero:
            trigger = True
            reason = 'stop_priority'
        elif (now - self.last_trigger_time) >= float(self.get_parameter('max_hold_sec').value):
            trigger = True
            reason = 'max_hold'
        else:
            z_last_x, z_last_y = self.last_trigger_z
            e_x = z_last_x - z_x
            e_y = z_last_y - z_y

            lhs = e_x * e_x + e_y * e_y

            epsilon_z = float(self.get_parameter('epsilon_z').value)
            delta_far = float(self.get_parameter('delta_far').value)
            delta_near = float(self.get_parameter('delta_near').value)
            rho = float(self.get_parameter('rho').value)

            sigma = delta_far if z_norm >= epsilon_z else delta_near
            rhs = sigma * (z_norm * z_norm) + rho

            if lhs >= rhs:
                trigger = True
                reason = f'setm_lhs={lhs:.5f}_rhs={rhs:.5f}'

        if trigger:
            self.held_cmd = self.clone_twist(self.raw_cmd)
            self.last_trigger_z = (z_x, z_y)
            self.last_trigger_time = now
            self.trigger_count += 1

        self.pub_cmd.publish(self.held_cmd)
        self.write_log(now, trigger, reason)

        ratio = self.trigger_count / max(1, self.timer_count)
        reduction = 100.0 * (1.0 - ratio)

        self.get_logger().info(
            f'SETM trigger={trigger} reason={reason} | '
            f'z=({z_x:.3f},{z_y:.3f}) | '
            f'raw=({self.raw_cmd.linear.x:.2f},{self.raw_cmd.linear.y:.2f},{self.raw_cmd.angular.z:.2f}) | '
            f'held=({self.held_cmd.linear.x:.2f},{self.held_cmd.linear.y:.2f},{self.held_cmd.angular.z:.2f}) | '
            f'count={self.trigger_count}/{self.timer_count}, reduction={reduction:.1f}%',
            throttle_duration_sec=0.5
        )

    def write_log(self, now, trigger, reason):
        if self.csv_writer is None:
            return

        z_x, z_y = self.current_z if self.current_z is not None else (0.0, 0.0)
        z_norm = math.sqrt(z_x * z_x + z_y * z_y)

        ratio = self.trigger_count / max(1, self.timer_count)
        reduction = 100.0 * (1.0 - ratio)

        self.csv_writer.writerow([
            f'{now:.6f}',
            self.current_target_x if self.current_target_x is not None else '',
            self.current_target_y if self.current_target_y is not None else '',
            z_x,
            z_y,
            z_norm,
            self.raw_cmd.linear.x,
            self.raw_cmd.linear.y,
            self.raw_cmd.angular.z,
            self.held_cmd.linear.x,
            self.held_cmd.linear.y,
            self.held_cmd.angular.z,
            int(trigger),
            reason,
            self.timer_count,
            self.trigger_count,
            ratio,
            reduction,
        ])

        if self.csv_file is not None:
            self.csv_file.flush()

    def destroy_node(self):
        try:
            self.pub_cmd.publish(Twist())
        except Exception:
            pass

        if self.csv_file is not None:
            self.csv_file.close()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Car2SETMCmdGateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
