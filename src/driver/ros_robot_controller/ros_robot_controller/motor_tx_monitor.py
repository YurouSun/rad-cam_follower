#!/usr/bin/env python3
# encoding: utf-8

import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class MotorTxMonitor(Node):
    def __init__(self):
        super().__init__('motor_tx_monitor')
        self.last_cmd_controller = None
        self.last_cmd_tracker = None

        self.sub_cmd_controller = self.create_subscription(Twist, '/controller/cmd_vel', self.on_cmd_vel_controller, 20)
        self.sub_cmd_tracker = self.create_subscription(Twist, '/tracker/cmd_vel', self.on_cmd_vel_tracker, 20)
        self.sub_tx = self.create_subscription(String, '/controller/motor_tx', self.on_motor_tx, 20)

        self.get_logger().info('motor_tx_monitor started, listening /tracker/cmd_vel, /controller/cmd_vel and /controller/motor_tx')

    def on_cmd_vel_controller(self, msg: Twist):
        self.last_cmd_controller = (msg.linear.x, msg.linear.y, msg.angular.z)

    def on_cmd_vel_tracker(self, msg: Twist):
        self.last_cmd_tracker = (msg.linear.x, msg.linear.y, msg.angular.z)

    def on_motor_tx(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception:
            self.get_logger().warn(f'invalid /controller/motor_tx payload: {msg.data}')
            return

        src = data.get('source', '')
        out = data.get('out', {})
        inp = data.get('in', {})
        payload_hex = data.get('payload_hex', '')

        vx = float(out.get('vx', 0.0))
        vy = float(out.get('vy', 0.0))
        wz = float(out.get('wz', 0.0))

        inx = inp.get('x', None)
        iny = inp.get('y', None)
        inz = inp.get('z', None)

        self.get_logger().info(
            f'TX src={src} in=({inx},{iny},{inz}) out=(vx={vx:.3f},vy={vy:.3f},wz={wz:.3f}) payload={payload_hex}',
            throttle_duration_sec=0.05,
        )


def main():
    rclpy.init()
    node = MotorTxMonitor()
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
