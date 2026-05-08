#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class TurnInPlace(Node):
    def __init__(self):
        super().__init__('turn_in_place_test')
        self.publisher_ = self.create_publisher(Twist, '/controller/cmd_vel', 10)
        self.timer = self.create_timer(0.05, self.publish_cmd)
        self.duration_sec = 10.0
        self.start_time = self.get_clock().now()

    def publish_cmd(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed > self.duration_sec:
            self.stop_and_shutdown()
            return

        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.3  # constant clockwise rotation
        self.publisher_.publish(cmd)
        self.get_logger().info(f'Rotating in place... {elapsed:.1f}s/{self.duration_sec:.0f}s')

    def stop_and_shutdown(self):
        self.timer.cancel()
        cmd = Twist()
        self.publisher_.publish(cmd)
        self.get_logger().info('Turn test complete, stopping.')
        rclpy.shutdown()

def main():
    rclpy.init()
    node = TurnInPlace()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_and_shutdown()

if __name__ == '__main__':
    main()
