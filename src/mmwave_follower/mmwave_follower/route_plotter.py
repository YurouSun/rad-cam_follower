#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
import math
import matplotlib
matplotlib.use('Agg')  # Headless mode for remote server
import matplotlib.pyplot as plt

class RoutePlotter(Node):
    def __init__(self):
        super().__init__('route_plotter')
        self.sub = self.create_subscription(Twist, '/car3/cmd_vel', self.cmd_cb, 10)
        self.x = 0.0
        self.y = 0.0
        self.xs = [0.0]
        self.ys = [0.0]
        self.last_time = time.time()
        
        # 打印与画图定时器
        self.timer = self.create_timer(1.0, self.report_and_plot)
        
        self.current_vx = 0.0
        self.current_vy = 0.0

    def cmd_cb(self, msg):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        
        if dt > 1.0 or dt < 0: dt = 0.05
        
        self.current_vx = msg.linear.x
        self.current_vy = msg.linear.y

        self.x += self.current_vx * dt
        self.y += self.current_vy * dt
        
        self.xs.append(self.x)
        self.ys.append(self.y)

    def report_and_plot(self):
        # 1. 打印每秒速度
        speed_m_s = math.hypot(self.current_vx, self.current_vy)
        self.get_logger().info(f"【速度播报】前向: {self.current_vx:.3f} m/s | 侧向: {self.current_vy:.3f} m/s | 综合速度: {speed_m_s:.3f} m/s")
        
        # 2. 生成路线图
        try:
            plt.figure(figsize=(6, 6))
            # ROS 中 Y 是左侧，X 是前方，为了画图直观，横轴设为Y，纵轴设为X
            plt.plot(self.ys, self.xs, '-b', linewidth=2, label='Trajectory')
            if len(self.xs) > 0:
                plt.scatter([self.ys[0]], [self.xs[0]], c='green', s=100, label='Start')
                plt.scatter([self.ys[-1]], [self.xs[-1]], c='red', s=100, label='Current')
            
            plt.title(f"Robot Active Route Map\nx:{self.x:.2f} y:{self.y:.2f}")
            plt.xlabel("Y (Lateral) [m]")
            plt.ylabel("X (Forward) [m]")
            plt.legend()
            plt.grid(True)
            plt.axis('equal') # 保持比例
            
            # 保存到工作空间根目录
            plt.savefig('/home/ubuntu/ros2_ws/route_map.png')
            plt.close()
        except Exception as e:
            self.get_logger().error(f"Plotting failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = RoutePlotter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
