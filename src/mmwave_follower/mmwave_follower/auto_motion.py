import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time  # 添加time模块
import math  # 添加math模块

class AutoMotion(Node):
    def __init__(self, mode=1):
        super().__init__('auto_motion')
        self.pub = self.create_publisher(Twist, '/controller/cmd_vel', 10)
        self.timer = self.create_timer(0.02, self.timer_callback)  # 50Hz 创建定时器
        self.twist = Twist()
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.stopped = False
        self.counter = 0
        self.run_count = 0
        self.mode = mode  # 运动模式：1蛇形 2匀速 3变速 4半圆
        self.get_logger().info(f'当前运动模式: {self.mode}')

    def timer_callback(self):
        now = time.time()
        # 检查是否应该停止运动（模式4为5秒，其他为10秒）
        stop_time = 5.0 if self.mode == 4 else 10.0

        # 主要修改点：将停止逻辑与运动逻辑分离
        if self.stopped:
            # 如果已进入停止状态，持续发布零速指令直到节点关闭
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.0
            self.pub.publish(self.twist)
            return # 直接返回，不再执行后续逻辑

        if now - self.start_time >= stop_time:
            self.get_logger().info(f'已到{stop_time}秒，开始发送停止指令...')
            self.stopped = True
            # 设置一个短暂的延时后关闭节点，确保有足够的时间发送停车指令
            self.create_timer(0.5, self.shutdown_node) # 0.5秒后执行关闭
            return # 进入停止状态，下一个timer周期将持续发送零速

        # --- 运动控制逻辑 ---
        # 运动模式选择
        if self.mode == 1:
            # 蛇形运动：线速度恒定，角速度周期性变化
            self.twist.linear.x = 0.25
            self.twist.angular.z = 0.8 * (1 if int((now - self.start_time) * 2) % 2 == 0 else -1)
        elif self.mode == 2:
            # 匀速直行
            self.twist.linear.x = 1.0
            self.twist.angular.z = 0.0
        elif self.mode == 3:
            # 正弦波变速：根据 sin 函数在一个周期内实现加速/减速和正反向运动
            period = 10.0  # 周期10秒
            max_v = 0.4
            t = (now - self.start_time) % period
            # 角频率 ω = 2π / T
            omega = 2 * math.pi / period
            # 正弦速度: 从0开始，先加速向前，再减速过0，加速后退，减速回0
            v = max_v * math.sin(omega * t)
            self.twist.linear.x = v
            self.twist.angular.z = 0.0
        elif self.mode == 4:
            # 绕半圆：线速度恒定，角速度恒定
            self.twist.linear.x = 0.25
            self.twist.angular.z = 0.8
        
        self.pub.publish(self.twist)
        self.counter += 1
        if now - self.last_log_time > 1.0:
            self.get_logger().info(f'Published: linear.x={self.twist.linear.x:.2f}, angular.z={self.twist.angular.z}, time={now:.2f}, counter={self.counter}, run_count={self.run_count}')
            self.last_log_time = now

    def shutdown_node(self):
        """安全关闭节点"""
        self.get_logger().info('运行结束，关闭节点。')
        # 在关闭前最后发布一次，以防万一
        final_twist = Twist()
        self.pub.publish(final_twist)
        time.sleep(0.1) # 留出最后一点时间
        rclpy.shutdown()

def main():
    import sys
    print("请选择运动模式: 1-蛇形 2-匀速 3-变速 4-半圆")
    try:
        mode = int(input("输入模式编号(1/2/3/4): ").strip())
        if mode not in [1,2,3,4]:
            print("输入无效，默认使用模式3(变速)")
            mode = 3
    except Exception:
        print("输入无效，默认使用模式3(变速)")
        mode = 3
    rclpy.init()
    node = AutoMotion(mode)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()