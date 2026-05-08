import sys
import time
import signal

# SDK 路径
sys.path.append("/home/ubuntu/ros2_ws/src 11/driver/ros_robot_controller/ros_robot_controller")
from ros_robot_controller_sdk import Board

def signal_handler(sig, frame):
    print('紧急停车...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    board = Board()
    board.enable_reception()
    time.sleep(1)

    try:
        # 右前方：两个电机同向前进，右侧更快，持续约 5 秒
        duration = 5.0
        left_speed = 0.6   # ID 1：左电机，略慢（原 3.0 / 5）
        right_speed = 1.0  # ID 2：右电机，更快以偏右（原 5.0 / 5）

        start = time.time()
        while time.time() - start < duration:
            board.set_motor_speed([[1, left_speed], [2, right_speed]])
            time.sleep(0.1)

        print("停车...")
        board.set_motor_speed([[1, 0.0], [2, 0.0]])
        
    except Exception as e:
        print(f"出错: {e}")
        board.set_motor_speed([[1, 0.0], [2, 0.0]])