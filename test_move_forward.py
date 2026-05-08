import time
import sys
import signal

# SDK 路径
sys.path.append('/home/ubuntu/ros2_ws/src 11/driver/ros_robot_controller/ros_robot_controller')
from ros_robot_controller_sdk import Board

def signal_handler(sig, frame):
    print('程序中断，紧急停车...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    board = Board()
    board.enable_reception()
    time.sleep(1) # 等待初始化

    try:
        SPEED = 0.5
        DURATION = 3.0
        print(f"测试: 向前移动 (X={SPEED}, Y=0) 持续 {DURATION}秒...")
        
        start_time = time.time()
        while time.time() - start_time < DURATION:
            # ID 1 = X (前后), ID 2 = Y (左右)
            board.set_motor_speed([[1, SPEED], [2, 0.0]])
            time.sleep(0.1)
            
        print("停车...")
        board.set_motor_speed([[1, 0.0], [2, 0.0]])
        time.sleep(0.5)

    except Exception as e:
        print(f"发生错误: {e}")
        board.set_motor_speed([[1, 0.0], [2, 0.0]])
