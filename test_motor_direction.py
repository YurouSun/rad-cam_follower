import time
import sys
import os

# Add the SDK path to sys.path
sys.path.append('/home/ubuntu/ros2_ws/src/driver/ros_robot_controller/ros_robot_controller')
from ros_robot_controller_sdk import Board

if __name__ == "__main__":
    board = Board()
    board.enable_reception()
    time.sleep(1) # Wait for init

    try:
        print("Test 1: Moving Left (Y=1.0) for 3 seconds...")
        for _ in range(30): 
            board.set_motor_speed([[1, 0.0], [2, 1.0]])
            time.sleep(0.1)
            
        print("Stopping for 2 seconds...")
        board.set_motor_speed([[1, 0], [2, 0]])
        time.sleep(2)

        print("Test 2: Moving Right (Y=-1.0) for 3 seconds...")
        for _ in range(30): 
            board.set_motor_speed([[1, 0.0], [2, -1.0]])
            time.sleep(0.1)

        print("Stopping...")
        board.set_motor_speed([[1, 0], [2, 0]])
        time.sleep(1)

    except KeyboardInterrupt:
        print("Interrupted")
        board.set_motor_speed([[1, 0], [2, 0]])
