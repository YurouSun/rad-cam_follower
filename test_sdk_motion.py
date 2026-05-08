#!/usr/bin/env python3
import sys
import time
import os

# Add the path to the SDK
sdk_path = '/home/ubuntu/ros2_ws/src 11/driver/ros_robot_controller'
sys.path.append(sdk_path)

try:
    from ros_robot_controller.ros_robot_controller_sdk import Board
except ImportError:
    # Fallback if the package structure is different
    sys.path.append('/home/ubuntu/ros2_ws/src 11/driver/ros_robot_controller/ros_robot_controller')
    try:
        from ros_robot_controller_sdk import Board
    except ImportError as e:
        print(f"Error importing SDK: {e}")
        sys.exit(1)

def main():
    print("Initializing Board...")
    try:
        board = Board()
        board.enable_reception()
        print("Board initialized.")
    except Exception as e:
        print(f"Failed to initialize board: {e}")
        return

    print("Attempting to move forward...")
    # ID 1=Vx, ID 2=Vy, ID 3=Vz
    # Move forward at 1.0 m/s
    board.set_motor_speed([[1, 10.0], [2, 0.0], [3, 0.0]])
    time.sleep(2)

    print("Stopping...")
    board.set_motor_speed([[1, 0.0], [2, 0.0], [3, 0.0]])
    time.sleep(1)

    print("Attempting to rotate...")
    # Rotate at 0.5 rad/s
    board.set_motor_speed([[1, 0.0], [2, 0.0], [3, 0.5]])
    time.sleep(2)

    print("Stopping...")
    board.set_motor_speed([[1, 0.0], [2, 0.0], [3, 0.0]])
    
    print("Test complete.")

if __name__ == "__main__":
    main()
