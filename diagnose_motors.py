#!/usr/bin/env python3
import time
import sys

# 调整为你实际的 SDK 路径（如果路径错误请修改此行）
sys.path.append('/home/ubuntu/ros2_ws/src 11/driver/ros_robot_controller/ros_robot_controller')
try:
    from ros_robot_controller_sdk import Board
except Exception as e:
    print('无法导入 Board SDK，请检查 sys.path 路径是否正确:', e)
    raise


def pulse_motor(board, mid, speed=3, t=0.5):
    print(f"PULSE motor {mid}: speed=+{speed}")
    board.set_motor_speed([[mid, speed]])
    time.sleep(t)
    print(f"PULSE motor {mid}: speed=-{speed}")
    board.set_motor_speed([[mid, -speed]])
    time.sleep(t)
    board.set_motor_speed([[mid, 0]])
    time.sleep(0.2)


if __name__ == '__main__':
    board = Board()
    board.enable_reception()
    time.sleep(1)

    # 测试 id 范围，通常从 1 开始，根据你的控制器调整 max_id
    max_id = 8
    print('开始逐个脉冲测试 motor id (请观察物理轮子反应)，按 Ctrl-C 可中断')
    try:
        for mid in range(1, max_id + 1):
            print('\n=== 测试 motor id:', mid, '===')
            pulse_motor(board, mid, speed=3, t=0.4)
            # 小间隔
            time.sleep(0.3)
        print('\n全部测试完成，已发送停止命令。')
    except KeyboardInterrupt:
        print('用户中断，发送停止命令...')
        for mid in range(1, max_id + 1):
            try:
                board.set_motor_speed([[mid, 0]])
            except Exception:
                pass
        print('已停止所有测试')
