#!/usr/bin/env python3
"""
run_motor_tests.py
小车电机诊断脚本（安全默认）：
- 先强制停止所有电机
- 单电机正/反脉冲测试（id 列表可配置）
- 组合测试 A/B（用于判别原地旋转符号组合）

用法示例：
    python3 ~/ros2_ws/run_motor_tests.py --sdk '/path/to/ros_robot_controller' --max-id 4

如果你不传 --sdk，会使用脚本内默认路径（请根据实际情况修改）。

运行时请在现场有人监护并能立刻停止。
"""

import time
import argparse
import sys

DEFAULT_SDK = '/home/ubuntu/ros2_ws/install/ros_robot_controller/lib/python3.10/site-packages/ros_robot_controller'
# 你当前脚本里看到的路径含空格可能是错误：'/home/ubuntu/ros2_ws/src 11/driver/...'

parser = argparse.ArgumentParser()
parser.add_argument('--sdk', default=DEFAULT_SDK, help='ros_robot_controller sdk path (folder that contains ros_robot_controller_sdk)')
parser.add_argument('--max-id', type=int, default=4, help='最大 motor id 测试到哪个编号 (默认 4)')
parser.add_argument('--speed', type=int, default=4, help='测试速度大小（正负两向测试）')
parser.add_argument('--pulse', type=float, default=0.4, help='单次脉冲持续时间(s)')
args = parser.parse_args()

# 把 SDK 路径加入 sys.path
sys.path.insert(0, args.sdk)
try:
    from ros_robot_controller_sdk import Board
except Exception as e:
    print('错误：无法导入 Board。请确认 --sdk 路径是否正确，或把 SDK 放到此路径。')
    print('尝试导入错误信息：', e)
    sys.exit(1)


def stop_all(board, max_id):
    cmds = [[i, 0] for i in range(1, max_id+1)]
    board.set_motor_speed(cmds)


def pulse_motor(board, mid, speed, t):
    print(f"PULSE motor {mid}: +{speed} for {t}s")
    board.set_motor_speed([[mid, speed]])
    time.sleep(t)
    print(f"PULSE motor {mid}: -{speed} for {t}s")
    board.set_motor_speed([[mid, -speed]])
    time.sleep(t)
    board.set_motor_speed([[mid, 0]])
    time.sleep(0.2)


def combo_test(board, ids, speeds, t):
    # ids: list of (id, speed) pairs
    print('COMBO:', ids, 'for', t, 's')
    board.set_motor_speed(ids)
    time.sleep(t)
    # stop those
    board.set_motor_speed([[i,0] for (i,_) in ids])
    time.sleep(0.2)


if __name__ == '__main__':
    print('Using SDK path:', args.sdk)
    board = Board()
    board.enable_reception()
    time.sleep(1)

    max_id = args.max_id
    speed = args.speed
    pulse = args.pulse

    print('\n=== STEP 0: FORCE STOP ALL MOTORS ===')
    stop_all(board, max_id)
    time.sleep(0.3)

    print('\n=== STEP 1: SINGLE MOTOR PULSE TEST ===')
    for mid in range(1, max_id+1):
        print(f'--- Testing id {mid} ---')
        try:
            pulse_motor(board, mid, speed, pulse)
        except Exception as e:
            print('Error pulsing motor', mid, e)
        time.sleep(0.3)

    print('\n=== STEP 2: COMBO TESTS (A/B) for rotation check ===')
    # 两个组合示例（按 4-wheel 假设）：
    if max_id >= 4:
        comboA = [[1,  speed],[2, -speed],[3,  speed],[4, -speed]]
        comboB = [[1, -speed],[2,  speed],[3, -speed],[4,  speed]]
        combo_test(board, comboA, speed, pulse)
        combo_test(board, comboB, speed, pulse)
    else:
        print('max_id < 4: will try 2-channel combos using id 1 and 2')
        combo_test(board, [[1, speed],[2, speed]], speed, pulse)
        combo_test(board, [[1, speed],[2, -speed]], speed, pulse)

    print('\n=== DONE: final STOP ===')
    stop_all(board, max_id)
    print('All tests finished. 记录观察结果并贴回给我（例如: id1->前左, id2->前右, comboA->原地右转）。')
