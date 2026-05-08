#!/usr/bin/env python3
"""
cli_drive.py

交互式 / 命令行 小车控制器（便捷封装）

功能：
- 支持一次性命令：前进/后退/左右/原地旋转/停止/自定义 vx,vy,rot
- 支持交互式 REPL，实时输入简短命令控制小车
- 使用与 `move_control.py` 相同的通道映射：
  id1 <- vx * k_vx
  id2 <- vy * k_vy + omega * k_rot
  id4 <- -vy * k_vy + omega * k_rot

用法示例：
  单次前进：
    python3 ~/ros2_ws/cli_drive.py --cmd forward --speed 0.2 --duration 1.0

  单次旋转（度/秒）：
    python3 ~/ros2_ws/cli_drive.py --cmd rotate --deg 30 --duration 0.5

  交互式模式：
    python3 ~/ros2_ws/cli_drive.py --interactive

交互命令速查：
  f <speed> <duration>    前进
  b <speed> <duration>    后退
  l <speed> <duration>    向左（侧向）
  r <speed> <duration>    向右（侧向）
  rot <deg/s> <duration>  原地旋转（deg/s，正为顺时针，若不对可取负）
  stop                    停止
  q / quit                退出

注意：先用 `--dry` 查看将发送的命令，再在现场有人监护下运行真实命令。
"""

import sys
import time
import argparse
import math
import signal

DEFAULT_SDK = '/home/ubuntu/ros2_ws/install/ros_robot_controller/lib/python3.10/site-packages/ros_robot_controller'

parser = argparse.ArgumentParser()
parser.add_argument('--sdk', default=DEFAULT_SDK, help='ros_robot_controller sdk path')
parser.add_argument('--cmd', choices=['forward','back','left','right','rotate','stop','custom'], help='一次性命令')
parser.add_argument('--speed', type=float, default=0.2, help='平移速度抽象值（用于 forward/back/left/right 或 custom vx）')
parser.add_argument('--vy', type=float, default=0.0, help='自定义侧向速度（用于 custom）')
parser.add_argument('--deg', type=float, default=0.0, help='旋转角速度，度/秒（用于 rotate 或 custom）')
parser.add_argument('--duration', type=float, default=0.5, help='持续时间（秒）')
parser.add_argument('--k_vx', type=float, default=1.0, help='vx 到 id1 增益')
parser.add_argument('--k_vy', type=float, default=1.0, help='vy 到 id2/id4 增益')
parser.add_argument('--k_rot', type=float, default=1.0, help='omega 到 id2/id4 增益')
parser.add_argument('--max_cmd', type=float, default=10.0, help='命令绝对值上限')
parser.add_argument('--dry', action='store_true', help='仅打印将发送的命令')
parser.add_argument('--interactive', action='store_true', help='交互式 REPL 模式')
args = parser.parse_args()


def clamp(x, lim):
    return max(-lim, min(lim, x))


def compose(vx, vy, omega, kvx, kvy, krot, max_cmd):
    c1 = clamp(vx * kvx, max_cmd)
    c2 = clamp(vy * kvy + omega * krot, max_cmd)
    c4 = clamp(-vy * kvy + omega * krot, max_cmd)
    return [[1, c1], [2, c2], [3, 0], [4, c4]]


def send_cmd(board, cmd, dry=False):
    print('Sending:', cmd)
    if dry:
        return
    board.set_motor_speed(cmd)


def stop_all(board, dry=False):
    send_cmd(board, [[1,0],[2,0],[3,0],[4,0]], dry=dry)


def run_once(cmd_name, speed, vy, deg, duration, board=None, dry=False):
    # 计算 vx, vy, omega
    if cmd_name == 'forward':
        vx = abs(speed)
        vyv = 0.0
    elif cmd_name == 'back':
        vx = -abs(speed)
        vyv = 0.0
    elif cmd_name == 'left':
        vx = 0.0
        vyv = -abs(speed)
    elif cmd_name == 'right':
        vx = 0.0
        vyv = abs(speed)
    elif cmd_name == 'rotate':
        vx = 0.0
        vyv = 0.0
    elif cmd_name == 'custom':
        vx = speed
        vyv = vy
    elif cmd_name == 'stop':
        stop_all(board, dry=dry)
        return
    else:
        print('Unknown cmd', cmd_name); return

    # deg -> rad/s
    omega = deg * math.pi / 180.0

    # compose
    cmd = compose(vx, vyv, omega, args.k_vx, args.k_vy, args.k_rot, args.max_cmd)
    send_cmd(board, cmd, dry=dry)
    if not dry:
        time.sleep(duration)
        stop_all(board, dry=dry)


def interactive_loop(board=None, dry=False):
    print('Interactive mode. Commands: f/b/l/r <speed> <dur>, rot <deg> <dur>, custom vx vy deg dur, stop, q')
    try:
        while True:
            line = input('> ').strip()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0]
            if cmd in ('q', 'quit'):
                break
            if cmd == 'stop':
                stop_all(board, dry=dry)
                continue
            if cmd in ('f','forward'):
                s = float(parts[1]) if len(parts) > 1 else 0.2
                d = float(parts[2]) if len(parts) > 2 else 0.5
                run_once('forward', s, 0.0, 0.0, d, board=board, dry=dry)
                continue
            if cmd in ('b','back'):
                s = float(parts[1]) if len(parts) > 1 else 0.2
                d = float(parts[2]) if len(parts) > 2 else 0.5
                run_once('back', s, 0.0, 0.0, d, board=board, dry=dry)
                continue
            if cmd in ('l','left'):
                s = float(parts[1]) if len(parts) > 1 else 0.2
                d = float(parts[2]) if len(parts) > 2 else 0.5
                run_once('left', s, 0.0, 0.0, d, board=board, dry=dry)
                continue
            if cmd in ('r','right'):
                s = float(parts[1]) if len(parts) > 1 else 0.2
                d = float(parts[2]) if len(parts) > 2 else 0.5
                run_once('right', s, 0.0, 0.0, d, board=board, dry=dry)
                continue
            if cmd in ('rot','rotate'):
                deg = float(parts[1]) if len(parts) > 1 else 30.0
                d = float(parts[2]) if len(parts) > 2 else 0.4
                run_once('rotate', 0.0, 0.0, deg, d, board=board, dry=dry)
                continue
            if cmd == 'custom':
                vx = float(parts[1]) if len(parts) > 1 else 0.0
                vy = float(parts[2]) if len(parts) > 2 else 0.0
                deg = float(parts[3]) if len(parts) > 3 else 0.0
                d = float(parts[4]) if len(parts) > 4 else 0.5
                run_once('custom', vx, vy, deg, d, board=board, dry=dry)
                continue
            print('Unknown input')
    except (EOFError, KeyboardInterrupt):
        print('\nExiting interactive mode')


def main():
    board = None
    if not args.dry:
        sys.path.insert(0, args.sdk)
        try:
            from ros_robot_controller_sdk import Board
        except Exception as e:
            print('错误：无法导入 Board。请确认 --sdk 路径或使用 --dry。')
            print('详细：', e)
            return
        board = Board()
        board.enable_reception()
        time.sleep(0.3)

    # signal safe stop
    def _stop(sig, frame):
        if board is not None:
            stop_all(board, dry=args.dry)
        sys.exit(0)
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    if args.interactive:
        interactive_loop(board=board, dry=args.dry)
        return

    # single-shot command
    if args.cmd:
        if args.cmd == 'stop':
            if board is not None:
                stop_all(board, dry=args.dry)
            else:
                print('Dry stop')
            return

        # map cmd to run_once
        if args.cmd in ('forward','back','left','right','rotate','custom'):
            run_once(args.cmd, args.speed, args.vy, args.deg, args.duration, board=board, dry=args.dry)
        else:
            print('Unsupported cmd')


if __name__ == '__main__':
    main()
