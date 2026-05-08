#!/usr/bin/env python3
"""
move_control.py

通过已知 SDK 通道合成车辆的平移（vx, vy）与原地旋转（omega）操作。

说明（基于你观测到的映射）：
- `id1` 控制整车前后通道（vx）
- `id2` 控制左上(前左)+右下(后右) 对角组
- `id4` 控制右上(前右)+左下(后左) 对角组

理论上，令 c = [c1,c2,c4] 为 SDK 通道命令，我们用线性组合近似把期望的机体速度 [vx, vy, omega]
映射到通道命令：
  c1 = vx * k_vx
  c2 =  vy * k_vy + omega * k_rot
  c4 = -vy * k_vy + omega * k_rot

其中增益 `k_vx/k_vy/k_rot` 与符号可通过命令行参数调整以匹配你的硬件。默认值为保守的低速增益。

使用示例（低速短脉冲测试）：
  python3 ~/ros2_ws/move_control.py --vx 0.2 --vy 0 --omega 0.5 --duration 0.5

安全注意：现场应有人监护并能立即停止动力（断电或按急停）。
"""

import sys
import time
import argparse
import signal

DEFAULT_SDK = '/home/ubuntu/ros2_ws/install/ros_robot_controller/lib/python3.10/site-packages/ros_robot_controller'

parser = argparse.ArgumentParser()
parser.add_argument('--sdk', default=DEFAULT_SDK, help='ros_robot_controller sdk path')
parser.add_argument('--vx', type=float, default=0.2, help='前后速度分量（正为向前）')
parser.add_argument('--vy', type=float, default=0.0, help='左右速度分量（正为向右）')
# 支持以度/秒输入角速度，优先使用 --rot-deg，如果同时提供 --omega 则以 --rot-deg 优先
parser.add_argument('--omega', type=float, default=0.0, help='偏航角速度（弧度/秒，保留向后兼容）')
parser.add_argument('--rot-deg', type=float, default=0.0, help='偏航角速度（度/秒），将转换为弧度/秒')
parser.add_argument('--duration', type=float, default=0.5, help='命令保持时间（秒），若只需旋转可与 --rot-duration 配合')
parser.add_argument('--rot-duration', type=float, default=None, help='仅旋转时的持续时间（秒），若设置将覆盖 --duration 对旋转的时间')
parser.add_argument('--k_vx', type=float, default=1.0, help='vx 到 id1 的增益')
parser.add_argument('--k_vy', type=float, default=1.0, help='vy 到 id2/id4 的增益')
parser.add_argument('--k_rot', type=float, default=0.8, help='omega 到 id2/id4 的增益')
parser.add_argument('--max_cmd', type=float, default=10.0, help='命令绝对值上限，避免过大')
parser.add_argument('--dry', action='store_true', help='干运行模式：只打印将发送的命令，不实际调用 SDK')
args = parser.parse_args()

sys.path.insert(0, args.sdk)
try:
    from ros_robot_controller_sdk import Board
except Exception as e:
    if not args.dry:
        print('错误：无法导入 Board。请确认 --sdk 路径是否正确，或使用 --dry 查看命令。')
        print('详细错误：', e)
        sys.exit(1)


def clamp(x, lim):
    if x > lim:
        return lim
    if x < -lim:
        return -lim
    return x


def stop_all(board):
    board.set_motor_speed([[1,0],[2,0],[3,0],[4,0]])


def compose_commands(vx, vy, omega, kvx, kvy, krot, max_cmd):
    c1 = vx * kvx
    c2 = vy * kvy + omega * krot
    c4 = -vy * kvy + omega * krot
    c1 = clamp(c1, max_cmd)
    c2 = clamp(c2, max_cmd)
    c4 = clamp(c4, max_cmd)
    return [[1, c1], [2, c2], [3, 0], [4, c4]]


def main():
    vx = args.vx
    vy = args.vy
    # 优先使用 rot-deg（度/秒）输入，兼容旧的 omega (rad/s)
    if args.rot_deg is not None:
        import math
        omega = args.rot_deg * math.pi / 180.0
    else:
        omega = args.omega

    # 决定持续时间：默认使用 --duration，若只设置了 --rot-duration 则旋转使用 rot-duration
    dur = args.duration
    rot_dur = args.rot_duration if args.rot_duration is not None else dur

    # 如果只想旋转而不平移，可将 vx,vy=0 并设置 rot_dur
    cmds = compose_commands(vx, vy, omega, args.k_vx, args.k_vy, args.k_rot, args.max_cmd)
    print('Composed command (id, value):', cmds)

    if args.dry:
        print('Dry run: not sending to hardware.')
        return

    b = Board()
    b.enable_reception()
    time.sleep(0.3)

    # Install signal handler to stop motors on Ctrl-C
    stop_called = {'v': False}

    def _stop(signum, frame):
        if not stop_called['v']:
            print('\nSignal received: stopping motors')
            stop_all(b)
            stop_called['v'] = True
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        # 如果旋转持续时间与总体持续时间不同，先以平移命令运行 dur，额外在旋转时间内发送包含 omega 的命令
        if (omega != 0.0) and (rot_dur != dur):
            # 先执行平移（若有）但不包含旋转
            cmds_trans = compose_commands(vx, vy, 0.0, args.k_vx, args.k_vy, args.k_rot, args.max_cmd)
            print('Sending translation-only command for', dur, 's:', cmds_trans)
            b.set_motor_speed(cmds_trans)
            time.sleep(dur)

            # 然后执行只含旋转的命令
            cmds_rot = compose_commands(0.0, 0.0, omega, args.k_vx, args.k_vy, args.k_rot, args.max_cmd)
            print('Sending rotation-only command for', rot_dur, 's:', cmds_rot)
            b.set_motor_speed(cmds_rot)
            time.sleep(rot_dur)
        else:
            print('Sending combined command for', dur, 's ...')
            b.set_motor_speed(cmds)
            time.sleep(dur)
    finally:
        print('Stopping all motors')
        stop_all(b)


if __name__ == '__main__':
    main()
