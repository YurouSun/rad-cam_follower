import time
import sys
import argparse

# 默认 SDK 路径（根据你的安装位置调整）
DEFAULT_SDK = '/home/ubuntu/ros2_ws/install/ros_robot_controller/lib/python3.10/site-packages/ros_robot_controller'

# =========================
# 可编辑的文件级默认参数（用户可直接在此处修改）
# =========================
# 设置 'ENABLED' 为 True 后脚本将在启动时直接使用这些参数。
# 为安全起见，默认把 'DRY' 设为 True（不会实际发送到硬件）。
FILE_CONFIG = {
    'ENABLED': True,                   # 是否启用文件配置（True：使用下面的值；False：使用命令行）
    'SDK': DEFAULT_SDK,                # SDK 路径
    'ACTION': 'move_then_rotate',      # 'pulse_x' | 'pulse_y' | 'rotate' | 'move_then_rotate'
    'SPEED': 2.0,                      # 线性速度或回退角速度
    'MOVE_SPEED': 0.2,                 # 平移速度（用于 move_then_rotate）
    'OMEGA': None,                     # 角速度 rad/s（优先于 ROT_DEG）
    'ROT_DEG': 30.0,                   # 角速度（度/秒），当 OMEGA 为 None 时使用
    'DURATION': 0.5,                   # 平移持续时间（秒）
    'ROT_DURATION': None,              # 旋转持续时间（秒），若为 None 则使用 DURATION
    'DIRECTION': 'cw',                 # 'cw' 或 'ccw'
    'DRY': True,                       # True = 仅打印命令，不发送到硬件
}


parser = argparse.ArgumentParser(description='test_motor_x.py — 增加旋转操作的单电机/组合测试')
parser.add_argument('--sdk', default=DEFAULT_SDK, help='ros_robot_controller sdk path')
parser.add_argument('--action', choices=['pulse_x','rotate','pulse_y','move_then_rotate'], default='pulse_x', help='要执行的测试动作')
# 旧的通用 speed 参数保留用于 pulse_x/pulse_y 或作为 rotate 的回退值
parser.add_argument('--speed', type=float, default=2.0, help='速度幅度（线性或作为 rotate 的回退角速度，单位取决于上下文）')
parser.add_argument('--omega', type=float, default=None, help='旋转角速度（rad/s，优先于 --speed，用于 action=rotate）')
parser.add_argument('--rot-deg', type=float, default=20, help='旋转角速度（度/秒，等价于 --omega 的度量单位，可替代 --omega）')
parser.add_argument('--duration', type=float, default=0.5, help='脉冲持续时间（秒）')
parser.add_argument('--direction', choices=['cw','ccw'], default='cw', help='旋转方向，顺时针或逆时针（仅用于 action=rotate；若使用 --omega/--rot-deg 可用正负值替代）')
parser.add_argument('--dry', action='store_true', help='干运行：只打印将发送的命令，不实际调用 SDK')
parser.add_argument('--move-speed', type=float, default=None, help='move_then_rotate 时的平移线速度（覆盖 --speed）')
parser.add_argument('--rot-duration', type=float, default=None, help='只旋转时的持续时间（秒），用于 move_then_rotate 或 rotate 的覆盖')
args = parser.parse_args()

# 如果启用了文件级配置，覆盖命令行参数以便用户直接在文件中修改
if FILE_CONFIG.get('ENABLED'):
    args.sdk = FILE_CONFIG.get('SDK', args.sdk)
    args.action = FILE_CONFIG.get('ACTION', args.action)
    args.speed = FILE_CONFIG.get('SPEED', args.speed)
    args.omega = FILE_CONFIG.get('OMEGA', args.omega)
    args.rot_deg = FILE_CONFIG.get('ROT_DEG', args.rot_deg)
    args.duration = FILE_CONFIG.get('DURATION', args.duration)
    args.direction = FILE_CONFIG.get('DIRECTION', args.direction)
    # 如果文件配置要求 dry-run，优先设置
    if FILE_CONFIG.get('DRY'):
        args.dry = True
    # 文件级配置中若提供 MOVE_SPEED 或 ROT_DURATION，合并到 args
    if 'MOVE_SPEED' in FILE_CONFIG:
        args.move_speed = FILE_CONFIG.get('MOVE_SPEED')
    if 'ROT_DURATION' in FILE_CONFIG and FILE_CONFIG.get('ROT_DURATION') is not None:
        args.rot_duration = FILE_CONFIG.get('ROT_DURATION')

# SDK 导入与 Board 构造应在实际发送前进行；
# 如果是 dry-run，则跳过导入以避免因路径问题退出
Board = None
if not args.dry:
    sys.path.insert(0, args.sdk)
    try:
        from ros_robot_controller_sdk import Board
    except Exception as e:
        print('错误：无法导入 Board。请确认 --sdk 路径是否正确，或使用 --dry 查看命令。')
        print('详细错误：', e)
        sys.exit(1)


def stop_all(board):
    board.set_motor_speed([[1,0],[2,0],[3,0],[4,0]])


def pulse_x(board, speed, duration):
    # ID 1 控制前后整车通道（vx）
    print(f'Pulse X: id1={speed} for {duration}s')
    board.set_motor_speed([[1, speed]])
    time.sleep(duration)
    board.set_motor_speed([[1, -speed]])
    time.sleep(duration)
    board.set_motor_speed([[1, 0]])


def pulse_y(board, speed, duration):
    # 简单示例：使用 id2/id4 产生侧向分量（具体符号/效果需现场观察）
    print(f'Pulse Y-like: apply id2={speed}, id4={-speed} for {duration}s')
    board.set_motor_speed([[2, speed],[4, -speed]])
    time.sleep(duration)
    board.set_motor_speed([[2, -speed],[4, speed]])
    time.sleep(duration)
    board.set_motor_speed([[2,0],[4,0]])


def rotate(board, direction, omega, duration):
    # 这里的 omega 单位为 rad/s（调用方需保证）；根据之前观测，
    # 使用 id2 与 id4 产生对角组反向以实现原地旋转
    mag = float(omega)
    if direction == 'cw':
        cmd = [[1, 0], [2, mag], [3, 0], [4, -mag]]
    else:
        cmd = [[1, 0], [2, -mag], [3, 0], [4, mag]]

    print(f'Rotate (omega={omega:.4f} rad/s) cmd:', cmd, 'for', duration, 's')
    board.set_motor_speed(cmd)
    time.sleep(duration)
    stop_all(board)


def move_then_rotate(board, move_speed, move_duration, direction, omega, rot_duration):
    """
    先执行平移（沿 id1 通道），然后执行原地旋转。
    move_speed: 线性速度（用于 pulse_x）
    omega: 角速度（rad/s）
    """
    # 平移阶段（短脉冲）
    print(f'Move-then-rotate: move id1={move_speed} for {move_duration}s')
    board.set_motor_speed([[1, move_speed]])
    time.sleep(move_duration)
    board.set_motor_speed([[1, 0]])
    time.sleep(0.05)

    # 旋转阶段
    print(f'Move-then-rotate: rotate omega={omega} for {rot_duration}s')
    rotate(board, direction, omega, rot_duration)


if __name__ == '__main__':
    # 如果是 dry-run，打印将要发送的命令并退出，不构造 Board
    if args.dry:
        import math
        if args.action == 'pulse_x':
            print(f'DRY: would pulse id1={args.speed} for {args.duration}s')
        elif args.action == 'pulse_y':
            print(f'DRY: would pulse id2={args.speed}, id4={-args.speed} for {args.duration}s')
        elif args.action == 'rotate':
            if args.rot_deg is not None:
                omega_val = args.rot_deg * math.pi / 180.0
            elif args.omega is not None:
                omega_val = args.omega
            else:
                omega_val = args.speed
            mag = float(omega_val)
            if args.direction == 'cw':
                cmd = [[1, 0], [2, mag], [3, 0], [4, -mag]]
            else:
                cmd = [[1, 0], [2, -mag], [3, 0], [4, mag]]
            print('DRY: would send rotation cmd:', cmd, 'for', args.duration, 's')
        sys.exit(0)

    # 非 dry-run：构造 Board 并执行真实命令
    board = Board()
    board.enable_reception()
    time.sleep(1)

    try:
        stop_all(board)
        time.sleep(0.2)

        if args.action == 'pulse_x':
            pulse_x(board, args.speed, args.duration)
        elif args.action == 'pulse_y':
            pulse_y(board, args.speed, args.duration)
        elif args.action == 'rotate':
            import math
            if args.rot_deg is not None:
                omega_val = args.rot_deg * math.pi / 180.0
            elif args.omega is not None:
                omega_val = args.omega
            else:
                omega_val = args.speed

            rotate(board, args.direction, omega_val, args.duration)
        elif args.action == 'move_then_rotate':
            # move_then_rotate: 先平移再旋转
            import math
            move_speed = args.move_speed if (args.move_speed is not None) else args.speed
            if args.rot_deg is not None:
                omega_val = args.rot_deg * math.pi / 180.0
            elif args.omega is not None:
                omega_val = args.omega
            else:
                omega_val = args.speed

            move_dur = args.duration
            rot_dur = args.rot_duration if (args.rot_duration is not None) else args.duration

            move_then_rotate(board, move_speed, move_dur, args.direction, omega_val, rot_dur)

        stop_all(board)
        time.sleep(0.2)
        print('Done. Motors stopped.')

    except KeyboardInterrupt:
        print('Interrupted — stopping motors')
        stop_all(board)
    except Exception as e:
        print('Error:', e)
        stop_all(board)
