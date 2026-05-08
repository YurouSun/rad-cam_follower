import time
from ros_robot_controller_sdk import Board
import struct
from ros_robot_controller_sdk import PacketFunction

def test_single_motor():
    print("Connecting to board...")
    board = Board()
    time.sleep(2)
    
    print("\n--- 正在直接测试底层电机映射 ---")
    print("如果这个测试里轮子能单独转动，说明底层X/Y/Z算法有严重Bug（可能是固件或者电机线插乱了）。")
    print("我们将发送原生的 4轮独立驱动指令 来绕过Bug！\n")
    
    for motor_id in [1, 2, 3, 4]:
        print(f"正在驱动 [电机 {motor_id}] 向前转动 (速度 0.5) ... 请观察是哪个轮子在转，以及正反向！")
        # 0x01 = Subcmd, 0x01 = Count=1
        data = [0x01, 0x01]
        data.append(motor_id)
        data.extend(struct.pack("<f", 0.5))
        
        board.buf_write(PacketFunction.PACKET_FUNC_MOTOR, data)
        time.sleep(2.5)
        
        # Stop
        data = [0x01, 0x01, motor_id] + list(struct.pack("<f", 0.0))
        board.buf_write(PacketFunction.PACKET_FUNC_MOTOR, data)
        time.sleep(1)

if __name__ == "__main__":
    test_single_motor()