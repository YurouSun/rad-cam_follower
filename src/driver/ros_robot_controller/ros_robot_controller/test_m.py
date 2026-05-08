import time
from ros_robot_controller_sdk import Board

def test_motors():
    print("Connecting to board...")
    board = Board()
    time.sleep(2)  # Wait for connection
    
    print("Testing sending 4 independent motor speeds...")
    # Attempt to send custom packet to motor
    # We will use board.set_motor_speed but wait, sdk's set_motor_speed overrides it.
    # Let's write the raw buffer!
    import struct
    from ros_robot_controller_sdk import PacketFunction
    
    # Standard Hiwonder 4-motor format: Subcmd=1, Count=4, then ID, Speed ...
    # Wait, some firms use count=4 directly. Let's try:
    data = [0x01, 0x04]
    data.append(0x01); data.extend(struct.pack("<f", 0.5))
    data.append(0x02); data.extend(struct.pack("<f", 0.5))
    data.append(0x03); data.extend(struct.pack("<f", 0.5))
    data.append(0x04); data.extend(struct.pack("<f", 0.5))
    
    board.buf_write(PacketFunction.PACKET_FUNC_MOTOR, data)
    print("Sent +0.5 to Motors 1,2,3,4. Moving for 2 seconds...")
    time.sleep(2)
    
    data = [0x01, 0x04]
    data.append(0x01); data.extend(struct.pack("<f", 0.0))
    data.append(0x02); data.extend(struct.pack("<f", 0.0))
    data.append(0x03); data.extend(struct.pack("<f", 0.0))
    data.append(0x04); data.extend(struct.pack("<f", 0.0))
    board.buf_write(PacketFunction.PACKET_FUNC_MOTOR, data)
    print("Stopped.")

if __name__ == "__main__":
    test_motors()
