#!/usr/bin/env python3
# encoding: utf-8
# stm32 python sdk

# --- 导入标准库 ---
import enum     # 用于定义枚举类型，使代码更具可读性（如状态机状态、功能码）
import time     # 用于延时 (sleep) 和时间戳获取
import queue    # 线程安全的队列，用于在接收线程（后台）和主线程（前台）之间传递数据
import struct   # 【核心】用于 Python 数据类型（如 float, int）与 C 语言二进制字节流之间的打包和解包
import serial   # PySerial 库，用于串口通信 (USB 转 TTL)
import threading # 多线程支持，用于并行处理串口接收

# --- 定义协议解析状态机的状态 ---
# 这是一个有限状态机 (FSM)，用于从连续的串口字节流中解析出完整的数据包
# 协议格式通常为: [0xAA] [0x55] [长度] [功能ID] [数据内容...] [校验码]

class PacketControllerState(enum.IntEnum):
    # 通信协议的格式(the format of the communication protocol)
    # 0xAA 0x55 Length Function ID Data Checksum
    PACKET_CONTROLLER_STATE_STARTBYTE1 = 0 # 等待第1个帧头 (0xAA)
    PACKET_CONTROLLER_STATE_STARTBYTE2 = 1 # 等待第2个帧头 (0x55)
    PACKET_CONTROLLER_STATE_LENGTH = 2     # 等待数据长度字节
    PACKET_CONTROLLER_STATE_FUNCTION = 3   # 等待功能ID字节
    PACKET_CONTROLLER_STATE_ID = 4         # (预留状态，代码中主要通过 FUNCTION 判断)
    PACKET_CONTROLLER_STATE_DATA = 5       # 正在读取数据体
    PACKET_CONTROLLER_STATE_CHECKSUM = 6   # 等待校验码

# --- 定义功能ID (Function ID) ---
# 这些 ID 对应 STM32 固件中不同的控制功能或传感器数据
class PacketFunction(enum.IntEnum):
    # 可通过串口实现的控制功能(achieve control function via the serial port)
    PACKET_FUNC_SYS = 0         # 系统信息（如电池电压）
    PACKET_FUNC_LED = 1         # LED控制(LED control)
    PACKET_FUNC_BUZZER = 2      # 蜂鸣器控制(buzzer control)
    PACKET_FUNC_MOTOR = 3       # 电机控制(motor control)
    PACKET_FUNC_PWM_SERVO = 4   # PWM舵机控制, 板子上从里到外依次为1-4
    PACKET_FUNC_BUS_SERVO = 5   # 总线舵机控制(bus servo control)
    PACKET_FUNC_KEY = 6         # 按键获取(obtain button)
    PACKET_FUNC_IMU = 7         # IMU获取(obtain IMU，即陀螺仪+加速度计)
    PACKET_FUNC_GAMEPAD = 8     # 手柄获取(obtain handle)
    PACKET_FUNC_SBUS = 9        # 航模遥控获取(obtain model aircraft remote control)
    PACKET_FUNC_OLED = 10       # OLED 显示内容设置(set OLED display content)
    PACKET_FUNC_RGB = 11        # RGB 灯珠控制
    PACKET_FUNC_NONE = 12       # 空指令

# --- 定义按键事件类型 ---
# 这是一个位掩码或者枚举值，表示按键的具体动作
class PacketReportKeyEvents(enum.IntEnum):
    # 按键的不同状态(different button status)
    KEY_EVENT_PRESSED = 0x01            # 按下
    KEY_EVENT_LONGPRESS = 0x02          # 长按
    KEY_EVENT_LONGPRESS_REPEAT = 0x04   # 长按连发
    KEY_EVENT_RELEASE_FROM_LP = 0x08    # 长按后松开
    KEY_EVENT_RELEASE_FROM_SP = 0x10    # 短按后松开
    KEY_EVENT_CLICK = 0x20              # 单击
    KEY_EVENT_DOUBLE_CLICK= 0x40        # 双击
    KEY_EVENT_TRIPLE_CLICK = 0x80       # 三击

# --- CRC8 校验表 ---
# 预计算好的查找表，用于快速计算数据的校验和，确保传输未出错
crc8_table = [
    0, 94, 188, 226, 97, 63, 221, 131, 194, 156, 126, 32, 163, 253, 31, 65,
    157, 195, 33, 127, 252, 162, 64, 30, 95, 1, 227, 189, 62, 96, 130, 220,
    35, 125, 159, 193, 66, 28, 254, 160, 225, 191, 93, 3, 128, 222, 60, 98,
    190, 224, 2, 92, 223, 129, 99, 61, 124, 34, 192, 158, 29, 67, 161, 255,
    70, 24, 250, 164, 39, 121, 155, 197, 132, 218, 56, 102, 229, 187, 89, 7,
    219, 133, 103, 57, 186, 228, 6, 88, 25, 71, 165, 251, 120, 38, 196, 154,
    101, 59, 217, 135, 4, 90, 184, 230, 167, 249, 27, 69, 198, 152, 122, 36,
    248, 166, 68, 26, 153, 199, 37, 123, 58, 100, 134, 216, 91, 5, 231, 185,
    140, 210, 48, 110, 237, 179, 81, 15, 78, 16, 242, 172, 47, 113, 147, 205,
    17, 79, 173, 243, 112, 46, 204, 146, 211, 141, 111, 49, 178, 236, 14, 80,
    175, 241, 19, 77, 206, 144, 114, 44, 109, 51, 209, 143, 12, 82, 176, 238,
    50, 108, 142, 208, 83, 13, 239, 177, 240, 174, 76, 18, 145, 207, 45, 115,
    202, 148, 118, 40, 171, 245, 23, 73, 8, 86, 180, 234, 105, 55, 213, 139,
    87, 9, 235, 181, 54, 104, 138, 212, 149, 203, 41, 119, 244, 170, 72, 22,
    233, 183, 85, 11, 136, 214, 52, 106, 43, 117, 151, 201, 74, 20, 246, 168,
    116, 42, 200, 150, 21, 75, 169, 247, 182, 232, 10, 84, 215, 137, 107, 53
]

def checksum_crc8(data):
    # 校验(check)函数
    # 输入字节数据，返回一个字节的校验码
    check = 0
    for b in data:
        check = crc8_table[check ^ b]
    return check & 0x00FF

# --- SBUS 协议辅助类 ---
# 用于存储航模遥控器的状态
class SBusStatus:
    def __init__(self):
        self.channels = [0] * 16 # 16个通道的值
        self.channel_17 = False
        self.channel_18 = False
        self.signal_loss = True  # 是否丢失信号
        self.fail_safe = False   # 失控保护状态

# --- 核心驱动类 Board ---
class Board:
    # 手柄按键掩码映射表：用于从一个 16bit 的整数中提取出具体哪个键被按下了
    buttons_map = {
            'GAMEPAD_BUTTON_MASK_L2':        0x0001,
            'GAMEPAD_BUTTON_MASK_R2':        0x0002,
            'GAMEPAD_BUTTON_MASK_SELECT':    0x0004,
            'GAMEPAD_BUTTON_MASK_START':     0x0008,
            'GAMEPAD_BUTTON_MASK_L3':        0x0020,
            'GAMEPAD_BUTTON_MASK_R3':        0x0040,
            'GAMEPAD_BUTTON_MASK_CROSS':     0x0100,
            'GAMEPAD_BUTTON_MASK_CIRCLE':    0x0200,
            'GAMEPAD_BUTTON_MASK_SQUARE':    0x0800,
            'GAMEPAD_BUTTON_MASK_TRIANGLE':  0x1000,
            'GAMEPAD_BUTTON_MASK_L1':        0x4000,
            'GAMEPAD_BUTTON_MASK_R1':        0x8000
    }

    def __init__(self, device="/dev/ttyACM0", baudrate=115200, timeout=10, debug: bool = False):
        self.enable_recv = False # 接收使能标志
        self.debug = bool(debug)
        self.device = device
        self.baudrate = baudrate
        self.timeout = timeout
        self.last_motor_trace = None
        self._last_recv_err_log_time = 0.0
        self._last_reopen_log_time = 0.0
        self.frame = []     # 临时存储接收到的数据帧
        self.recv_count = 0 # 当前已接收的数据字节计数

        # 初始化串口连接
        self.port = None
        self._open_port()

        self.state = PacketControllerState.PACKET_CONTROLLER_STATE_STARTBYTE1 # 初始化状态机状态

        # 线程锁：防止多线程同时读写造成混乱
        self.servo_read_lock = threading.Lock()
        self.pwm_servo_read_lock = threading.Lock()

        # 队列用来存储数据(use queue to store data)
        # maxsize=1 意味着只保留最新的数据，旧数据如果没处理会被丢弃
        self.sys_queue = queue.Queue(maxsize=1)
        self.bus_servo_queue = queue.Queue(maxsize=1)
        self.pwm_servo_queue = queue.Queue(maxsize=1)
        self.key_queue = queue.Queue(maxsize=1)
        self.imu_queue = queue.Queue(maxsize=1)
        self.gamepad_queue = queue.Queue(maxsize=1)
        self.sbus_queue = queue.Queue(maxsize=1)

        # 注册解析回调函数：功能码 -> 处理函数
        self.parsers = {
            PacketFunction.PACKET_FUNC_SYS: self.packet_report_sys,
            PacketFunction.PACKET_FUNC_KEY: self.packet_report_key,
            PacketFunction.PACKET_FUNC_IMU: self.packet_report_imu,
            PacketFunction.PACKET_FUNC_GAMEPAD: self.packet_report_gamepad,
            PacketFunction.PACKET_FUNC_BUS_SERVO: self.packet_report_serial_servo,
            PacketFunction.PACKET_FUNC_SBUS: self.packet_report_sbus,
            PacketFunction.PACKET_FUNC_PWM_SERVO: self.packet_report_pwm_servo
        }

        time.sleep(0.5)
        # 启动接收线程，设置为 daemon 守护线程，主程序退出时它也会自动退出
        threading.Thread(target=self.recv_task, daemon=True).start()

    def _open_port(self):
        self.port = serial.Serial(None, self.baudrate, timeout=self.timeout)
        self.port.rts = False
        self.port.dtr = False
        try:
            # Linux 下启用独占可更早暴露“串口被其它进程占用”的问题
            self.port.exclusive = True
        except Exception:
            pass
        self.port.setPort(self.device)
        self.port.open() # 打开串口

    def _reopen_port(self):
        try:
            if self.port is not None and self.port.is_open:
                self.port.close()
        except Exception:
            pass

        time.sleep(0.2)
        try:
            self._open_port()
            self.state = PacketControllerState.PACKET_CONTROLLER_STATE_STARTBYTE1
            self.frame = []
            self.recv_count = 0
            now = time.time()
            if now - self._last_reopen_log_time > 1.0:
                print(f"[Serial] Reconnected to {self.device}")
                self._last_reopen_log_time = now
        except Exception as e:
            now = time.time()
            if now - self._last_reopen_log_time > 1.0:
                print(f"[Serial] Reconnect failed on {self.device}: {e}")
                self._last_reopen_log_time = now


    # --- 数据分发回调函数 (Putters) ---
    # 这些函数由 recv_task 调用，将解析好的数据放入对应的队列中
    
    def packet_report_sys(self, data):
        try:
            # put_nowait: 非阻塞放入。如果队列满(maxsize=1)，直接抛 Full 异常，忽略旧数据
            self.sys_queue.put_nowait(data)
        except queue.Full:
            pass

    def packet_report_key(self, data):
        try:
            self.key_queue.put_nowait(data)
        except queue.Full:
            pass

    def packet_report_imu(self, data):
        try:
            self.imu_queue.put_nowait(data)
        except queue.Full:
            pass

    def packet_report_gamepad(self, data):
        try:
            self.gamepad_queue.put_nowait(data)
        except queue.Full:
            pass

    def packet_report_serial_servo(self, data):
        try:
            self.bus_servo_queue.put_nowait(data)
        except queue.Full:
            pass

    def packet_report_pwm_servo(self, data):
        try:
            self.pwm_servo_queue.put_nowait(data)
        except queue.Full:
            pass

    def packet_report_sbus(self, data):
        try:
            self.sbus_queue.put_nowait(data)
        except queue.Full:
            pass

    # --- 数据获取接口 (Getters) ---
    # 这些函数供外部（如 ROS 节点）调用，从队列中取出 Python 格式的数据

    def get_battery(self):
        # 获取电压，单位mAh(obtain voltage, which is in the unit of mAh)
        # 注意注释写的是 mAh，但通常这个接口返回的是 mV 或者电压值
        if self.enable_recv:
            try:
                # get(block=False): 非阻塞获取。如果队列空，抛 Empty 异常
                data = self.sys_queue.get(block=False)
                if data[0] == 0x04:
                    # struct.unpack('<H'): 解包为 Little-endian 的 Unsigned Short (2字节)
                    return struct.unpack('<H', data[1:])[0]
                else:
                    None
            except queue.Empty:
                return None
        else:
            print('get_battery enable reception first!')
            return None

    def get_button(self):
        # 获取按键key1， key2状态，返回按键ID(1表示按键1，2表示按键2)和状态(0表示按下，1表示松开)
        if self.enable_recv:
            try:
                data = self.key_queue.get(block=False)
                key_id = data[0]
                key_event = PacketReportKeyEvents(data[1])
                # 将复杂的事件类型简化为 0/1 状态
                if key_event == PacketReportKeyEvents.KEY_EVENT_CLICK:
                    return key_id, 0
                elif key_event == PacketReportKeyEvents.KEY_EVENT_PRESSED:
                    return key_id, 1
            except queue.Empty:
                return None
        else:
            print('get_button enable reception first!')
            return None

    def get_imu(self):
        # 获取IMU数据，返回ax, ay, az, gx, gy, gz(obtain IMU data to return ax, ay, az, gx, gy, and gz)
        if self.enable_recv:
            try:
                # ax, ay, az, gx, gy, gz
                # struct.unpack('<6f'): 解包为 6 个 float (每个4字节，共24字节)
                return struct.unpack('<6f', self.imu_queue.get(block=False))
            except queue.Empty:
                return None
        else:
            print('get_imu enable reception first!')
            return None

    def get_gamepad(self):
        # 获取手柄数据(obtain handle data)
        # 手柄数据解析相对复杂，需要处理位掩码和模拟量归一化
        if self.enable_recv:
            try:
                # buttons(Unsigned Short), hat(Byte), lx, ly, rx, ry (4 bytes)
                gamepad_data = struct.unpack("<HB4b", self.gamepad_queue.get(block=False))
                
                # 初始化 8 个轴数据，16 个按钮数据
                # 'lx', 'ly', 'rx', 'ry', 'r2', 'l2', 'hat_x', 'hat_y'
                axes = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                # 'cross', 'circle', '', 'square', 'triangle', '', 'l1', 'r1', 'l2', 'r2', 'select', 'start', '', 'l3', 'r3', ''
                buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
                
                # 遍历字典，检查按键掩码
                for b in self.buttons_map:
                    if self.buttons_map[b] & gamepad_data[0]:
                        if b == 'GAMEPAD_BUTTON_MASK_R2':
                            axes[4] = 1.0 # R2 既是按钮也是轴
                        elif b == 'GAMEPAD_BUTTON_MASK_L2':
                            axes[5] = 1.0 # L2 同上
                        elif b == 'GAMEPAD_BUTTON_MASK_CROSS':
                            buttons[0] = 1
                        elif b == 'GAMEPAD_BUTTON_MASK_CIRCLE':
                            buttons[1] = 1
                        elif b == 'GAMEPAD_BUTTON_MASK_SQUARE':
                            buttons[3] = 1
                        elif b == 'GAMEPAD_BUTTON_MASK_TRIANGLE':
                            buttons[4] = 1
                        elif b == 'GAMEPAD_BUTTON_MASK_L1':
                            buttons[6] = 1
                        elif b == 'GAMEPAD_BUTTON_MASK_R1':
                            buttons[7] = 1
                        elif b == 'GAMEPAD_BUTTON_MASK_SELECT':
                            buttons[10] = 1
                        elif b == 'GAMEPAD_BUTTON_MASK_START':
                            buttons[11] = 1
                
                # 处理左摇杆 X 轴 (死区处理与归一化)
                if gamepad_data[2] > 0:
                    axes[0] = -gamepad_data[2] / 127
                elif gamepad_data[2] < 0:
                    axes[0] = -gamepad_data[2] / 128

                # 处理左摇杆 Y 轴
                if gamepad_data[3] > 0:
                    axes[1] = gamepad_data[3] / 127
                elif gamepad_data[3] < 0:
                    axes[1] = gamepad_data[3] / 128

                # 处理右摇杆 X 轴
                if gamepad_data[4] > 0:
                    axes[2] = -gamepad_data[4] / 127
                elif gamepad_data[4] < 0:
                    axes[2] = -gamepad_data[4] / 128

                # 处理右摇杆 Y 轴
                if gamepad_data[5] > 0:
                    axes[3] = gamepad_data[5] / 127
                elif gamepad_data[5] < 0:
                    axes[3] = gamepad_data[5] / 128
            
                # 处理 HAT (方向键)
                if gamepad_data[1] == 9:
                    axes[6] = 1.0
                elif gamepad_data[1] == 13:
                    axes[6] = -1.0
                
                if gamepad_data[1] == 11:
                    axes[7] = -1.0
                elif gamepad_data[1] == 15:
                    axes[7] = 1.0
                return axes, buttons
            except queue.Empty:
                return None
        else:
            print('get_gamepad enable reception first!')
            return None

    def get_sbus(self):
        # 解析航模接收机 SBUS 信号
        if self.enable_recv:
            try:
                sbus_data = self.sbus_queue.get(block=False)
                status = SBusStatus()
                # 结构: 16个short通道, ch17, ch18, 丢失标志, 失控保护标志
                *status.channels, ch17, ch18, sig_loss, fail_safe = struct.unpack("<16hBBBB", sbus_data)
                status.channel_17 = ch17 != 0
                status.channel_18 = ch18 != 0
                status.signal_loss = sig_loss != 0
                status.fail_safe = fail_safe != 0
                data = []
                if status.signal_loss:
                    # 信号丢失时的默认值处理
                    data = 16 * [0.5]
                    data[4] = 0
                    data[5] = 0
                    data[6] = 0
                    data[7] = 0
                else:
                    # 正常数据的归一化处理 (映射到 -1.0 到 1.0)
                    for i in status.channels:
                        data.append(2*(i - 192)/(1792 - 192) - 1)
                return data
            except queue.Empty:
                return None
        else:
            print('get_sbus enable reception first!')
            return None

    # --- 底层发送函数 ---
    def buf_write(self, func, data):
        # 封装协议包：[0xAA, 0x55, 功能码, 长度, 数据..., 校验码]
        buf = [0xAA, 0x55, int(func)]
        buf.append(len(data))
        buf.extend(data)
        buf.append(checksum_crc8(bytes(buf[2:]))) # 计算校验码
        buf = bytes(buf)
        if self.debug:
            print(buf) # 调试模式打印发送的字节
        self.port.write(buf)
        # print(buf)


    # --- 控制指令接口 (Setters) ---

    def set_led(self, on_time, off_time, repeat=1, led_id=1):
        # 设置 LED 闪烁参数
        on_time = int(on_time*1000)
        off_time = int(off_time*1000)
        # 打包: ID(Byte), OnTime(UShort), OffTime(UShort), Repeat(UShort)
        self.buf_write(PacketFunction.PACKET_FUNC_LED, struct.pack("<BHHH", led_id, on_time, off_time, repeat))

    def set_buzzer(self, freq, on_time, off_time, repeat=1):
        # 设置蜂鸣器参数
        on_time = int(on_time*1000)
        off_time = int(off_time*1000)
        self.buf_write(PacketFunction.PACKET_FUNC_BUZZER, struct.pack("<HHHH", freq, on_time, off_time, repeat))

    def set_motor_speed(self, speeds):
        # 设置电机速度 (这是最常用的控制函数)
        angle = 0.0
        vx = 0.0
        vy = 0.0
        
        # 遍历输入列表，解析速度
        for s in speeds:
            if s[0] == 0:
                angle = float(s[1]) # ID 0: 角速度 (Angular Z)
            elif s[0] == 1:
                vx = float(s[1])    # ID 1: 线性速度 X
            elif s[0] == 2:
                vy = float(s[1])    # ID 2: 线性速度 Y
        
        # Format: Angle(4bytes) + 0x02(1byte) + 0x01(X_ID) + Vx(4bytes) + 0x02(Y_ID) + Vy(4bytes)
        # 第五到第八字节：角度
        # 第九字节：写死的0x02标识
        # 第十字节：0x01代表X方向速度ID
        # 第十一到十四字节：X速度值
        # 第十五字节：0x02代表Y方向速度ID
        # 第十六到十九字节：Y速度值
        data = []
        # 添加角度（4字节浮点）
        data.extend(struct.pack("<f", angle))
        # 添加写死的标识0x02
        data.append(0x02)
        # 添加X方向速度信息
        data.append(0x01)  # X方向ID
        data.extend(struct.pack("<f", vx))
        # 添加Y方向速度信息
        data.append(0x02)  # Y方向ID
        data.extend(struct.pack("<f", vy))

        self.last_motor_trace = {
            'time': time.time(),
            'angle': angle,
            'vx': vx,
            'vy': vy,
            'payload_hex': bytes(data).hex(),
        }

        if self.debug:
            print(f"[SDK_MOTOR] angle={angle:.3f} vx={vx:.3f} vy={vy:.3f} payload={bytes(data).hex()}")
        
        self.buf_write(PacketFunction.PACKET_FUNC_MOTOR, data)

    
    # 下面这段是原始代码中注释掉的 set_rgb 实现，保留原貌
    #def set_rgb(self, pixels):
        #data = [0x01, len(pixels), ]
        #data = pixels
       # data = []
        #print('data:',data)
        #if len(pixels) > 4:
        #data = [0]
        #for index, r, g, b in pixels:
        #   data.extend(struct.pack("<BBBB", int(index), int(r), int(g), int(b)))
        #data.extend(struct.pack("<BBBBBBB", int(0), int(0), int(0), int(222),int(0), int(0), int(222)))
       # data.extend(struct.pack("<BBBBBBB", 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00))
       # print('data:',data)
        #self.buf_write(PacketFunction.PACKET_FUNC_RGB, data)
    
    def set_rgb(self , pixels):
        # 实际使用的 RGB 灯控制函数
        data = [0x01 , len(pixels),] # 子命令 + 像素个数
        for index , r , g , b in pixels:
            # 打包每个像素的 Index, R, G, B
            data.extend(struct.pack("<BBBB", int(index - 1) , int(r), int(g) , int(b)))
        self.buf_write(PacketFunction.PACKET_FUNC_RGB , data)
        
    def set_oled_text(self, line, text):
        # 设置 OLED 显示文本
        data = [line, len(text)] # 子命令为行号, 第二个字节是字符串长度，该长度包含'\0'字符串结束符
        data.extend(bytes(text, encoding='utf-8'))
        self.buf_write(PacketFunction.PACKET_FUNC_OLED, data)

    # --- PWM 舵机相关 ---
    def pwm_servo_set_position(self, duration, positions):
        # 批量设置 PWM 舵机位置
        duration = int(duration * 1000)
        # 包头: 子命令0x01 + 持续时间(2字节) + 舵机数量
        data = [0x01, duration & 0xFF, 0xFF & (duration >> 8), len(positions)]
        for i in positions:
            # 舵机数据: ID + 位置
            data.extend(struct.pack("<BH", i[0], i[1]))
        self.buf_write(PacketFunction.PACKET_FUNC_PWM_SERVO, data)

    def pwm_servo_set_offset(self, servo_id, offset):
        # 设置 PWM 舵机偏移量 (校准用)
        # 0x07 是设置 offset 的子命令
        data = struct.pack("<BBb", 0x07, servo_id, int(offset))
        self.buf_write(PacketFunction.PACKET_FUNC_PWM_SERVO, data)

    def pwm_servo_read_and_unpack(self, servo_id, cmd, unpack):
        # 这是一个同步读取函数：发送请求 -> 阻塞等待 -> 返回结果
        with self.servo_read_lock: # 加锁确保原子操作
            self.buf_write(PacketFunction.PACKET_FUNC_PWM_SERVO, [cmd, servo_id])
            # block=True: 必须等待队列有数据
            data = self.pwm_servo_queue.get(block=True)
            servo_id, cmd, info = struct.unpack(unpack, data)
            return info

    def pwm_servo_read_offset(self, servo_id):
        # 读取舵机偏移量 (子命令 0x09)
        return self.pwm_servo_read_and_unpack(servo_id, 0x09, "<BBb")

    def pwm_servo_read_position(self, servo_id):
        # 读取舵机当前位置 (子命令 0x05)
        return self.pwm_servo_read_and_unpack(servo_id, 0x05, "<BBH")

    # --- 总线舵机 (Bus Servo) 相关 ---
    # 逻辑与 PWM 舵机类似，但功能码和子命令定义不同

    def bus_servo_enable_torque(self, servo_id, enable):
        # 开启/关闭 扭矩输出 (锁力)
        if enable:
            data = struct.pack("<BB", 0x0B, servo_id) # 0x0B 开启
        else:
            data = struct.pack("<BB", 0x0C, servo_id) # 0x0C 关闭
        self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, data)
        time.sleep(0.02) # 总线通信需要一点间隔

    def bus_servo_set_id(self, servo_id_now, servo_id_new):
        # 修改舵机 ID (慎用)
        data = struct.pack("<BBB", 0x10, servo_id_now, servo_id_new)
        self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, data)
        time.sleep(0.02)

    def bus_servo_set_offset(self, servo_id, offset):
        data = struct.pack("<BBb", 0x20, servo_id, int(offset))
        self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, data)
        time.sleep(0.02)

    def bus_servo_save_offset(self, servo_id):
        # 保存偏移量到舵机内部存储
        data = struct.pack("<BB", 0x24, servo_id)
        self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, data)
        time.sleep(0.02)

    def bus_servo_set_angle_limit(self, servo_id, limit):
        # 设置角度限制
        data = struct.pack("<BBHH", 0x30, servo_id, int(limit[0]), int(limit[1]))
        self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, data)
        time.sleep(0.02)

    def bus_servo_set_vin_limit(self, servo_id, limit):
        # 设置电压限制
        data = struct.pack("<BBHH", 0x34, servo_id, int(limit[0]), int(limit[1]))
        self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, data)
        time.sleep(0.02)

    def bus_servo_set_temp_limit(self, servo_id, limit):
        # 设置温度限制
        data = struct.pack("<BBb", 0x38, servo_id, int(limit))
        self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, data)
        time.sleep(0.02)

    def bus_servo_stop(self, servo_id):
        # 舵机急停
        data = [0x03, len(servo_id)] 
        data.extend(struct.pack("<"+'B'*len(servo_id), *servo_id))
        self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, data)

    def bus_servo_set_position(self, duration, positions):
        # 批量设置总线舵机位置
        duration = int(duration * 1000)
        data = [0x01, duration & 0xFF, 0xFF & (duration >> 8), len(positions)] # 0x01 是子命令
        for i in positions:
            data.extend(struct.pack("<BH", i[0], i[1]))
        self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, data)

    def bus_servo_read_and_unpack(self, servo_id, cmd, unpack):
        # 同步读取总线舵机数据
        with self.servo_read_lock:
            self.buf_write(PacketFunction.PACKET_FUNC_BUS_SERVO, [cmd, servo_id])
            # 阻塞等待返回
            data = self.bus_servo_queue.get(block=True)
            # 总线舵机返回包多了一个 success 字节
            servo_id, cmd, success, *info = struct.unpack(unpack, data)
            if success == 0: # 0 代表读取成功
                return info

    def bus_servo_read_id(self, servo_id=254):
        # 读取 ID (254 是广播地址)
        return self.bus_servo_read_and_unpack(servo_id, 0x12, "<BBbB")

    def bus_servo_read_offset(self, servo_id):
        return self.bus_servo_read_and_unpack(servo_id, 0x22, "<BBbb")
    
    def bus_servo_read_position(self, servo_id):
        return self.bus_servo_read_and_unpack(servo_id, 0x05, "<BBbh")

    def bus_servo_read_vin(self, servo_id):
        return self.bus_servo_read_and_unpack(servo_id, 0x07, "<BBbH")
    
    def bus_servo_read_temp(self, servo_id):
        return self.bus_servo_read_and_unpack(servo_id, 0x09, "<BBbB")

    def bus_servo_read_temp_limit(self, servo_id):
        return self.bus_servo_read_and_unpack(servo_id, 0x3A, "<BBbB")

    def bus_servo_read_angle_limit(self, servo_id):
        return self.bus_servo_read_and_unpack(servo_id, 0x32, "<BBb2H")

    def bus_servo_read_vin_limit(self, servo_id):
        return self.bus_servo_read_and_unpack(servo_id, 0x36, "<BBb2H")

    def bus_servo_read_torque_state(self, servo_id):
        return self.bus_servo_read_and_unpack(servo_id, 0x0D, "<BBbb")

    def enable_reception(self, enable=True):
        self.enable_recv = enable

    # --- 【核心】接收线程任务：状态机 (Finite State Machine) ---
    def recv_task(self):
        while True:
            if self.enable_recv:
                try:
                    # 检查串口缓冲区是否有数据
                    if self.port.in_waiting > 0:
                        recv_data = self.port.read(self.port.in_waiting)
                        if recv_data:
                            for dat in recv_data:
                                # --- 状态 0: 寻找第一个帧头 0xAA ---
                                if self.state == PacketControllerState.PACKET_CONTROLLER_STATE_STARTBYTE1:
                                    if dat == 0xAA:
                                        self.state = PacketControllerState.PACKET_CONTROLLER_STATE_STARTBYTE2
                                    continue
                                
                                # --- 状态 1: 寻找第二个帧头 0x55 ---
                                elif self.state == PacketControllerState.PACKET_CONTROLLER_STATE_STARTBYTE2:
                                    if dat == 0x55:
                                        self.state = PacketControllerState.PACKET_CONTROLLER_STATE_FUNCTION
                                    else:
                                        self.state = PacketControllerState.PACKET_CONTROLLER_STATE_STARTBYTE1 # 匹配失败，重置
                                    continue
                                
                                # --- 状态 2: 读取功能码 ---
                                elif self.state == PacketControllerState.PACKET_CONTROLLER_STATE_FUNCTION:
                                    if dat < int(PacketFunction.PACKET_FUNC_NONE):
                                        self.frame = [dat, 0] # 初始化帧缓冲 [功能码, 长度占位]
                                        self.state = PacketControllerState.PACKET_CONTROLLER_STATE_LENGTH
                                    else:
                                        self.frame = []
                                        self.state = PacketControllerState.PACKET_CONTROLLER_STATE_STARTBYTE1
                                    continue
                                
                                # --- 状态 3: 读取数据长度 ---
                                elif self.state == PacketControllerState.PACKET_CONTROLLER_STATE_LENGTH:
                                    self.frame[1] = dat # 记录长度
                                    self.recv_count = 0
                                    if dat == 0:
                                        # 如果数据长度为0，直接跳去校验
                                        self.state = PacketControllerState.PACKET_CONTROLLER_STATE_CHECKSUM
                                    else:
                                        self.state = PacketControllerState.PACKET_CONTROLLER_STATE_DATA
                                    continue
                                
                                # --- 状态 4: 读取数据体 ---
                                elif self.state == PacketControllerState.PACKET_CONTROLLER_STATE_DATA:
                                    self.frame.append(dat)
                                    self.recv_count += 1
                                    # 如果读够了长度，就进入下一状态
                                    if self.recv_count >= self.frame[1]:
                                        self.state = PacketControllerState.PACKET_CONTROLLER_STATE_CHECKSUM
                                    continue
                                
                                # --- 状态 5: 校验与分发 ---
                                elif self.state == PacketControllerState.PACKET_CONTROLLER_STATE_CHECKSUM:
                                    # 计算接收到的数据的 CRC8
                                    crc8 = checksum_crc8(bytes(self.frame))
                                    if crc8 == dat: # 如果校验码匹配
                                        func = PacketFunction(self.frame[0])
                                        data = bytes(self.frame[2:]) # 提取数据部分
                                        
                                        # 查找解析器并调用
                                        if func in self.parsers:
                                            self.parsers[func](data)
                                        # 校验成功后清零错误计数，说明链路已同步
                                        self._checksum_err_count = 0
                                    else:
                                        self._checksum_err_count = getattr(self, '_checksum_err_count', 0) + 1
                                        # 只有连续多次校验失败，或者调试模式开启且不是刚启动时，才打印错误。
                                        # 因为串口刚打开时，硬件缓冲区残留的不完整帧极易导致一两次校验失败，这是正常现象。
                                        if self._checksum_err_count > 2 and self.debug:
                                            print(f"[SDK_WARN] 串口连续校验失败 (帧未对齐/丢包): func={self.frame[0]} len={self.frame[1]}")
                                    
                                    # 一帧处理完毕，重置状态机，准备接收下一帧
                                    self.state = PacketControllerState.PACKET_CONTROLLER_STATE_STARTBYTE1
                                    continue
                        else:
                            time.sleep(0.001)
                except Exception as e:
                    now = time.time()
                    if now - self._last_recv_err_log_time > 1.0:
                        print(f"Recv error: {e}")
                        self._last_recv_err_log_time = now
                    self._reopen_port()
                    time.sleep(0.1)
            else:
                time.sleep(0.01) # 如果未开启接收，小睡一下避免占满 CPU
        self.port.close()
        print("END...")

# --- 测试函数 ---
# 这些函数通常用于调试驱动本身，不通过 ROS 调用

def bus_servo_test(board):
    # 总线舵机测试流程
    board.bus_servo_set_position(1, [[1, 500], [2, 500]])
    time.sleep(1)
    board.bus_servo_set_position(2, [[1, 0], [2, 0]])
    time.sleep(1)
    board.bus_servo_stop([1, 2])
    time.sleep(1)
    
    servo_id = 1
    board.bus_servo_set_id(254, servo_id) # 254 是广播
    servo_id = board.bus_servo_read_id()
    if servo_id is not None:
        servo_id = servo_id[0]
        
        offset_set = -10
        board.bus_servo_set_offset(servo_id, offset_set)
        board.bus_servo_save_offset(servo_id)
        
        vin_l, vin_h = 4500, 14500
        board.bus_servo_set_vin_limit(servo_id, [vin_l, vin_h])

        temp_limit = 85
        board.bus_servo_set_temp_limit(servo_id, temp_limit)

        angle_l, angle_h = 0, 1000
        board.bus_servo_set_angle_limit(servo_id, [angle_l, angle_h])
        
        board.bus_servo_enable_torque(servo_id, 1)

        print('id:', board.bus_servo_read_id(servo_id))
        print('offset:', board.bus_servo_read_offset(servo_id), offset_set)
        print('vin:', board.bus_servo_read_vin(servo_id))
        print('temp:', board.bus_servo_read_temp(servo_id))
        print('position:', board.bus_servo_read_position(servo_id))
        print('angle_limit:', board.bus_servo_read_angle_limit(servo_id), [angle_l, angle_h])
        print('vin_limit:', board.bus_servo_read_vin_limit(servo_id), [vin_l, vin_h])
        print('temp_limit:', board.bus_servo_read_temp_limit(servo_id), temp_limit)
        print('torque_state:', board.bus_servo_read_torque_state(servo_id))

def pwm_servo_test(board):
    # PWM 舵机测试流程
    servo_id = 1
    board.pwm_servo_set_position(0.5, [[servo_id, 500]])
    board.pwm_servo_set_offset(servo_id, 0)
    board.pwm_servo_set_position(0.5, [[servo_id, 1500]])
    print('offset:', board.pwm_servo_read_offset(servo_id))
    print('position:', board.pwm_servo_read_position(servo_id))

if __name__ == "__main__":
    # 程序入口：如果直接运行此文件，将执行以下测试逻辑
    board = Board(device="/dev/ttyACM0")
    board.enable_reception()
    print("START...")
    #time.sleep(2)
    # board.set_led(0.1, 0.9, 1,1)
    #board.set_led(0.1, 0.9, 5,2)
    # board.set_buzzer(1900, 0.05, 0.01, 1)
    #time.sleep(1)0
    #board.set_buzzer(1900, 0.05, 0.01, 1)
    #time.sleep(1)
    #board.set_rgb([[2, 100, 0, 0],[1,100,0,0]])
    #time.sleep(0.5)
    #board.set_rgb([[2, 0, 0, 255],[1,0,0,255]])
    #time.sleep(0.5)
    #board.set_rgb([[2, 255, 0, 0],[1,255,0,0]])
    #time.sleep(0.5)
    #board.set_rgb([[1, 0, 255, 0]])
    board.set_motor_speed([[0, 0], [1, 1], [2, 0]])  # 分别是角速度，X速度，Y速度
    #time.sleep(1)
    #board.set_motor_speed([[1, 0], [2, 0], [3, 0], [4, 0]])
    
    #bus_servo_test(board)
    #board.bus_servo_set_position(1, [[1, 700], [2, 500]])
    # pwm_servo_test(board)
    # last_time = time.time()
    while True:
        try:
            # board.set_buzzer(3000, 0.05, 0.01, 1)
            res = board.get_imu()
            if res is not None:
                for item in res:
                   print("  {: .8f} ".format(item), end='')
                print()
            # res = board.get_button()
            # if res is not None:
                # print(res)
            # data = board.get_gamepad()
            # if data is not None:
                # print(data[0])
                # print(data[1])
            # res = board.get_sbus()
            # if res is not None:
                # print(res)
            # res = board.get_battery()
            # if res is not None:
                # print(res)
            #board.set_rgb([[2, 50, 0, 0],[1,50,0,0]])
            #time.sleep(0.05)
            #board.set_rgb([[2, 0, 50, 0],[1,0,50,0]])
            #time.sleep(0.05)
            #board.set_rgb([[2, 255, 0, 0],[1,255,0,0]])
            
            #time.sleep(0.1)
            # t = time.time()
            # print(1/(t - last_time))
            # last_time = t
        except KeyboardInterrupt:
            break
