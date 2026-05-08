#!/usr/bin/env python3
# encoding: utf-8
# @Author: hsc
# @Date: 2026.1.19
# stm32 ros2 package

import math
import time
import json
import rclpy
import signal 
import threading
try:
    import yaml  # type: ignore
except Exception:  # PyYAML may be missing in some environments
    yaml = None
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Imu, Joy
from std_msgs.msg import UInt16, Bool, String
from geometry_msgs.msg import Twist
from ros_robot_controller.ros_robot_controller_sdk import Board, PacketReportKeyEvents
from ros_robot_controller_msgs.srv import GetBusServoState, GetPWMServoState
from ros_robot_controller_msgs.msg import (
    ButtonState, BuzzerState, MotorsState, BusServoState, LedState,
    SetBusServoState, ServosPosition, SetPWMServoState, Sbus, OLEDState,
    RGBStates, PWMServoState
)

class RosRobotController(Node):
    gravity = 9.80665

    def __init__(self, name):
        if not rclpy.ok():
            rclpy.init()
        
        super().__init__(name)
        
        self.declare_parameter('device_name', '/dev/ttyACM0')
        self.declare_parameter('sdk_debug', False)
        device_name = self.get_parameter('device_name').value
        sdk_debug = bool(self.get_parameter('sdk_debug').value)
        self.get_logger().info(f"Attempting to connect to robot controller on {device_name}...")

        try:
            self.board = Board(device=device_name, debug=sdk_debug)
            self.board.enable_reception()
            self.get_logger().info(f"Successfully connected to {device_name}")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to board on {device_name}: {e}")
            self.board = None

        self.running = True

        self.declare_parameter('imu_frame', 'imu_link')
        self.declare_parameter('init_finish', False)
        self.IMU_FRAME = self.get_parameter('imu_frame').value

        self.imu_pub = self.create_publisher(Imu, '~/imu_raw', 1)
        self.joy_pub = self.create_publisher(Joy, '~/joy', 1)
        self.sbus_pub = self.create_publisher(Sbus, '~/sbus', 1)
        self.button_pub = self.create_publisher(ButtonState, '~/button', 1)
        self.battery_pub = self.create_publisher(UInt16, '~/battery', 1)
        self.motor_tx_pub = self.create_publisher(String, '/controller/motor_tx', 20)
        self.create_subscription(LedState, '~/set_led', self.set_led_state, 5)
        self.create_subscription(BuzzerState, '~/set_buzzer', self.set_buzzer_state, 5)
        self.create_subscription(OLEDState, '~/set_oled', self.set_oled_state, 5)
        self.create_subscription(MotorsState, '~/set_motor', self.set_motor_state, 10)
        self.create_subscription(Bool, '~/enable_reception', self.enable_reception, 1)
        self.create_subscription(SetBusServoState, '~/bus_servo/set_state', self.set_bus_servo_state, 10)
        self.create_subscription(ServosPosition, '~/bus_servo/set_position', self.set_bus_servo_position, 10)
        self.create_subscription(SetPWMServoState, '~/pwm_servo/set_state', self.set_pwm_servo_state, 10)
        self.create_service(GetBusServoState, '~/bus_servo/get_state', self.get_bus_servo_state)
        self.create_service(GetPWMServoState, '~/pwm_servo/get_state', self.get_pwm_servo_state)
        self.create_subscription(RGBStates, '~/set_rgb', self.set_rgb_states, 10)
        
        # 速度参数
        self.declare_parameter('vel_scale', 1.0)  # 默认放大倍数
        self.declare_parameter('max_v', 2.0)
        self.declare_parameter('max_w', 10.0)
        self.declare_parameter('min_v', 0.2)
        self.declare_parameter('min_vy', 0.0)
        self.declare_parameter('min_wz', 0.0)
        self.declare_parameter('angle_scale', 1.0)
        self.declare_parameter('min_w', 1.0)
        # 指令死区与 x-only 模式
        self.declare_parameter('cmd_deadband_y', 0.05)
        self.declare_parameter('cmd_deadband_z', 0.05)
        self.declare_parameter('x_only_mode', False)
        # 零速度抑制/保持
        self.declare_parameter('zero_cmd_hold_sec', 0.25)
        self.declare_parameter('zero_cmd_epsilon', 1e-4)
        # 忽略零指令（在指定超时内）
        self.declare_parameter('ignore_zero_cmd', True)
        self.declare_parameter('zero_cmd_stop_timeout', 1.0)
        # 零指令日志节流
        self.declare_parameter('zero_cmd_log_throttle_sec', 1.0)
        # 指令超时保持上一条速度
        self.declare_parameter('hold_last_cmd_on_timeout', True)
        self.declare_parameter('cmd_timeout_sec', 0.2)
        self.declare_parameter('cmd_resend_period_sec', 0.05)
        # 方向符号，可根据底盘实际坐标或接线调整
        # 默认使用 ROS 坐标系：x 前进为正，y 向左为正，z 逆时针为正
        self.declare_parameter('sign_vx', 1.0)
        self.declare_parameter('sign_vy', 1.0)
        self.declare_parameter('sign_wz', -1.0)
        self.declare_parameter('enable_cmd_vel', True)
        self.declare_parameter('cmd_vel_topic', '/tracker/cmd_vel')
        self.declare_parameter('allow_direct_set_motor', False)
        # 诊断开关：用于严格验证 tracker -> node -> sdk 是否一致
        self.declare_parameter('cmd_passthrough_mode', False)
        self.declare_parameter('cmd_trace_log', True)

        if bool(self.get_parameter('enable_cmd_vel').value):
            cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
            self.create_subscription(Twist, cmd_vel_topic, self.cmd_vel_callback, 10)
            self.get_logger().info(f'cmd_vel subscribed topic: {cmd_vel_topic}')
        else:
            self.get_logger().info('cmd_vel subscription disabled (enable_cmd_vel=false)')

        # 记录最近一次非零指令
        self.last_nonzero_cmd = (0.0, 0.0, 0.0)
        self.last_nonzero_time = 0.0
        self.last_cmd_log_time = 0.0
        self.last_cmd = (0.0, 0.0, 0.0)
        self.last_cmd_time = 0.0
        self.last_cmd_valid = False

        self.load_servo_offsets()

        # 初始化停止
        self._send_motor_speed(0.0, 0.0, 0.0, source='init')

        self.clock = self.get_clock()
        threading.Thread(target=self.pub_callback, daemon=True).start()
        resend_period = float(self.get_parameter('cmd_resend_period_sec').value)
        if resend_period > 0.0:
            self.create_timer(resend_period, self.resend_last_cmd)
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def load_servo_offsets(self): #设置舵机
        if yaml is None:
            self.get_logger().warn('PyYAML 未安装，跳过舵机偏移量加载')
            return
        config_path = '/home/ubuntu/software/Servo_upper_computer/servo_config.yaml'
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            if not isinstance(config, dict):
                return
            for servo_id in range(1, 5):
                offset = config.get(servo_id, 0)
                try:
                    self.board.pwm_servo_set_offset(servo_id, offset)
                except Exception:
                    pass
        except Exception as e:
            self.get_logger().error(f"读取舵机配置出错: {e}")

    def get_node_state(self, request, response):
        response.success = True
        return response

    def pub_callback(self): #传感器发布循环
        while self.running: #无限循环
            if self.board is None:
                time.sleep(1)
                continue
            if getattr(self, 'enable_reception', False): #尝试获取self.enable_reception 直接写 self.enable_reception 会报错，但用 getattr 并给个默认值 False 就很安全。
                self.pub_button_data(self.button_pub)
                self.pub_joy_data(self.joy_pub)
                self.pub_imu_data(self.imu_pub)
                self.pub_sbus_data(self.sbus_pub)
                self.pub_battery_data(self.battery_pub)
                time.sleep(0.02)
            else:
                time.sleep(0.02)
        rclpy.shutdown()
   # 下面130-220行是各种控制和服务回调函数，不涉及小车运动 可以暂时不关注。
    def enable_reception(self, msg):
        self.get_logger().info('\033[1;32m%s\033[0m' % ('enable_reception ' + str(msg.data)))
        self.enable_reception = msg.data
        self.board.enable_reception(msg.data)

    def set_led_state(self, msg):
        self.board.set_led(msg.on_time, msg.off_time, msg.repeat, msg.id)

    def set_buzzer_state(self, msg):
        self.board.set_buzzer(msg.freq, msg.on_time, msg.off_time, msg.repeat)
    
    def set_rgb_states(self, msg):
        pixels = []
        for state in msg.states:
            pixels.append((state.index, state.red, state.green, state.blue))
        self.board.set_rgb(pixels)

    def set_motor_state(self, msg):
        if self.board is None:
            return
        allow_direct = bool(self.get_parameter('allow_direct_set_motor').value)
        if not allow_direct:
            self.get_logger().warn('ignore ~/set_motor because allow_direct_set_motor=false', throttle_duration_sec=1.0)
            return
        data = []
        for i in msg.data:
            data.extend([[i.id, i.rps]])
        self.board.set_motor_speed(data)

    def _send_motor_speed(self, vx, vy, wz, source='cmd_vel', input_cmd=None):
        if self.board is None:
            return

        self.board.set_motor_speed([[0, wz], [1, vx], [2, vy]])

        trace = getattr(self.board, 'last_motor_trace', None)
        payload_hex = ''
        if isinstance(trace, dict):
            payload_hex = str(trace.get('payload_hex', ''))

        record = {
            't': round(time.time(), 6),
            'source': source,
            'in': {
                'x': float(input_cmd[0]) if input_cmd is not None else None,
                'y': float(input_cmd[1]) if input_cmd is not None else None,
                'z': float(input_cmd[2]) if input_cmd is not None else None,
            },
            'out': {'vx': float(vx), 'vy': float(vy), 'wz': float(wz)},
            'payload_hex': payload_hex,
        }
        msg = String()
        msg.data = json.dumps(record, ensure_ascii=True)
        self.motor_tx_pub.publish(msg)

    def set_oled_state(self, msg):
        self.board.set_oled_text(int(msg.index), msg.text)

    def set_pwm_servo_state(self, msg):
        data = []
        for i in msg.state:
            if i.id and i.position:
                data.extend([[i.id[0], i.position[0]]])
            if i.id and i.offset:
                self.board.pwm_servo_set_offset(i.id[0], i.offset[0])
        if data != []:
            self.board.pwm_servo_set_position(msg.duration, data)

    def get_pwm_servo_state(self, msg):
        states = []
        for i in msg.cmd:
            data = PWMServoState()
            if i.get_position:
                state = self.board.pwm_servo_read_position(i.id)
                if state is not None:
                    data.position = state
            if i.get_offset:
                state = self.board.pwm_servo_read_offset(i.id)
                if state is not None:
                    data.offset = state
            states.append(data)
        return [True, states]

    def set_bus_servo_position(self, msg):
        data = []
        for i in msg.position:
            data.extend([[i.id, i.position]])
        if data:
            self.board.bus_servo_set_position(msg.duration, data)

    def set_bus_servo_state(self, msg):
        pass # 省略总线舵机具体实现以保持简洁

    def get_bus_servo_state(self, request, response):
        response.success = True
        return response

    def pub_battery_data(self, pub):
        data = self.board.get_battery()
        if data is not None:
            msg = UInt16()
            msg.data = data
            pub.publish(msg)

    def pub_button_data(self, pub): #电池电压
        data = self.board.get_button()
        if data is not None:
            key_id, key_event = data
            state_map = {
                PacketReportKeyEvents.KEY_EVENT_PRESSED: 1,
                PacketReportKeyEvents.KEY_EVENT_LONGPRESS: 2,
                PacketReportKeyEvents.KEY_EVENT_LONGPRESS_REPEAT: 3,
                PacketReportKeyEvents.KEY_EVENT_RELEASE_FROM_LP: 4,
                PacketReportKeyEvents.KEY_EVENT_RELEASE_FROM_SP: 0,
                PacketReportKeyEvents.KEY_EVENT_CLICK: 5,
                PacketReportKeyEvents.KEY_EVENT_DOUBLE_CLICK: 6,
                PacketReportKeyEvents.KEY_EVENT_TRIPLE_CLICK: 7,
            }
            state = state_map.get(key_event, -1)
            if state != -1:
                msg = ButtonState()
                msg.id = key_id
                msg.state = state
                pub.publish(msg)

    def pub_joy_data(self, pub): #手柄数据
        data = self.board.get_gamepad()
        if data is not None:
            msg = Joy()
            msg.axes = data[0]
            msg.buttons = data[1]
            msg.header.stamp = self.clock.now().to_msg()
            pub.publish(msg)

    def pub_sbus_data(self, pub): #sbus数据发布
        data = self.board.get_sbus()
        if data is not None:
            msg = Sbus()
            msg.channel = data
            msg.header.stamp = self.clock.now().to_msg()
            pub.publish(msg)

    def pub_imu_data(self, pub): #imu数据发布
        data = self.board.get_imu()
        if data is not None:
            ax, ay, az, gx, gy, gz = data
            msg = Imu()
            msg.header.frame_id = self.IMU_FRAME
            msg.header.stamp = self.clock.now().to_msg()
            msg.orientation.w = 0.0
            msg.linear_acceleration.x = ax * self.gravity
            msg.linear_acceleration.y = ay * self.gravity
            msg.linear_acceleration.z = az * self.gravity
            msg.angular_velocity.x = math.radians(gx)
            msg.angular_velocity.y = math.radians(gy)
            msg.angular_velocity.z = math.radians(gz)
            msg.orientation_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
            pub.publish(msg)

    def resend_last_cmd(self):
        if self.board is None:
            return
        if not self.last_cmd_valid:
            return
        hold_on_timeout = bool(self.get_parameter('hold_last_cmd_on_timeout').value)
        if not hold_on_timeout:
            return
        timeout_sec = float(self.get_parameter('cmd_timeout_sec').value)
        now = time.time()
        if (now - self.last_cmd_time) >= timeout_sec:
            vx, vy, wz = self.last_cmd
            self._send_motor_speed(vx, vy, wz, source='resend')

    def cmd_vel_callback(self, msg): #小车运动控制回调 重要。
        """
        根据测试结果修正的运动控制逻辑:
        - ID 1: 控制前后 (正数向前) -> 对应 msg.linear.x
        - ID 2: 控制左右 (正数向右) -> 对应 -msg.linear.y (ROS左正右负，所以取反)
        - ID 0: 控制旋转 (正数左转) -> 对应 msg.angular.z
        """
        if self.board is None:
            return

        trace_log = bool(self.get_parameter('cmd_trace_log').value)
        passthrough = bool(self.get_parameter('cmd_passthrough_mode').value)

        scale = float(self.get_parameter('vel_scale').value)
        max_v = float(self.get_parameter('max_v').value)
        max_w = float(self.get_parameter('max_w').value)
        min_v = float(self.get_parameter('min_v').value)
        min_vy = float(self.get_parameter('min_vy').value)
        min_wz = float(self.get_parameter('min_wz').value)
        min_w = float(self.get_parameter('min_w').value)
        sign_vx = float(self.get_parameter('sign_vx').value)
        sign_vy = float(self.get_parameter('sign_vy').value)
        sign_wz = float(self.get_parameter('sign_wz').value)
        deadband_y = float(self.get_parameter('cmd_deadband_y').value)
        deadband_z = float(self.get_parameter('cmd_deadband_z').value)
        x_only_mode = bool(self.get_parameter('x_only_mode').value)
        hold_sec = float(self.get_parameter('zero_cmd_hold_sec').value)
        zero_eps = float(self.get_parameter('zero_cmd_epsilon').value)
        ignore_zero = bool(self.get_parameter('ignore_zero_cmd').value)
        stop_timeout = float(self.get_parameter('zero_cmd_stop_timeout').value)
        zero_log_throttle = float(self.get_parameter('zero_cmd_log_throttle_sec').value)

        # 诊断模式：不做保持/死区/最小速度抬升，只保留必要轴映射，便于核对传输链路
        if passthrough:
            vx = msg.linear.x * sign_vx
            raw_vy = 0.0 if x_only_mode else -msg.linear.y
            vy = raw_vy * sign_vy
            raw_wz = 0.0 if x_only_mode else msg.angular.z
            wz = raw_wz * sign_wz

            if trace_log:
                self.get_logger().info(
                    f"TRACE passthrough in(x={msg.linear.x:.3f}, y={msg.linear.y:.3f}, z={msg.angular.z:.3f}) -> out(vx={vx:.3f}, vy={vy:.3f}, wz={wz:.3f})",
                    throttle_duration_sec=0.1
                )

            self._send_motor_speed(vx, vy, wz, source='passthrough', input_cmd=(msg.linear.x, msg.linear.y, msg.angular.z))
            self.last_cmd = (vx, vy, wz)
            self.last_cmd_time = time.time()
            self.last_cmd_valid = True
            return

        # 过滤短暂 0 指令：在 hold 时间内保持上一条指令
        is_zero_cmd = (abs(msg.linear.x) < zero_eps and
                       abs(msg.linear.y) < zero_eps and
                       abs(msg.angular.z) < zero_eps)
        now = time.time()
        hold_on_timeout = bool(self.get_parameter('hold_last_cmd_on_timeout').value)
        if is_zero_cmd and hold_on_timeout and self.last_cmd_valid:
            if (now - self.last_cmd_time) <= stop_timeout:
                vx, vy, wz = self.last_cmd
                self._send_motor_speed(vx, vy, wz, source='hold_last')
                return
        if not is_zero_cmd:
            self.last_nonzero_cmd = (msg.linear.x, msg.linear.y, msg.angular.z)
            self.last_nonzero_time = now
        else:
            if self.last_nonzero_time == 0.0 and ignore_zero:
                return
            if ignore_zero and (now - self.last_nonzero_time) <= stop_timeout:
                return
            if hold_sec > 0.0 and (now - self.last_nonzero_time) <= hold_sec:
                msg = Twist()
                msg.linear.x, msg.linear.y, msg.angular.z = self.last_nonzero_cmd

        # 小的非零指令容易克服不了静摩擦，给个最小速度并限幅
        def clamp(value, min_abs, max_abs):
            if abs(value) < 0.01:
                return 0.0
            mag = min(max_abs, max(abs(value), min_abs))
            return math.copysign(mag, value)

        # 1. 前进/后退 (ID 1)
        vx = clamp(msg.linear.x * scale * sign_vx, min_v, max_v)

        # 2. 横移 (ID 2) - ROS 为左正右负，这里取反后保持右正
        raw_vy = 0.0 if x_only_mode else -msg.linear.y
        if abs(raw_vy) < deadband_y:
            raw_vy = 0.0
        vy = clamp(raw_vy * scale * sign_vy, min_vy, max_v)

        # 3. 旋转 (ID 0) - 默认按实际硬件方向做一次翻转
        raw_wz = 0.0 if x_only_mode else msg.angular.z
        if abs(raw_wz) < deadband_z:
            raw_wz = 0.0
        wz = clamp(raw_wz * scale * sign_wz, min_wz, max_w)
        
        if is_zero_cmd:
            if zero_log_throttle > 0.0 and (now - self.last_cmd_log_time) < zero_log_throttle:
                pass
            else:
                self.last_cmd_log_time = now
                self.get_logger().info(f"cmd_vel hit vx={vx:.3f} vy={vy:.3f} wz={wz:.3f}")
        else:
            self.last_cmd_log_time = now
            self.get_logger().info(f"cmd_vel hit vx={vx:.3f} vy={vy:.3f} wz={wz:.3f}")

        if trace_log:
            self.get_logger().info(
                f"TRACE normal in(x={msg.linear.x:.3f}, y={msg.linear.y:.3f}, z={msg.angular.z:.3f}) -> out(vx={vx:.3f}, vy={vy:.3f}, wz={wz:.3f})",
                throttle_duration_sec=0.1
            )

        self._send_motor_speed(vx, vy, wz, source='cmd_vel', input_cmd=(msg.linear.x, msg.linear.y, msg.angular.z))
        self.last_cmd = (vx, vy, wz)
        self.last_cmd_time = now
        self.last_cmd_valid = True

def main():
    node = RosRobotController('ros_robot_controller')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.board.set_motor_speed([[0, 0], [1, 0], [2, 0]])
        node.destroy_node()
        rclpy.shutdown()
        print('shutdown')
    finally:
        print('shutdown finish')

if __name__ == '__main__':
    main()