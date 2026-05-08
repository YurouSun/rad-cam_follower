import rclpy
from rclpy.node import Node
from ti_mmwave_rospkg_msgs.msg import RadarTrackArray
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
from interfaces.msg import ObjectsInfo
import math
import struct
import time

class MmwaveTrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_node')
        self.get_logger().info('毫米波雷达跟踪节点已启动 (V3.0 重构版)')

        # 订阅
        # 保留毫米波原始话题用于可视化/调试，但不再直接用来控制运动
        self.create_subscription(RadarTrackArray, '/ti_mmwave/radar_track_array', self.radar_callback, 10)
        self.create_subscription(PointCloud2, '/ti_mmwave/radar_scan_pcl', self.pcl_callback, 10)
        # 新增：订阅融合后的多目标跟踪结果，作为控制依据
        self.create_subscription(ObjectsInfo, '/tracked_objects_3d', self.fusion_callback, 10)
        self.publisher_ = self.create_publisher(Twist, '/controller/cmd_vel', 10)
        self.cluster_marker_pub = self.create_publisher(MarkerArray, '/mmwave_tracker/cluster_markers', 10)

        # --- 核心控制参数 ---
        # 转向/横向控制参数（整体偏保守，减少乱转）
        self.kp_angular = 1.2        # 将横向速度映射到角速度的增益
        self.min_angular_speed = 0.1
        self.max_angular_velocity = 1.0
        self.angular_from_lateral_gain = 0.8

        # 前进方向控制参数 (X 方向)
        self.kp_linear = 1.0
        self.min_linear_speed = 0.0
        # 再降低最大线速度，先保证稳定性
        self.max_linear_velocity = 0.12
        self.linear_deadband = 0.05

        # 目前先不根据左右偏差做侧移控制，避免乱动
        # 如需开启侧移，可重新设置 kp_lateral / max_lateral_speed
        self.kp_lateral = 0.0
        self.max_lateral_speed = 0.0
        self.lateral_deadband = 0.02
        
        # 距离阈值 / 前向速度策略
        # 0.2m 以内停车，0.2~0.8m 线性加速，>=0.8m 匀速前进，不后退
        self.stop_distance = 0.2       # <= 0.2m 停止
        self.full_speed_distance = 0.8 # >= 0.8m 视为全速区间
        self.target_distance = 0.8     # 保持距离（与 full_speed_distance 对齐，便于理解）

        # 平滑参数
        self.alpha = 0.15            # 速度低通滤波
        # 记录上一次发布的速度分量（麦氏轮底盘：vx, vy, wz）
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_wz = 0.0

        # 融合距离滤波（用于抑制前后抖动）
        self.forward_dist_filtered = 0.0
        # 略加大平滑，减小前后忽快忽慢
        self.forward_filter_alpha = 0.2

        # 目标坐标滤波
        self.target_x_filtered = 0.0
        self.target_y_filtered = 0.0
        self.coord_filter_alpha = 0.12 # 目标坐标低通滤波

        # 点云聚类参数
        self.min_cluster_points = 4      # 降低聚类门槛，远距离点云稀疏也能识别
        self.cluster_distance_threshold = 0.35 # 稍微放宽聚类距离

        # 状态管理
        self.last_track_time = 0.0
        self.track_timeout = 1.0     # 丢失目标 1.0秒 后才认为丢失 (防止闪烁)
        self.is_searching = False
        # 搜索模式：丢失目标后原地转圈的角速度
        self.search_angular_speed = 0.6  # rad/s，可根据实际再调
        # 当前版本：不再区分具体 ID，只要存在任意轨迹就跟随；
        # 当一条有效轨迹都没有时，进入搜索模式。

    def radar_callback(self, msg: RadarTrackArray):
        """
        已不再使用毫米波自带 track 数据进行控制，这里预留做调试/统计用。
        """
        return

    def pcl_callback(self, msg: PointCloud2):
        """
        仅用于点云聚类可视化，不再用于控制逻辑。
        """
        now = self.get_clock().now().nanoseconds / 1e9
        
        # 如果最近刚收到过跟踪数据，忽略点云，避免冲突
        if now - self.last_track_time < 0.2:
            return

        # 解析点云
        points = []
        point_step = msg.point_step
        num_points = msg.width
        for i in range(num_points):
            offset = i * point_step
            x, y, z = struct.unpack_from('<fff', msg.data, offset)
            
            # 过滤范围优化:
            # 1. Y (前方): 0.05m ~ 6.0m (扩大视野)
            # 2. X (左右): -1.5m ~ 1.5m (稍微放宽左右视野)
            if 0.05 < y < 6.0 and -1.5 < x < 1.5:
                points.append((x, y))

        if len(points) < self.min_cluster_points:
            if now - self.last_track_time > self.track_timeout:
                if not self.is_searching:
                    self.get_logger().info(f'有效点云不足({len(points)} < {self.min_cluster_points})，进入原地搜索模式。')
                    self.start_search()
            # 也要清空聚类可视化
            self.publish_cluster_markers([], msg.header.frame_id)
            return

        clusters = self.cluster_points(points)
        valid_clusters = [cluster for cluster in clusters if len(cluster) >= self.min_cluster_points]

        if not valid_clusters:
            if now - self.last_track_time > self.track_timeout and not self.is_searching:
                self.get_logger().info('未找到符合条件的聚类，进入原地搜索模式。')
                self.start_search()
            self.publish_cluster_markers([], msg.header.frame_id)
            return

        # 发布聚类可视化供 RViz 查看
        self.publish_cluster_markers(valid_clusters, msg.header.frame_id)

        # 选择最近的有效聚类作为目标
        target_cluster = min(valid_clusters, key=self._cluster_average_distance)
        avg_x, avg_y = self._cluster_centroid(target_cluster)

        # 找到了点云目标，这里仅更新搜索状态和可视化，不再驱动底盘
        self.is_searching = False

    def fusion_callback(self, msg: ObjectsInfo):
        """使用 radar_camera_fusion 发布的 /tracked_objects_3d 作为控制依据"""
        now = self.get_clock().now().nanoseconds / 1e9

        if not msg.objects:
            # 长时间没有目标则停车
            if now - self.last_track_time > self.track_timeout and not self.is_searching:
                self.get_logger().info('融合结果中无目标，进入原地搜索模式。')
                self.start_search()
            return

        # 从 ObjectsInfo 中选择一个目标：不关心具体 ID，
        # 只要有任意带有效距离的信息，就选择距离最近的那个。
        best_obj = None
        best_dist = None

        for obj in msg.objects:
            cls_name = obj.class_name or ''
            dist = None
            # 解析 class_name 中编码的距离后缀: 形如 "person_Y1.23m"
            if '_Y' in cls_name:
                try:
                    base, y_part = cls_name.rsplit('_Y', 1)
                    if y_part.endswith('m'):
                        y_part = y_part[:-1]
                    dist_val = float(y_part)
                    dist = dist_val
                except Exception:
                    dist = None

            if dist is None or dist <= 0.0:
                continue

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_obj = obj

        if best_obj is None or best_dist is None:
            # 有对象但都没有有效距离，也视为“未识别成功目标”
            if now - self.last_track_time > self.track_timeout and not self.is_searching:
                self.get_logger().info('融合结果中存在对象但无有效距离，进入原地搜索模式。')
                self.start_search()
            return

        # 由融合节点给出的前向距离（Y 轴前方），做一次低通滤波，减少抖动
        raw_forward_dist = best_dist

        if self.forward_dist_filtered == 0.0:
            self.forward_dist_filtered = raw_forward_dist
        else:
            self.forward_dist_filtered = (
                self.forward_filter_alpha * raw_forward_dist
                + (1.0 - self.forward_filter_alpha) * self.forward_dist_filtered
            )

        forward_dist = self.forward_dist_filtered

        # 根据 bbox 像素位置 + 图像宽度估计左右偏移
        # ObjectInfo.box: [x1,y1,x2,y2]; width 为整幅图像宽度
        use_x = 0.0
        if len(best_obj.box) >= 4 and best_obj.width > 0:
            x1, y1, x2, y2 = best_obj.box[:4]
            img_w = float(best_obj.width)
            center_x = (float(x1) + float(x2)) / 2.0
            img_center_x = img_w / 2.0
            # 归一化像素偏移 [-1,1]，>0 表示目标在图像右侧
            offset_norm = (center_x - img_center_x) / max(img_center_x, 1.0)

            # 将归一化偏移粗略映射为雷达坐标系下的 X（右为正），比例系数可根据实际再微调
            lateral_scale = 0.3  # 稍微减小横向估计，避免过度转向
            use_x = offset_norm * forward_dist * lateral_scale

        # 与 radar_camera_fusion 约定：forward_dist 即为雷达 Y 轴（前方）
        use_y = forward_dist

        self.last_track_time = now
        self.is_searching = False
        self.process_target(use_x, use_y)

    def cluster_points(self, points):
        clusters = []
        visited = [False] * len(points)

        for idx in range(len(points)):
            if visited[idx]:
                continue

            queue = [idx]
            visited[idx] = True
            cluster = []

            while queue:
                current = queue.pop()
                cluster.append(points[current])

                for neighbor in range(len(points)):
                    if visited[neighbor]:
                        continue
                    if self._point_distance(points[current], points[neighbor]) <= self.cluster_distance_threshold:
                        visited[neighbor] = True
                        queue.append(neighbor)

            clusters.append(cluster)

        return clusters

    @staticmethod
    def _cluster_centroid(cluster):
        avg_x = sum(p[0] for p in cluster) / len(cluster)
        avg_y = sum(p[1] for p in cluster) / len(cluster)
        return avg_x, avg_y

    @staticmethod
    def _cluster_average_distance(cluster):
        return sum(p[1] for p in cluster) / len(cluster)

    @staticmethod
    def _point_distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def publish_cluster_markers(self, clusters, frame_id):
        if self.cluster_marker_pub is None:
            return

        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        if not clusters:
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = 'mmwave_clusters'
            marker.id = 0
            marker.action = Marker.DELETEALL
            marker_array.markers.append(marker)
            self.cluster_marker_pub.publish(marker_array)
            return

        for idx, cluster in enumerate(clusters):
            cx, cy = self._cluster_centroid(cluster)
            size = 0.12 + min(len(cluster) * 0.01, 0.2)

            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = 'mmwave_clusters'
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = cx
            marker.pose.position.y = cy
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = size
            marker.scale.y = size
            marker.scale.z = size

            r, g, b = self._color_from_index(idx)
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 0.85

            marker.lifetime = Duration(sec=0, nanosec=200_000_000)

            marker_array.markers.append(marker)

        self.cluster_marker_pub.publish(marker_array)

    @staticmethod
    def _color_from_index(index):
        palette = [
            (0.93, 0.26, 0.21),  # red
            (0.26, 0.56, 0.98),  # blue
            (0.30, 0.85, 0.39),  # green
            (0.98, 0.74, 0.18)   # amber
        ]
        return palette[index % len(palette)]

    def process_target(self, x, y):
        # --- 0. 坐标滤波 (关键：消除雷达噪点导致的抽搐) ---
        if self.target_x_filtered == 0.0 and self.target_y_filtered == 0.0:
            self.target_x_filtered = x
            self.target_y_filtered = y
        else:
            self.target_x_filtered = self.coord_filter_alpha * x + (1 - self.coord_filter_alpha) * self.target_x_filtered
            self.target_y_filtered = self.coord_filter_alpha * y + (1 - self.coord_filter_alpha) * self.target_y_filtered
        
        # 使用滤波后的坐标进行计算
        use_x = self.target_x_filtered
        use_y = self.target_y_filtered

        # --- 1. 坐标转换 ---
        # 雷达坐标: X=右, Y=前
        # 机器人坐标: X=前, Y=左
        # 目标在机器人坐标系下的位置:
        target_dx = use_y              # 前向距离
        target_dy = -use_x             # 横向距离 (左为正)

        # --- 2. 基于 X/Y 的分量控制 ---
        distance = math.sqrt(use_x**2 + use_y**2)

        # 2.a 前向速度：由前向距离直接映射，不后退
        #   <= 0.2m: vx = 0
        #   0.2~0.8m: 在 [0, max_linear_velocity] 之间线性增加
        #   >= 0.8m: vx = max_linear_velocity
        distance_forward = max(target_dx, 0.0)

        if distance_forward <= self.stop_distance:
            vx = 0.0
        elif distance_forward >= self.full_speed_distance:
            vx = self.max_linear_velocity
        else:
            ratio = (distance_forward - self.stop_distance) / (self.full_speed_distance - self.stop_distance)
            vx = self.max_linear_velocity * ratio

        # 保护性限幅（确保不为负，也不超过上限）
        vx = max(0.0, min(vx, self.max_linear_velocity))

        # 2.b 先禁用侧移控制，减少横向乱动
        lateral_error = target_dy
        vy = 0.0

        # --- 3. 指令生成（麦氏轮：直接使用 vx、vy，不再依赖角速度转向） ---
        # 这里不再根据偏差计算角速度，默认只用平移去跟随目标
        cmd_vx = vx
        cmd_vy = vy
        cmd_wz = 0.0

        # --- 4. 调试日志 ---
        self.get_logger().info(
            f'Tgt:({x:.2f}, {y:.2f}) Dist:{distance:.2f} vx:{cmd_vx:.2f} vy:{cmd_vy:.2f} wz:{cmd_wz:.2f}',
            throttle_duration_sec=0.2
        )

        self.smooth_and_publish(cmd_vx, cmd_vy, cmd_wz)

    def smooth_and_publish(self, target_vx, target_vy, target_wz):
        # 低通滤波平滑三个分量
        vx = self.alpha * target_vx + (1 - self.alpha) * self.last_vx
        vy = self.alpha * target_vy + (1 - self.alpha) * self.last_vy
        wz = self.alpha * target_wz + (1 - self.alpha) * self.last_wz

        # 绝对禁止后退：前向速度再做一次非负夹紧
        if vx < 0.0:
            self.get_logger().warn(
                f"检测到负线速度 {vx:.3f}，已强制置为 0，禁止后退。",
                throttle_duration_sec=0.5
            )
            vx = 0.0

        # 更新历史
        self.last_vx = vx
        self.last_vy = vy
        self.last_wz = wz

        # 发布（麦氏轮底盘支持 x/y 两个线速度分量）
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.angular.z = wz
        self.publisher_.publish(msg)

    def start_search(self):
        """丢失目标后进入原地转圈搜索模式。"""
        if self.is_searching:
            return

        # 进入搜索状态，清理目标滤波，避免旧目标残留影响
        self.is_searching = True
        self.target_x_filtered = 0.0
        self.target_y_filtered = 0.0
        self.forward_dist_filtered = 0.0

        # 原地转圈：vx、vy 为 0，只给一个恒定角速度
        self.smooth_and_publish(0.0, 0.0, self.search_angular_speed)

    def stop_robot(self):
        # 立即停止：清零历史速度并直接发布 0 速度指令
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_wz = 0.0

        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.angular.z = 0.0
        self.publisher_.publish(msg)

        # 重置滤波器状态，以便下次检测到目标时能快速响应
        self.target_x_filtered = 0.0
        self.target_y_filtered = 0.0
        self.forward_dist_filtered = 0.0

def main(args=None):
    rclpy.init(args=args)
    node = MmwaveTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_robot()
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
