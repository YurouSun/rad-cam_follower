#!/bin/bash

# Source ROS2环境和当前工作空间
source /opt/ros/humble/setup.bash
source /home/ubuntu/ros2_ws/install/local_setup.bash

# 定义输出目录名（时间戳格式）
BAG_DIRECTORY_NAME="bag_$(date +%Y%m%d_%H%M%S)"

# 定义要录制的所有话题列表

TOPICS=(
"/ascamera/camera_publisher/rgb0/image"
"/ascamera/camera_publisher/depth0/image_raw"
"/ti_mmwave/radar_scan_pcl"
"/car3/cmd_vel"
"/controller/motor_tx"
"/car3/ros_robot_controller/imu_raw"
"/odom"
"/tf"
"/tf_static"
"/tracked_objects_3d"
)

echo "========================================="
echo "  端到端自动驾驶模型 - 本地数据采集启动  "
echo "========================================="
echo "将要保存的完整路径: $(pwd)/${BAG_DIRECTORY_NAME}"
echo "设置单个包大小限制: 3GB (超过会自动分包)"
echo "-----------------------------------------"

echo "-----------------------------------------"

# ================= 话题检测已禁用 =================

echo ">>> 跳过话题检测，直接开始录制 <<<" 

# =========================================================

echo ">>> 注意观察下方是否有 'Dropped messages' 掉帧警告 <<<"
echo ">>> 按 Ctrl+C 即可安全停止录制 <<<"
echo "========================================="

# 执行 ros2 bag record 命令，保存在当前工作区
# 可选：如果环境变量 DEFER_DRIVER_CMD 被设置，先在后台启动录制，
# 然后执行该驱动命令（适用于发布者为“懒发布”场景）。

if [ -n "${DEFER_DRIVER_CMD}" ]; then
	echo ">>> DEFER_DRIVER_CMD 已设置，录制将后台运行并随后执行驱动命令。"
	echo "    驱动命令: ${DEFER_DRIVER_CMD}"

	# 启动录制（后台）
	ros2 bag record -o ${BAG_DIRECTORY_NAME} -b 3000000000 ${TOPICS[@]} &
	BAG_PID=$!

	# 捕获 Ctrl+C，确保后台进程被清理
	trap "echo '停止录制并清理...'; kill -INT ${BAG_PID} 2>/dev/null; kill 0; exit 0" INT

	# 等待一点时间再启动驱动（让录制订阅器就绪）
	sleep 1
	# 启动驱动命令（在子shell中运行，便于用户使用复合命令）
	bash -c "${DEFER_DRIVER_CMD}" &
	DRIVER_PID=$!

	# 等待 bag 进程结束（用户按 Ctrl+C 时会触发 trap）
	wait ${BAG_PID}

	echo ""
	echo "录制已停止！本次采集的数据已安全保存至: $(pwd)/${BAG_DIRECTORY_NAME}"
else
	ros2 bag record -o ${BAG_DIRECTORY_NAME} -b 3000000000 ${TOPICS[@]}
	echo ""
	echo "录制已停止！本次采集的数据已安全保存至: $(pwd)/${BAG_DIRECTORY_NAME}"
fi