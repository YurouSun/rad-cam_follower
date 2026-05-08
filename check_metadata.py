
import sys
import os
import importlib.metadata

print(f"sys.path: {sys.path}")
try:
    print(f"yolov5_ros2: {importlib.metadata.distribution('yolov5_ros2')}")
except Exception as e:
    print(f"yolov5_ros2 error: {e}")

try:
    print(f"yolov5-ros2: {importlib.metadata.distribution('yolov5-ros2')}")
except Exception as e:
    print(f"yolov5-ros2 error: {e}")
