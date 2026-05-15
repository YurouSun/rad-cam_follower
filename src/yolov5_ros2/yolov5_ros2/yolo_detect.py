import os
# 限制底层库线程数，留出资源给ROS调度
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, Detection2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

try:
    from interfaces.msg import ObjectInfo, ObjectsInfo
except ImportError:
    ObjectInfo = None
    ObjectsInfo = None
from std_srvs.srv import Trigger
import time
from yolov5 import YOLOv5 

try:
    import ncnn
except ImportError:
    ncnn = None

ros_distribution = os.environ.get("ROS_DISTRO")
package_share_directory = get_package_share_directory('yolov5_ros2')

# --- 颜色标签识别工具函数 (优化版) ---
def _enhance_dark_roi_for_color(roi_rgb):
    # 暗光下先拉高局部亮度，减少颜色标签被阈值误杀
    if roi_rgb is None or roi_rgb.size == 0:
        return None
    if roi_rgb.shape[0] > 64 or roi_rgb.shape[1] > 64:
        roi_rgb = cv2.resize(roi_rgb, (64, 64), interpolation=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    return cv2.merge((h, s, v))

def detect_color_tag(image_rgb, bbox):
    try:
        if image_rgb is None: return None
        x1, y1, x2, y2 = map(int, bbox)
        
        # 即使在边界内，也只取中心的一小部分
        w_box, h_box = x2 - x1, y2 - y1
        if w_box < 10 or h_box < 10: return None

        # 仅取中心 30% 区域
        cx, cy = x1 + w_box // 2, y1 + h_box // 2
        w_crop, h_crop = int(w_box * 0.3), int(h_box * 0.3)
        
        x1_c = max(0, cx - w_crop//2)
        y1_c = max(0, int(y1 + h_box * 0.10))
        x2_c = min(image_rgb.shape[1], x1_c + w_crop)
        y2_c = min(image_rgb.shape[0], int(y2 - h_box * 0.05))
        center_roi = image_rgb[y1_c:y2_c, x1_c:x2_c]
        if center_roi.size == 0: return None
        hsv = _enhance_dark_roi_for_color(center_roi)
        if hsv is None:
            return None
        
        # 阈值
        lower_red1 = np.array([0, 70, 35]); upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 35]); upper_red2 = np.array([180, 255, 255])
        lower_blue = np.array([100, 70, 35]); upper_blue = np.array([130, 255, 255])
        lower_yellow = np.array([20, 60, 35]); upper_yellow = np.array([35, 255, 255])
        
        # 红色检测
        mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
        c_red = cv2.countNonZero(mask_red)
        
        total = hsv.shape[0] * hsv.shape[1]
        thresh = max(8, total * 0.08)
        
        if c_red > thresh: return "RED"
        
        # 蓝色检测
        c_blue = cv2.countNonZero(cv2.inRange(hsv, lower_blue, upper_blue))
        if c_blue > thresh: return "BLUE"
        
        # 黄色检测
        c_yellow = cv2.countNonZero(cv2.inRange(hsv, lower_yellow, upper_yellow))
        if c_yellow > thresh: return "YELLOW"
        
        return None
    except:
        return None

class YoloV5Ros2(Node):
    def __init__(self):
        super().__init__('yolov5_ros2')
        self.get_logger().info(f"YOLOv5 Node Started on {ros_distribution}")
        
        cv2.setNumThreads(1)
        
        # Stats
        self.fps_val = 0.0
        self.last_time = time.time()
        self.last_log_time = time.time() # 记录上次打印日志的绝对时间
        self.frame_count = 0
        self.t_pre = 0.0
        self.t_infer = 0.0
        self.t_post = 0.0

        self.declare_parameter("device", "cuda", ParameterDescriptor(name="device", description="Compute device"))
        self.declare_parameter("model", "yolov5s", ParameterDescriptor(name="model", description="Model path"))
        self.declare_parameter("backend", "auto", ParameterDescriptor(name="backend", description="Backend"))
        self.declare_parameter("image_topic", "/ascamera/camera_publisher/rgb0/image", ParameterDescriptor(name="image_topic"))
        self.declare_parameter("show_result", False, ParameterDescriptor(name="show_result", description="Show GUI"))
        self.declare_parameter("pub_result_img", False, ParameterDescriptor(name="pub_result_img", description="Publish overlay"))
        self.declare_parameter("input_size", 320, ParameterDescriptor(name="input_size", description="Input size"))
        self.declare_parameter('class_names_file', os.path.join(package_share_directory, 'config', 'classes.txt'))
        self.declare_parameter('conf_thres', 0.3)

        self.create_service(Trigger, '/yolov5/start', self.start_srv_callback)
        self.create_service(Trigger, '/yolov5/stop', self.stop_srv_callback) 
        self.create_service(Trigger, '~/init_finish', self.get_node_state)

        # Config Logic
        model_param = self.get_parameter('model').value
        target_size = self.get_parameter('input_size').value
        self.net_input_size = (target_size, target_size)
        
        if os.path.isabs(model_param): model_path = model_param
        else:
            exts = ('.pt', '.tflite', '.onnx', '.param')
            if model_param.endswith(exts): model_path = os.path.join(package_share_directory, 'config', model_param)
            else: model_path = os.path.join(package_share_directory, 'config', model_param + '.pt')
        
        self.use_tflite = False
        self.use_onnx = False
        self.use_ncnn = False

        # Explicit Backend Selection
        backend = str(self.get_parameter('backend').value).lower()
        
        if backend == 'onnx':
            self.use_onnx = True
            if not model_path.endswith('.onnx'):
                candidate = model_path.rsplit('.', 1)[0] + '.onnx'
                if os.path.isfile(candidate): model_path = candidate
        elif backend == 'ncnn':
            self.use_ncnn = True
            if not model_path.endswith('.param'):
                candidate = model_path.rsplit('.', 1)[0] + '.param'
                if os.path.isfile(candidate): model_path = candidate
        elif backend == 'tflite':
            self.use_tflite = True
        
        # Auto detection
        if not any([self.use_onnx, self.use_ncnn, self.use_tflite]):
            base_path = model_path.rsplit('.', 1)[0]
            if os.path.isfile(base_path + '.param'):
                self.use_ncnn = True
                model_path = base_path + '.param'
            elif os.path.isfile(base_path + '.onnx'):
                self.use_onnx = True
                model_path = base_path + '.onnx'

        self.interpreter = None
        self.onnx_sess = None
        self.ncnn_net = None
        
        # 1. NCNN Init
        if self.use_ncnn:
            if ncnn is None: 
                self.get_logger().error("NCNN library missing")
                self.use_ncnn = False
            else:
                try:
                    self.ncnn_net = ncnn.Net()
                    self.ncnn_net.opt.use_vulkan_compute = False 
                    self.ncnn_net.opt.num_threads = 2
                    self.ncnn_net.load_param(model_path)
                    self.ncnn_net.load_model(model_path.replace('.param', '.bin'))
                    self.ncnn_input_name, self.ncnn_output_name = self.parse_ncnn_param(model_path)
                    self.get_logger().info(f"NCNN Loaded: {model_path}")
                except Exception as e: 
                    self.get_logger().error(f"NCNN Init Failed: {e}")
                    self.use_ncnn = False

        # 2. TFLite Init
        if self.use_tflite:
            try:
                try: import tflite_runtime.interpreter as tflite
                except: import tensorflow.lite as tflite
                self.interpreter = tflite.Interpreter(model_path=model_path, num_threads=4)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except: self.use_tflite = False
                
        # 3. ONNX Init
        if self.use_onnx:
            try:
                import onnxruntime as ort
                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                so.intra_op_num_threads = 2
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL 
                self.onnx_sess = ort.InferenceSession(model_path, sess_options=so, providers=['CPUExecutionProvider'])
                self.get_logger().info(f"ONNX Loaded: {model_path} (Threads=2)")
            except Exception as e:
                self.get_logger().error(f"ONNX Init Failed: {e}")
                self.use_onnx = False

        # 4. Fallback PyTorch
        if not any([self.use_ncnn, self.use_tflite, self.use_onnx]):
            self.yolov5 = YOLOv5(model_path=model_path, device=self.get_parameter('device').value)

        # Class Names
        self.class_names = getattr(self, 'class_names', None)
        if self.class_names is None: self.class_names = {0: 'object'}
        cnf = self.get_parameter('class_names_file').value
        if cnf and os.path.isfile(cnf):
            with open(cnf, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            self.class_names = {i: l for i, l in enumerate(lines)}

        # IO
        self.yolo_result_pub = self.create_publisher(Detection2DArray, "/yolo_result", 10)
        self.result_msg = Detection2DArray()
        self.object_pub = None
        if ObjectInfo is not None:
            self.object_pub = self.create_publisher(ObjectsInfo, '~/object_detect', 1)
        
        self.result_img_pub = self.create_publisher(Image, "result_img", 10)
        self.image_sub = self.create_subscription(Image, self.get_parameter('image_topic').value, self.image_callback, qos_profile_sensor_data)
        self.bridge = CvBridge()
        self.show_result = self.get_parameter('show_result').value
        self.pub_result_img = self.get_parameter('pub_result_img').value

    def parse_ncnn_param(self, param_path):
        in_name, out_name = "images", "output"
        try:
            with open(param_path, 'r') as f: lines = [l.strip() for l in f.readlines() if l.strip()]
            for line in lines:
                p = line.split()
                if len(p) >= 5 and p[0] == "Input": in_name = p[4]; break
            last = lines[-1].split()
            if len(last) >= 3:
                in_cnt = int(last[2])
                out_idx = 4 + in_cnt
                if len(last) > out_idx: out_name = last[out_idx]
        except: pass
        return in_name, out_name

    def get_node_state(self, request, response): response.success = True; return response
    def start_srv_callback(self, request, response): self.start = True; response.success = True; return response
    def stop_srv_callback(self, request, response): self.start = False; response.success = True; return response

    def image_callback(self, msg: Image):
        t0 = time.time()
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception: return
        t1 = time.time() # Pre-process start

        if self.use_ncnn: preds, names = self.ncnn_predict(image)
        elif self.use_tflite: preds, names = self.tflite_predict(image)
        elif self.use_onnx: preds, names = self.onnx_predict(image)
        else: 
            res = self.yolov5.predict(image)
            preds, names = res.pred[0], res.names
        
        t2 = time.time() # Inference Done

        self.publish_results(preds, names, image, msg.header)
        
        t3 = time.time() # All Done

        # 统计分布
        self.t_pre += (t1 - t0)
        self.t_infer += (t2 - t1)
        self.t_post += (t3 - t2)
        self.frame_count += 1

        # [关键修改] 使用真实时间差来计算FPS，而不是累加推理时间
        # 这样能反映真实的系统处理速度（包含等待时间）
        if self.frame_count % 30 == 0:
            curr_time = time.time()
            elapsed = curr_time - self.last_log_time
            real_fps = 30.0 / (elapsed + 1e-6)
            
            # 计算纯算法延迟 (Latency)
            avg_latency = (self.t_pre + self.t_infer + self.t_post) / 30.0 * 1000
            
            self.get_logger().info(f"Real FPS: {real_fps:.1f} | Latency: {avg_latency:.1f}ms | Backend: {'NCNN' if self.use_ncnn else 'ONNX'}")
            
            # 重置统计
            self.last_log_time = curr_time
            self.t_pre = 0; self.t_infer = 0; self.t_post = 0

    def publish_results(self, predictions, class_names, image_rgb, header):
        self.result_msg.detections.clear()
        self.result_msg.header = header
        
        if len(predictions) == 0: boxes, scores, categories = [], [], []
        else:
            boxes = predictions[:, :4]; scores = predictions[:, 4]; categories = predictions[:, 5]

        h_img, w_img = image_rgb.shape[:2]
        objects_info = [] if ObjectInfo is not None else None

        for index in range(len(categories)):
            cls_id = int(categories[index])
            name = class_names[cls_id] if isinstance(class_names, dict) and cls_id in class_names else str(cls_id)
            x1, y1, x2, y2 = map(int, boxes[index])
            
            # --- 颜色识别 ---
            if scores[index] > 0.5: 
                bbox = [x1, y1, x2, y2]
                color_tag = detect_color_tag(image_rgb, bbox)
                if color_tag: name = f"{name}_{color_tag}"
            
            detection2d = Detection2D()
            detection2d.id = name
            detection2d.bbox.center.position.x = float(x1 + x2) * 0.5
            detection2d.bbox.center.position.y = float(y1 + y2) * 0.5
            detection2d.bbox.size_x = float(x2 - x1); detection2d.bbox.size_y = float(y2 - y1)
            
            obj_pose = ObjectHypothesisWithPose()
            obj_pose.hypothesis.class_id = name; obj_pose.hypothesis.score = float(scores[index])
            detection2d.results.append(obj_pose)
            self.result_msg.detections.append(detection2d)

            if ObjectInfo is not None:
                object_info = ObjectInfo()
                object_info.class_name = name; object_info.box = [x1, y1, x2, y2]
                object_info.score = round(float(scores[index]), 2)
                object_info.width = w_img; object_info.height = h_img
                objects_info.append(object_info)

            # Draw (只有在需要显示时才绘制，节省时间)
            if self.show_result or self.pub_result_img:
                color = (0, 255, 0)
                if "RED" in name: color = (255, 0, 0)
                elif "BLUE" in name: color = (0, 0, 255)
                elif "YELLOW" in name: color = (255, 255, 0)
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image_rgb, name, (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if ObjectsInfo is not None and self.object_pub is not None and objects_info:
            object_msg = ObjectsInfo(); object_msg.objects = objects_info
            self.object_pub.publish(object_msg)

        # 始终发布
        self.yolo_result_pub.publish(self.result_msg)


        if self.show_result:
            self.update_fps()
            img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(img_bgr, f"FPS: {self.fps_val:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow('result', img_bgr); cv2.waitKey(1)

        if self.pub_result_img:
            result_img_msg = self.bridge.cv2_to_imgmsg(image_rgb, encoding="rgb8")
            result_img_msg.header = header
            self.result_img_pub.publish(result_img_msg)

    def update_fps(self):
        curr_time = time.time()
        self.fps_val = 1.0 / (curr_time - self.last_time + 1e-6)
        self.last_time = curr_time

    def letterbox(self, im, new_shape=(320, 320), color=(114, 114, 114)):
        shape = im.shape[:2]
        if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        r = min(r, 1.0)
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2; dh /= 2
        if shape[::-1] != new_unpad: im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color) 
        if im.shape[0] != new_shape[0] or im.shape[1] != new_shape[1]: im = cv2.resize(im, (new_shape[1], new_shape[0]))
        return im, r, (left, top)

    def nms(self, boxes, scores, iou_thres=0.45):
        x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1); h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(ovr <= iou_thres)[0]; order = order[inds + 1]
        return keep

    def ncnn_predict(self, image_rgb):
        t0 = time.time()
        if self.ncnn_net is None: return np.zeros((0,6)), {}

        ih, iw = image_rgb.shape[:2]
        net_w, net_h = self.net_input_size
        inp, r, (dw, dh) = self.letterbox(image_rgb, (net_w, net_h))
        
        try:
            inp_bytes = inp.astype(np.uint8).tobytes()
            mat_in = ncnn.Mat.from_pixels(inp_bytes, ncnn.Mat.PixelType.PIXEL_RGB, net_w, net_h)
        except Exception as e:
            self.get_logger().error(f"NCNN input conversion failed: {e}")
            return np.zeros((0,6)), self.class_names

        mat_in.substract_mean_normalize([0.0]*3, [1/255.0]*3)

        ex = self.ncnn_net.create_extractor()
        ex.input(self.ncnn_input_name, mat_in)
        
        ret, mat_out = ex.extract(self.ncnn_output_name)
        if ret != 0:
             candidates = ["output0", "output", "outputs", "predictions", "transpose_215"]
             for c in candidates:
                 if c == self.ncnn_output_name: continue
                 ret, mat_out = ex.extract(c)
                 if ret == 0: 
                     self.ncnn_output_name = c 
                     break
        
        if ret != 0: return np.zeros((0,6)), self.class_names

        output_data = np.array(mat_out)
        if output_data.ndim == 1: output_data = output_data.reshape(-1, 6)
        
        preds = output_data
        boxes = preds[:, :4]; scores = preds[:, 4]; cls_ids = preds[:, 5]
        
        conf_thres = self.get_parameter('conf_thres').value
        mask = scores > conf_thres
        boxes = boxes[mask]; scores = scores[mask]; cls_ids = cls_ids[mask]

        x1 = boxes[:, 0] - boxes[:, 2]/2; y1 = boxes[:, 1] - boxes[:, 3]/2
        x2 = boxes[:, 0] + boxes[:, 2]/2; y2 = boxes[:, 1] + boxes[:, 3]/2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        if len(boxes_xyxy) > 0:
            keep = self.nms(boxes_xyxy, scores, 0.45)
            boxes_xyxy = boxes_xyxy[keep]; scores = scores[keep]; cls_ids = cls_ids[keep]
            boxes_xyxy[:, [0, 2]] -= dw; boxes_xyxy[:, [1, 3]] -= dh
            boxes_xyxy /= r
            boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, iw)
            boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, ih)
            final_preds = np.concatenate([boxes_xyxy, scores[:, None], cls_ids[:, None]], axis=1)
        else:
            final_preds = np.zeros((0, 6))

        return final_preds, self.class_names

    def onnx_predict(self, image_rgb):
        if self.onnx_sess is None: return np.zeros((0, 6)), self.class_names
        inp, r, (dw, dh) = self.letterbox(image_rgb, self.net_input_size)
        inp = inp.transpose(2, 0, 1)
        inp = np.ascontiguousarray(inp, dtype=np.float32)
        inp /= 255.0
        inp = inp[None, ...] 
        try:
            input_name = self.onnx_sess.get_inputs()[0].name
            preds = self.onnx_sess.run(None, {input_name: inp})[0][0]
        except Exception: return np.zeros((0, 6)), self.class_names
        
        if preds.shape[0] == 0: return np.zeros((0, 6)), self.class_names
        boxes = preds[:, :4]; scores = preds[:, 4]; cls_ids = preds[:, 5]
        conf_thres = self.get_parameter('conf_thres').value
        mask = scores > conf_thres
        boxes = boxes[mask]; scores = scores[mask]; cls_ids = cls_ids[mask]
        
        if len(boxes) == 0: return np.zeros((0, 6)), self.class_names
        x = boxes[:, 0]; y = boxes[:, 1]; w = boxes[:, 2]; h = boxes[:, 3]
        x1 = x - w/2; y1 = y - h/2; x2 = x + w/2; y2 = y + h/2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        keep = self.nms(boxes_xyxy, scores, 0.45)
        boxes_xyxy = boxes_xyxy[keep]; scores = scores[keep]; cls_ids = cls_ids[keep]
        boxes_xyxy[:, [0, 2]] -= dw; boxes_xyxy[:, [1, 3]] -= dh
        boxes_xyxy /= r
        h_orig, w_orig = image_rgb.shape[:2]
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w_orig)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h_orig)
        final_preds = np.concatenate([boxes_xyxy, scores[:, None], cls_ids[:, None]], axis=1)
        return final_preds, self.class_names

    def tflite_predict(self, image): return np.zeros((0,6)), {}

def main():
    rclpy.init()
    rclpy.spin(YoloV5Ros2())
    rclpy.shutdown()

if __name__ == "__main__":
    main()