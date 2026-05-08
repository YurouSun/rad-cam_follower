import onnxruntime as ort
import cv2, numpy as np, os, sys, argparse

script_dir = os.path.dirname(__file__)
default_model = os.path.abspath(os.path.join(script_dir, '..', 'config', 'best_onnx.onnx'))
default_img = os.path.abspath(os.path.join(script_dir, '..', 'resource', 'fish.jpg'))

parser = argparse.ArgumentParser(description='Run ONNX inference on a single image')
parser.add_argument('--model', '-m', default=default_model, help='Path to ONNX model')
parser.add_argument('--image', '-i', default=default_img, help='Path to test image')
args = parser.parse_args()

MODEL = args.model
IMG = args.image

# Preprocess (letterbox to 640x640 + RGB + normalize)
def letterbox(im, new_shape=(640,640), color=(114,114,114)):
    h,w = im.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w/w, new_h/h)
    rw, rh = int(round(w*r)), int(round(h*r))
    img_resized = cv2.resize(im, (rw, rh), interpolation=cv2.INTER_LINEAR)
    dw, dh = new_w - rw, new_h - rh
    top, bottom = dh//2, dh-dh//2
    left, right = dw//2, dw-dw//2
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, (left, top)

sess = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
print('provider:', sess.get_providers())
inp_meta = sess.get_inputs()[0]
out_meta = sess.get_outputs()[0]
print('input meta:', inp_meta.name, inp_meta.shape, inp_meta.type)
print('output meta:', out_meta.name, out_meta.shape, out_meta.type)

img = cv2.imread(IMG)
if img is None:
    print('cannot read image', IMG); sys.exit(1)
# convert BGR->RGB (if your pipeline uses RGB)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

net_h, net_w = 640, 640  # set per your model
inp, r, (dw, dh) = letterbox(img, (net_w, net_h))
# If model expects NCHW, transpose; if NHWC, leave as is. Check inp_meta.shape
if len(inp_meta.shape) == 4:
    # common: [1,3,640,640] -> NCHW
    if inp_meta.shape[1] == 3:
        # convert HWC->CHW
        inp_arr = inp.astype(np.float32) / 255.0
        inp_arr = np.transpose(inp_arr, (2,0,1))[None, ...]
    else:
        # NHWC [1,640,640,3]
        inp_arr = (inp.astype(np.float32) / 255.0)[None, ...]
else:
    inp_arr = (inp.astype(np.float32) / 255.0)[None, ...]

# Run inference
import time
outputs = sess.run(None, {inp_meta.name: inp_arr})
print('num outputs:', len(outputs))
t0 = time.time()
out = outputs[0]
t1 = time.time()
print('out shape', out.shape)
print(f'inference time: {(t1-t0)*1000:.1f} ms')
# postprocess like node: preds -> xywh or xyxy
preds = out[0]

def nms(boxes, scores, iou_thres=0.45):
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        x1 = np.maximum(boxes[i,0], boxes[rest,0])
        y1 = np.maximum(boxes[i,1], boxes[rest,1])
        x2 = np.minimum(boxes[i,2], boxes[rest,2])
        y2 = np.minimum(boxes[i,3], boxes[rest,3])
        inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
        area1 = (boxes[i,2]-boxes[i,0]) * (boxes[i,3]-boxes[i,1])
        area2 = (boxes[rest,2]-boxes[rest,0]) * (boxes[rest,3]-boxes[rest,1])
        union = area1 + area2 - inter + 1e-6
        ious = inter / union
        idxs = rest[ious <= iou_thres]
    return keep

# Interpret preds
if preds.ndim == 2 and preds.shape[1] >= 5:
    if preds.shape[1] == 5:
        boxes_xywh = preds[:, :4]
        scores = preds[:, 4]
        cls_ids = np.zeros((preds.shape[0],), dtype=np.int32)
    elif preds.shape[1] == 6:
        boxes_xywh = preds[:, :4]
        scores = preds[:, 4]
        cls_ids = preds[:, 5].astype(np.int32)
    else:
        boxes_xywh = preds[:, :4]
        scores = preds[:, 4]
        cls_scores = preds[:, 5:]
        cls_ids = np.argmax(cls_scores, axis=1)

mask = scores > 0.1
boxes_xywh = boxes_xywh[mask]
scores = scores[mask]
cls_ids = cls_ids[mask]

# xywh -> xyxy
boxes_xyxy = np.zeros_like(boxes_xywh)
boxes_xyxy[:,0] = boxes_xywh[:,0] - boxes_xywh[:,2]/2
boxes_xyxy[:,1] = boxes_xywh[:,1] - boxes_xywh[:,3]/2
boxes_xyxy[:,2] = boxes_xywh[:,0] + boxes_xywh[:,2]/2
boxes_xyxy[:,3] = boxes_xywh[:,1] + boxes_xywh[:,3]/2

if boxes_xyxy.shape[0] > 0:
    keep = nms(boxes_xyxy, scores, iou_thres=0.45)
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    cls_ids = cls_ids[keep]

print('Final detections (x1,y1,x2,y2,score,class):')
for i in range(len(scores)):
    print(f" {boxes_xyxy[i].tolist()}, {float(scores[i]):.3f}, {int(cls_ids[i])}")