import onnx, sys, os, argparse
from onnx import numpy_helper

parser = argparse.ArgumentParser(description='Inspect an ONNX model')
parser.add_argument('--model', '-m', help='Path to ONNX model',
                    default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'best_onnx.onnx')))
args = parser.parse_args()

model_path = args.model
print('using model:', model_path)
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"ONNX model not found: {model_path}")

m = onnx.load(model_path)
print('ir_version:', m.ir_version)
print('producer_name:', m.producer_name)
print('opset imports:', m.opset_import)
print('num nodes:', len(m.graph.node))
print('num inputs:', len(m.graph.input))
print('num outputs:', len(m.graph.output))

print('\\nInputs:')
for i in m.graph.input:
    name = i.name
    shape = []
    try:
        for dim in i.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            elif dim.dim_value:
                shape.append(dim.dim_value)
            else:
                shape.append('?')
    except Exception:
        shape = ['?']
    dtype = i.type.tensor_type.elem_type
    print(' ', name, shape, 'dtype:', dtype)

print('\\nOutputs:')
for o in m.graph.output:
    name = o.name
    shape = []
    try:
        for dim in o.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            elif dim.dim_value:
                shape.append(dim.dim_value)
            else:
                shape.append('?')
    except Exception:
        shape = ['?']
    dtype = o.type.tensor_type.elem_type
    print(' ', name, shape, 'dtype:', dtype)

# 检查可疑节点（简单打印前20个）
print('\\nFirst 20 nodes (op type):')
for n in m.graph.node[:20]:
    print(' ', n.op_type, '->', n.name if n.name else '')