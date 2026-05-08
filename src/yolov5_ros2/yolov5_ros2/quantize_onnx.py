import os
import sys
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(input_model_path, output_model_path):
    """
    将 FP32 ONNX 模型转换为 INT8 量化模型 (动态量化)
    """
    if not os.path.exists(input_model_path):
        print(f"错误: 找不到输入模型文件: {input_model_path}")
        print("请检查路径是否正确。")
        return

    print(f"正在量化模型: {input_model_path} ...")
    print(f"输出路径: {output_model_path}")
    
    try:
        # 使用动态量化 (Dynamic Quantization)
        # 这种方式不需要校准数据集，适合 RNN/LSTM 和 Transformer，对 CNN (YOLO) 也有一定的加速效果
        # 最重要的是它非常稳定，且使用简便
        quantize_dynamic(
            model_input=input_model_path,
            model_output=output_model_path,
            weight_type=QuantType.QUInt8  # 将权重从 FP32 转为 UINT8
        )
        
        print("-" * 30)
        print("量化完成！SUCCESS")
        
        size_fp32 = os.path.getsize(input_model_path) / (1024 * 1024)
        size_int8 = os.path.getsize(output_model_path) / (1024 * 1024)
        
        print(f"原始模型大小 (FP32): {size_fp32:.2f} MB")
        print(f"量化模型大小 (INT8): {size_int8:.2f} MB")
        print(f"体积减小: {(1 - size_int8/size_fp32)*100:.1f}%")
        print("-" * 30)
        print("现在你可以使用以下命令运行新模型:")
        print(f"ros2 launch yolov5_ros2 yolov5_ros2.launch.py model:={os.path.basename(output_model_path)} backend:=onnx")
        
    except Exception as e:
        print(f"量化过程中发生错误: {e}")

if __name__ == "__main__":
    # 自动定位路径配置
    # 获取脚本当前所在目录 (src/yolov5_ros2/yolov5_ros2/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # config 目录在上一级 (src/yolov5_ros2/config/)
    base_dir = os.path.normpath(os.path.join(script_dir, "../config"))
    
    input_name = "best_simplified_320.onnx"
    output_name = "best_simplified_320_int8.onnx"
    
    input_path = os.path.join(base_dir, input_name)
    output_path = os.path.join(base_dir, output_name)
    
    # 支持命令行传参: python3 quantize_onnx.py [input_path]
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        if len(sys.argv) > 2:
            output_path = sys.argv[2]
        else:
            output_path = input_path.replace(".onnx", "_int8.onnx")

    quantize_model(input_path, output_path)