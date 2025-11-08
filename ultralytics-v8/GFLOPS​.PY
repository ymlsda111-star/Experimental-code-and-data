import torch
from thop import profile
from ultralytics import YOLO
import os


def calculate_yolov8_gflops_with_config(cfg_path, model_path, input_size=(640, 640)):
    """
    使用配置文件计算YOLOv8模型的GFLOPS

    参数:
        cfg_path: 模型配置文件路径
        model_path: 模型权重文件路径
        input_size: 输入图像尺寸 (高度, 宽度)
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        print("正在加载模型配置和权重...")
        print(f"配置文件: {cfg_path}")
        print(f"模型文件: {model_path}")

        # 方法1: 使用配置文件构建模型，然后加载权重
        model = YOLO(cfg_path)  # 先通过配置文件构建模型结构
        model.load(model_path)  # 然后加载训练好的权重

        # 获取PyTorch模型
        pytorch_model = model.model
        pytorch_model.eval()  # 设置为评估模式

        # 创建虚拟输入
        batch_size = 1
        channels = 3
        height, width = input_size
        dummy_input = torch.randn(batch_size, channels, height, width)

        # 计算FLOPs和参数数量
        print("正在计算GFLOPS和参数量...")
        flops, params = profile(pytorch_model, inputs=(dummy_input,), verbose=False)

        # 转换为GFLOPS
        gflops = flops / 1e9

        # 打印详细结果
        print(f"\n=== YOLOv8模型计算复杂度分析 ===")
        print(f"配置文件: {os.path.basename(cfg_path)}")
        print(f"模型文件: {os.path.basename(model_path)}")
        print(f"输入尺寸: {input_size[1]}x{input_size[0]}")
        print(f"FLOPs: {flops:,.0f}")
        print(f"GFLOPS: {gflops:.2f} G")
        print(f"参数量: {params:,.0f}")
        print(f"参数量: {params / 1e6:.2f} M")

        return gflops, params

    except Exception as e:
        print(f"计算过程中出现错误: {e}")
        # 尝试备选方案
        return calculate_gflops_fallback(model_path, input_size)


def calculate_gflops_fallback(model_path, input_size=(640, 640)):
    """
    备选方案：直接加载模型文件计算GFLOPS
    """
    try:
        print("尝试备选方案...")
        # 直接加载模型
        model = YOLO(model_path)
        pytorch_model = model.model
        pytorch_model.eval()

        # 创建虚拟输入
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

        # 计算FLOPs
        flops, params = profile(pytorch_model, inputs=(dummy_input,), verbose=False)
        gflops = flops / 1e9

        print(f"\n=== 备选方案计算结果 ===")
        print(f"GFLOPS: {gflops:.2f} G")
        print(f"参数量: {params / 1e6:.2f} M")

        return gflops, params

    except Exception as e:
        print(f"备选方案也失败: {e}")
        return None, None


def get_model_info(model_path):
    """
    获取模型基本信息
    """
    try:
        # 加载模型但不初始化（快速获取信息）
        model = YOLO(model_path, task='detect')
        print(f"\n=== 模型基本信息 ===")
        print(f"模型类型: {model.task}")
        print(f"模型类别数: {model.model.model[-1].nc if hasattr(model.model, 'model') else '未知'}")

        # 尝试获取输入尺寸
        if hasattr(model, 'args') and hasattr(model.args, 'imgsz'):
            print(f"训练输入尺寸: {model.args.imgsz}")

    except Exception as e:
        print(f"获取模型信息失败: {e}")


# 主程序
if __name__ == "__main__":
    # 文件路径
    cfg_path = r"F:\yolov8_lichi\ultralytics-main\ultralytics\cfg\models\v8\yolov8-AC-head.yaml"
    model_path = r"F:\yolov8_lichi\runs\detect\yolov8_ACmix_head\weights\best.pt"

    print("开始计算YOLOv8模型复杂度...")

    # 先获取模型基本信息
    get_model_info(model_path)

    # 计算GFLOPS
    gflops, params = calculate_yolov8_gflops_with_config(cfg_path, model_path)

    if gflops is not None:
        print(f"\n🎉 计算完成!")
        print(f"您的YOLOv8模型的GFLOPS为: {gflops:.2f} G")
        print(f"模型参数量为: {params / 1e6:.2f} M")

        # 与标准YOLOv8模型对比
        print(f"\n=== 与标准YOLOv8模型对比 ===")
        standard_gflops = {
            'yolov8n': 8.9,
            'yolov8s': 28.8,
            'yolov8m': 79.3,
            'yolov8l': 165.7,
            'yolov8x': 258.5
        }

        closest_model = min(standard_gflops.items(), key=lambda x: abs(x[1] - gflops))
        print(f"最接近的标准模型: {closest_model[0]} ({closest_model[1]:.1f} GFLOPS)")
        print(
            f"您的模型比{closest_model[0]} {'大' if gflops > closest_model[1] else '小'} {abs(gflops - closest_model[1]):.1f} GFLOPS")

    else:
        print("❌ 计算失败，请检查错误信息")