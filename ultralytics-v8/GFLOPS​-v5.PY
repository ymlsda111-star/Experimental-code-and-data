import torch
import torch.nn as nn
from ultralytics import YOLO


def manual_flops_calculation(yolov5_model_path, input_size=(640, 640)):
    """
    手动计算FLOPs（最可靠的方法）
    """
    try:
        print("使用手动方法计算FLOPs...")

        # 加载模型
        model = YOLO(yolov5_model_path)
        pytorch_model = model.model
        pytorch_model.eval()

        # 手动计算FLOPs
        total_flops = 0

        def conv_flops_hook(module, input, output):
            # 卷积层FLOPs计算
            batch_size, output_channels, output_height, output_width = output.shape
            kernel_height, kernel_width = module.kernel_size
            input_channels = module.in_channels / module.groups

            # 每个输出元素的FLOPs: kernel_h * kernel_w * in_channels
            flops_per_element = kernel_height * kernel_width * input_channels
            # 总FLOPs: flops_per_element * output_h * output_w * out_channels * batch_size
            flops = flops_per_element * output_height * output_width * output_channels * batch_size
            # 考虑乘加操作（MACs到FLOPs的转换）
            flops *= 2  # 因为每个MAC包含一个乘法和一个加法

            nonlocal total_flops
            total_flops += flops

        # 注册钩子
        hooks = []
        for name, module in pytorch_model.named_modules():
            if isinstance(module, nn.Conv2d):
                hook = module.register_forward_hook(conv_flops_hook)
                hooks.append(hook)

        # 前向传播一次以计算FLOPs
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
        with torch.no_grad():
            _ = pytorch_model(dummy_input)

        # 移除钩子
        for hook in hooks:
            hook.remove()

        gflops = total_flops / 1e9

        # 计算参数量
        total_params = sum(p.numel() for p in pytorch_model.parameters())

        print(f"\n=== 手动FLOPs计算结果 ===")
        print(f"总FLOPs: {total_flops:,}")
        print(f"GFLOPS: {gflops:.2f} G")
        print(f"参数量: {total_params:,}")
        print(f"参数量: {total_params / 1e6:.2f} M")

        return gflops, total_params

    except Exception as e:
        print(f"手动计算失败: {e}")
        return None, None


# 使用示例
yolov5_model_path = r"G:\runs\exp3-ACmix\weights\best.pt"
manual_flops_calculation(yolov5_model_path)