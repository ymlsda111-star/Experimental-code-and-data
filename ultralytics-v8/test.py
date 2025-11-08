import torch
import ultralytics
import sys

print("=" * 50)
print("YOLOv11 GPU环境验证报告")
print("=" * 50)

# 1. 检查Python和基础库版本
print(f"> Python版本: {sys.version}")
print(f"> Ultralytics版本: {ultralytics.__version__}")

# 2. 检查PyTorch和CUDA
print(f"> PyTorch版本: {torch.__version__}")
print(f"> CUDA是否可用: {torch.cuda.is_available()}")
print(f"> 可用GPU数量: {torch.cuda.device_count()}")
print(f"> 当前GPU索引: {torch.cuda.current_device()}")
print(f"> GPU设备名称: {torch.cuda.get_device_name(0)}")

# 3. 检查GPU内存
if torch.cuda.is_available():
    gpu_props = torch.cuda.get_device_properties(0)
    total_mem_gb = gpu_props.total_memory / (1024 ** 3)
    print(f"> GPU显存总量: {total_mem_gb:.1f} GB")

    # 测试GPU计算功能
    x = torch.randn(3000, 3000).cuda()
    y = torch.randn(3000, 3000).cuda()
    z = torch.matmul(x, y)
    print("> GPU计算测试: 成功")

    # 清理显存
    del x, y, z
    torch.cuda.empty_cache()
    print("> 显存清理: 完成")
else:
    print("> ⚠️ 警告: 未检测到GPU支持")

# 4. 检查YOLO模型加载
try:
    from ultralytics import YOLO

    # 尝试加载一个最小模型来验证
    model = YOLO("yolov11n.pt")  # 这会自动下载预训练权重
    print("> YOLO模型加载: 成功")
except Exception as e:
    print(f"> YOLO模型加载: 失败 - {e}")

print("=" * 50)
print("环境验证完成！")
print("=" * 50)
# 成功运行 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# 成功  pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118