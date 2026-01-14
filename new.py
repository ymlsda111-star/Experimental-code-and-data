import os
import cv2
import time
import torch
import glob
import sys
from ultralytics import YOLO

try:
    from thop import profile
except ImportError:
    print("❌ 未检测到 thop 库，请运行 'pip install thop' 安装")
    sys.exit(1)


# ==========================================
# 0. 日志记录工具类 (新增功能)
# ==========================================
class DualLogger:
    """
    双向日志记录器：同时输出到控制台和文件
    """

    def __init__(self, filepath):
        self.terminal = sys.stdout
        # 确保目录存在
        log_dir = os.path.dirname(filepath)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.log = open(filepath, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 实时写入，防止崩溃丢失

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ==========================================
# 1. 核心功能模块 (保持不变)
# ==========================================

def analyze_compatibility(cfg_path, pt_path):
    """
    分析 YAML 配置文件和 PT 权重文件是否配套
    """
    print(f"\n{'=' * 20} 正在分析文件配套性 {'=' * 20}")
    print(f"配置文件 (YAML): {cfg_path}")
    print(f"权重文件 (PT):   {pt_path}")

    if not os.path.exists(cfg_path) or not os.path.exists(pt_path):
        print("❌ 文件不存在，无法分析。")
        return False

    try:
        print("1. 尝试使用 YAML 构建模型骨架...", end=" ")
        model_from_yaml = YOLO(cfg_path)
        print("✅ 成功")

        print("2. 读取 PT 权重文件...", end=" ")
        pt_data = torch.load(pt_path, map_location='cpu')

        if isinstance(pt_data, dict):
            if 'model' in pt_data:
                state_dict = pt_data['model'].state_dict() if hasattr(pt_data['model'], 'state_dict') else pt_data[
                    'model']
            else:
                state_dict = pt_data
        else:
            state_dict = pt_data.state_dict()
        print("✅ 成功")

        print("3. 正在比对权重与模型结构...")
        model_dict = model_from_yaml.model.state_dict()

        matched_keys = 0
        mismatched_keys = []
        shape_mismatch = []

        total_keys_yaml = len(model_dict)
        total_keys_pt = len(state_dict)

        for k, v in model_dict.items():
            if k in state_dict:
                if v.shape == state_dict[k].shape:
                    matched_keys += 1
                else:
                    shape_mismatch.append(f"{k}: YAML={v.shape} vs PT={state_dict[k].shape}")
            else:
                mismatched_keys.append(k)

        print(f"   - YAML 定义层数(Keys): {total_keys_yaml}")
        print(f"   - PT 包含层数(Keys):   {total_keys_pt}")
        print(f"   - 完美匹配层数:        {matched_keys}")

        if shape_mismatch:
            print("\n❌ [严重不匹配] 形状冲突 (部分示例):")
            for msg in shape_mismatch[:5]:
                print(f"   - {msg}")
            print("   (结论: 模型结构已被修改，YAML 与 PT 不匹配)")
            return False

        elif len(mismatched_keys) > 0:
            print(f"\n⚠️ [部分不匹配] 缺失键值: {len(mismatched_keys)} 个")
            print("   (结论: 只有部分结构匹配，可能是迁移学习模型或结构有微调)")
            return False

        else:
            print("\n✅ [完美匹配] YAML 与 PT 文件结构完全一致！")
            return True

    except Exception as e:
        print(f"\n❌ 分析过程中发生错误: {e}")
        return False


def print_network_structure(model):
    """
    打印网络结构详情
    """
    print(f"\n{'=' * 20} 网络结构详情 {'=' * 20}")
    print(f"{'Idx':<5} | {'Module Type':<35} | {'Params':<10} | {'Arguments'}")
    print("-" * 80)

    py_model = model.model if hasattr(model, 'model') else model

    if hasattr(py_model, 'model') and isinstance(py_model.model, torch.nn.Sequential):
        for i, module in enumerate(py_model.model):
            module_name = str(type(module)).split("'")[1]
            params = sum(p.numel() for p in module.parameters())

            args_info = ""
            if hasattr(module, 'f'): args_info += f"from={module.f} "
            if hasattr(module, 'i'): args_info += f"idx={module.i} "

            print(f"{i:<5} | {module_name:<35} | {params:<10} | {args_info}")
    else:
        print("无法解析 Sequential 结构，打印简略信息:")
        print(py_model)


def calculate_gflops(model, input_size=(640, 640)):
    """
    计算 GFLOPS 和 参数量
    """
    print(f"\n{'=' * 20} GFLOPS 计算 {'=' * 20}")
    try:
        device = next(model.parameters()).device
        model.eval()

        dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)

        print("正在进行 thop 性能分析...")
        flops, params = profile(model.model if hasattr(model, 'model') else model,
                                inputs=(dummy_input,),
                                verbose=False)

        gflops = flops / 1e9
        params_m = params / 1e6

        print(f"输入尺寸: {input_size}")
        print(f"GFLOPS:   {gflops:.4f} G")
        print(f"Params:   {params_m:.4f} M")
        return gflops, params_m
    except Exception as e:
        print(f"GFLOPS 计算失败: {e}")
        return None, None


def calculate_fps(model, image_folder, max_images=500):
    """
    计算纯推理 FPS
    """
    print(f"\n{'=' * 20} FPS 纯推理测试 {'=' * 20}")

    device = next(model.parameters()).device
    print(f"测试设备: {device}")

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))

    if not image_files:
        print(f"❌ 文件夹中未找到图片: {image_folder}")
        return

    image_files = image_files[:max_images]
    images_buffer = []
    print(f"预读取 {len(image_files)} 张图片到内存...")

    for f in image_files:
        img = cv2.imread(f)
        if img is not None:
            images_buffer.append(img)

    if not images_buffer:
        print("❌ 图片读取失败")
        return

    print("🔥 预热模型 (Warmup 10次)...")
    warmup_img = images_buffer[0]
    for _ in range(10):
        model(warmup_img, verbose=False)

    print("🚀 开始计时...")
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start_time = time.time()

    for img in images_buffer:
        _ = model(img, verbose=False)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()

    count = len(images_buffer)
    total_time = end_time - start_time
    fps = count / total_time

    print(f"处理数量: {count} 张")
    print(f"总耗时:   {total_time:.4f} s")
    print(f"平均延迟: {total_time / count * 1000:.2f} ms")
    print(f"推理 FPS: {fps:.2f}")
    return fps


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # ---------------- 配置区域 ----------------
    # 1. 模型权重文件 (.pt)
    model_pt_path = r"F:\yolov8_lichi\runs\detect\yolov8_BK_CBAM_0.777\weights\best.pt"

    # 2. 模型配置文件 (.yaml) - 用于配套性检测
    model_yaml_path = r"F:\yolov8_lichi\ultralytics-main\ultralytics\cfg\models\v8\yolov8-CBAM9.yaml"
    #r"F:\yolov11\ultralytics-main\ultralytics\cfg\models\11\yolo11.yaml"
    #r"F:\yolov8_lichi\ultralytics-main\ultralytics\cfg\models\v8\yolov8.yaml"

    # 3. FPS测试用的图片文件夹
    img_folder = r"F:\yolov8_lichi\datasats-new\train\images"
    # ----------------------------------------

    # 自动设置日志保存路径 (保存在 weights 文件夹下)
    weights_dir = os.path.dirname(model_pt_path)
    verify_txt_path = os.path.join(weights_dir, "verify.txt")

    # 重定向 stdout 到 DualLogger
    original_stdout = sys.stdout
    logger = DualLogger(verify_txt_path)
    sys.stdout = logger

    print(f"📄 日志文件将保存至: {verify_txt_path}")
    print("🚀 程序启动...")
    print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    try:
        # 1. 检测配套性
        is_compatible = analyze_compatibility(model_yaml_path, model_pt_path)

        if is_compatible:
            print("\n✅ 将使用 YAML 配置加载模型 (最准确)...")
            try:
                model = YOLO(model_yaml_path)
                model.load(model_pt_path)
            except Exception as e:
                print(f"YAML加载失败，回退到直接加载 PT: {e}")
                model = YOLO(model_pt_path)
        else:
            print("\n⚠️ 文件不匹配或无法加载 YAML，将直接加载 PT 文件进行后续测试...")
            model = YOLO(model_pt_path)

        # 确保模型在 CUDA 上 (如果有)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        # 2. 打印网络结构
        print_network_structure(model)

        # 3. 计算 GFLOPS
        calculate_gflops(model)

        # 4. 计算 FPS
        calculate_fps(model, img_folder)

        print("\n🎉 所有任务完成!")
        print(f"结果已保存到: {verify_txt_path}")

    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
    finally:
        # 恢复 stdout 并关闭日志文件
        sys.stdout = original_stdout
        logger.close()