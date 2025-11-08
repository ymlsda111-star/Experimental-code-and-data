import os
import cv2
import time
from ultralytics import YOLO
import glob


def calculate_batch_fps(model_path, image_folder, batch_size=1):
    """
    通过批量处理图像计算YOLOv8模型的FPS

    参数:
        model_path: 模型文件路径
        image_folder: 包含测试图像的文件夹路径
        batch_size: 批量大小（1为逐帧处理）
    """
    # 加载YOLOv8模型
    model = YOLO(model_path)

    # 获取图像文件列表
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))

    if not image_files:
        print(f"在文件夹 {image_folder} 中未找到图像文件")
        return

    print(f"找到 {len(image_files)} 张测试图像")

    # 预热模型（先运行几次避免初始化时间影响）
    print("预热模型...")
    warmup_image = cv2.imread(image_files[0])
    for _ in range(10):
        _ = model(warmup_image)

    # 开始正式测试
    frame_count = 0
    start_time = time.time()

    print("开始批量FPS计算...")

    # 批量处理图像
    for i, image_path in enumerate(image_files):
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            continue

        # 使用模型进行推理
        results = model(image)

        frame_count += 1

        # 每处理10张图像打印一次进度
        if frame_count % 10 == 0:
            current_time = time.time() - start_time
            current_fps = frame_count / current_time
            print(f"已处理 {frame_count}/{len(image_files)} 张图像, 当前FPS: {current_fps:.2f}")

    # 计算最终结果
    total_time = time.time() - start_time
    fps = frame_count / total_time if total_time > 0 else 0

    print(f"\n=== 批量FPS统计 ===")
    print(f"总图像数: {frame_count}")
    print(f"总时间: {total_time:.2f}秒")
    print(f"平均FPS: {fps:.2f}")
    print(f"平均每张图像推理时间: {total_time / frame_count * 1000:.2f}ms")

    return fps


# 使用方法
if __name__ == "__main__":
    model_path = r"F:\yolov8_lichi\runs\detect\yolov8_ACmix_head\weights\best.pt" # 替换为您的模型路径
    image_folder = r"F:\yolov8_lichi\datasats-new\train\images"  # 替换为您的测试图像文件夹路径
    calculate_batch_fps(model_path, image_folder)