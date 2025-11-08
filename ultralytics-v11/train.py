
from ultralytics import YOLO

def main():
    # 加载预训练模型（可选：'yolov8n.pt', 'yolov8s.pt' 等）
    #model = YOLO(r'F:\yolov11\yolo11s.pt')
    model = YOLO(r'F:\yolov11\ultralytics-main\ultralytics\cfg\models\11\yolo11-ACmixH.yaml')  # 或从头训练：YOLO('yolov8n.yaml')
    print(model)
    model.load(r'F:\yolov11\yolo11s.pt')
    # 训练模型
    results = model.train(
        data=r'F:\yolov11\new.yaml',       # 数据集配置文件路径
        epochs=300,            # 训练轮次 <1万张：40-100轮次。 >10万张：10-30轮次
        batch=16,                 # 批次大小 if报CUDA out of memory，需降低 batch 或 imgsz
        imgsz=640,                # 图像尺寸 640 * 640检测小目标但增加显存占用+高性能场景
        device='0',               # GPU（如 '0' 或 'cpu'）cpu速度可能会慢50-100倍
        name='yolov11_litch',     #保存训练结果的文件夹名称
        save=True,                 # 是否保存训练过程和结果
    )



if __name__ == '__main__':
    main()



