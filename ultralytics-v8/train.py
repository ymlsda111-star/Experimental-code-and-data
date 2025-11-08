from ultralytics import YOLO

def main():
    # 加载预训练模型（可选：'yolov8n.pt', 'yolov8s.pt' 等）
    model = YOLO(r'F:\yolov8_lichi\ultralytics-main\ultralytics\cfg\models\v8\yolov8.yaml')  # 或从头训练：YOLO('yolov8n.yaml')
    print(model)
    model.load(r'F:\yolov8_lichi\yolov8s.pt')
    # 训练模型
    results = model.train(
        data=r'F:\yolov8_lichi\new.yaml',       # 数据集配置文件路径
        epochs=300,            # 训练轮次 <1万张：40-100轮次。 >10万张：10-30轮次
        batch=16,                 # 批次大小 if报CUDA out of memory，需降低 batch 或 imgsz
        imgsz=640,                # 图像尺寸 640 * 640检测小目标但增加显存占用+高性能场景
        device='0',               # GPU（如 '0' 或 'cpu'）cpu速度可能会慢50-100倍

        # 增强数据增强强度
        # 增强数据增强强度（调整后）
        #hsv_h=0.03,  # 降低色相扰动（原0.05）：避免果实颜色偏离自然范围（如荔枝不会变成紫色）
        #hsv_s=0.5,  # 降低饱和度扰动（原0.9）：防止颜色过于鲜艳或暗淡，保留果实固有色差特征
        #hsv_v=0.4,  # 降低明度扰动（原0.6）：避免过亮/过暗导致果实纹理丢失
        #degrees=30.0,  # 降低旋转角度（原60.0）：减少果实因过度旋转导致的形态失真（荔枝多为圆形，旋转30°内特征稳定）
        #translate=0.15,  # 降低平移幅度（原0.2）：避免果实移出画面或过度截断
        #scale=0.6,  # 降低缩放幅度（原0.8）：防止果实被过度放大/缩小导致细节丢失（如小果实缩放后模糊）
        #shear=5.0,  # 降低剪切幅度（原10.0）：减少非自然形变（荔枝生长形态较规则，剪切过强易失真）
        #perspective=0.0005,  # 降低透视畸变（原0.001）：弱化画面扭曲，保留自然拍摄视角的特征
        #flipud=0.1,  # 降低上下翻转比例（原0.2）：荔枝多生长在树上，上下翻转不符合自然分布，减少此类样本占比
        #fliplr=0.5,  # 保持左右翻转（原0.5）：左右翻转不改变果实特征，且符合自然场景中果实的左右分布多样性
        #mosaic=0.6,  # 降低mosaic比例（原1）：减少多图混合导致的果实重叠干扰（尤其密集场景下）
        #mixup=0.1,  # 降低mixup比例（原0.2）：减少样本混合带来的特征混淆，优先让模型学习单一果实的特征

        name='yolov8_litch',     #保存训练结果的文件夹名称
        save=True,                 # 是否保存训练过程和结果
    )



if __name__ == '__main__':
    main()

'''
# 增强数据增强强度
        hsv_h=0.05,
        hsv_s=0.9,
        hsv_v=0.6,
        degrees=60.0,
        translate=0.2,
        scale=0.8,
        shear=10.0,
        perspective=0.001,
        flipud=0.2,
        fliplr=0.5,
        mosaic=1,
        mixup=0.2,
'''
