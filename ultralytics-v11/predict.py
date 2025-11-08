from ultralytics import YOLO
import cv2

# 加载训练好的模型
model = YOLO(r'F:\yolov11\runs\detect\yolov11_litch_O_0.9\weights\best.pt')

# 单张图片推理
results = model(r'G:\new', save=True, conf=0.7)

