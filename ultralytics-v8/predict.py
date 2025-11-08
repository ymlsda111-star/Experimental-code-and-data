from ultralytics import YOLO
import cv2

# 加载训练好的模型
model = YOLO('runs/detect/yolov8_flower4/weights/best.pt')

# 单张图片推理
results = model('test_image.jpg', save=True, conf=0.5)

# 视频流推理
# cap = cv2.VideoCapture(0)  # 0表示默认摄像头
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     results = model(frame, stream=True)
#
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             # 绘制边界框
#             b = box.xyxy[0]
#             cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
#             # 添加标签和置信度
#             cv2.putText(frame, f"{model.names[int(box.cls)]} {box.conf:.2f}",
#                         (int(b[0]), int(b[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#
#     cv2.imshow('YOLOv8 Detection', frame)
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()