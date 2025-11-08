# iou_tracker.py
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# ------------------- KalmanFilter 类 -------------------
class KalmanFilter:
    def __init__(self):
        # 定义8维状态向量 (x_center,y_center,w,h,vx,vy,vw,vh)
        self.kf = cv2.KalmanFilter(8, 4)

        # 状态转移矩阵 (假设匀速运动模型)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]], np.float32)

        # 观测矩阵 (只能观测位置和尺寸)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]], np.float32)

        # 初始化噪声参数
        self.kf.processNoiseCov = 1e-4 * np.eye(8, dtype=np.float32)
        self.kf.measurementNoiseCov = 1e-2 * np.eye(4, dtype=np.float32)
        self.kf.errorCovPost = 1e-1 * np.eye(8, dtype=np.float32)

    def init(self, bbox):
        """用初始检测框初始化状态"""
        x_center, y_center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        self.kf.statePost = np.array([x_center, y_center, w, h, 0, 0, 0, 0], dtype=np.float32)

    def predict(self):
        """返回预测后的bbox (xyxy格式)"""
        state = self.kf.predict()
        x_center, y_center, w, h = state[0], state[1], state[2], state[3]
        return np.array([
            x_center - w / 2,  # x_min
            y_center - h / 2,  # y_min
            x_center + w / 2,  # x_max
            y_center + h / 2],  # y_max
            dtype=np.float32)

    def update(self, bbox):
        """用新检测框更新滤波器"""
        measurement = np.array([
            (bbox[0] + bbox[2]) / 2,  # x_center
            (bbox[1] + bbox[3]) / 2,  # y_center
            bbox[2] - bbox[0],  # width
            bbox[3] - bbox[1]  # height
        ], dtype=np.float32)
        self.kf.correct(measurement)


# ------------------- Track 类 -------------------
class Track:
    def __init__(self, track_id, bbox, conf, cls):
        self.track_id = track_id
        self.bbox = bbox  # xyxy格式
        self.conf = conf
        self.cls = cls
        self.missed = 0  # 连续未匹配次数
        self.predicted_bbox = None  # 新增：存储卡尔曼预测的bbox


# ------------------- 基类 IOUTracker -------------------
class IOUTracker:
    def __init__(self, max_age=5, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    def update(self, boxes, confs, clses):
        current_detections = list(zip(boxes, confs, clses))
        # 创建IOU矩阵
        iou_matrix = np.zeros((len(self.tracks), len(current_detections)), dtype=np.float32)

        # 填充IOU矩阵
        for t, track in enumerate(self.tracks):
            for d, (det_box, _, _) in enumerate(current_detections):
                iou_matrix[t, d] = self.compute_iou(track.bbox, det_box)

        # 匈牙利算法匹配
        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_pairs = []
        for t, d in zip(*matched_indices):
            if iou_matrix[t, d] >= self.iou_threshold:
                matched_pairs.append((t, d))

        # 更新匹配的track
        for t, d in matched_pairs:
            self.tracks[t].bbox = current_detections[d][0]
            self.tracks[t].conf = current_detections[d][1]
            self.tracks[t].cls = current_detections[d][2]
            self.tracks[t].missed = 0

        # 处理未匹配的detections（新建track）
        unmatched_detections = set(range(len(current_detections))) - set(d for _, d in matched_pairs)
        for d in unmatched_detections:
            new_track = Track(self.next_id, *current_detections[d])
            self.tracks.append(new_track)
            self.next_id += 1

        # 处理未匹配的tracks（增加missed计数）
        unmatched_tracks = set(range(len(self.tracks))) - set(t for t, _ in matched_pairs)
        for t in unmatched_tracks:
            self.tracks[t].missed += 1

        # 移除超过max_age的track
        self.tracks = [t for t in self.tracks if t.missed <= self.max_age]

        # 返回当前帧有效的track（missed=0的）
        return [t for t in self.tracks if t.missed == 0]

    @staticmethod
    def compute_iou(box1, box2):
        # 计算两个xyxy格式框的IOU
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        return inter_area / (area1 + area2 - inter_area) if (area1 + area2 - inter_area) > 0 else 0.0


# ------------------- 子类 EnhancedIOUTracker -------------------
class EnhancedIOUTracker(IOUTracker):
    def __init__(self, max_age=5, iou_threshold=0.3):
        super().__init__(max_age, iou_threshold)
        self.kalman_filters = {}  # 存储每个目标的卡尔曼滤波器

    def update(self, boxes, confs, clses):
        # 步骤1: 对现有跟踪器进行卡尔曼预测
        for track in self.tracks:
            if track.track_id in self.kalman_filters:
                predicted_bbox = self.kalman_filters[track.track_id].predict()
                track.predicted_bbox = predicted_bbox  # 保存预测结果
            else:
                track.predicted_bbox = track.bbox  # 无滤波器时使用原bbox

        # 步骤2: 使用预测后的bbox计算IOU矩阵
        current_detections = list(zip(boxes, confs, clses))
        iou_matrix = np.zeros((len(self.tracks), len(current_detections)), dtype=np.float32)

        for t, track in enumerate(self.tracks):
            for d, (det_box, _, _) in enumerate(current_detections):
                iou_matrix[t, d] = self.compute_iou(track.predicted_bbox, det_box)

        # 匈牙利算法匹配
        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_pairs = []
        for t, d in zip(*matched_indices):
            if iou_matrix[t, d] >= self.iou_threshold:
                matched_pairs.append((t, d))

        # 步骤3: 处理匹配成功的跟踪器
        for t, d in matched_pairs:
            track = self.tracks[t]
            detection = current_detections[d]

            # 更新卡尔曼滤波器
            if track.track_id not in self.kalman_filters:
                self.kalman_filters[track.track_id] = KalmanFilter()
                self.kalman_filters[track.track_id].init(detection[0])
            else:
                self.kalman_filters[track.track_id].update(detection[0])

            # 更新跟踪器状态
            track.bbox = detection[0]
            track.conf = detection[1]
            track.cls = detection[2]
            track.missed = 0

        # 步骤4: 处理新检测（初始化滤波器）
        unmatched_detections = set(range(len(current_detections))) - set(d for _, d in matched_pairs)
        for d in unmatched_detections:
            new_track = Track(self.next_id, *current_detections[d])
            self.kalman_filters[new_track.track_id] = KalmanFilter()
            self.kalman_filters[new_track.track_id].init(current_detections[d][0])
            self.tracks.append(new_track)
            self.next_id += 1

        # 步骤5: 处理未匹配的跟踪器
        unmatched_tracks = set(range(len(self.tracks))) - set(t for t, _ in matched_pairs)
        for t in unmatched_tracks:
            self.tracks[t].missed += 1
            # 使用预测结果作为当前bbox
            if self.tracks[t].track_id in self.kalman_filters:
                self.tracks[t].bbox = self.tracks[t].predicted_bbox

        # 移除超过max_age的track
        self.tracks = [t for t in self.tracks if t.missed <= self.max_age]

        return [t for t in self.tracks if t.missed == 0]