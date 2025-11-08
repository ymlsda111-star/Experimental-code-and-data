# --------------- 新增跟踪器类 ---------------
import numpy as np
from scipy.optimize import linear_sum_assignment


class Track:
    def __init__(self, track_id, bbox, conf, cls):
        self.track_id = track_id
        self.bbox = bbox  # xyxy格式
        self.conf = conf
        self.cls = cls
        self.missed = 0  # 连续未匹配次数


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
# --------------- 跟踪器类添加结束 ---------------