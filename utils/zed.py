import pyzed.sl as sl
import cv2
import numpy as np


def setup_zed(svo_path):
    """Initialize ZED camera for SVO playback"""
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_maximum_distance = 15  # 最大深度距离

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera:", err)
        return None

    depth_map = sl.Mat()
    point_cloud = sl.Mat()
    runtime_params = sl.RuntimeParameters()
    return zed, runtime_params, depth_map, point_cloud


def get_zed_frame(zed, runtime_params):
    """Get frame and depth data from ZED camera"""
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        # 获取左视图
        left_image = sl.Mat()
        zed.retrieve_image(left_image, sl.VIEW.LEFT)

        # 获取点云数据
        point_cloud = sl.Mat()
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        # 转换图像格式
        frame = left_image.get_data()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        return frame, None, point_cloud
    return None, None, None