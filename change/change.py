import numpy as np


def uv2xyz(uv, K, depth):
    '''
    Args:
        uv: pixel coordinates shape (n, 2)
        K: camera instrincs, shape (3, 3)
        depth: depth values of uv, shape (n, 1)
    Returns: point cloud coordinates xyz, shape (n, 3)
    '''
    assert depth.ndim == 2, f'depth shape should be (n, 1) instead of {depth.shape}'
    assert uv.ndim == 2, f'uv shape should be (n, 2) instead of {uv.shape}'

    # Another form
    u = uv[:, 0]
    v = uv[:, 1]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) / fx
    y = (v - cy) / fy
    xyz = np.hstack((x.reshape(-1, 1) * depth, y.reshape(-1, 1) * depth, depth))

    # xy = cv2.undistort(np.float32(uv), np.float32(K), distCoeffs=np.zeros(5))
    # xyz = np.hstack((xy * depth, depth))
    return xyz


if __name__ == '__main__':
    ########### 造数据 #######
    uv = np.array([[100, 200]])  # 像素坐标点 (100, 200)
    depth = np.array([[0.5]])  # 像素坐标点 (100, 200) 对应的深度 0.5(一般单位为m)
    fx, fy, cx, cy = 540, 540, 320, 240  # 相机内参
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    ###########################
    xyz = uv2xyz(uv, K, depth)