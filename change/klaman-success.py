import numpy as np
from pykalman import KalmanFilter

# 已知的前5次迭代的位置信息
'''positions = np.array([
    [0.342, 0.148, 0.63],
    [0.348, 0.148, 0.63],
    [0.349, 0.148, 0.65],
    [0.349, 0.148, 0.64],
    [0.348, 0.148, 0.64]
])'''
positions = np.array([
    [0.35679, 0.20399, 0.57],
    [0.35719, 0.20399, 0.56],
    [0.35957, 0.20960, 0.58],
    [0.35533, 0.20373, 0.57],
    [0.35600, 0.20386, 0.56]

])

# 初始化卡尔曼滤波器
kf = KalmanFilter(initial_state_mean=positions[-1], n_dim_obs=3)

# 使用已知的数据拟合卡尔曼滤波器
kf = kf.em(positions, n_iter=5)

# 预测后面的运动状态
n_steps = 5  # 预测的步数
(filtered_state_means, filtered_state_covariances) = kf.filter(positions)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(positions)

print("Filtered state means:")
print(filtered_state_means[-n_steps:])

print("Smoothed state means:")
print(smoothed_state_means[-n_steps:])
