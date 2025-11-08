import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

# 给定的10个点的坐标
points = np.array([
    [0.34832, 0.14790, 0.62],
    [0.34846, 0.14803, 0.64],
    [0.34700, 0.14803, 0.62],
    [0.34872, 0.14817, 0.63],
    [0.34740, 0.14790, 0.62]
])

points1 = np.array([
    [0.343945,0.1341264,0.62088],
    [0.34118,0.13586,0.61232],
    [0.32728,0.1234559,0.618],
    [0.334241,0.1234559,0.615],
    [0.330454,0.13945,0.622]
])

# 提取x和y坐标
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

x1 = points1[:, 0]
y1 = points1[:, 1]
z1 = points1[:, 2]

# 计算样条插值参数
tck, u = splprep([x, y], s=0)
tck1, u1 = splprep([x1, y1], s=0)

# 生成插值后的点
new_points = splev(np.linspace(0, 1, 100), tck)
new_points1 = splev(np.linspace(0, 1, 100), tck1)

# 绘制原始点和插值后的曲线
plt.figure()
plt.plot(x, y, 'ro', label='Original Points')
plt.plot(x1, y1, 'bo', label='Prediction Points')
plt.plot(new_points[0], new_points[1], 'b-', label='Spline Interpolation')
plt.plot(new_points1[0], new_points1[1], 'r-', label='Spline Interpolation1')
plt.legend()
plt.xlim(0, 0.5)  # 设置x坐标范围
plt.ylim(0, 0.3)  # 设置y坐标范围
plt.show()
