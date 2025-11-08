import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 已知的前5次迭代的位置信息
positions = np.array([
    [0.342, 0.148, 0.63],
    [0.348, 0.148, 0.63],
    [0.349, 0.148, 0.65],
    [0.349, 0.148, 0.64],
    [0.348, 0.148, 0.64]
])

# 初始化卡尔曼滤波器参数  后期根据实际情况修改参数
n = len(positions)
A = np.eye(3)  # 状态转移矩阵
H = np.eye(3)  # 观测矩阵
Q = np.eye(3)  # 过程噪声协方差矩阵
R = np.eye(3)  # 观测噪声协方差矩阵
x = positions[-1]  # 初始状态估计
P = np.eye(3)  # 初始状态协方差矩阵

print(n)
print("状态转移矩阵A:")
print(A)
print("观测矩阵H:")
print(H)
print("过程噪声协方差矩阵Q:")
print(Q)
print("观测噪声协方差矩阵:")
print(R)
print("初始状态估计x:")
print(x)
print("初始状态协方差矩阵P:")
print(P)


# 卡尔曼滤波器更新过程
for i in range(1, len(positions)):
    # 预测步骤
    x_pred = np.dot(A, x)
    P_pred = np.dot(np.dot(A, P), A.T) + Q

    # 更新步骤（这里我们假设没有观测到新的位置信息）
    K = np.dot(np.dot(P_pred, H.T), np.linalg.inv(np.dot(np.dot(H, P_pred), H.T) + R))
    x = x_pred + np.dot(K, (positions[i - 1] - np.dot(H, x_pred)))
    P = np.dot((np.eye(3) - np.dot(K, H)), P_pred)

    print("预测位置 {}: {}".format(i, x))

# 绘制三维视图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', marker='o', label='Observed')
ax.scatter(x[0], x[1], x[2], c='b', marker='^', label='Predicted')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
