import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 已知的位置信息
X = np.array([
    [0.35679, 0.20399, 0.57],
    [0.35719, 0.20399, 0.56],
    [0.35957, 0.20960, 0.58],
    [0.35533, 0.20373, 0.57],
    [0.35600, 0.20386, 0.56]
])
y = np.array([0.63, 0.63, 0.65, 0.64, 0.64])

# 创建高斯过程回归模型
kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
gpr = GaussianProcessRegressor(kernel=kernel)

# 拟合模型
gpr.fit(X, y)

# 预测新位置
X_new = np.array([
    [0.350, 0.148],
    [0.351, 0.148],
    [0.352, 0.148],
    [0.353, 0.148],
    [0.354, 0.148]
])
y_pred, y_std = gpr.predict(X_new, return_std=True)

# 打印预测结果
print("Predicted positions:")
for i in range(len(X_new)):
    print(f"Position {i+1}: ({X_new[i][0]:.3f}, {X_new[i][1]:.3f}, {y_pred[i]:.3f})")

# 可视化结果
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='o', label='Known Positions')
ax.scatter(X_new[:, 0], X_new[:, 1], y_pred, c='b', marker='^', label='Predicted Positions')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.show()
