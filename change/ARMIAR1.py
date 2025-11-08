import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.tsa.arima.model import ARIMA

data = np.array([
    [0.342, 0.148, 0.63],
    [0.348, 0.148, 0.63],
    [0.349, 0.148, 0.65],
    [0.349, 0.148, 0.64],
    [0.348, 0.148, 0.64]
])

# 将数据转换为适合ARMA模型的格式
arma_data = data[:, 2].reshape(-1, 1)

# 创建ARMA模型并拟合数据
model = ARIMA(arma_data, order=(1, 0, 1))
results = model.fit()

# 预测下一个位置
next_position = results.forecast(steps=1)[0]
print("预测的位置信息：", next_position)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制已知数据点
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', marker='o', label='Known Data')

# 绘制预测数据点
ax.scatter(data[-1, 0], data[-1, 1], next_position, c='r', marker='^', label='Predicted Data')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
