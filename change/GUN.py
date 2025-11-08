import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential
from keras.layers import GRU, Dense

# 已知的前5次迭代的位置信息
positions = np.array([
    [0.342, 0.148, 0.63],
    [0.348, 0.148, 0.63],
    [0.349, 0.148, 0.65],
    [0.349, 0.148, 0.64],
    [0.348, 0.148, 0.64]
])

# 数据预处理
X = positions[:-1].reshape(1, 4, 3)  # 输入数据，去掉最后一个位置作为预测的起点
y = positions[1:].reshape(1, 4, 3)   # 输出数据，去掉第一个位置作为预测的起点

# 构建GRU模型
model = Sequential()
model.add(GRU(50, activation='relu', input_shape=(4, 3)))
model.add(Dense(3))
model.compile(optimizer='adam', loss='mse')

# 训练模型（这里假设已经训练好了）
# model.fit(X, y, epochs=100, verbose=0)

# 预测下一个位置信息
next_position = model.predict(X)
print("预测的下一个位置信息：", next_position[0])

# 绘制三维视图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b', marker='o', label='Known Data')
ax.scatter(*next_position[0], c='r', marker='^', label='Predicted Data')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
