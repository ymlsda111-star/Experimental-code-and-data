import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

# 将数据转换为适合LSTM输入的形状 (samples, timesteps, features)
X = positions.reshape(1, 5, 3)
y = positions[-1].reshape(1, 3)

# 创建LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel(3, 50, 3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# 训练模型
for epoch in range(1000):
    model.zero_grad()
    output = model(torch.tensor(X).float())
    loss = criterion(output, torch.tensor(y).float())
    loss.backward()
    optimizer.step()

# 预测后面5次迭代的位置信息
predictions = []
with torch.no_grad():
    for _ in range(5):
        prediction = model(torch.tensor(y).float().view(1, 1, 3))
        predictions.append(prediction.numpy())
        y = prediction.numpy()

# 打印三维预测位置信息
print("三维预测位置信息：")
for i, pos in enumerate(predictions):
    print(f"第{i+1}次迭代： {pos}")

# 绘制三维视图
'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b', label='实际位置')
ax.scatter(predictions[:, 0], predictions[:, 1], predictions[:, 2], c='r', label='预测位置')
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_zlabel('Z轴')
ax.legend()
plt.show()
'''