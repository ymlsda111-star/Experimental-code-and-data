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

# 假设粒子数量为100
num_particles = 100

# 初始化粒子位置
particles = np.random.normal(loc=positions[-1], scale=0.01, size=(num_particles, 3))


# 定义状态转移函数（这里我们假设状态转移是线性的）
def state_transition(x):
    return x + np.random.normal(loc=0, scale=0.01, size=3)


# 定义观测函数（这里我们假设观测是线性的）
def observation(x):
    return x + np.random.normal(loc=0, scale=0.01, size=3)


# 运行粒子滤波器进行预测
for i in range(5):
    # 更新粒子位置
    particles = np.array([state_transition(p) for p in particles])

    # 重采样（这里我们简单地采用均匀采样）
    weights = np.ones(num_particles) / num_particles
    particles = particles[np.random.choice(num_particles, size=num_particles, p=weights)]

    # 计算观测值
    observations = np.array([observation(p) for p in particles])

    # 更新权重（这里我们简单地将所有权重设为相等）
    weights = np.ones(num_particles) / num_particles

    # 输出当前预测位置的均值
    predicted_position = np.mean(particles, axis=0)
    print(f"Predicted position at iteration {i + 1}: {predicted_position}")

# 绘制三维视图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2], c='r', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
