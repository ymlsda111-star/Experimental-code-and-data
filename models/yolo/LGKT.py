import numpy as np

def rk4_step(f, x, t, dt):
    k1 = f(x, t)
    k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(x + dt * k3, t + dt)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def predict_positions(initial_positions, time_intervals, f, dt=0.01):
    positions = [initial_positions]
    for i in range(len(time_intervals) - 1):
        t = time_intervals[i]
        x = positions[-1]
        new_position = rk4_step(f, x, t, dt)
        positions.append(new_position)
    return positions

# 示例：预测物体在三维空间中的运动轨迹
def motion_function(x, t):
    # 假设物体受到一个恒定的加速度 a = [1, 2, 3] m/s^2
    a = np.array([1, 2, 3])
    return a

initial_positions = np.array([0, 0, 0])  # 初始位置
time_intervals = np.linspace(0, 10, 1001)  # 时间间隔，从0秒到10秒，共1001个点

predicted_positions = predict_positions(initial_positions, time_intervals, motion_function)
print(predicted_positions)
