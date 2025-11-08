import os
import re
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # 仍然使用 MinMaxScaler 进行数据预处理

# --- 1. 文件路径定义 ---
INPUT_FILE_PATH = r"D:\YOLO5\yolov5-master-new\depeth\txt\PF\1\1-3.5\useful\T1-3.5time-Z.txt"
OUTPUT_FILE_PATH = r"D:\YOLO5\yolov5-master-new\depeth\txt\PF\1\1-3.5\useful\P1-3.5time-Z.txt"


# --- 2. 数据读取和分析函数 (保持不变) ---
def analyze_and_load_data(file_path):
    coordinates, indices = [], []
    if not os.path.exists(file_path):
        print(f"错误：文件未找到 -> {file_path}")
        return coordinates, indices
    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = re.search(r'\(\s*(\d+)\s*,\s*(\d+\.?\d*)\s*\)', line)
                if match:
                    indices.append(int(match.group(1)))
                    coordinates.append(float(match.group(2)))
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
    return coordinates, indices


# --- 3. 粒子滤波 (PF) 参数配置 ---
NUM_PARTICLES = 500  # 粒子数量
SYSTEM_NOISE = 0.5  # 系统模型（运动）的噪声标准差
OBSERVATION_NOISE = 1.0  # 观测模型（测量）的噪声标准差
LOOKBACK = 10  # 预测所需的历史数据点数量 (用于初始化)


# --- 4. 粒子滤波核心函数 ---

def initialize_particles(initial_state, num_particles, noise_std):
    """初始化粒子集，以初始状态为中心分布。"""
    # 假设初始状态是序列前几个点的平均值
    return initial_state + np.random.randn(num_particles) * noise_std


def predict_step(particles, system_noise):
    """预测步骤：根据系统模型移动粒子。"""
    # 简化的系统模型：随机游走 (下一个状态 = 当前状态 + 噪声)
    # 假设状态是坐标值本身
    return particles + np.random.randn(len(particles)) * system_noise


def update_step(particles, observation, observation_noise):
    """更新步骤：根据观测值计算粒子的权重。"""
    # 观测模型：假设观测值 z_t = 状态 x_t + 噪声 v_t
    # 权重与观测值和粒子状态之间的误差呈高斯分布关系
    weights = np.exp(-0.5 * ((observation - particles) / observation_noise) ** 2)
    weights /= np.sum(weights)  # 归一化
    return weights


def resample_step(particles, weights):
    """重采样步骤：根据权重重新选择粒子。"""
    # 使用系统重采样 (Systematic Resampling)
    N = len(particles)
    indices = np.arange(N)
    cumulative_sum = np.cumsum(weights)

    # 随机起始点
    u = np.random.rand() / N

    # 重采样索引
    resample_indices = []
    i = 0
    for j in range(N):
        while u > cumulative_sum[i]:
            i += 1
        resample_indices.append(indices[i])
        u += 1.0 / N

    return particles[resample_indices]


def estimate_state(particles, weights):
    """估计当前状态（预测值）：使用加权平均。"""
    return np.sum(particles * weights)


# --- 5. 主预测逻辑 ---

def run_particle_filter(data, indices, lookback):
    # 1. 数据归一化 (PF通常在原始空间工作，但为了与DL代码格式一致，这里保留)
    data_np = np.array(data).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_np).flatten()

    predicted_scaled_coordinates = []

    # 2. 初始化粒子集
    # 使用前 LOOKBACK 个点的平均值作为初始状态
    initial_state = np.mean(scaled_data[:lookback])
    particles = initialize_particles(initial_state, NUM_PARTICLES, SYSTEM_NOISE)
    weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES  # 初始权重均匀分布

    # 3. 迭代预测
    # 从 LOOKBACK 索引开始，因为前 LOOKBACK 个点用于初始化
    for t in range(lookback, len(scaled_data)):
        observation = scaled_data[t]

        # 3.1. 预测 (Predict)
        particles = predict_step(particles, SYSTEM_NOISE)

        # 3.2. 更新 (Update)
        weights = update_step(particles, observation, OBSERVATION_NOISE)

        # 3.3. 估计 (Estimate)
        # 预测值是基于当前粒子集和权重的估计
        predicted_state = estimate_state(particles, weights)
        predicted_scaled_coordinates.append(predicted_state)

        # 3.4. 重采样 (Resample)
        particles = resample_step(particles, weights)
        # 重采样后，权重重新均匀分布
        weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

    # 4. 反归一化
    predicted_coordinates = scaler.inverse_transform(np.array(predicted_scaled_coordinates).reshape(-1, 1)).flatten()

    # 5. 格式化输出
    output_lines = []
    for i in range(lookback):
        output_lines.append(f"({indices[i]:03d}, {data[i]:.1f}, N/A)")
    for i in range(len(predicted_coordinates)):
        original_index = lookback + i
        output_lines.append(
            f"({indices[original_index]:03d}, {data[original_index]:.1f}, {predicted_coordinates[i]:.3f})")

    # 6. 写入输出文件
    try:
        os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
        with open(OUTPUT_FILE_PATH, 'w') as f:
            for line in output_lines:
                f.write(line + '\n')
        print("--- 预测结果输出成功 ---")
        print(f"结果已写入文件：{OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"写入文件时发生错误：{e}")


# --- 6. 执行主函数 ---
if __name__ == '__main__':
    data, indices = analyze_and_load_data(INPUT_FILE_PATH)
    if data:
        run_particle_filter(data, indices, LOOKBACK)
    else:
        print("无法执行粒子滤波，因为没有加载到有效数据。")