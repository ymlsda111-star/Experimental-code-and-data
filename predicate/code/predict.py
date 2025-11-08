import os
import re
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# --- 1. 文件路径定义 ---
# 输入文件路径
INPUT_FILE_PATH = r"D:\YOLO5\yolov5-master-new\depeth\txt\LSTM\1\1-3.5\useful\T1-3.5time-X.txt"
OUTPUT_FILE_PATH = r"D:\YOLO5\yolov5-master-new\depeth\txt\LSTM\1\1-3.5\useful\P1-3.5time-X-Transformer.txt"


# --- 2. 数据读取和分析函数 ---
def analyze_and_load_data(file_path):
    """读取文件并提取坐标数据和原始索引。"""
    coordinates = []
    indices = []

    if not os.path.exists(file_path):
        print(f"错误：文件未找到，请检查路径：{file_path}")
        return coordinates, indices

    try:
        with open(file_path, 'r') as f:
            for line in f:
                # 匹配 (INDEX, VALUE)
                match = re.search(r'\(\s*(\d+)\s*,\s*(\d+\.?\d*)\s*\)', line)
                if match:
                    index = int(match.group(1))
                    coordinate = float(match.group(2))
                    indices.append(index)
                    coordinates.append(coordinate)
    except Exception as e:
        print(f"读取或处理文件时发生错误：{e}")
        return coordinates, indices

    return coordinates, indices


# --- 3. 执行数据加载 ---
data, indices = analyze_and_load_data(INPUT_FILE_PATH)

print("--- 文件分析结果 ---")
print(f"输入文件：{INPUT_FILE_PATH}")
print(f"文件中的总坐标数量：{len(data)} 个")

if not data:
    print("没有加载到有效数据，程序终止。")
    exit()

print("\n" + "=" * 40 + "\n")

# --- 4. LSTM 参数配置 ---
LOOKBACK = 10  # 用前10个数据点预测下一个点
HIDDEN_SIZE = 50
NUM_LAYERS = 1
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 5. 数据预处理 ---

# 将列表数据转换为 NumPy 数组并进行归一化
data_np = np.array(data).reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_np)


# 创建 LSTM 所需的序列数据 (X: 1..N-1, Y: N)
def create_dataset(dataset, lookback=LOOKBACK):
    X, Y = [], []
    for i in range(len(dataset) - lookback):
        # X: 序列 i 到 i+lookback-1 (共 lookback 个点)
        X.append(dataset[i:(i + lookback), 0])
        # Y: 序列 i+lookback (第 i+lookback+1 个点)
        Y.append(dataset[i + lookback, 0])
    return np.array(X), np.array(Y)


X, Y = create_dataset(scaled_data)

# 转换为 PyTorch 张量
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(DEVICE)  # 形状: (样本数, lookback, 1)
Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1).to(DEVICE)  # 形状: (样本数, 1)

# 创建 DataLoader
train_loader = DataLoader(
    TensorDataset(X_tensor, Y_tensor),
    batch_size=32,
    shuffle=True
)


# --- 6. 定义 LSTM 模型 ---

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 实例化模型
model = SimpleLSTM(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=1).to(DEVICE)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 7. 模型训练 ---
print("--- 模型开始训练 ---")
for epoch in range(NUM_EPOCHS):
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.6f}')

print("--- 模型训练完成 ---\n")

# --- 8. 生成预测序列 ---

# 切换到评估模式
model.eval()

# 1. 准备用于预测的完整数据集 (与训练集 X 相同)
# 预测的输入是 X_tensor，它包含了所有长度为 LOOKBACK 的子序列
# 预测的输出将对应于原始数据中索引从 LOOKBACK 开始到结束的部分

with torch.no_grad():
    # 得到所有预测值的归一化结果
    predicted_scaled_all = model(X_tensor).cpu().numpy()

# 2. 反归一化，得到真实坐标值
predicted_coordinates = scaler.inverse_transform(predicted_scaled_all).flatten()

# 3. 准备输出数据
# 预测序列的起始索引是 LOOKBACK + 1 (因为前 LOOKBACK 个点没有历史数据来预测)
# 原始数据中，索引从 LOOKBACK 开始的部分是实际被预测的 Y 值
# 预测序列的长度是 len(data) - LOOKBACK

output_lines = []

# 前 LOOKBACK 个点无法预测，用原始值代替或标记为 N/A
for i in range(LOOKBACK):
    # 格式: (索引, 原始坐标, 预测坐标)
    output_lines.append(f"({indices[i]:03d}, {data[i]:.1f}, N/A)")

# 从 LOOKBACK 索引开始，是模型实际预测的部分
for i in range(len(predicted_coordinates)):
    original_index = LOOKBACK + i
    # 格式: (索引, 原始坐标, 预测坐标)
    output_lines.append(
        f"({indices[original_index]:03d}, {data[original_index]:.1f}, {predicted_coordinates[i]:.3f})"
    )

# --- 9. 写入输出文件 ---
try:
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)

    with open(OUTPUT_FILE_PATH, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')

    print("--- 预测结果输出成功 ---")
    print(f"结果已写入文件：{OUTPUT_FILE_PATH}")
    print(f"预测序列长度：{len(predicted_coordinates)} 个点")
    print(f"前 {LOOKBACK} 个点 (N/A) 是因为没有足够的历史数据进行预测。")

except Exception as e:
    print(f"写入文件时发生错误：{e}")