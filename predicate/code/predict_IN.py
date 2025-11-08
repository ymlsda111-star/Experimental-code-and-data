import os
import re
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# --- 1. 文件路径定义 ---
INPUT_FILE_PATH = r"D:\YOLO5\yolov5-master-new\depeth\txt\Informer\5\5-9\useful\T5-9time-15-Z.txt"
OUTPUT_FILE_PATH = r"D:\YOLO5\yolov5-master-new\depeth\txt\Informer\5\5-9\useful\P5-9time-15-Z.txt"


# --- 2. 数据读取和分析函数 (与之前相同) ---
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


# --- 3. 执行数据加载 (与之前相同) ---
data, indices = analyze_and_load_data(INPUT_FILE_PATH)
print("--- 文件分析结果 ---")
print(f"输入文件：{INPUT_FILE_PATH}")
print(f"总坐标数量：{len(data)} 个")
if not data:
    print("无有效数据，程序终止。")
    exit()
print("\n" + "=" * 40 + "\n")

# --- 4. Informer 参数配置 ---
LOOKBACK = 10  # 编码器输入序列长度 (Seq_len)
PRED_LEN = 1  # 预测长度 (我们只需要预测下一个点)
D_MODEL = 64  # 模型内部的特征维度 (必须是 nhead 的整数倍)
NHEAD = 4  # 注意力头数
NUM_ENCODER_LAYERS = 2  # 编码器层数
DIM_FEEDFORWARD = 256  # 前馈网络中间层的维度
DROPOUT = 0.1
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 5. 数据预处理 (与之前相同) ---
data_np = np.array(data).reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_np)


def create_dataset(dataset, lookback=LOOKBACK):
    X, Y = [], []
    for i in range(len(dataset) - lookback):
        X.append(dataset[i:(i + lookback), 0])
        Y.append(dataset[i + lookback, 0])
    return np.array(X), np.array(Y)


X, Y = create_dataset(scaled_data)
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
train_loader = DataLoader(TensorDataset(X_tensor, Y_tensor), batch_size=32, shuffle=True)


# =================================================================
# --- 6. Informer 核心组件 (attn.py 和 embed.py 的简化实现) ---
# =================================================================

# 6.1. Positional Embedding (来自 embed.py)
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


# 6.2. Token Embedding (来自 embed.py)
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=1, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


# 6.3. Data Embedding (来自 embed.py)
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


# 6.4. ProbSparse Attention (来自 attn.py - 简化版)
class ProbSparseAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbSparseAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        # 简化实现：仅使用标准点积注意力，忽略 ProbSparse 采样逻辑
        # 实际 Informer 会在这里进行稀疏化采样
        return torch.matmul(Q, K.transpose(-1, -2))

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        # 简化：使用标准注意力
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            # 仅在训练时需要掩码，这里简化为无掩码
            pass

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


# 6.5. Multi-Head Attention (来自 attn.py)
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


# 6.6. Informer Encoder Layer (来自 encoder.py)
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # Self-Attention
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        # Feed Forward
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


# 6.7. Informer Encoder (来自 encoder.py)
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


# 6.8. Informer 主模型 (来自 model.py - 仅编码器部分)
class InformerModel(nn.Module):
    def __init__(self, enc_in, c_out, seq_len, d_model, n_heads, e_layers, d_ff, dropout):
        super(InformerModel, self).__init__()

        # 1. 嵌入层
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)

        # 2. 编码器
        attn_layers = [
            EncoderLayer(
                AttentionLayer(
                    ProbSparseAttention(False, factor=5, attention_dropout=dropout),
                    d_model, n_heads
                ),
                d_model, d_ff, dropout
            ) for _ in range(e_layers)
        ]
        self.encoder = Encoder(attn_layers, norm_layer=nn.LayerNorm(d_model))

        # 3. 预测层 (只取编码器最后一个时间步的输出)
        self.projection = nn.Linear(d_model, c_out)

    def forward(self, x_enc):
        # x_enc shape: (batch_size, seq_len, 1)

        # 嵌入
        enc_out = self.enc_embedding(x_enc)

        # 编码
        enc_out, attns = self.encoder(enc_out)

        # 预测：只使用最后一个时间步的输出
        # enc_out shape: (batch_size, seq_len, d_model)
        output = self.projection(enc_out[:, -1, :])

        return output


# 实例化模型
model = InformerModel(
    enc_in=1, c_out=1, seq_len=LOOKBACK,
    d_model=D_MODEL, n_heads=NHEAD,
    e_layers=NUM_ENCODER_LAYERS, d_ff=DIM_FEEDFORWARD, dropout=DROPOUT
).to(DEVICE)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 7. 模型训练 ---
print("--- Informer 模型开始训练 (自包含简化版) ---")
for epoch in range(NUM_EPOCHS):
    for inputs, targets in train_loader:
        # Informer 编码器只需要一个输入
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.6f}')
print("--- 模型训练完成 ---\n")

# --- 8. 生成预测序列 (与之前相同) ---
model.eval()
with torch.no_grad():
    predicted_scaled_all = model(X_tensor).cpu().numpy()
predicted_coordinates = scaler.inverse_transform(predicted_scaled_all).flatten()

output_lines = []
for i in range(LOOKBACK):
    output_lines.append(f"({indices[i]:03d}, {data[i]:.1f}, N/A)")
for i in range(len(predicted_coordinates)):
    original_index = LOOKBACK + i
    output_lines.append(f"({indices[original_index]:03d}, {data[original_index]:.1f}, {predicted_coordinates[i]:.3f})")

# --- 9. 写入输出文件 (与之前相同) ---
try:
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    with open(OUTPUT_FILE_PATH, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')
    print("--- 预测结果输出成功 ---")
    print(f"结果已写入文件：{OUTPUT_FILE_PATH}")
except Exception as e:
    print(f"写入文件时发生错误：{e}")