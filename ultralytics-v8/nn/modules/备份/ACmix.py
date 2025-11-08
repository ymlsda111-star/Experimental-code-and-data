import torch
import torch.nn as nn


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head

        # 确保head_dim计算正确
        if head <= 0:
            raise ValueError(f"head参数必须大于0，当前为: {head}")

        self.head_dim = out_planes // head
        if self.head_dim == 0:
            # 如果head_dim为0，调整head参数
            self.head = min(head, out_planes)
            self.head_dim = out_planes // self.head if self.head > 0 else out_planes

        print(f"ACmix初始化参数: in_planes={in_planes}, out_planes={out_planes}, head={head}")
        print(f"计算得到的head_dim: {self.head_dim}")

        # 简化版本：使用标准的卷积层
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)

        # 使用分组卷积代替复杂的dep_conv
        # 确保groups参数有效
        groups = min(self.head_dim, out_planes)
        if groups == 0:
            groups = 1

        self.dep_conv = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                                  padding=1, groups=groups, stride=stride)

        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)

        # 简化的注意力机制
        b, c, h, w = q.shape

        # 确保head_dim有效
        if self.head_dim > 0 and c >= self.head * self.head_dim:
            att = torch.softmax((q * k).view(b, self.head, self.head_dim, h * w).mean(2), dim=-1)
            out_att = (att.unsqueeze(2) * v.view(b, self.head, self.head_dim, h * w)).view(b, c, h, w)
        else:
            # 备用方案：直接使用卷积结果
            out_att = v

        out_conv = self.dep_conv(v)

        return self.rate1 * out_att + self.rate2 * out_conv