import torch
import torch.nn as nn


class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=3, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        # 确保head数量合理
        if out_planes < head:
            head = 1
        self.head = head

        # 简化版本：只实现核心功能
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)

        # 添加一个1x1卷积确保输出通道数正确
        self.output_conv = nn.Conv2d(out_planes, out_planes, kernel_size=1)

        # 初始化参数
        self.rate1 = nn.Parameter(torch.tensor(0.5))
        self.rate2 = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        b, c, h, w = x.shape

        # 确保输入通道数正确
        if c != self.in_planes:
            # 如果通道数不匹配，使用1x1卷积适配
            if not hasattr(self, 'adapter'):
                self.adapter = nn.Conv2d(c, self.in_planes, kernel_size=1).to(x.device)
            x = self.adapter(x)

        # 正常处理
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)

        # 简化的注意力机制
        att = torch.sigmoid(q)  # 简单的注意力权重
        out = att * v + (1 - att) * k  # 加权融合

        # 确保输出通道数正确
        out = self.output_conv(out)

        return out