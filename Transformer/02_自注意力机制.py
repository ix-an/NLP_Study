import torch
import torch.nn as nn


class Self_Attention(nn.Module):
    def __init__(self, dim, dk, dv):
        super(Self_Attention, self).__init__()
        self.scale = dk ** -0.5
        self.q = nn.Linear(dim, dk)
        self.k = nn.Linear(dim, dk)
        self.v = nn.Linear(dim, dv)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 对最后一个维度进行softmax归一化
        attn = attn.softmax(dim=-1)

        x = attn @ v    # 计算注意力输出
        return x

if __name__ == '__main__':
    att = Self_Attention(dim=2, dk=2, dv=3)
    x = torch.randn(1, 4, 2)    # [batch_size, seq_len, dim]
    # 其中，seq_len表示序列长度（token数），dim表示输入维度（每个token的维度）
    output = att(x)    # [batch_size, seq_len, dv]
    print(output, output.shape)

