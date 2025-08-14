import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim_in, d_model, num_heads=3):
        super(MultiHeadSelfAttention, self).__init__()

        self.dim_in = dim_in
        self.d_model = d_model
        self.num_heads = num_heads

        # 维度必须能被 num_heads整除
        assert d_model % num_heads == 0, "d_model must be multiple of num_heads"

        # 定义线性变换矩阵
        self.linear_q = nn.Linear(dim_in, d_model)
        self.linear_k = nn.Linear(dim_in, d_model)
        self.linear_v = nn.Linear(dim_in, d_model)
        self.scale = 1 / math.sqrt(d_model // num_heads)

        # 最后的线性层
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: tensor of shape (batch_size, n, dim_in)
        batch, n, dim_in = x.shape
        # 检查x的token维度是否与全连接层指定的输入维度一致
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.d_model // nh    # dim_k of each head

        # (1,4,6) -> reshape -> (1,4,3,2) -> transpose -> (1,3,4,2)
        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)

        # 相似度分数
        dist = torch.matmul(q, k.transpose(2, 3)) * self.scale  # (batch, nh, n, n)
        dist = torch.softmax(dist, dim=-1)  # (batch, nh, n, n)

        # 注意力输出
        att = torch.matmul(dist, v)  # (batch, nh, n, dv)
        att = att.transpose(1, 2).reshape(batch, n, self.d_model)  # (batch, n, dim_v)

        # 通过最后一个线性层进行变换
        out = self.fc(att)
        return out


if __name__ == '__main__':
    torch.manual_seed(42)
    x = torch.rand((1, 4, 2))  # (batch, n, dim_in)
    multi_head_att = MultiHeadSelfAttention(x.shape[2], 6, 3)  # (6, 3)
    output = multi_head_att(x)
    print(output)
    print(output.shape)


