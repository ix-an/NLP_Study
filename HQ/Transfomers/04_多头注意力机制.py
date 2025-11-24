import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, d_model, num_heads=3):
        super().__init__()
        
        self.dim_in = dim_in        # 输入中每个token的维度（x的最后一个维度）
        self.d_model = d_model      # qkv向量长度
        self.num_heads = num_heads  # head个数

        # 维度必须能被num_heads整除
        assert d_model % num_heads == 0, "d_model must be multiple of num_heads"

        # 定义线性变换矩阵
        self.linear_q = nn.Linear(dim_in, d_model)
        self.linear_k = nn.Linear(dim_in, d_model)
        self.linear_v = nn.Linear(dim_in, d_model)
        self.scale = 1 / math.sqrt(d_model // num_heads)

        # 最后的线性层：把输出的m个head的向量拼接起来
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (batch, seq_len, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.d_model // nh    # 每个头的维度

        # q,k,v : (batch, nh, n, dk)
        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)
        v = self.linear_v(x).reshape(batch, n, nh, dk).transpose(1, 2)

        # dist: (batch, nh, n, n)
        dist = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # dist: (batch, nh, n, n)
        dist = torch.softmax(dist, dim=-1)

        # att: (batch, nh, n, dk)
        att = torch.matmul(dist, v)
        # (batch, nh, n, dk) -> (batch, n, nh, dk) -> (batch, n, d_model)
        att = att.transpose(1, 2).reshape(batch, n, self.d_model)

        # 通过最后一个线性层变换
        output = self.fc(att)  # (batch, n, d_model)
        
        return output
    


if __name__ == '__main__':
    x = torch.randn(1, 4, 2)
    multi_head_attention = MultiHeadAttention(x.shape[2], 6, 3)
    output = multi_head_attention(x)
    print(output.shape)  # torch.Size([1, 4, 6])


        
    


        
