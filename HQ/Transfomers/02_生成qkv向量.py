import torch
import torch.nn as nn

# 假设词表
vocab = {'我':0, '可':1, '是':2, '职业选手':3, '啊':4}

# 词嵌入：向量化
embedding = nn.Embedding(5, 128)

x1 = vocab['我']
print(x1)
x1 = embedding(torch.tensor([x1]))
print(x1.shape)  # torch.Size([1, 128])

# 0 1 2 3 4 0
# 我 可 是 职业选手 啊 我
x = embedding(torch.tensor([0,1,2,3,4,0]))
print(x.shape)  # torch.Size([6, 128])


q_w = torch.randn(128, 64)
k_w = torch.randn(128, 64)
v_w = torch.randn(128, 64)

q = q_w * x 
k = k_w * x
v = v_w * x

print(q.shape, k.shape, v.shape)  # torch.Size([6, 64]) 