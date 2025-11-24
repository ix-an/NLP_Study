'''词嵌入：把一个token转换为向量'''

import torch
import torch.nn as nn

embedding = nn.Embedding(5, 128)

# 假设一个有3个token的句子
x = torch.tensor([1, 2, 3])
# 把每个token都转换为了一个128维的向量
print(embedding(x), embedding(x).shape) # (3, 128)