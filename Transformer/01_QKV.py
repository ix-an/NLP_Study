import torch
import torch.nn as nn


if __name__ == '__main__':
    vocab = {
        '我们' : 0,
        '还有' : 1,
        '很多' : 2,
        '个' : 3,
        '属于' : 4,
        '蓝雨' : 5,
        '的' : 6,
        '夏天' : 7,
    }
    num_embeddings = len(vocab)
    # 词嵌入向量
    embedding = nn.Embedding(num_embeddings, 256)
    # 获取 '我们' 的词向量
    embed = embedding(torch.tensor([0]))
    # print(embed)
    # print(embed.shape)

    # 映射QKV -> 提升表达能力和灵活度
    Q = nn.Linear(256, 4)(embed)
    print(Q)
    print(Q.shape)
    K = nn.Linear(256, 4)(embed)
    print(K)
    print(K.shape)
    V = nn.Linear(256, 4)(embed)
    print(V)
    print(V.shape)


    

