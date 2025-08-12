import torch
import torch.nn as nn


# 单向单层RNN模型
def rnn_test01():
    # 定义RNN模型
    rnn = nn.RNN(
        input_size=10,       # 输入特征维度
        hidden_size=20,      # 隐藏状态维度
        num_layers=1,        # 默认1层，即单层RNN
        batch_first=True,    # 默认False，为True时，输入为(batch_size, seq_len, input_size)
        bidirectional=False  # 默认为False,单向RNN
    )

    # 构造输入数据：(batch_size, seq_len, input_size)
    input_data = torch.randn(2, 3, 10)  # 2个样本，每个样本3个时间步，每个时间步10维特征

    # 前向传播
    output, hidden = rnn(input_data)

    print("Output shape:", output.shape)  # (batch_size, seq_len, hidden_size)
    print("Hidden shape:", hidden.shape)  # (num_layers, batch_size, hidden_size)

    # 查看输出
    print("Output:", output)
    print("Hidden:", hidden)


# 双向单层RNN模型
def rnn_test02():
    # 双向RNN：num_directions=2，输出维度=2*hidden_size
    bi_rnn = nn.RNN(
        input_size=10,
        hidden_size=16,
        num_layers=1,
        batch_first=True,
        bidirectional=True    # 双向RNN
    )

    x = torch.randn(32, 5, 10)
    output, h_n = bi_rnn(x)
    print(output.shape)  # (32,5,32) （32=16*2）
    print(h_n.shape)  # (2,32,16) （2个方向的最后隐藏状态）


if __name__ == '__main__':
    # rnn_test01()
    rnn_test02()
