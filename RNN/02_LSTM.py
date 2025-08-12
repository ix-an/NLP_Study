import torch
import torch.nn as nn

# 定义一个简单的LSTM模型
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)  # LSTM层
        self.fc = nn.Linear(hidden_size, output_size)  # 全连接层

    def forward(self, x):
        # 初始化隐藏状态h0和细胞状态c0
        h0 = torch.zeros(1, x.size(1), self.hidden_size)
        c0 = torch.zeros(1, x.size(1), self.hidden_size)

        # 前向传播
        out, _ = self.lstm(x, (h0, c0))  # _是最后一个时间步的h和c
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        # out: [seq_length, batch_size, 1*hidden_size]
        # out[:, -1, :] 取出最后一个batch的输出，即最后一个时间步的输出
        # 整数索引会自动降维处理，而切片索引不会，这里自动降维成[seq_length, 1*hidden_size]
        # 再通过FC层，输出维度为2 -> 最终 out.shape [seq_length, output_size]=[5,2]
        return out

# 测试模型
if __name__ == '__main__':
    # 定义模型参数
    input_size = 10
    hidden_size = 20
    output_size = 2

    # 创建模型
    model = SimpleLSTM(input_size, hidden_size, output_size)

    # 生成随机输入数据
    # 输入数据的形状为 (seq_length, batch_size, input_size)
    input_data = torch.randn(5, 3, 10)  # 序列长度为5，批量大小为3，输入特征维度为10

    # 运行模型（前向传播）
    output = model(input_data)

    # 打印输出和形状
    print("Output:\n", output)
    print("Output shape:", output.shape)  # torch.Size([5, 2])
