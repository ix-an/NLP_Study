import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 公共变量（模型相关配置）
device = "cuda" if torch.cuda.is_available() else "cpu"
dict_file = "data/hotel_dict.txt"  # 字典文件路径
encoding_file = "data/hotel_encoding.txt"  # 编码后的样本文件路径
max_lens = 256  # 序列最大长度


def get_dict_len(dict_path):
    """获取字典长度"""
    with open(dict_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
        new_dict = eval(lines[0])
    return len(new_dict.keys())


class LSTMModel(nn.Module):
    """LSTM模型用于文本分类任务"""

    def __init__(self, vocab_size, target_size, embedding_dim=512, hidden_dim=512):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # MLP层用于增强嵌入特征
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

        # LSTM层
        # embedding_dim 是输入数据的特征维度，通常是嵌入向量的维度。
        # hidden_dim 是 LSTM 隐层的维度，即每个时间步的隐藏状态和细胞状态的大小。
        # num_layers=4 指定了 LSTM 的层数，这里是 4 层堆叠的 LSTM。
        # batch_first=True 表示输入数据的形状为 (batch_size, seq_length, input_size)，即批次大小是第一个维度。
        # dropout=0.1 在层与层之间添加了 dropout，以防止过拟合。
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=4,
            batch_first=True,
            dropout=0.2
        )

        # 输出层（结合LSTM特征和嵌入特征）
        self.hidden2tag = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),    # 层归一化
            nn.Linear(hidden_dim, target_size)
        )

        # 池化层用于序列特征聚合
        self.avg_lstm = nn.AdaptiveAvgPool1d(1)
        self.avg_linear = nn.AdaptiveAvgPool1d(1)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids, lengths):
        """前向传播过程"""
        # 嵌入层计算
        embeds = self.embedding(input_ids)
        embeds = self.mlp(embeds)

        # 嵌入特征池化
        # transpose(-1, -2) 的作用是交换张量的最后两个维度，
        # 这里是将 (batch_size, seq_length, embedding_dim) 转换为 (batch_size, embedding_dim, seq_length)
        # 然后进行序列池化，得到 (batch_size, embedding_dim) 的张量
        linear_out = self.avg_linear(embeds.transpose(-1, -2))
        # print("l",linear_out.shape)

        # LSTM处理（支持变长序列）
        # pack_padded_sequence 将填充的序列打包，便于LSTM处理变长序列
        # 这个对象包含了所有非填充的元素及其对应的长度信息
        packed_embedded = pack_padded_sequence(
            embeds,               # 嵌入向量序列
            lengths,              # 包含每个序列实际长度的列表或张量
            batch_first=True,     # 输入(batch_size, seq_length, embedding_dim)
            enforce_sorted=False  # 输入的序列不需要事先按长度递减排序
        )
        lstm_out, _ = self.lstm(packed_embedded)
        # pad_packed_sequence 函数用于将LSTM的输出从打包格式还原为填充的序列格式
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.norm(lstm_out)

        # LSTM特征池化
        lstm_out = self.avg_lstm(lstm_out.transpose(-1, -2))
        # print("lstm",lstm_out.shape)

        # 特征拼接与输出
        output = torch.cat([lstm_out[:, :, 0], linear_out[:, :, 0]], dim=-1)
        # print("out",output.shape)

        lstm_feats = self.hidden2tag(output)

        return lstm_feats


def load_encoded_data(encoding_file_path=encoding_file):
    """加载编码后的文本数据和标签"""
    values = []
    labels = []

    with open(encoding_file_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
        np.random.shuffle(lines)  # 打乱样本数据

        for line in lines:
            data, label = line.split("\t")
            # 解析编码数据
            val = [int(word) for word in data.split(",") if word.isdigit()]

            # 序列长度统一化（填充或截断）
            for _ in range(max_lens - len(val)):
                val.append(0)
            val = val[:max_lens]

            values.append(val)
            labels.append(int(label))

    return np.array(values, dtype="int32"), np.array(labels, dtype="int64")