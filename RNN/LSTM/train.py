import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from matplotlib import pyplot as plt
from model import LSTMModel, get_dict_len, load_encoded_data, device, dict_file

# 训练相关配置
train_rate = 0.8  # 训练集比例
lr = 1e-4  # 学习率
epochs = 20  # 训练轮数


def main():
    # 设置随机种子确保可复现性
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed_all(42)

    # 加载并准备数据
    values, labels = load_encoded_data()
    train_x, test_x, train_y, test_y = train_test_split(
        values, labels,
        train_size=train_rate,
        random_state=1
    )

    # 转换为Tensor并创建数据加载器
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    # 初始化模型、优化器和损失函数
    vocab_size = get_dict_len(dict_file)
    model = LSTMModel(vocab_size, 2)  # 二分类任务
    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fc = nn.CrossEntropyLoss()

    # 训练过程记录
    best_loss = float('inf')
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    train_result, test_result = [], []

    # 开始训练
    for epoch in range(1, epochs + 1):
        # 训练阶段
        train_pbar = tqdm(train_loader)
        train_loss_all, train_acc_all = 0.0, 0.0

        for step, (x, y) in enumerate(train_pbar):
            x, y = x.to(device), y.to(device)
            lengths = torch.sum(x != 0, dim=-1).cpu().long()  # 计算序列实际长度

            # 前向传播
            out = model(x, lengths)
            loss = loss_fc(out, y)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算指标
            train_loss_all += loss.item()
            train_loss_avg = train_loss_all / (step + 1)

            acc = torch.mean((y == torch.argmax(out, dim=-1)).float())
            train_acc_all += acc.item()
            train_acc_avg = train_acc_all / (step + 1)

            # 更新进度条
            desc = f"train => epoch:{epoch} - step:{step} - loss:{loss:.4f} - avg_loss:{train_loss_avg:.4f} - acc:{acc:.4f} - avg_acc:{train_acc_avg:.4f}"
            train_pbar.set_description(desc)
            train_result.append(desc + "\n")

        # 记录训练指标
        train_losses.append(train_loss_avg)
        train_accuracies.append(train_acc_avg)

        # 测试阶段（无梯度计算）
        with torch.no_grad():
            test_pbar = tqdm(test_loader)
            test_loss_all, test_acc_all = 0.0, 0.0

            for step, (x, y) in enumerate(test_pbar):
                x, y = x.to(device), y.to(device)
                lengths = torch.sum(x != 0, dim=-1).cpu().long()

                out = model(x, lengths)
                loss = loss_fc(out, y)

                # 计算指标
                test_loss_all += loss.item()
                test_loss_avg = test_loss_all / (step + 1)

                acc = torch.mean((y == torch.argmax(out, dim=-1)).float())
                test_acc_all += acc.item()
                test_acc_avg = test_acc_all / (step + 1)

                # 更新进度条
                desc = f"test => epoch:{epoch} - step:{step} - loss:{loss:.4f} - avg_loss:{test_loss_avg:.4f} - acc:{acc:.4f} - avg_acc:{test_acc_avg:.4f}"
                test_pbar.set_description(desc)
                test_result.append(desc + "\n")

        # 记录测试指标
        test_losses.append(test_loss_avg)
        test_accuracies.append(test_acc_avg)

        # 保存结果
        with open("train_result.txt", "w") as f:
            f.writelines(train_result)
        with open("test_result.txt", "w") as f:
            f.writelines(test_result)

        # 保存最佳模型
        if test_loss_avg < best_loss:
            best_loss = test_loss_avg
            torch.save(model.state_dict(), "model.pkl")
            print(f"Epoch {epoch} 保存最佳模型，测试损失: {best_loss:.4f}")

    # 可视化训练过程
    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies)


def plot_metrics(train_losses, test_losses, train_acc, test_acc):
    """可视化训练和测试的损失与准确率"""
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    # 损失图
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='训练损失')
    plt.plot(epochs_range, test_losses, label='测试损失')
    plt.xlabel('轮次')
    plt.ylabel('损失值')
    plt.title('训练与测试损失')
    plt.legend()

    # 准确率图
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc, label='训练准确率')
    plt.plot(epochs_range, test_acc, label='测试准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.title('训练与测试准确率')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_testing_metrics.png")
    plt.show()


if __name__ == "__main__":
    main()