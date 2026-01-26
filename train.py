import torch
import torch.nn as nn
import os

import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard.writer import SummaryWriter
import torch_directml


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 检查GPU是否可用
device = torch_directml.device()
print(f"Using device: {device}")
model = CIFAR10CNN().to(device)


tensorboard = SummaryWriter("./logs")


def main():
    train_dataset = datasets.CIFAR10(
        root="./dataset", train=True, download=True, transform=transforms.ToTensor()
    )
    test_dataset = datasets.CIFAR10(
        root="./dataset", train=False, download=True, transform=transforms.ToTensor()
    )

    # 开始训练

    train_epoch = 20
    # 当前训练的次数
    current_train_step = 1

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    time_start = time.time()
    for epoch in range(train_epoch):
        print(f"=============第{epoch}轮训练开始=============")
        dataloader = DataLoader(train_dataset, batch_size=64)
        total_loss = 0
        for batch in dataloader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            # 前向传播
            outputs = model(images)
            # 计算损失
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            print(outputs.shape)
            # 梯度清零，否则训练时梯度会累加，导致不准
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 根据计算出的梯度更新模型参数
            optimizer.step()
            # 打印训练信息
            if current_train_step % 100 == 0:
                print(f"第{current_train_step}次训练，损失为:{loss.item():.4f}")
                tensorboard.add_scalar("train_loss", loss.item(), current_train_step)
            current_train_step += 1
        print(
            f"=============第{epoch}轮训练结束，总的损失为{total_loss:.4f}============="
        )

        # 开始测试阶段
        model.eval()
        with torch.no_grad():
            test_dataloader = DataLoader(test_dataset, batch_size=64)
            total_accuracy = 0
            for batch in test_dataloader:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                accuracy = (outputs.argmax(1) == labels).sum().item()
                total_accuracy += accuracy
            print(
                f"=============第{epoch + 1}轮测试结束，准确率为{total_accuracy / len(test_dataset):.4f}============="
            )
            tensorboard.add_scalar(
                "test_accuracy",
                total_accuracy / len(test_dataset),
                epoch + 1,
            )
        # 保存模型
        if (epoch + 1) % 10 == 0:
            if not os.path.exists("./train_models"):
                os.makedirs("./train_models")
            torch.save(
                model.state_dict(), f"./train_models/cifar10_cnn_{epoch + 1}.pth"
            )

    time_end = time.time()
    print(f"训练耗时{time_end - time_start:.4f}秒")

    tensorboard.close()


if __name__ == "__main__":
    print("开始训练")
    main()
