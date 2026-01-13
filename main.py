import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class CIFAR10CNN(nn.Module):
    """一个用于CIFAR10分类的卷积神经网络（CNN），结构如下：
    - 输入：3@32x32（3通道RGB图像，32x32像素）
    - 第一层卷积：5x5卷积核，32个输出通道，padding=2保持尺寸
    - 第一层池化：2x2最大池化，步长2，尺寸减半
    - 第二层卷积：5x5卷积核，32个输出通道，padding=2保持尺寸
    - 第二层池化：2x2最大池化，步长2，尺寸减半
    - 第三层卷积：5x5卷积核，64个输出通道，padding=2保持尺寸
    - 第三层池化：2x2最大池化，步长2，尺寸减半
    - 扁平层：将64@4x4展平为1024个特征
    - 全连接层：1024输入，10输出（对应CIFAR10的10个类别）
    """

    def __init__(self):
        super().__init__()
        #  (32 - 5 + 2*padding)/1 + 1 = 32 => padding = 2
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,padding=2)
        # (32 - 2)/s+1 = 16 => s = 2
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # (16 - 5 + 2*p) +1 = 16 => padding = 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)

        # (16 - 2)/s +1 = 8 => s = 2
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # (8 - 5 + 2*p) +1 = 8 => padding = 2
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)

        # (8 - 2)/s +1 = 4 => s = 2
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.linear = nn.Linear(64 * 4 * 4, 10)




        
       

    def forward(self, x):
        # 输入形状: [batch_size, 3, 32, 32]
        
        # 第一层卷积 + 激活 + 池化
        x = self.conv1(x)  # 输出形状: [batch_size, 32, 32, 32]
        x = self.p1(x)               # 输出形状: [batch_size, 32, 16, 16]
        
        # 第二层卷积 + 激活 + 池化
        x = self.conv2(x)  # 输出形状: [batch_size, 32, 16, 16]
        x = self.p2(x)               # 输出形状: [batch_size, 32, 8, 8]
        
        # 第三层卷积 + 激活 + 池化
        x = self.conv3(x)  # 输出形状: [batch_size, 64, 8, 8]
        x = self.p3(x)               # 输出形状: [batch_size, 64, 4, 4]
        
        # 扁平层
        x = torch.flatten(x, 1)        # 输出形状: [batch_size, 64*4*4=1024]
        
        # 全连接层
        x = self.linear(x)                 # 输出形状: [batch_size, 10]
        
        return x


def main():
    dataset = datasets.CIFAR10(root="./dataset", train=False, download=True, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=1,shuffle=False)
    for batch in dataloader:
        images, labels = batch
        model = CIFAR10CNN()
        # 前向传播
        outputs = model(images)
        # 找到概率最高的类别索引
        pred_idx = outputs[0].argmax().item()
        print(labels,"预测类别索引:", pred_idx, "对应类别:", dataset.classes[pred_idx])

        break


if __name__ == "__main__":
    main()
