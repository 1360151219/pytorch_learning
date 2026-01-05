import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def get_device():
    """获取当前可用的设备（GPU/MPS/CPU）。

    说明：
    - 在苹果芯片的 Mac 上若安装了 PyTorch 的 MPS 后端，会优先使用 mps；
    - 若有支持的 Nvidia GPU 则使用 cuda；
    - 否则回落到 cpu。
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_dataloaders(batch_size: int = 64):
    """创建并返回 CIFAR10 的训练与测试 DataLoader。

    参数：
    - batch_size：每个批次的样本数量。

    返回：
    - train_loader：训练集 DataLoader
    - test_loader：测试集 DataLoader

    关键点讲解：
    - CIFAR10 是 32x32 的彩色图像，共 10 类；
    - transforms.ToTensor 会将 [0,255] 的像素转为 [0,1] 的张量；
    - Normalize 按通道做标准化，有助于加速训练与稳定收敛；
    - DataLoader 将数据打包成批次，并在训练集上 shuffle 打乱顺序。
    """
    # 官方常用的 CIFAR10 标准化参数（RGB 三通道）
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    # 定义训练与测试的图像预处理流水线
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),  # 转为张量，范围变为 [0,1]
        ]
    )

    # 加载数据集（若本地无数据会自动下载到 ./dataset 目录）
    train_dataset = datasets.CIFAR10(
        root="./dataset", train=True, transform=train_transform, download=True
    )

    # 使用 DataLoader 对数据进行批处理与打乱（训练集）
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    # 打印一些基础信息，帮助理解
    print(f"训练集样本数量: {len(train_dataset)}, 类别: {train_dataset.classes}")

    return train_loader


class SimpleCNN(nn.Module):
    """一个非常简洁的卷积神经网络（CNN），用于 CIFAR10 分类。

    模型结构：
    - Conv2d(3->16, kernel=3, padding=1) + ReLU
    - MaxPool2d(kernel=2, stride=2)      # 尺寸从 32x32 -> 16x16
    - Conv2d(16->32, kernel=3, padding=1) + ReLU
    - MaxPool2d(kernel=2, stride=2)      # 尺寸从 16x16 -> 8x8
    - Flatten + Linear(32*8*8 -> 128) + ReLU
    - Linear(128 -> 10) 输出 10 类的 logits
    """

    def __init__(self):
        super().__init__()
        # 第一层卷积：输入通道 3（RGB），输出通道 16，3x3 卷积核，padding=1 保持尺寸
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        # 第二层卷积：通道从 16 -> 32
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        # 2x2 的最大池化，降低特征图尺寸同时引入平移不变性
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层，将卷积提取到的特征映射到类别空间
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """前向计算流程。

        说明：
        - 卷积 + 非线性激活提取空间特征；
        - 池化减小尺寸防止过拟合；
        - 展平后通过全连接层进行最终分类。
        """
        x = F.relu(self.conv1(x))  # 32x32 -> 32x32（因为 padding=1）
        x = self.pool(x)  # 32x32 -> 16x16
        x = F.relu(self.conv2(x))  # 16x16 -> 16x16
        x = self.pool(x)  # 16x16 -> 8x8
        x = torch.flatten(x, 1)  # 展平为 [batch, 32*8*8]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 输出未归一化的 logits（交叉熵会内部做 softmax）
        return x


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """训练模型一个 epoch，并返回平均损失与准确率。

    关键点：
    - model.train() 打开训练模式（使 BN/Dropout 等按训练逻辑工作）；
    - 将数据与标签移到相同设备上；
    - 计算损失，反向传播，优化更新；
    - 记录训练过程中的准确率。
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # 清空上一轮的梯度
        outputs = model(images)  # 前向计算得到 logits
        loss = criterion(outputs, labels)  # 交叉熵损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device):
    """在验证/测试集上评估模型，返回平均损失与准确率。"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def demo_naive_convolution():
    """演示一个“朴素卷积”的实现与效果（教学用途）。

    内容：
    - 手写一个 2D 卷积的核心逻辑（单通道、无 padding、stride=1）；
    - 与 nn.Conv2d 的结果进行对比，帮助理解卷积含义与维度变化。
    """
    # 构造一个简单的单通道输入（1 张 5x5 的图像）
    x = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [5.0, 4.0, 3.0, 2.0, 1.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [5.0, 4.0, 3.0, 2.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ]
    )  # 形状 [1, 5, 5]

    # 增加批次与通道维度，使形状变为 [N=1, C=1, H=5, W=5]
    x = x.unsqueeze(0).unsqueeze(0)

    # 定义一个 3x3 的卷积核（单输入通道 -> 单输出通道）
    kernel = torch.tensor(
        [
            [
                [1.0, 0.0, -1.0],
                [1.0, 0.0, -1.0],
                [1.0, 0.0, -1.0],
            ]
        ]
    )  # 形状 [1, 3, 3]

    # 朴素卷积：用滑动窗口计算每个位置的加权和
    def naive_conv2d(
        input_1n1hw: torch.Tensor, kernel_1kk: torch.Tensor
    ) -> torch.Tensor:
        """一个不使用 PyTorch 高级 API 的朴素卷积实现。

        参数：
        - input_1n1hw：形状 [1, 1, H, W]
        - kernel_1kk：形状 [1, K, K]

        返回：
        - out：形状 [1, 1, H-K+1, W-K+1]
        """
        N, _, H, W = input_1n1hw.shape
        K = kernel_1kk.shape[-1]
        out_h, out_w = H - K + 1, W - K + 1
        out = torch.zeros((N, 1, out_h, out_w), dtype=input_1n1hw.dtype)
        # 双层循环滑动窗口
        for i in range(out_h):
            for j in range(out_w):
                window = input_1n1hw[:, :, i : i + K, j : j + K]  # 取出 KxK 的局部区域
                out[:, :, i, j] = (window * kernel_1kk).sum(dim=(1, 2, 3))
        return out

    out_naive = naive_conv2d(x, kernel)
    print(
        "朴素卷积输出形状:", tuple(out_naive.shape), "\n", out_naive.squeeze().numpy()
    )

    # 使用 nn.Conv2d 做对照（权重设为与 kernel 一致，偏置为 0）
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)
    with torch.no_grad():
        conv.weight[:] = kernel.unsqueeze(0)  # 形状匹配为 [out_c, in_c, K, K]
    out_torch = conv(x)
    print(
        "nn.Conv2d 输出形状:",
        tuple(out_torch.shape),
        "\n",
        out_torch.squeeze().detach().numpy(),
    )


def main():
    """整合 DataLoader、CNN 搭建与训练的完整示例，并演示简单卷积。"""
    device = get_device()
    print(f"使用设备: {device}")

    # 1) 准备数据与 DataLoader
    train_loader, test_loader = create_dataloaders(batch_size=128)

    # 2) 搭建模型、优化器
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3) 训练与评估（演示 1 个 epoch，可自行调大）
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, device)

    print(f"训练集: loss={train_loss:.4f}, acc={train_acc:.4f}")
    print(f"测试集: loss={test_loss:.4f}, acc={test_acc:.4f}")

    # 4) 额外演示：一个朴素卷积的实现与 nn.Conv2d 对比
    print("\n====== 朴素卷积演示 ======")
    demo_naive_convolution()


if __name__ == "__main__":
    main()
