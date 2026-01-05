from torch import reshape
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn


class SimpleCNN(nn.Module):
    """一个非常简洁的卷积神经网络（CNN），用于 CIFAR10 分类。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


def main():
    train_dataset = datasets.CIFAR10(
        root="./dataset", train=False, download=True, transform=transforms.ToTensor()
    )
    writer = SummaryWriter("logs")
    dataloader = DataLoader(train_dataset, batch_size=64)
    step = 0
    for batch in dataloader:
        images, labels = batch
        model = SimpleCNN()
        # 前向传播
        outputs = model(images)
        print(outputs.shape)
        # torch.Size([64, 6, 32, 32])
        # 因为 add_images 要求输入为 [N, C, H, W]，而当前 outputs 是 [64, 6, 32, 32]（N=64, C=6），
        # 为了把 6 个通道拆成 3 通道，需要把 6 拆成 2 组 3 通道，于是把 N 和 C 合并成 64×2=128 张图，
        # 即 reshape 成 [128, 3, 32, 32]
        outputs = reshape(outputs, (-1, 3, 32, 32))
        writer.add_images("Batch Origin2", images, step)
        writer.add_images("Batch Outputs2", outputs, step)
        print(step)
        step += 1
    writer.close()


if __name__ == "__main__":
    main()
