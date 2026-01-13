from torch import reshape
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn


class SimpleCNN(nn.Module):
    """一个非常简洁的卷积神经网络（CNN），用于 CIFAR10 分类。"""

    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool2d(
           kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = self.max_pool(x)
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
        writer.add_images("Maxpool Batch Origin", images, step)
        writer.add_images("Maxpool Batch Output", outputs, step)
        step += 1
    writer.close()


if __name__ == "__main__":
    main()
