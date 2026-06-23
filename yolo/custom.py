import os
import torch
import torch.nn as nn
import xmltodict
from torchvision.models import vgg16, VGG16_Weights


class MyYolo(nn.Module):
    def __init__(self):
        super().__init__()

        # self.backbone = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     # 第二层卷积：通道从 32 -> 64
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     # 第三层卷积：通道从 64 -> 128
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     # 第四层卷积：通道从 128 -> 256
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        # )

        self.backbone = vgg16(weights=VGG16_Weights.DEFAULT).features

        # 自适应平均池化层：无论输入特征图的空间尺寸是多少，都能输出指定大小的特征图。
        self.pool = nn.AdaptiveAvgPool2d((10, 10))
        self.share = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 10 * 10, 1024),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.share(x)
        return x


def get_dataset():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, "custom_dataset")
    annotation_dir = os.path.join(dataset_dir, "annotation")
    images_dir = os.path.join(dataset_dir, "images")

    # 定义类别
    # 分别是坦克模型、高级咖啡豆、信息终端以及量子存储
    classes = ["tank", "coffee_bean", "info_device", "quantum_memory"]

    xml_files = [f for f in os.listdir(annotation_dir) if f.endswith(".xml")]

    for name in xml_files:
        xml_path = os.path.join(annotation_dir, name)
        with open(xml_path, "r", encoding="utf-8") as f:
            data = xmltodict.parse(f.read())
        # 提取位置信息

        


def main():
    # model = MyYolo()
    # output = model(torch.randn(1, 3, 64, 64))
    # print(output.shape)
    get_dataset()


if __name__ == "__main__":
    main()
