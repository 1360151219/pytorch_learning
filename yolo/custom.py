import os
import torch
import torch.nn as nn
import xmltodict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms


class MyYolo(nn.Module):
    def __init__(self, classes_length):
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

        self.bbox_head = nn.Linear(1024, 4)
        self.class_head = nn.Linear(1024, classes_length)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.share(x)
        bbox_logit = self.bbox_head(x)
        class_logit = self.class_head(x)
        return bbox_logit, class_logit


current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, "custom_dataset")
annotation_dir = os.path.join(dataset_dir, "annotation")
images_dir = os.path.join(dataset_dir, "images")
my_annotation_dir = os.path.join(dataset_dir, "my_annotation")
# 定义类别
# 分别是坦克模型、高级咖啡豆、信息终端以及量子存储
classes = ["tank", "coffee_bean", "info_device", "quantum_memory"]


class MyDataset(Dataset):
    def __init__(self) -> None:
        self.files = os.listdir(my_annotation_dir)
        print(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        trans = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        name = self.files[index]
        file_name = os.path.splitext(name)[0]

        with open(
            os.path.join(my_annotation_dir, file_name + ".txt"), "r", encoding="utf-8"
        ) as f:
            content = f.read()
            # 单目标
            content = content.split("\n")[0].split(" ")
        target_str_bbox = content[0:4]
        target_str_class = content[4:]

        target_bbox = torch.tensor(
            [float(i) for i in target_str_bbox], dtype=torch.float32
        )

        # target_class = torch.tensor(
        #     [float(i) for i in target_str_class], dtype=torch.float32
        # )
        # 转换为类别索引
        target_class = torch.tensor(target_str_class.index("1"), dtype=torch.long)
        image = trans(
            Image.open(os.path.join(images_dir, file_name + ".png")).convert("RGB")
        )

        return image, target_bbox, target_class


# VOC 数据格式转
def get_dataset():
    if not os.path.exists(my_annotation_dir):
        os.makedirs(my_annotation_dir)

    xml_files = [f for f in os.listdir(annotation_dir) if f.endswith(".xml")]

    for name in xml_files:
        xml_path = os.path.join(annotation_dir, name)
        my_file_path = os.path.join(
            my_annotation_dir, os.path.splitext(name)[0] + ".txt"
        )
        with open(xml_path, "r", encoding="utf-8") as f:
            data = xmltodict.parse(f.read())
        # 提取位置信息
        # print(data["annotation"])
        annotation = data["annotation"]
        file_width = float(annotation["size"]["width"])
        file_height = float(annotation["size"]["height"])
        object = annotation["object"]

        if isinstance(object, dict):
            object = [object]

        my_line = []

        for obj in object:
            label_name = obj["name"]
            label_index = classes.index(label_name)
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            # 输出转换
            x_center = (xmin + xmax) / 2.0 / file_width
            y_center = (ymin + ymax) / 2.0 / file_height

            width = (xmax - xmin) / file_width
            height = (ymax - ymin) / file_height

            # one_hot 格式化
            one_hot = [0] * len(classes)
            one_hot[label_index] = 1
            one_hot_str = " ".join(str(h) for h in one_hot)

            my_line.append(
                f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {one_hot_str}"
            )
        with open(my_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(my_line))


def main():
    model = MyYolo(len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    class_loss_fn = nn.CrossEntropyLoss()
    bbox_loss_fn = nn.MSELoss()
    # output = model(torch.randn(1, 3, 64, 64))
    # print(output.shape)
    # get_dataset()
    dataloader = DataLoader(MyDataset(), batch_size=4, shuffle=True)

    for epoch in range(20):
        total_loss = 0
        for image, target_bbox, target_class in dataloader:
            predict_bbox, predict_class = model(image)

            bbox_loss = bbox_loss_fn(predict_bbox, target_bbox)
            class_loss = class_loss_fn(predict_class, target_class)

            loss = bbox_loss + class_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"epoch {epoch}, loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "my_model.pth")


if __name__ == "__main__":
    main()
