import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms

from eval_utils import compute_iou, visualize_batch
from voc_convert import voc_to_yolo


class MyYolo(nn.Module):
    def __init__(self, classes_length):
        super().__init__()

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
        if not os.path.isdir(my_annotation_dir):
            raise FileNotFoundError(
                f"{my_annotation_dir} 不存在，请先运行 voc_to_yolo() 生成标签。"
            )
        self.files = sorted(
            f for f in os.listdir(my_annotation_dir) if f.endswith(".txt")
        )
        if not self.files:
            raise ValueError(f"{my_annotation_dir} 中没有可用的 .txt 标签文件。")

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
            content = content.splitlines()[0].split()
        target_str_bbox = content[0:4]
        target_str_class = content[4:]

        target_bbox = torch.tensor(
            [float(i) for i in target_str_bbox], dtype=torch.float32
        )

        # 转换为类别索引
        target_class = torch.tensor(target_str_class.index("1"), dtype=torch.long)
        image = trans(
            Image.open(os.path.join(images_dir, file_name + ".png")).convert("RGB")
        )

        return image, target_bbox, target_class


def main():
    model = MyYolo(len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    class_loss_fn = nn.CrossEntropyLoss()
    bbox_loss_fn = nn.MSELoss()

    # voc_to_yolo(annotation_dir, my_annotation_dir, classes)
    dataset = MyDataset()
    if len(dataset) < 2:
        raise ValueError("至少需要 2 个样本，才能划分训练集和验证集。")

    # 8:2 划分训练集与验证集
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False)

    best_val_loss = float("inf")
    save_path = os.path.join(current_dir, "my_model.pth")
    vis_dir = os.path.join(current_dir, "val_vis")

    for epoch in range(20):
        # 训练
        model.train()
        train_loss = 0
        for image, target_bbox, target_class in train_loader:
            predict_bbox, predict_class = model(image)

            bbox_loss = bbox_loss_fn(predict_bbox, target_bbox)
            class_loss = class_loss_fn(predict_class, target_class)

            loss = bbox_loss + class_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        iou_sum = 0.0
        iou_count = 0
        val_batches = []
        with torch.no_grad():
            for image, target_bbox, target_class in val_loader:
                predict_bbox, predict_class = model(image)
                bbox_loss = bbox_loss_fn(predict_bbox, target_bbox)
                class_loss = class_loss_fn(predict_class, target_class)
                val_loss += (bbox_loss + class_loss).item()

                ious = compute_iou(predict_bbox, target_bbox)
                iou_sum += ious.sum().item()
                iou_count += ious.numel()
                val_batches.append(
                    (
                        image,
                        predict_bbox,
                        target_bbox,
                        predict_class,
                        target_class,
                        ious,
                    )
                )

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        avg_iou = iou_sum / max(iou_count, 1)
        print(
            f"epoch {epoch}, train_loss: {avg_train:.4f}, "
            f"val_loss: {avg_val:.4f}, val_iou: {avg_iou:.4f}"
        )

        # 保留验证 loss 最低的权重，并落盘当轮的可视化结果
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            epoch_vis_dir = os.path.join(vis_dir, f"epoch_{epoch}")
            for batch_idx, batch in enumerate(val_batches):
                visualize_batch(
                    *batch, classes=classes, save_dir=epoch_vis_dir, batch_idx=batch_idx
                )
            print(
                f"  -> best model saved (val_loss: {best_val_loss:.4f}), "
                f"vis -> {epoch_vis_dir}"
            )


if __name__ == "__main__":
    main()
