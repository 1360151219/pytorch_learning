import os
import yaml
import xmltodict
import torch
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms


class CustomVocDataset:
    def __init__(self, images_dir, labels_dir, img_transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_transform = img_transform
        self.classes = ["tank", "coffee_bean", "info_device", "quantum_memory"]

    def __len__(self):
        return len(os.listdir(self.images_dir))

    def __getitem__(self, idx):
        img_name = os.listdir(self.images_dir)[idx]
        img_id = img_name.split(".")[0]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_id + ".xml")

        # 读取图片和标注文件
        image = Image.open(img_path)
        if self.img_transform is not None:
            image = self.img_transform(image)
        with open(label_path, "r") as f:
            content = f.read()
            content = xmltodict.parse(content)
            objects = content["annotation"]["object"]
            if isinstance(objects, dict):
                objects = [objects]
            labels = []
            for obj in objects:
                label = obj["name"]
                bndbox = obj["bndbox"]
                label_index = self.classes.index(label)
                # {'label': 'tank', 'bndbox': {'xmin': '374', 'ymin': '12', 'xmax': '726', 'ymax': '358'}}
                labels.append(
                    {"label": label, "label_id": label_index, "bndbox": bndbox}
                )

        return image, labels


def convert_voc_to_yolo(annotations_dir, labels_dir, classes):
    """
    将 VOC XML 格式的标注转换为 YOLO TXT 格式
    """
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    print(f"正在将标注从 {annotations_dir} 转换为 YOLO 格式并保存到 {labels_dir}...")

    # 获取所有 XML 文件
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith(".xml")]

    for xml_file in xml_files:
        xml_path = os.path.join(annotations_dir, xml_file)

        with open(xml_path, "r", encoding="utf-8") as f:
            xml_content = f.read()
            data = xmltodict.parse(xml_content)

        # 获取图片尺寸
        image_width = float(data["annotation"]["size"]["width"])
        image_height = float(data["annotation"]["size"]["height"])

        # 获取文件名（不含扩展名）
        file_id = os.path.splitext(xml_file)[0]
        txt_path = os.path.join(labels_dir, f"{file_id}.txt")

        # 处理对象
        objects = data["annotation"].get("object", [])
        if isinstance(objects, dict):
            objects = [objects]

        yolo_lines = []
        for obj in objects:
            cls_name = obj["name"]
            if cls_name not in classes:
                continue

            cls_id = classes.index(cls_name)
            bndbox = obj["bndbox"]

            xmin = float(bndbox["xmin"])
            ymin = float(bndbox["ymin"])
            xmax = float(bndbox["xmax"])
            ymax = float(bndbox["ymax"])

            # 转换为 YOLO 格式 (x_center, y_center, width, height) 归一化
            x_center = (xmin + xmax) / 2.0 / image_width
            y_center = (ymin + ymax) / 2.0 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            yolo_lines.append(
                f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        # 写入 txt 文件
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

    print("转换完成！")


def create_data_yaml(data_dir, classes):
    """
    创建 YOLO 训练所需的 data.yaml 文件
    """
    yaml_content = {
        "path": data_dir,  # 数据集根目录
        "train": "images",  # 训练集图片目录 (相对于 path)
        "val": "images",  # 验证集图片目录 (这里简单起见使用同一份数据，实际应划分)
        "names": {i: name for i, name in enumerate(classes)},
    }

    yaml_path = os.path.join(data_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, allow_unicode=True)

    return yaml_path


def custom_collate_fn(batch):
    """
    自定义整理函数，用于处理 DataLoader 中的批次数据。
    因为目标检测中每张图片的边界框数量不同，所以不能简单的 stack。
    """
    images = []
    targets = []
    for img, label in batch:
        images.append(img)
        targets.append(label)
    return images, targets


def verify_dataset_with_dataloader(images_dir, annotations_dir):
    """
    使用 PyTorch 原生 DataLoader 验证 CustomVocDataset 是否能正确加载数据
    """
    print("-" * 30)
    print("正在使用 DataLoader 验证数据集...")

    # 定义变换：调整大小并转为 Tensor
    transform = transforms.Compose(
        [transforms.Resize((640, 640)), transforms.ToTensor()]
    )

    # 实例化自定义数据集
    # 注意：这里我们传入的是 XML 所在的 annotations_dir
    dataset = CustomVocDataset(images_dir, annotations_dir, img_transform=transform)

    # 实例化 DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn
    )

    try:
        # 尝试加载一个 batch
        images, targets = next(iter(dataloader))
        print(f"成功加载一个 Batch！")
        print(f"Batch 图片数量: {len(images)}")
        print(f"第一张图片 Tensor 形状: {images[0].shape}")
        print(f"第一张图片对应的标注数量: {len(targets[0])}")
        if len(targets[0]) > 0:
            print(f"第一张图片的第一个标注: {targets[0][0]}")
        print("Dataset 和 DataLoader 工作正常！")
    except Exception as e:
        print(f"DataLoader 加载失败: {e}")
        import traceback

        traceback.print_exc()
    print("-" * 30)


def run_inference(model_path, images_path: list):
    """
    加载训练好的模型并进行推理验证
    """
    print("-" * 30)
    print(f"开始使用模型 {model_path} 进行推理验证...")

    if not os.path.exists(model_path):
        print(f"警告：模型文件 {model_path} 不存在。可能训练尚未完成或路径错误。")
        return

    # 加载模型
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 推理
    # conf=0.25 是置信度阈值
    # save=True 会自动保存带有边框的图片
    results = model.predict(
        source=images_path,
        save=True,
        conf=0.25,
        project=os.path.dirname(
            os.path.dirname(model_path)
        ),  # 保存到与 train_results 同级的目录
        name="inference_demo",
    )

    print(f"推理完成！")
    for result in results:
        save_dir = result.save_dir
        print(f"结果已保存到: {save_dir}")
        boxes = result.boxes
        print(f"检测到 {len(boxes)} 个目标:")
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            print(f"  - 类别: {cls_name}, 置信度: {conf:.2f}")
    print("-" * 30)


def main():
    # 路径配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(current_dir, "custom_dataset")
    images_dir = os.path.join(dataset_root, "images")
    annotations_dir = os.path.join(dataset_root, "annotation")
    labels_dir = os.path.join(dataset_root, "labels")

    # 定义类别
    classes = ["tank", "coffee_bean", "info_device", "quantum_memory"]

    # 1. 数据准备：将 XML 转为 YOLO 格式 TXT
    # 检查 labels 目录是否为空或不存在，如果需要则进行转换
    if not os.path.exists(labels_dir) or not os.listdir(labels_dir):
        convert_voc_to_yolo(images_dir, annotations_dir, labels_dir, classes)

    # 2. 创建 data.yaml 配置文件
    yaml_path = create_data_yaml(dataset_root, classes)

    # 3. 加载模型并开始训练
    model_path = os.path.join(current_dir, "yolo26n.pt")
    # 如果本地没有预训练模型，YOLO 会自动下载 yolov8n.pt，或者你可以指定其他模型
    # 注意：如果 'yolo26n.pt' 是你自定义的模型名，请确保它存在

    print("开始训练...")
    try:
        model = YOLO(model_path)
    except Exception:
        print(f"未找到 {model_path}，尝试使用 yolov8n.pt")
        model = YOLO("yolo26n.pt")

    # 定义训练结果保存的绝对路径
    train_project_dir = os.path.join(current_dir, "train_results")

    # 训练参数设置
    results = model.train(
        data=yaml_path,
        epochs=100,  # 训练轮数，可根据需要调整
        imgsz=1000,  # 图片大小
        batch=4,  # 批次大小，显存小可以调小
        project=train_project_dir,  # 训练结果保存目录 (绝对路径)
        name="custom_train",  # 训练实验名称
    )

    # print(f"训练完成！结果保存在 {os.path.join(train_project_dir, 'custom_train')} 中")

    # 4. 模型推理验证
    # 训练完成后，最佳模型通常保存在 weights/best.pt
    best_model_path = os.path.join(
        train_project_dir,
        "custom_train",
        "weights",
        "best.pt",
    )

    # 选择一张图片进行测试，这里简单取 images 目录下的第一张图片

    test_image_path = os.path.join(images_dir)
    images_path = [
        os.path.join(test_image_path, f)
        for f in os.listdir(test_image_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    # run_inference(best_model_path, images_path)
    print(yaml_path)


if __name__ == "__main__":
    main()
