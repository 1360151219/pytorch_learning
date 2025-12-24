# PyTorch Learning

A repository for learning PyTorch concepts and implementing various examples.

## torchvision 介绍与使用指南

### 1. torchvision 基本概念

`Torchvision` 是 `PyTorch` 的一个重要扩展库，专门用于计算机视觉任务，提供了丰富的工具和资源，主要包括以下核心组件：

官方文档：[Torchvision 官方文档](https://docs.pytorch.org/vision/stable/index.html)

- **torchvision.datasets**: 提供常用的计算机视觉数据集下载和加载功能，如MNIST、COCO、VOC等
- **torchvision.io**: 提供视频读写、图像读写和编码解码功能，支持JPEG、PNG等格式
- **torchvision.models**: 提供预训练的深度学习模型，如AlexNet、VGG、ResNet、Inception等
- **torchvision.ops**: 提供计算机视觉相关的操作函数
- **torchvision.transforms**: 提供图像预处理和数据增强的功能
- **torchvision.utils**: 提供一些实用工具函数

#### 1.1. TensorBoard 可视化图像变换

`TensorBoard` 是一个强大的可视化工具，可以用于展示图像变换的过程。在后续的深度学习代码中，我们会频繁的使用到这个工具，来观察图像的各种变换过程。

在示例代码中，我们使用 `TensorBoard` 来可视化各种 `Transforms` 对图像的影响。

**使用方法**

1. 安装 `TensorBoard`：`pip install tensorboard`
2. 在代码中创建 `SummaryWriter`：
   ```python
   from torch.utils.tensorboard import SummaryWriter
   writer = SummaryWriter("logs")
   ```
3. 写入图像数据：
   ```python
   writer.add_image("Image Title", image_tensor, global_step)
   ```
4. 运行TensorBoard：`tensorboard --logdir=logs`
5. 在浏览器中访问TensorBoard界面（通常是http://localhost:6006）

#### 1.2. Transforms 的基本使用

`Transforms` 是 `torchvision` 中用于图像预处理和数据增强的重要模块，可以对图像进行各种变换操作。

以下是一些常用的 `Transforms`：

#### 1.3. 常用的 `Transforms` 操作

- **ToTensor()**: 将 PIL 图像或 numpy 数组转换为 Tensor 格式。这个操作我们非常常用，因为它可以将图像转换为 PyTorch 中的张量格式，方便后续的深度学习模型处理。

- **Normalize(mean, std)**: 对Tensor图像进行标准化（归一化）。所谓归一化是指将图像的像素值从原始范围（通常是0到255）映射到标准范围（通常是-1到1或0到1），以提高模型的训练效果和收敛速度。

`output[channel] = (input[channel] - mean[channel]) / std[channel]`

这样的话可以使得图片颜色分布更加均匀，提高模型的泛化能力，有助于模型的训练。

- **Resize(size)**: 调整图像尺寸。这个操作可以将图像调整为指定的尺寸，通常用于图像预处理或模型输入。

---

下面还有一些操作

- **CenterCrop(size)**: 从图像中心裁剪指定尺寸
- **RandomCrop(size)**: 随机裁剪指定尺寸
- **RandomHorizontalFlip(p=0.5)**: 随机水平翻转图像
- **RandomVerticalFlip(p=0.5)**: 随机垂直翻转图像
- **ColorJitter(brightness, contrast, saturation, hue)**: 随机调整图像的亮度、对比度、饱和度和色调


可以使用`transforms.Compose()`将多个Transforms组合在一起，形成一个变换序列：

```python
composed_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

这里介绍了这么多操作，可以使用以下代码亲自跑一下，看看效果：
```py
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

def main():
    # 加载图片
    img_path = "demo1.png"
    
    img = Image.open(img_path)
    # 将RGBA图像转换为RGB图像
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # 创建各种Transforms
    transform_list = [
        ("ToTensor", transforms.ToTensor()),
        ("Resize", transforms.Resize((256, 256))),
        ("CenterCrop", transforms.CenterCrop(200)),
        ("RandomCrop", transforms.RandomCrop(200)),
        ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=1)),
        ("RandomVerticalFlip", transforms.RandomVerticalFlip(p=1)),
        ("ColorJitter", transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)),
        ("Normalize", transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))
    ]
    
    # 创建SummaryWriter
    writer = SummaryWriter("logs")
    
    # 写入原始图片
    img_tensor = transforms.ToTensor()(img)
    writer.add_image("Original Image", img_tensor, 0)
    
    # 应用各种变换并写入TensorBoard
    for i, (transform_name, transform) in enumerate(transform_list):
        if transform_name == "Normalize":
            # Normalize需要先转换为Tensor
            transformed_img = transform(img)
        else:
            transformed_img = transform(img)
            # 如果不是Tensor，转换为Tensor以便写入TensorBoard
            if not isinstance(transformed_img, torch.Tensor):
                transformed_img = transforms.ToTensor()(transformed_img)
        
        writer.add_image(f"{transform_name}", transformed_img, i+1)
        print(f"应用 {transform_name} 变换后的图片形状: {transformed_img.shape}")
    
    
    # 关闭SummaryWriter
    writer.close()

if __name__ == "__main__":
    main()
```



#### 1.4. Dataset 数据集的使用

`Dataset` 是 `torchvision` 中用于加载和处理数据集的重要模块。它可以帮助我们方便地加载图像数据集、标注数据集等。

`torchvision.datasets` 模块提供了许多常用的数据集加载函数，以下是其中的一些：

- **CIFAR10**: 加载 CIFAR-10 数据集。CIFAR-10 是一个包含 60000 张 32x32 彩色图像的数据集，分为 10 个类别。

- **CIFAR100**: 加载 CIFAR-100 数据集。CIFAR-100 是一个包含 60000 张 32x32 彩色图像的数据集，分为 100 个类别。

- **Country211**：该数据集是通过从 YFCC100m 数据集中筛选出具有与 ISO-3166 国家代码对应的 GPS 坐标的图像构建而成的。为了实现数据集的平衡，每个国家会抽取 150 张训练图像、50 张验证图像和 100 张测试图像。
```

加载方式也很简单：

```python
from torchvision.datasets import Country211

# 加载训练集
train_dataset = Country211(root='./data', Train=True, download=True)
```

#### 1.5. dataLoader 数据加载器

`DataLoader` 是 `torch.utils.data` 模块中的一个类，用于将数据集（`Dataset`）封装为一个可迭代的对象，以便于在训练模型时批量加载数据。



