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

下面是一个简单的示例，展示了如何使用 `DataLoader` 加载 CIFAR-10 数据集：
- `batch_size=64`：每个批次包含 64 张图像。
- `shuffle=True`：在每个 epoch 开始时，随机打乱数据集的顺序。

```py
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def main():
    # 加载训练集
    train_dataset = datasets.CIFAR10(
        root="./dataset", train=False, download=True, transform=transforms.ToTensor()
    )
    writer = SummaryWriter("logs")
    dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    step = 0
    for batch in dataloader:
        images, labels = batch
        writer.add_images("Batch Images2", images, step)
        print(step)
        step += 1
    writer.close()


if __name__ == "__main__":
    main()

```


### 2. 神经网络

#### 2.1. 卷积层

卷积层是神经网络中最基本的层之一，用于提取图像的特征。它通过卷积操作将输入图像与一组可学习的卷积核进行卷积，从而生成特征图。

我们可以通过 `torch.nn` 来实现一个神经网络

```python
import torch
import torch.nn as nn

class SimpleConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = self.conv1(x)
        return x
```

##### 2.1.1 卷积层输出尺寸计算

**为什么 padding=1 时输出尺寸保持 32×32？**

当你运行代码时会发现，输入 CIFAR10 的 32×32 图像经过 `kernel_size=3, padding=1, stride=1` 的卷积层后，输出仍然是 32×32 尺寸。这是通过以下公式计算的：

```
输出尺寸 = (输入尺寸 - 卷积核大小 + 2×padding) ÷ stride + 1
```

**具体计算过程：**
- 输入尺寸 (CIFAR10 图像)：32×32
- 卷积核大小 (kernel_size)：3
- 填充 (padding)：1
- 步幅 (stride)：1

代入公式：
```
输出高度 = (32 - 3 + 2×1) ÷ 1 + 1 = (31) ÷ 1 + 1 = 32
输出宽度 = (32 - 3 + 2×1) ÷ 1 + 1 = (31) ÷ 1 + 1 = 32
```

##### 2.1.2 卷积过程示例

为了更直观地理解卷积操作，让我们通过一个简单的例子来演示卷积过程。

**输入特征图**（单通道，5×5）：
```
输入特征图：
1  2  3  0  1
4  5  6  1  0
7  8  9  2  1
0  1  2  3  4
3  2  1  0  5
```

**卷积核**（3×3，单输出通道）：
```
卷积核：
1  0  -1
1  0  -1
1  0  -1
```

**卷积操作步骤**（步长=1，无填充）：

1. **第一个窗口**（左上角3×3区域）：
   ```
   输入区域：
   1  2  3
   4  5  6
   7  8  9
   
   与卷积核相乘求和：
   (1×1) + (2×0) + (3×-1) +
   (4×1) + (5×0) + (6×-1) +
   (7×1) + (8×0) + (9×-1)
   = 1 + 0 - 3 + 4 + 0 - 6 + 7 + 0 - 9 = -6
   ```

2. **第二个窗口**（右移一步）：
   ```
   输入区域：
   2  3  0
   5  6  1
   8  9  2
   
   与卷积核相乘求和：
   (2×1) + (3×0) + (0×-1) +
   (5×1) + (6×0) + (1×-1) +
   (8×1) + (9×0) + (2×-1)
   = 2 + 0 + 0 + 5 + 0 - 1 + 8 + 0 - 2 = 12
   ```

3. **第三个窗口**（继续右移一步）：
   ```
   输入区域：
   3  0  1
   6  1  0
   9  2  1
   
   与卷积核相乘求和：
   (3×1) + (0×0) + (1×-1) +
   (6×1) + (1×0) + (0×-1) +
   (9×1) + (2×0) + (1×-1)
   = 3 + 0 - 1 + 6 + 0 + 0 + 9 + 0 - 1 = 16
   ```

4. **后续窗口**：
   继续按照步长=1向右滑动，处理完第一行后向下滑动一行，重复上述过程。

**最终输出特征图**（3×3）：
```
输出特征图：
-6  12  16
-12  18  20
-21  15  14
```

**多通道卷积示例**：

当输入有多个通道时，每个通道会有对应的卷积核，最终结果是各通道卷积结果的总和。

假设输入有2个通道，每个通道有对应的3×3卷积核：

**输入通道1**：
```
1  2
3  4
```

**输入通道2**：
```
5  6
7  8
```

**卷积核通道1**：
```
1  0
0  1
```

**卷积核通道2**：
```
0  1
1  0
```

**计算过程**：
- 通道1卷积结果：(1×1)+(2×0)+(3×0)+(4×1) = 1+0+0+4 = 5
- 通道2卷积结果：(5×0)+(6×1)+(7×1)+(8×0) = 0+6+7+0 = 13
- 最终输出：5 + 13 = 18

在本章节一开始的例子中，一共会生成 6 个卷积核进行卷积操作，每个卷积核的结构是 `(3, 3, 3)`，其值是随机初始化的。。

#### 2.2. 最大池化层

最大池化（Max Pooling）最大池化是卷积神经网络中常用的操作，具有以下重要作用：

- 🔹 降维（Dimensionality Reduction）
**最大池化可以减少特征图的空间维度（高度和宽度），不能改变通道数**，从而减少后续层的参数数量和计算量。这有助于降低模型的复杂度，提高训练速度。

- 🔹 提取主要特征
池化窗口会选择区域内的最大值，这相当于保留了该区域内最显著、最关键的特征，丢弃了次要信息。这样可以使模型更加关注重要特征。

- 🔹 防止过拟合
通过减少特征数量和复杂度，最大池化有助于防止模型对训练数据过度拟合，提高模型的泛化能力。

- 🔹 增强鲁棒性
最大池化可以过滤掉一些噪声和细节，使模型更加鲁棒。

最大池化的操作步骤跟卷积很相似，只是池化窗口在输入上滑动，每次取窗口内的最大值作为输出。举个例子：

假设输入是一个 5x5 的特征图（为了简单起见，我们只看一个通道）：

```
输入特征图：
1  3  2  4  0
5  2  7  1  3
9  6  3  8  2
2  5  1  7  4
8  3  6  2  9
```

使用 3x3 最大池化，步长为 `1`，处理过程如下：

```
1. 第一个窗口（左上角 3x3 区域）：(最大值为9)
1  3  2
5  2  7
9  6  3

2. 第二个窗口（左移一步，保持高度，宽度 +1）：(最大值为8)
3  2  4
2  7  1
6  3  8
```

最终得到的特征图大小如下：
```
9  8  8
9  8  8
9  7  9
```

实操代码示例如下：

```py
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

```

![alt text](<截屏2026-01-09 00.16.25.png>)

#### 2.3. 非线性层（激活函数层）


非线性层（也叫激活函数层）是神经网络中引入非线性变换的组件，常见的有 `ReLU` 、 `Sigmoid` 、 `Tanh` 等。非线性层比较简单，主要作用是引入非线性变换，使神经网络能够学习复杂的特征表示，而不是简单的线性变换。

我们可以通过查看 PyTorch 文档中的[非线性层部分](https://docs.pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)来了解更多信息。



比如 `nn.ReLU` 层，它的作用是对输入进行非线性变换，将所有负值设为 0，保持正值不变。

$$
\text{ReLU}(x) = (x)^+ = \max(0, x)
$$

python 代码示例如下：
```py
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """一个非常简洁的卷积神经网络（CNN），用于 CIFAR10 分类。"""

    def __init__(self):
        super().__init__()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.ReLU(x)
        return x


def main():
    a = torch.randn(10)
    model = SimpleCNN()
    outputs = model(a)
    print(a,outputs)
    # tensor([ 0.3132, -0.0041, -0.9163,  1.1990, -0.4604, -1.4164, -0.2908, -1.6122,0.8373,  0.5947]) 
    # tensor([0.3132, 0.0000, 0.0000, 1.1990, 0.0000, 0.0000, 0.0000, 0.0000, 0.8373,0.5947])
```

#### 2.4. 全连接层（线性层）

全连接层（也叫线性层）是神经网络中最基本的层，它的作用是对输入进行线性变换，输出的维度可以任意指定。

- 1. 线性层的数学原理
线性层的核心是执行一个线性变换操作，数学公式表示为：

$$
Y = X \times W^T + b
$$

- 2. 线性层的作用
   - **特征转换**：将输入特征从一个维度空间转换到另一个维度空间
   - **特征组合**：通过权重矩阵对输入特征进行加权组合，学习特征之间的关联
   - **信息传递**：作为神经网络各层之间的信息传递桥梁

全连接层通常用于神经网络的最后几层，用于将前面提取的特征转换为最终的输出（如分类任务的类别概率）。

- 3. 代码示例

```py
import torch
import torch.nn as nn

# 创建一个线性层：输入维度10，输出维度5
linear_layer = nn.Linear(in_features=10, out_features=5)

# 随机生成输入张量 (batch_size=2, input_features=10)
input_tensor = torch.randn(2, 10)

output_tensor = linear_layer(input_tensor)

print("输入形状:", input_tensor.shape)    # 输出: torch.Size([2, 10])
print("输出形状:", output_tensor.shape)   # 输出: torch.Size([2, 5])
print("权重形状:", linear_layer.weight.shape)  # 输出: torch.Size([5, 10])
print("偏置形状:", linear_layer.bias.shape)    # 输出: torch.Size([5])
```

#### 2.5. 小实战：实现一个CIFAR10分类的神经网络模型

本节，我们将实现一个简单的卷积神经网络（CNN），用于 CIFAR10 分类任务。我们先搜一下 CIFAR10 的实现模型：

![](./Structure-of-CIFAR10-quick-model.png)



