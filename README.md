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
```python
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

```python
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

在本章节一开始的例子中，一共会生成 6 个卷积核进行卷积操作，每个卷积核的结构是 `(3, 3, 3)`，其值是随机初始化的。

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

```python
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

![alt text](截屏2026-01-09%2000.16.25.png)

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

根据上图我们分析，CIFAR10 分类模型的结构如下：

- 输入：3@32x32（3通道RGB图像，32x32像素）
- 第一层卷积：5x5卷积核，32个输出通道
- 第一层池化：2x2最大池化
- 第二层卷积：5x5卷积核，32个输出通道
- 第二层池化：2x2最大池化
- 第三层卷积：5x5卷积核，64个输出通道
- 第三层池化：2x2最大池化
- 扁平层：将64@4x4展平为1024个特征
- 全连接层：1024输入，10输出（对应CIFAR10的10个类别）

分析得出模型结构后，我们还需要考虑各个层的参数选择：

比如第一层卷积中，卷积核大小为 5x5，输入时图片尺寸是32，输出时图片尺寸也是32，那么肯定设置了 `padding`。根据卷积层的输出尺寸公式：

$$H_{out} = \left\lfloor \frac{H_{in} - KernelSize + 2Padding}{Stride} + 1 \right\rfloor$$

我们需要在卷积层中添加 `padding=2` 来保持输出尺寸为 32x32。

现在，我们就可以来实现这个模型了：

```py
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
        # (32 - 5 + 2*padding)/1 + 1 = 32 => padding = 2
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
```


#### 2.6. 损失函数（Loss Function）和反向传播（Backpropagation）

损失函数（Loss Function）是神经网络中用于衡量模型预测值与真实值之间差异的函数。它的作用是在训练过程中，通过最小化损失函数的值，来优化模型的参数，从而实现对数据的准确预测。下面我们来介绍一下常用的损失函数。

##### 2.6.1. L1Loss

拿最简单的 L1Loss 损失函数为例：

$$
L1Loss = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
$$

其中，$y_i$ 是真实值，$\hat{y}_i$ 是模型预测值，$n$ 是样本数量。L1Loss 的作用是计算模型预测值与真实值之间的平均绝对误差。

```py
def loss_fn():
    loss_mean = nn.L1Loss(reduction="mean")
    loss_sum = nn.L1Loss(reduction="sum")
    input = torch.tensor([1, 2, 3, 4, 5])
    target = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5])
    output_mean = loss_mean(input, target)
    output_sum = loss_sum(input, target)
    print(output_mean)
    print(output_sum)

    # tensor(0.5000)
    # tensor(2.5000)
```

##### 2.6.2. MSELoss

比较常用的还有 `nn.MSELoss` 损失函数，它的作用是计算模型预测值与真实值之间的均方误差。

$$
MSELoss = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

其中，$y_i$ 是真实值，$\hat{y}_i$ 是模型预测值，$n$ 是样本数量。MSELoss 的作用是计算模型预测值与真实值之间的均方误差。

```py
def loss_fn2():
    loss_mean = nn.MSELoss(reduction="mean")
    loss_sum = nn.MSELoss(reduction="sum")
    input = torch.tensor([1, 2, 3, 4, 5])
    target = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5])
    output_mean = loss_mean(input, target)
    output_sum = loss_sum(input, target)
    print(output_mean)
    print(output_sum)
    # tensor(0.2500): (0.5^2 + 0.5^2 + 0.5^2 + 0.5^2 + 0.5^2) / 5 = 0.25
    # tensor(1.2500)
```

##### 2.6.3. CrossEntropyLoss 和 NLLLoss 交叉熵损失函数

交叉熵损失函数（CrossEntropyLoss）是专门用于C 分类问题（即数据有 C 个类别，如 10 分类的图像识别），它的作用是计算模型预测值与真实值之间的交叉熵。它支持通过weight参数（1 维 Tensor，长度为 C）为每个类别分配权重，可解决训练集类别不平衡问题（例如某类样本极少，需提高其权重以避免模型忽略该类）。

其数学计算公式如下：

$$
\text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right) = -x[class] + \log\left(\sum_j \exp(x[j])\right)
$$


NLLLoss 是 CrossEntropyLoss 的一个特殊情况，当模型输出的是对数概率（log probabilities）时，可以直接使用 NLLLoss 。

其数学计算公式如下：

$$
Loss = \frac{1}{N} \sum_{n=1}^N -x_{n, y_n}
$$

举个例子，我想通过将一些动物图片进行分类，可选类别共有三种：猫、狗、猪。假设我们手头有两个参数不同的模型，它们都通过 sigmoid/softmax 输出每个类别的概率分布：

> 参考文档：https://www.zhihu.com/tardis/zm/art/35709485?source_id=1003

**模型1：**

| 预测 | 真实 | 是否正确 |
| :--- | :--- | :--- |
| 0.3 0.3 0.4 | 0 0 1 (猪) | 正确 |
| 0.3 0.4 0.3 | 0 1 0 (狗) | 正确 |
| 0.1 0.2 0.7 | 1 0 0 (猫) | 错误 |

模型1对于样本1和样本2以非常微弱的优势判断正确，对于样本3的判断则彻底错误。

---

**模型2：**

| 预测 | 真实 | 是否正确 |
| :--- | :--- | :--- |
| 0.1 0.2 0.7 | 0 0 1 (猪) | 正确 |
| 0.1 0.7 0.2 | 0 1 0 (狗) | 正确 |
| 0.3 0.4 0.3 | 1 0 0 (猫) | 错误 |

模型2对于样本1和样本2判断非常准确，对于样本3判断错误，但是相对来说没有错得太离谱。

我们可以通过计算模型1和模型2的交叉熵损失来比较它们的性能：

**模型1的交叉熵损失：**

在这里，$p_{target}^{(i)}$ 代表模型对第 $i$ 个样本**真实类别**的预测概率。

- **样本1**：真实类别是“猪”，模型1预测“猪”的概率是 **0.4**。
- **样本2**：真实类别是“狗”，模型1预测“狗”的概率是 **0.4**。
- **样本3**：真实类别是“猫”，模型1预测“猫”的概率是 **0.1**。

所以计算过程如下：

$$
\begin{aligned}
Loss_1 &= \frac{1}{3} \sum_{i=1}^3 -\log(p_{target}^{(i)}) \\
&= \frac{1}{3} ( \underbrace{-\log(0.4)}_{\text{sample1(pig)}} + \underbrace{-\log(0.4)}_{\text{sample2(dog)}} + \underbrace{-\log(0.1)}_{\text{sample3(cat)}} ) \\
&\approx \frac{1}{3} (0.916 + 0.916 + 2.302) \\
&= 1.378
\end{aligned}
$$

**模型2的交叉熵损失：**

同理，对于模型2：
- **样本1**（猪）：预测概率 **0.7**。
- **样本2**（狗）：预测概率 **0.7**。
- **样本3**（猫）：预测概率 **0.3**。

$$
\begin{aligned}
Loss_2 &= \frac{1}{3} \sum_{i=1}^3 -\log(p_{target}^{(i)}) \\
&= \frac{1}{3} ( \underbrace{-\log(0.7)}_{\text{sample1}} + \underbrace{-\log(0.7)}_{\text{sample2}} + \underbrace{-\log(0.3)}_{\text{sample3}} ) \\
&\approx \frac{1}{3} (0.357 + 0.357 + 1.204) \\
&= 0.639
\end{aligned}
$$

从结果可以看出，$Loss_2 < Loss_1$，说明模型2的预测结果与真实值更接近，性能优于模型1。

##### 2.6.4. CrossEntropyLoss 计算实战（代码详解）

这里在写代码前要注意一下，由于 nn.CrossEntropyLoss 接收的是“未归一化的对数概率”（logits），
而我们这里已知的是经过 Softmax 后的“概率”。所以我们需要先手动取对数 (torch.log)，
然后使用 NLLLoss (Negative Log Likelihood Loss)。

公式关系：`CrossEntropyLoss(logits) = NLLLoss(log_softmax(logits))`

```py
import torch
import torch.nn as nn
import math


def calculate_pytorch_loss(probs_tensor, targets_tensor, model_name):
    criterion = nn.NLLLoss(reduction="mean")
    log_probs = torch.log(probs_tensor)
    loss = criterion(log_probs, targets_tensor)
    return loss.item()


def main():
    # 数据准备
    # 类别映射: 0:猫, 1:狗, 2:猪

    # --- 模型 1 数据 ---
    # 样本1: 预测[0.3, 0.3, 0.4], 真实: 猪(2)
    # 样本2: 预测[0.3, 0.4, 0.3], 真实: 狗(1)
    # 样本3: 预测[0.1, 0.2, 0.7], 真实: 猫(0)
    model1_probs = torch.tensor(
        [[0.3, 0.3, 0.4], [0.3, 0.4, 0.3], [0.1, 0.2, 0.7]], dtype=torch.float32
    )
    model1_targets = torch.tensor([2, 1, 0], dtype=torch.long)

    # --- 模型 2 数据 ---
    # 样本1: 预测[0.1, 0.2, 0.7], 真实: 猪(2)
    # 样本2: 预测[0.1, 0.7, 0.2], 真实: 狗(1)
    # 样本3: 预测[0.3, 0.4, 0.3], 真实: 猫(0)
    model2_probs = torch.tensor(
        [[0.1, 0.2, 0.7], [0.1, 0.7, 0.2], [0.3, 0.4, 0.3]], dtype=torch.float32
    )
    model2_targets = torch.tensor([2, 1, 0], dtype=torch.long)
    # --- 执行计算 ---
    calculate_pytorch_loss(model1_probs, model1_targets, "模型 1")
    calculate_pytorch_loss(model2_probs, model2_targets, "模型 2")


if __name__ == "__main__":
    main()

```

##### 2.6.5. 反向传播（Backpropagation）

反向传播（Backpropagation）是神经网络训练中重要的一步，在训练过程中，我们最终希望能把损失函数的值最小化，从而使模型的预测结果与真实值更接近。为了实现这一目标，我们需要计算损失函数关于模型参数的梯度，然后根据梯度更新参数。

反向传播的过程可以简单描述为：
1. 前向传播：从输入层开始，依次计算每个神经元的输出，直到输出层。
2. 计算损失：将输出层的输出与真实值进行比较，计算损失函数的值。
3. 反向传播：从输出层开始，依次计算每个神经元的梯度，直到输入层。
4. 更新参数：根据计算得到的梯度，使用优化算法（如梯度下降）更新模型参数。

通过重复执行前向传播、计算损失、反向传播和参数更新步骤，我们可以逐渐优化模型的参数，使损失函数的值不断减小，从而提高模型的性能。

以之前我们自己实现的 CIFAR10 分类模型为例，我们可以通过反向传播来更新模型参数。

```python
def main():
    dataset = datasets.CIFAR10(
        root="./dataset", train=False, download=True, transform=transforms.ToTensor()
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss = nn.CrossEntropyLoss()

    for batch in dataloader:
        images, labels = batch
        model = CIFAR10CNN()
        # 前向传播
        outputs = model(images)
        # 计算损失
        loss_value = loss(outputs, labels)
        # 反向传播
        # 计算出的 loss 值具备 backward 方法，调用后可自动完成反向传播，为每个可训练参数计算梯度。  
        # 初始时网络参数的 grad 属性为空，执行 backward 后梯度被写入，供优化器更新网络参数。
        loss_value.backward()
```


#### 2.7. 优化器

优化器（Optimizer）是神经网络训练中用于更新模型参数的算法。它的作用是根据损失函数关于参数的梯度，
通过迭代地调整各网络层的参数，使损失函数的值不断减小，从而提高模型的性能。

常用的优化器包括：
- 梯度下降（Gradient Descent）
- 动量梯度下降（Momentum Gradient Descent）
- 自适应学习率方法（Adaptive Learning Rate Methods）
  - Adam
  - RMSProp
- 其他优化器（如 SGD with Nesterov Momentum）


```py
def main():
    dataset = datasets.CIFAR10(
        root="./dataset", train=False, download=True, transform=transforms.ToTensor()
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss = nn.CrossEntropyLoss()
    # 初始化优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for batch in dataloader:
        images, labels = batch
        model = CIFAR10CNN()
        # 前向传播
        outputs = model(images)
        # 计算损失
        loss_value = loss(outputs, labels)
        # 优化器更新参数前，需要先将梯度清零, 否则梯度会累加, 导致错误的参数更新
        optimizer.zero_grad()
        loss_value.backward()
        # 优化器根据计算得到的梯度，更新模型参数
        optimizer.step()
```

#### 2.8. 神经网络的下载和修改

除了我们自己搭建神经网络外，我们更多地是去下载现成的神经网络来直接使用。 [torchvision.models](https://pytorch.org/vision/stable/models.html)

`torchvision.models` 子包包含用于解决不同任务的模型定义，包括：图像分类、逐像素语义分割、目标检测、实例分割、人体关键点检测、视频分类和光流。

比如我们要下载一个用于图像分类的模型 `ResNet50`，我们可以这样做：

```py
import torchvision.models as models, ResNet50_Weights

# 下载预训练的 ResNet - 50 模型
# Old weights with accuracy 76.130%
resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# New weights with accuracy 80.858%
resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# no weights
resnet50()
```

使用方式也很简单：

```py
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# 下载预训练的 ResNet - 50 模型
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

print(model)


"""
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (layer2): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (3): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (layer3): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (3): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (4): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (5): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (layer4): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=2048, out_features=1000, bias=True)
)
"""

```

ResNet50 是计算机视觉领域非常著名的深度学习模型，它曾获得 ImageNet 图像分类竞赛的冠军。这里的 "50" 指的是这个网络一共有 50 层（包含卷积层和全连接层）。

这里主要分为 4 个阶段：

- **1. 预处理阶段（图像刚进入网络）**

```py
(conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
(bn1): BatchNorm2d(64, ...)
(relu): ReLU(inplace=True)
(maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, ...)
```
- Conv2d (卷积层) ：这是流水线的第一道工序。
  - `(3, 64)` ：输入是 3 个通道（因为图片是 RGB 彩色的），输出变成了 64 个特征通道。你可以理解为它提取了 64 种不同的基础特征（比如边缘、颜色斑点等）。
  - `kernel_size=(7, 7)` ：这是一个很大的“扫描窗口”，一次看 7x7 像素的区域。
  - `stride=(2, 2)` ：步长是 2，说明窗口每次移动 2 格。这会让图片的 长宽缩小一半 （例如从 224x224 变成 112x112）。
- BatchNorm2d (BN层) ：这相当于“质检和标准化”。它把数据调整到一个标准的分布，防止数据在传输过程中变得过大或过小，让网络更容易训练。
- ReLU (激活函数) ：这相当于一个“开关”。它把所有负数值变成 0，正数值保持不变。这给网络引入了非线性，让它能处理复杂任务。
- MaxPool2d (最大池化层) ：这相当于“精简数据”。它在 3x3 的区域里只取最大的那个值。 stride=2 再次让图片的 长宽缩小一半 （例如从 112x112 变成 56x56）。

总结 ：这一阶段主要是快速降低图片尺寸，提取初步特征。

- **2. 核心加工阶段 （4 个 Layer）**

接下来是 ResNet 的核心部分，由 layer1 到 layer4 组成。它们是由一种叫 Bottleneck（瓶颈结构） 的模块重复堆叠而成的。

什么是 Bottleneck？ 你可以把它看作一个“三明治”结构，它包含三个卷积层：

- **1. 1x1 卷积 ：先把通道数降下来（降维），减少计算量（像把面团压实）。**
- **2. 3x3 卷积 ：在低维空间进行处理，提取特征（像在面团上刻花）。**
- **3. 1x1 卷积 ：再把通道数升上去（升维），恢复特征维度（像把面团发酵变大）。**

- **3. 输出阶段**


```py
(avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
(fc): Linear(in_features=2048, out_features=1000, bias=True)
```

- AdaptiveAvgPool2d (平均池化) ：不管前面出来的尺寸是多少（这里是 7x7），它都把每个通道的数值求平均，压缩成 1x1 的一个点。
    - 输入：2048 个通道，每个通道 7x7。
    - 输出：2048 个数值（每个数值代表该通道的平均特征强度）。
- Linear (全连接层/FC) ：这是最后的裁判。
    - in_features=2048 ：接收这 2048 个特征。
    - out_features=1000 ：输出 1000 个数值。
    - 这 1000 个数值分别对应 ImageNet 数据集中的 1000 个类别（如猫、狗、飞机等）。数值最大的那个类别，就是网络认为的预测结果。

**修改模型：**

如果我们想要修改这个模型的话，我们可以通过类似 `model.layer1[2].conv1` 的方式去访问以及修改对应的结构层。