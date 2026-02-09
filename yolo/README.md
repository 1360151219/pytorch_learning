# 目标检测实战教程

目标检测是计算机视觉中的一个重要任务，它的目标是在图像中识别并定位出感兴趣的对象。

在这篇教程中，我们将使用 YOLOv26 来进行目标检测。

---

## 目录
1. [简介：为什么选择 YOLOv26？](#1-简介为什么选择-yolov26)
2. [环境准备](#2-环境准备)
3. [快速体验：你的第一次检测](#3-快速体验你的第一次检测)
4. [实战：训练自己的数据集](#4-实战训练自己的数据集)
5. [模型预测](#5-模型预测)
6. [总结](#6-总结)

---

## 1. 简介：为什么选择 YOLO26？

在目标检测领域，YOLO (You Only Look Once) 一直是速度和精度的代名词。
到了 2026 年，YOLOv26 带来了几个革命性的改进，非常适合我们初学者：

1.  **端到端 (End-to-End)**：以前的 YOLO 需要一个叫 "NMS" 的复杂后处理步骤来去除重复的框，现在 YOLOv26 直接输出最终结果，速度更快，代码更简单！
2.  **MuSGD 优化器**：这是一种结合了 SGD 和 Muon 的新型优化器，让模型训练收敛更快，就像给汽车换了更高级的引擎。
3.  **CPU 友好**：官方宣称 CPU 推理速度提升了 43%，这对我们使用 Mac (尤其是没有独立显卡) 的同学来说是巨大的福音。

---

## 2. 环境准备

尽管是 2026 年的最新模型，YOLOv26 依然保持了极好的兼容性，集成在 `ultralytics` 库中。

### 安装
在终端中运行以下命令：

```bash
# 安装 ultralytics 库 (确保更新到支持 YOLOv26 的最新版本)
pip install ultralytics --upgrade

# 安装 opencv 用于图像处理
pip install opencv-python
```

---

## 3. 快速体验：你的第一次检测

我们直接加载官方预训练好的 **YOLOv26 Nano** 模型 (`yolo26n.pt`)。这是最小、最快的版本。

打开或新建 `quick_start.py`，输入以下代码：

```python
from ultralytics import YOLO

# 1. 加载模型
# 'yolov26n.pt' 是 2026 年最新的 nano 版本模型
# 'n' 代表 nano，体积最小，速度最快
# 第一次运行时，系统会自动从云端下载这个模型文件
model = YOLO('yolo26n.pt')
# 2. 进行预测
# source: 图片来源，可以是本地路径，也可以是 URL
# 我们依然用经典的巴士图来测试
model.predict(source='https://ultralytics.com/images/bus.jpg', save=True)
```

**运行结果：**
你会在 `runs/detect/predict` 目录下看到结果。

当然，你也可以通过 `save_dir` 参数指定保存目录。

```python
model.predict(source='https://ultralytics.com/images/bus.jpg', save=True, save_dir='images')
```

**运行结果：**
你会在 `images` 目录下看到结果。

---

## 4. 实战：训练自己的数据集

假设我们要训练一个模型来检测 **“皮卡丘”** (Pikachu)。

### 4.1 数据集准备

业界常用于进行目标检测的数据集格式有 COCO、 VOC、YOLO 等，它们的数据格式各有千秋。不过唯一相同的是，他们都是以一个矩形框的形式来标注目标的。

- **[COCO](https://cocodataset.org/#download)**：COCO 是一个常用的数据集，包含 80 个类别，是目标检测领域的基准数据集。

coco 针对于每一个对象的标注格式是只标注**矩形框的左上角坐标 + 矩形框的宽高。**

    ```text
    coco/
    ├── annotations/
    │   ├── instances_train2017.json
    │   └── instances_val2017.json
    ├── train2017/
    │   ├── 000000000009.jpg
    │   └── ...
    └── val2017/
        ├── 000000000139.jpg
        └── ...
    ```


- **VOC**：VOC 是另一个常用的数据集，包含 20 个类别，也是目标检测领域的基准数据集。

voc 针对于每一个对象的标注格式是只标注**矩形框的左上角坐标 + 右下角坐标**

    ```text
    voc/
    ├── annotations/
    │   ├── train/
    │   └── val/
    ├── images/
    │   ├── train/
    │   └── val/
    ```

- **[YOLO](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)**：YOLO 数据集格式是最简洁的，每个图片对应一个文本文件，文件中包含了所有的标注信息。

yolo 针对于每一个对象的标注格式是只标注**矩形框的中心点坐标 + 矩形框的宽高**，且每个坐标都归一化到 [0, 1] 之间。

即：
- 矩形框的中心点坐标 = (x_center / img_width, y_center / img_height)
- 矩形框的宽高 = (width / img_width, height / img_height)

所有的坐标值都归一化到 [0, 1] 之间，这是 YOLO 数据集的一个标准要求。


结构如下：
```text
datasets/
└── pikachu_data/
    ├── images/
    │   ├── train/  # 训练图片
    │   └── val/    # 验证图片
    └── labels/
        ├── train/  # 训练标签
        └── val/    # 验证标签
```

### 4.2. 图片标注

强烈推荐使用 **LabelImg**。

这是最经典、最适合新手的开源标注工具。它的优点是**轻量级、不用联网、免费**，而且**原生支持**导出你需要的 **PascalVOC** (`.xml`) 和 **YOLO** (`.txt`) 格式，非常方便。


 📖 LabelImg 保姆级使用教程

启动软件后，请按照以下步骤操作：

**第一步：加载图片**
*   点击左侧的 **"Open Dir"** 按钮，选择存放图片的文件夹。
*   点击 **"Change Save Dir"** 按钮，选择一个文件夹用来存放生成的标签文件（建议新建一个 `labels` 文件夹，保持整洁）。

**第二步：切换格式（最重要的一步！）**
*   在左侧工具栏，你会看到一个按钮，默认显示 **"PascalVOC"**。
    *   如果你需要 **VOC** 格式（生成 `.xml` 文件），保持默认即可。
    *   如果你需要 **YOLO** 格式（生成 `.txt` 文件），**点击一下这个按钮**，它会变成 **"YOLO"**。

**第三步：开始标注**
*   按快捷键 **`W`**：进入画框模式，鼠标变成十字，在目标物体上拉一个框。
*   输入标签：画好框后，会弹窗让你输入类别名称（比如 `dog`, `cat`）。
*   点击 **"OK"** 确认。

**第四步：保存与切换**
*   按 **`Ctrl + S`** (Mac 上是 `Command + S`) 保存当前图片的标签。
*   按 **`D`**：切换到**下一张**图片。
*   按 **`A`**：切换到**上一张**图片。

**💡 小贴士：**
*   **自动保存**：点击菜单栏的 `View` -> 勾选 `Auto Save mode`，这样切换图片时会自动保存，不用每次都按保存键，效率翻倍！
*   **classes.txt**：当你选择 YOLO 格式时，软件会自动在保存目录下生成一个 `classes.txt` 文件，里面记录了所有的类别名称。**千万不要删掉它**，训练的时候需要用到。

### 4.2 配置文件 (pikachu.yaml)
在 `yolo` 目录下确认 `pikachu.yaml` 文件存在：

```yaml
# 数据的绝对路径
path: /Users/bytedance/workspace/pytorch_learning/yolo/datasets/pikachu_data
train: images/train
val: images/val

# 类别设置
nc: 1
names: ['pikachu']
```

### 4.3 开始训练 (使用 MuSGD)
新建 `train_custom.py`。注意，这里我们要利用 YOLOv26 的特性。

```python
from ultralytics import YOLO

def main():
    # 1. 加载模型
    # 加载预训练的 yolov26n.pt，利用迁移学习 (Transfer Learning)
    # 这样我们不需要从头训练，站在巨人的肩膀上
    model = YOLO('yolov26n.pt')

    # 2. 开始训练
    # data: 配置文件路径
    # epochs: 训练轮数，YOLOv26 收敛很快，50 轮通常就够了
    # device: Mac 使用 'mps' (Metal Performance Shaders) 加速
    # optimizer: 我们可以指定 'MuSGD' (如果是默认的可以不写，但为了展示特性我们写上)
    model.train(
        data='pikachu.yaml', 
        epochs=50, 
        imgsz=640, 
        device='mps',      # Mac M芯片加速神器
        optimizer='auto'   # YOLOv26 会自动选择最佳优化器 (通常是 MuSGD)
    )

if __name__ == '__main__':
    # 多进程保护，Mac 下必须写
    main()
```

---

## 5. 模型预测

训练完成后，权重会保存在 `runs/detect/train/weights/best.pt`。
此时这个 `best.pt` 就是一个 **YOLOv26** 架构的皮卡丘检测器了！

新建 `predict_custom.py`：

```python
from ultralytics import YOLO

# 1. 加载我们训练好的 YOLOv26 模型
model = YOLO('runs/detect/train/weights/best.pt')

# 2. 预测
# conf=0.5: 只显示置信度大于 0.5 的框，过滤掉不确定的结果
model.predict(source='new_pikachu.jpg', save=True, conf=0.5) 
```

---

## 6. 总结

你现在已经掌握了 2026 年最前沿的目标检测技术！
**YOLOv26** 的核心优势在于：
*   **快**：CPU 推理速度提升，Mac 上跑起来飞快。
*   **准**：MuSGD 优化器让训练更稳定。
*   **简**：去除了 NMS 后处理，模型输出就是最终结果。

快去收集一些图片，训练一个属于你自己的 AI 助手吧！
