# 2026小土堆a - 李沐动手学深度学习V2

## 1. 数据预处理

> 同学们好！今天我们正式开启深度学习之旅。
>
> 在第一堂课里，我们要聊一个听起来简单、但至关重要的话题：**数据预处理（Data Preprocessing）**。
>
> 机器学习模型就像一个刚上小学的孩子——你不能直接把一堆杂乱无章的原始数据甩给他，他消化不了。我们需要把数据"咀嚼"成模型能理解的形式，这个过程就是数据预处理。

---

## 本讲目标

通过一个真实的小案例（房价预测），你将学会：

1. 用 **pandas** 读取和操作数据
2. 如何处理数据中的 **缺失值（NaN）**
3. 如何将**文本特征**转换为数值特征（One-Hot 编码）

---

## 场景引入

假设你是一个房产中介，收到了 4 条房屋信息：

| NumRooms | Alley | Price |
|----------|-------|-------|
| 未知     | Pave  | 10000 |
| 2        | 未知  | 20000 |
| 4        | 未知  | 30000 |
| 未知     | 未知  | 40000 |

我们的目标：**根据 `NumRooms`（房间数）和 `Alley`（巷道类型）来预测 `Price`（价格）**。

但你可能已经发现问题了——有些数据是**缺失**的（表格里显示为 `NA`），而且 `Alley` 列是**文字**（"Pave"），模型不认识文字。

别着急，我们一步步来解决。

---

## Step 1 · 准备数据：创建一个 CSV 文件

首先，我们需要一个文件来存放数据。请看 `1-preprocess/main.py` 中的 `make_file()` 函数：

```python
def make_file():
    os.makedirs("dataset", exist_ok=True)
    data_file = os.path.join("dataset", "index.csv")
    with open(data_file, "w") as f:
        f.write("NumRooms,Alley,Price\n")
        f.write("NA,Pave,10000\n")
        f.write("2,NA,20000\n")
        f.write("4,NA,30000\n")
        f.write("NA,NA,40000\n")
```

**讲解时间 📝**

- `os.makedirs("dataset", exist_ok=True)`：创建一个 `dataset` 文件夹来存放数据。`exist_ok=True` 的意思是"如果文件夹已经存在就跳过，不要报错"。
- CSV 文件的每一行代表一个样本，逗号分隔不同的字段。
- `NA` 代表缺失值（Not Available），pandas 会自动把它识别为 `NaN`（Not a Number）。

---

## Step 2 · 读取数据：用 pandas 打开 CSV

```python
data_file = os.path.join("dataset", "index.csv")
data = pd.read_csv(data_file)
```

运行 `print(data)` 你会看到：

```
   NumRooms Alley  Price
0       NaN  Pave  10000
1       2.0   NaN  20000
2       4.0   NaN  30000
3       NaN   NaN  40000
```

**讲解时间 📝**

- `pd.read_csv()` 是 pandas 读取 CSV 文件的标准方式。
- 注意 pandas 自动把 `NA` 解析成了 `NaN`（浮点数类型的缺失值）。
- 第一列 `NumRooms` 因为含有 `NaN`，pandas 自动把它转为了 `float64` 类型（整数列不能存放 NaN，所以升级为浮点数）。

---

## Step 3 · 分离输入和输出

```python
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
```

**讲解时间 📝**

- `iloc` 是 pandas 的**位置索引**（integer location），用数字而不是列名来选取数据。
- `data.iloc[:, 0:2]`：取所有行（`:`），取第 0 和第 1 列（`0:2`，左闭右开）→ 得到 `NumRooms` 和 `Alley`。
- `data.iloc[:, 2]`：取所有行的第 2 列 → 得到 `Price`。
- 我们把 `inputs` 作为**特征**（输入），`outputs` 作为**标签**（预测目标）。

> 💡 **思考**：为什么用 `iloc` 而不是列名？因为在数据预处理阶段，我们希望代码对列的位置更稳健，即使列名变了也不影响。

---

## Step 4 · 处理缺失值

这是本讲的**核心难点**。直接对整个 DataFrame 调用 `.mean()` 会报错，因为字符串列 `Alley` 无法计算均值。

### 正确的做法：区分数值列和字符串列

```python
# 找出数值列和字符串列
numeric_cols = inputs.select_dtypes(include="number").columns
string_cols = inputs.select_dtypes(include=["object", "string"]).columns
```

### 数值列：用均值填充

```python
inputs[numeric_cols] = inputs[numeric_cols].fillna(
    inputs[numeric_cols].mean()
)
```

填充过程：`NumRooms` 列的 NaN 被 `(2 + 4) / 2 = 3.0` 填充。


**讲解时间 📝**

| 列类型 | 推荐填充策略 | 原因 |
|--------|-------------|------|
| **数值列** | 均值 `mean()` 或中位数 `median()` | 用中心趋势值填补，对模型影响最小 |

> ⚠️ **常见错误提醒**：直接对 DataFrame 调用 `df.mean()` 会遇到 `TypeError: Cannot perform reduction 'mean' with string dtype`——这正是我们一开始踩过的坑！记住：**先分类型，再处理**。

处理完之后，数据变成：

```
   NumRooms Alley
0       3.0  Pave
1       2.0  Nan
2       4.0  Nan
3       3.0  Nan
```

---

## Step 5 · One-Hot 编码：让模型认识文字

现在所有缺失值都处理好了，但 `Alley` 列还是文字，模型只认识数字。我们需要把文字转换成数字：

```python
inputs = pd.get_dummies(inputs, dummy_na=True, dtype=int)
```

运行结果：

```
   NumRooms  Alley_Pave  Alley_nan
0       3.0           1          0
1       2.0           0          1
2       4.0           0          1
3       3.0           0          1
```

**讲解时间 📝**

`pd.get_dummies()` 是 pandas 的独热编码（One-Hot Encoding）函数，关键参数：

| 参数 | 作用 |
|------|------|
| `dummy_na=True` | 为 NaN 单独创建一列（`Alley_nan`），表示"该行原本是缺失的" |
| `dtype=int` | 输出 0/1 整数，而不是 True/False 布尔值 |

编码逻辑：

- `Alley` 列有两个取值："Pave" 和 NaN
- 所以它被拆成两列：`Alley_Pave` 和 `Alley_nan`
- "Pave" → `Alley_Pave=1, Alley_nan=0`
- NaN → `Alley_Pave=0, Alley_nan=1`

> 💡 **为什么要 `dummy_na=True`？** 让模型知道"这行数据原本是缺失的"这个事实，而不是简单地当作"不是 Pave"。

---

## 课堂总结 🎓

同学们，我们今天完整地走完了一条数据预处理流水线：

```
原始数据 → 读取 → 分离特征/标签 → 处理缺失值 → One-Hot 编码 → 可用数据
```

回顾每一步的核心思想：

| 步骤 | 做什么 | 关键方法 |
|------|--------|---------|
| **准备数据** | 创建 CSV 文件 | `open()` + 文件写入 |
| **读取数据** | 让 pandas 认识数据 | `pd.read_csv()` |
| **分离特征/标签** | 区分输入和输出 | `iloc[:, 0:2]` |
| **处理缺失值** | 数值用均值，类别用众数 | `select_dtypes()` + `fillna()` |
| **One-Hot 编码** | 文字→数字 | `pd.get_dummies()` |

处理完成后，我们就得到了一张**完全由数字组成**的表格，深度学习模型可以直接用来训练了！

---

## 课后作业 📚

1. 修改 `make_file()`，增加一列 `Age`（房龄），并尝试用**中位数**（`median()`）而不是均值来填充缺失值
2. 如果 `Alley` 列有 3 种不同的取值（比如 "Pave"、"Gravel"、"NaN"），One-Hot 编码后会产生几列？试着改一下数据验证你的猜想
3. 查阅 `pd.get_dummies()` 的 `drop_first` 参数，理解它是如何避免**虚拟变量陷阱（Dummy Variable Trap）**的

下节课，我们将正式进入 **PyTorch 基础**，开始搭建你的第一个神经网络模型！🚀