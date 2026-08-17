## 矩阵乘法

### 1. 二维矩阵乘法

规则：`(m, n) @ (n, p) = (m, p)`

- 第一个矩阵的列数 `n` 必须等于第二个矩阵的行数 `n`。
- 结果矩阵的 shape 为 `(m, p)`。

```python
import torch

a = torch.randn(2, 3)   # shape: (2, 3)
b = torch.randn(3, 4)   # shape: (3, 4)

c = a @ b               # 等价于 torch.matmul(a, b)
print(c.shape)          # torch.Size([2, 4])
```

### 2. 高阶（批量）矩阵乘法

对于维度大于 2 的张量，`@` / `torch.matmul` 只对**最后两个维度**做矩阵乘法，前面的维度视为 **batch 维**，并遵循 broadcasting 规则。

#### 2.1 相同 batch 维

```python
a = torch.randn(4, 2, 3)   # 4 个 batch，每个是 (2, 3)
b = torch.randn(4, 3, 5)   # 4 个 batch，每个是 (3, 5)

c = a @ b
print(c.shape)             # torch.Size([4, 2, 5])
```

可以理解为：对 `i = 0..3`，分别计算 `a[i] @ b[i]`，最后堆叠起来。

#### 2.2 batch 维 broadcasting

当 batch 维形状不同但可以 broadcast 时，PyTorch 会自动广播：

```python
a = torch.randn(3, 4, 5)   # batch: (3,)
b = torch.randn(   5, 2)   # 没有 batch 维，会被广播

c = a @ b
print(c.shape)             # torch.Size([3, 4, 2])
```

但如果 batch 维无法 broadcast，则会报错：

```python
a = torch.randn(3, 4, 5)
b = torch.randn(2, 5, 6)

c = a @ b                  # RuntimeError: 批量维度无法 broadcast (3 vs 2)
```

### 3. 实际例子：Transformer 的 Attention

Transformer / 大模型里的 Self-Attention，核心计算就是一次矩阵乘法：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中最关键的一步是 $QK^T$，它用来计算每个词和其他所有词之间的关系（Attention Score）。

**假设输入一句话"我 喜欢 学习 AI"，共 4 个词，每个词用 512 维向量表示：**

```python
import torch

Q = torch.randn(4, 512)    # 4 个 query 向量，每个 512 维
K = torch.randn(4, 512)    # 4 个 key 向量，每个 512 维

# K^T 的 shape 是 (512, 4)
# (4, 512) @ (512, 4) -> (4, 4)
scores = Q @ K.T
print(scores.shape)        # torch.Size([4, 4])
```

结果矩阵 `scores[i][j]` 表示第 `i` 个词和第 `j` 个词之间的相关性大小：

```
            我      喜欢    学习    AI
我         0.1     0.7    0.1    0.1
喜欢       0.2     0.1    0.6    0.1
学习       0.1     0.2    0.1    0.6
AI         ...
```

- 从 shape 角度看：`(4, 512) @ (512, 4) = (4, 4)`，完全符合 `(m, n) @ (n, p) = (m, p)`。
- 从语义角度看：这个 `(4, 4)` 矩阵就是每个词对句子中所有词的"注意力权重"，所以 Transformer 的核心本质上也是矩阵乘法。

