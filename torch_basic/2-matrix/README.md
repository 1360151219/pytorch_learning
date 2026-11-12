# 深度学习笔记之反向传播与梯度下降

## 203-矩阵乘法

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

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V$$

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

## 204-梯度、自动求导

### 1. 题目1 - 矩阵梯度

```py
x = torch.tensor([1.0,2.0], requires_grad=True)

w = torch.tensor([3.0,4.0], requires_grad=True)

y = x[0]*w[0] + x[1]*w[1]

loss = y**2
loss.backward()
```

1. `y` 的数学表达式是什么？

$$y = x w^T = \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 3 \\ 4 \end{bmatrix} = 1 \times 3 + 2 \times 4 = 11$$

```
print(y) # y = xw^T = 1*3+2*4 = 11
```

2. `loss` 对 `y` 的梯度是多少？


$$\frac{\partial \text{loss}}{\partial y} = \frac{\partial (y^2)}{\partial y} = 2y = 2 \times 11 = 22$$




3. `loss.backward()` 后，`w.grad` 应该是多少？


$$\frac{\partial \text{loss}}{\partial w} = \frac{\partial \text{loss}}{\partial y}\frac{\partial \text{y}}{\partial w} = 2y \times \begin{bmatrix} 1&2\end{bmatrix} = \begin{bmatrix} 22&44\end{bmatrix}$$


### 2. 题目2 - 为什么反向传播一定是从后往前？为什么梯度可以一层层传递？

1. 为什么必须从后往前？
假设一个简单网络：
$$
x \rightarrow a \rightarrow b \rightarrow L
$$

对应：
\[
a=f(x)
\]\[
b=g(a)
\]\[
L=h(b)
\]现在我们想知道：
参数 $x$ 对最终 loss 的影响是多少？

也就是：
\[
\frac{\partial L}{\partial x}
\]如果直接求：
\[
L=h(g(f(x)))
\]
对于神经网络有几十层甚至几百层，这样计算的数据量非常巨大。
用链式法则，则变成了（方向相反）：

\[\frac{\partial L}{\partial x} = \frac{\partial L}{\partial b}\frac{\partial b}{\partial a}\frac{\partial a}{\partial x}\]

3. 为什么反向传播效率高

- 1. 核心：复用中间梯度，避免重复计算


$$x \rightarrow a \rightarrow b \rightarrow L$$

如果**直接展开** $L = h(g(f(x)))$ 对每个参数逐个求导，靠近输入的公共部分 $\frac{\partial L}{\partial b}\frac{\partial b}{\partial a}$ 会被反复重算。层数越深，重复量越大（naive 展开是 $O(n^2)$ 甚至指数级冗余）。

反向传播从后往前，把上游梯度存下来复用：

$$\frac{\partial L}{\partial b} \to \frac{\partial L}{\partial a}=\frac{\partial L}{\partial b}\frac{\partial b}{\partial a} \to \frac{\partial L}{\partial x}=\frac{\partial L}{\partial a}\frac{\partial a}{\partial x}$$

每个 $\frac{\partial L}{\partial (\cdot)}$ **只算一次**，后一层直接拿前一层的结果乘上局部导数。复杂度降到 $O(n)$——和一次前向传播同一量级（实测约 2 倍前向开销）。

- 2. 为什么"从后往前"而不是"从前往后"：标量输出 + 海量参数

这是效率的真正来源。求 $\frac{\partial L}{\partial x}$ 等价于把三个雅可比矩阵连乘：

$$J = \underbrace{J_3}_{\partial L/\partial b}\cdot \underbrace{J_2}_{\partial b/\partial a}\cdot \underbrace{J_1}_{\partial a/\partial x}$$

矩阵连乘满足结合律，但**乘的顺序决定代价**。因为 loss 是标量，$J_3$ 是 `1×|b|` 的行向量：

| 顺序 | 中间结果 | 代价 |
|------|---------|------|
| **从后往前**（reverse / backprop）| $(J_3 J_2)$ 是 `1×|a|` 向量 | 全程都是**向量×矩阵**，便宜 |
| 从前往后（forward mode）| $(J_2 J_1)$ 是 `|b|×|x|` **大矩阵** | 要构造并存储巨大雅可比，昂贵 |

从后往前，起点是 $\frac{\partial L}{\partial L}=1$（标量），梯度全程保持成"和该层激活同形状的向量"，永远不需要把完整雅可比矩阵展开出来。这就是所谓的**向量-雅可比积（VJP）**。

- 3. 一次反传拿到全部参数梯度

反向模式的代价**与参数数量几乎无关**：不管有 100 个还是 10 亿个参数，一次 backward 就全拿到了。

对比前向数值/前向模式求导——每个参数要单独跑一次前向：

$$\text{前向模式代价} \approx n_{\text{参数}} \times \text{一次前向}$$

对一个 10 亿参数的模型，就是 **10 亿次前向 vs 1 次反向**。当"输入维度（参数）≫ 输出维度（标量 loss）"时，reverse-mode 恰好是最优选择——这正是深度学习的典型场景。

---

**一句话总结**：反向传播高效 = 复用公共子表达式（省重复计算）+ 标量输出让梯度始终是向量（省矩阵展开）+ 一次遍历覆盖所有参数（省重复遍历）。

---


### 3. 题目3 - 向量对向量求导 Jacobian

> Jacobian 是**向量对向量**求导的产物，是针对所有输入对所有输出的影响表：输入 $n$ 维、输出 $m$ 维，Jacobian 就是一个 $m \times n$ 矩阵，第 $i$ 行第 $j$ 列 = 第 $i$ 个输出对第 $j$ 个输入的偏导 $\dfrac{\partial y_i}{\partial x_j}$。

设输入 $x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$，输出 $y = \begin{bmatrix} y_1 \\ y_2 \end{bmatrix}$，函数关系为：

$$y_1 = x_1^2 + x_2, \qquad y_2 = x_1 \cdot x_2$$

在点 $x = \begin{bmatrix} 3 \\ 2 \end{bmatrix}$ 处，Jacobian 定义为：

$$J = \frac{\partial y}{\partial x} = \begin{bmatrix} \dfrac{\partial y_1}{\partial x_1} & \dfrac{\partial y_1}{\partial x_2} \\[2mm] \dfrac{\partial y_2}{\partial x_1} & \dfrac{\partial y_2}{\partial x_2} \end{bmatrix}$$

1. 写出 $J$ 的符号形式（4 个偏导数，用 $x_1, x_2$ 表示）。

\[J=\begin{bmatrix}2x_1 & 1 \\ x_2 & x_1\end{bmatrix} = \begin{bmatrix}6 & 1 \\ 2 & 3\end{bmatrix}\]

给你一个判断"该不该转置、往哪边乘"的**通用心法**，比背公式管用：

对于 $Y = XW$：
- 对 $X$ 求导：$\dfrac{\partial L}{\partial X} = G\, W^T$ （上游梯度 $G$ 在左，$W$ 转置在右）
- 对 $W$ 求导：$\dfrac{\partial L}{\partial W} = X^T G$ （$X$ 转置在左，$G$ 在右）

规律：**求谁的导，谁的位置就放上游梯度 $G$；另一个乘数转置后，摆在它原来的位置上。** $X$ 原来在左边，它的梯度里 $G$ 就在左边；$W$ 原来在右边，它的梯度里 $G$ 就在右边。

这套规则是所有全连接层反向传播的模板，PyTorch 里 `nn.Linear` 的 backward 就是这两行。要不要我给你出一道"对 $W$ 求导"的配套小题练手？或者把题目4 的答案（含这个非对称变体）整理进文件当参考解？

## 205-综合实战-梯度下降计算

上一节讲了雅可比矩阵 $J$（所有输入对所有输出的影响表）。这一节回答一个更实际的问题：**真要算梯度时，是"从输入往前推"还是"从结果往回追"？两种方式各花多少次计算？**

### 0. 先摆一个具体例子

设一个函数，把 2 个输入变成 3 个输出：

$$f(x_1, x_2) = \begin{bmatrix} x_1 + x_2 \\ x_1 x_2 \\ x_1^2 \end{bmatrix}$$

在点 $(x_1, x_2) = (2, 3)$ 处，它的雅可比矩阵是：

$$J = \begin{bmatrix} \dfrac{\partial y_1}{\partial x_1} & \dfrac{\partial y_1}{\partial x_2} \\[2mm] \dfrac{\partial y_2}{\partial x_1} & \dfrac{\partial y_2}{\partial x_2} \\[2mm] \dfrac{\partial y_3}{\partial x_1} & \dfrac{\partial y_3}{\partial x_2} \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 3 & 2 \\ 4 & 0 \end{bmatrix}$$

再把三个输出加起来，得到一个标量损失 $L = y_1 + y_2 + y_3$。于是整条链变成"2 个数进，1 个数出"：

$$\mathbb{R}^2 \xrightarrow{\ f\ } \mathbb{R}^3 \xrightarrow{\ L\ } \mathbb{R}$$

本节要算的就是 $\nabla_x L$（最终答案是 $\begin{bmatrix} 8 \\ 3 \end{bmatrix}$），并说明**为什么反向传播一次就能全拿到**。

### 1. 两种模式，其实在问两个相反的问题

用一个类比先建立直觉——把这个函数想成**两个旋钮（输入）控制三个仪表（输出）**：

- **正向模式**问："我拨动旋钮，仪表会怎么动？"
  → 先定好输入怎么变，再看所有输出跟着变多少。计算的是 $J\mathbf{v}$。
- **反向模式**问："某个最终结果变了，这笔账该算到哪些旋钮头上？"
  → 先定好在意哪个输出，再往回追每个输入该负多少责任。计算的是 $J^\top \mathbf{u}$。

**用形状就能记住谁是谁：**

$$\underbrace{J}_{3\times 2}\ \underbrace{\mathbf{v}}_{2\times 1} = \underbrace{J\mathbf{v}}_{3\times 1} \qquad\Rightarrow\qquad \text{给 2 个输入的变化方向，得到 3 个输出的变化}$$

$$\underbrace{J^\top}_{2\times 3}\ \underbrace{\mathbf{u}}_{3\times 1} = \underbrace{J^\top\mathbf{u}}_{2\times 1} \qquad\Rightarrow\qquad \text{给 3 个输出的"重要程度"，得到 2 个输入该背的责任}$$

### 2. 正向模式：给输入配个"变化速度"，顺着往前推

做法：给每个输入贴一个"变化速度"$\dot x_1, \dot x_2$（这里的点不是对时间求导，就是"沿某个方向 $\mathbf{v}$ 变得多快"的意思）。然后**按原来的计算顺序**，一边算值、一边把这个速度传下去。

$$y_1 = x_1 + x_2 \quad\Rightarrow\quad \dot y_1 = \dot x_1 + \dot x_2 = v_1 + v_2$$

$$y_2 = x_1 x_2 \quad\Rightarrow\quad \dot y_2 = x_2\,\dot x_1 + x_1\,\dot x_2 = 3v_1 + 2v_2 \quad(\text{乘法求导，代入 } x_1{=}2, x_2{=}3)$$

$$y_3 = x_1^2 \quad\Rightarrow\quad \dot y_3 = 2x_1\,\dot x_1 = 4v_1$$

合起来正好就是 $J\mathbf{v}$：

$$\dot{\mathbf{y}} = \begin{bmatrix} v_1 + v_2 \\ 3v_1 + 2v_2 \\ 4v_1 \end{bmatrix} = J\mathbf{v}$$

比如只拨第二个旋钮，取 $\mathbf{v} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$，得到 $J\mathbf{v} = \begin{bmatrix} 1 \\ 2 \\ 0 \end{bmatrix}$——这正是 $J$ 的**第 2 列**。

> 关键点：正向模式**一次只能选定一个输入方向**，看全部输出的反应。想凑齐整个 $J$，就得一列一列来，有几个输入就跑几次。而且它不必真的把 $J$ 存出来，沿计算图直接就能算出 $J\mathbf{v}$。

### 3. 反向模式：从结果出发，往回分摊责任

做法分两步：

**第一步，先正常前向算一遍，把中间值全存下来：**

$$x_1 = 2,\quad x_2 = 3,\quad y_1 = 5,\quad y_2 = 6,\quad y_3 = 4$$

**第二步，从最终的标量 $L$ 往回走**，给每个量记一个"责任值"（记作 $\bar q = \dfrac{\partial L}{\partial q}$，读作"$L$ 对 $q$ 有多敏感"）。

因为 $L = y_1 + y_2 + y_3$，三个输出对 $L$ 的贡献都是 1，所以起点是：

$$\bar y_1 = \bar y_2 = \bar y_3 = 1$$

现在往回追 $x_1$。注意 $x_1$ **同时出现在三个输出里**（三条路径都影响 $L$），所以要把三条路的责任加起来：

$$\bar x_1 = \bar y_1\frac{\partial y_1}{\partial x_1} + \bar y_2\frac{\partial y_2}{\partial x_1} + \bar y_3\frac{\partial y_3}{\partial x_1} = 1\cdot 1 + 1\cdot 3 + 1\cdot 4 = 8$$

同理追 $x_2$：

$$\bar x_2 = \bar y_1\frac{\partial y_1}{\partial x_2} + \bar y_2\frac{\partial y_2}{\partial x_2} + \bar y_3\frac{\partial y_3}{\partial x_2} = 1\cdot 1 + 1\cdot 2 + 1\cdot 0 = 3$$

$$\Rightarrow\quad \nabla_x L = \begin{bmatrix} 8 \\ 3 \end{bmatrix}$$

> 这里又用到了题目4 学过的那条规则：**一个变量若通过多条分支影响结果，反向传播要把各条路径的梯度相加。**
> 关键点：反向模式跑**一次**（$\mathbf{u} = [1,1,1]^\top$）就把所有输入的梯度全拿到了。

### 4. 为什么两种模式的"跑几次"不一样

| 模式 | 一次选定什么 | 一次得到什么 | 求完整 $J$ 要几次 |
|---|---|---|---|
| 正向模式 | 一个输入方向 $\mathbf{v}$ | $J\mathbf{v}$（一列） | 约 $n$ 次（$n$ = 输入个数） |
| 反向模式 | 一个输出方向 $\mathbf{u}$ | $J^\top\mathbf{u}$（一行方向） | 约 $m$ 次（$m$ = 输出个数） |

结论（这就是深度学习为什么清一色用反向传播）：

- **输入少、输出多** → 正向模式划算。
- **输入多、输出少** → 反向模式划算。
- 神经网络有**海量参数（输入极多）但只有一个标量 loss（输出=1）**，所以永远用反向模式：一次反传，全部参数梯度到手。

> ⚠️ 别误解："一次反传拿到所有梯度"只在**最终是一个标量**时成立。如果你要的是某个向量函数的**完整雅可比矩阵**，反向模式仍然得按输出方向跑多次。


### 5. 叶子节点和中间节点

```py
w = torch.tensor([1.0, 2.0], requires_grad=True)
a = 3 * w
z = (a ** 2).sum()

z.backward()


w.is_leaf = True
a.is_leaf = False

w.grad = [16,36]
a.grad = None
```

PyTorch 默认只把梯度累积到需要求导的叶子张量中。

这不表示梯度没有经过 a。梯度先在 a 处计算，再继续传给 w，只是没有长期保存到 a.grad。

如果在反向传播前执行 `a.retain_grad()`，则 ：

$$a.grad = \frac{\partial z}{\partial a} = 2a = [6, 12]$$

### 6. detach 

```py
 w = torch.tensor(2.0, requires_grad=True)
  a = w ** 2
  b = a.detach()
  z = a + 3 * b
  z.backward()
  # 1*2*2+ (3*0*2*2 切断)
  print(w.grad)
  print(a.requires_grad) # True
  print(b.requires_grad) # False
```


### 7. 综合挑战：一次完整的线性回归训练（梯度下降）


有 3 个样本，每个样本有 2 个特征：

$$X = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \qquad \mathbf{w} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \qquad \mathbf{y} = \begin{bmatrix} 2 \\ 1 \\ 3 \end{bmatrix}$$

模型没有偏置：

$$\hat{\mathbf{y}} = X\mathbf{w}$$

残差和损失定义为：

$$\mathbf{r} = \hat{\mathbf{y}} - \mathbf{y}, \qquad L = \sum_{i=1}^{3} r_i^2 = \mathbf{r}^\top \mathbf{r}$$

请依次计算：

1. **各变量的形状：**

$$X:\ [3,2], \quad \mathbf{w}:\ [2,1], \quad \hat{\mathbf{y}}:\ [3,1], \quad \mathbf{r}:\ [3,1], \quad L:\ []， L是个标量$$

2. **前向传播：**

$$\hat{\mathbf{y}} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \qquad \mathbf{r} = \begin{bmatrix} -1 \\ 1 \\ 0 \end{bmatrix}, \qquad L = 2$$

3. **反向传播，补全链条：**

$$\nabla_{\mathbf{r}} L = 2r = \begin{bmatrix} -2 \\ 2 \\ 0 \end{bmatrix}$$

因为 $\mathbf{r} = X\mathbf{w} - \mathbf{y}$，所以：

$$\nabla_{\mathbf{w}} L = X^{\top}\, \nabla_{\mathbf{r}} L = \begin{bmatrix} 1& 0&  1\\0&1&1\end{bmatrix} \begin{bmatrix} -2 \\ 2 \\ 0 \end{bmatrix}   = \begin{bmatrix} -2 \\ 2 \end{bmatrix}$$

4. **一次梯度下降**（学习率 $\eta = 0.25$）：

$$\mathbf{w}_{\text{new}} = \mathbf{w} - \eta\, \nabla_{\mathbf{w}} L = \begin{bmatrix} 1.5 \\ 1.5 \end{bmatrix}$$

5. **再做一次前向计算**，求新损失 $L_{\text{new}}$，判断损失是增大还是减小。

$$\hat{\mathbf{y}}_{\text{new}} = X\mathbf{w}_{\text{new}} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1.5 \\ 1.5 \end{bmatrix} = \begin{bmatrix} 1.5 \\ 1.5 \\ 3 \end{bmatrix}$$

新残差：

$$\mathbf{r}_{\text{new}} = \hat{\mathbf{y}}_{\text{new}} - \mathbf{y} = \begin{bmatrix} -0.5 \\ 0.5 \\ 0 \end{bmatrix}$$

新损失：

$$L_{\text{new}} = (-0.5)^2 + (0.5)^2 + 0^2 = 0.5$$

因此损失从 $2$ 降到 $0.5$，一次梯度下降后下降了 $75\%$。

---

6. 手算流程 ↔ 真实训练循环的对应

上面这套"前向 → 算损失 → 反向 → 更新参数"就是 之前文章中 里训练循环的**手算版本**。逐一对应：

| 本节手算步骤 | 数学 | `train.py` 中的代码 |
|---|---|---|
| 前向传播 | $\hat{\mathbf{y}} = X\mathbf{w}$ | `outputs = model(images)` |
| 计算损失 | $L = \mathbf{r}^\top \mathbf{r}$（平方差之和） | `loss = MSELoss(y_hat, y)` |
| 梯度清零 | （手算每轮从头算，天然无残留） | `optimizer.zero_grad()` |
| 反向传播求梯度 | $\nabla_{\mathbf{w}} L = X^\top \nabla_{\mathbf{r}} L$ | `loss.backward()` |
| 更新参数 | $\mathbf{w}_{\text{new}} = \mathbf{w} - \eta\,\nabla_{\mathbf{w}} L$ | `optimizer.step() # torch.optim.SGD` |

代码实现如下：

```py
def main3():
  x = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0]])
  # target
  y = torch.tensor([[2.0],
                    [1.0],
                    [3.0]])
  # weight
  w = torch.tensor([[1.0],
                    [2.0]], requires_grad=True)

  optimizer = torch.optim.SGD([w], lr=0.25)
  optimizer.zero_grad()
  # predict
  y_hat = x @ w
  r = y_hat - y
  loss = r.T @ r

  loss.backward()
  optimizer.step()

  print(w.grad)
  print(w)
```

## 206-线性回归、平方损失与梯度下降

### 1. 线性模型、偏置与形状

有 $m$ 个样本、每个样本有 $d$ 个特征时，线性回归模型为：

$$\hat{\mathbf y}=X\mathbf w+b\mathbf 1_m$$

各变量的形状为：

$$X\in\mathbb R^{m\times d},\qquad
\mathbf w\in\mathbb R^{d\times1},\qquad
b\in\mathbb R,\qquad
\hat{\mathbf y}\in\mathbb R^{m\times1}$$

- $\mathbf w$ 有 $d$ 个可学习权重，$b$ 是一个可学习的标量偏置，总参数量为 $d+1$。
- $b\mathbf 1_m$ 是偏置广播到整个批次后的结果，形状为 $(m,1)$；它不代表模型学习了 $m$ 个偏置。
- 样本数改变时，$X$、$\hat{\mathbf y}$ 和 $b\mathbf 1_m$ 的第一维会改变，$\mathbf w$ 和 $b$ 的形状不变。
- 可以增加一个恒为 $1$ 的特征，把偏置合并进矩阵乘法：

$$\widetilde X=[X,\mathbf 1_m],\qquad
\widetilde{\mathbf w}=
\begin{bmatrix}
\mathbf w\\b
\end{bmatrix},
\qquad
\hat{\mathbf y}=\widetilde X\widetilde{\mathbf w}$$

#### 示例题

已知：

$$X=
\begin{bmatrix}
0&0\\
1&0\\
0&1
\end{bmatrix},
\qquad
\mathbf y=
\begin{bmatrix}
4\\6\\1
\end{bmatrix}$$

求满足 $\hat{\mathbf y}=X\mathbf w+b\mathbf 1_3=\mathbf y$ 的 $\mathbf w,b$，并写出增广形式。

第一行输入为 $[0,0]$，权重项恒为 $0$，因此：

$$b=4$$

再由后两行得到：

$$w_1+4=6,\qquad w_2+4=1$$

所以：

$$\mathbf w=
\begin{bmatrix}
2\\-3
\end{bmatrix},
\qquad b=4$$

增广形式为：

$$\widetilde X=
\begin{bmatrix}
0&0&1\\
1&0&1\\
0&1&1
\end{bmatrix},
\qquad
\widetilde{\mathbf w}=
\begin{bmatrix}
2\\-3\\4
\end{bmatrix}$$

### 2. 平方损失

定义残差：

$$\mathbf r=\hat{\mathbf y}-\mathbf y$$

平均误差 $\frac1n\sum_i r_i$ 不能衡量回归效果，因为正负残差可能互相抵消。平方损失把每个误差都变成非负惩罚，并让大误差受到更重惩罚：

$$L=\frac1{2n}\sum_{i=1}^{n}r_i^2
=\frac1{2n}\mathbf r^\top\mathbf r
=\frac1{2n}\lVert X\mathbf w-\mathbf y\rVert^2$$

- $\frac1n$ 用于消除样本数量对损失尺度的直接影响。
- $\frac12$ 会与平方求导产生的 $2$ 抵消，使梯度公式更简洁。
- 乘上任意正常数不会改变损失的最小值位置，但会改变梯度的数值尺度。

#### 示例题

真实值和两个模型的预测为：

$$\mathbf y=
\begin{bmatrix}
10\\10
\end{bmatrix},
\qquad
\hat{\mathbf y}_A=
\begin{bmatrix}
0\\20
\end{bmatrix},
\qquad
\hat{\mathbf y}_B=
\begin{bmatrix}
11\\11
\end{bmatrix}$$

两个模型的残差分别为：

$$\mathbf r_A=
\begin{bmatrix}
-10\\10
\end{bmatrix},
\qquad
\mathbf r_B=
\begin{bmatrix}
1\\1
\end{bmatrix}$$

若使用平均误差，模型 A 的结果为 $0$，会被错误地判断为完美预测。使用平方损失：

$$L_A=\frac14(100+100)=50$$

$$L_B=\frac14(1+1)=0.5$$

因此模型 B 明显更好。

### 3. 从残差反向传播到权重

对于：

$$\mathbf r=X\mathbf w-\mathbf y,\qquad
L=\frac1{2n}\mathbf r^\top\mathbf r$$

先对残差求梯度：

$$\nabla_{\mathbf r}L=\frac1n\mathbf r$$

残差的正负号必须保留，因为它表示参数应该向哪个方向修正。由于 $\frac{\partial\mathbf r}{\partial\mathbf w}=X$，反向传播得到：

$$\nabla_{\mathbf w}L
=X^\top\nabla_{\mathbf r}L
=\frac1nX^\top\mathbf r
=\frac1nX^\top(X\mathbf w-\mathbf y)$$

形状为：

$$\underbrace{X^\top}_{d\times n}
\underbrace{\mathbf r}_{n\times1}
=
\underbrace{\nabla_{\mathbf w}L}_{d\times1}$$

$X^\top$ 的作用是把 $n$ 个样本的残差责任汇总回 $d$ 个权重。形状检查只能排除错误，真正决定使用 $X^\top$ 的是链式法则。

#### 示例题

已知：

$$X=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix},
\qquad
\mathbf r=
\begin{bmatrix}
-1\\1\\0
\end{bmatrix},
\qquad n=3$$

求 $\nabla_{\mathbf r}L$ 和 $\nabla_{\mathbf w}L$。

$$\nabla_{\mathbf r}L
=\frac13
\begin{bmatrix}
-1\\1\\0
\end{bmatrix}$$

$$\nabla_{\mathbf w}L
=\frac13
\begin{bmatrix}
1&0&1\\
0&1&1
\end{bmatrix}
\begin{bmatrix}
-1\\1\\0
\end{bmatrix}
=
\begin{bmatrix}
-\frac13\\[1mm]\frac13
\end{bmatrix}$$

梯度与 $\mathbf w$ 同形状，分别给出两个权重的更新方向。

### 4. 正规方程与解析解

平方损失关于 $\mathbf w$ 是凸二次函数。在最低点令梯度为零：

$$\frac1nX^\top(X\mathbf w-\mathbf y)=0$$

得到正规方程：

$$X^\top X\mathbf w=X^\top\mathbf y$$

如果 $X^\top X$ 可逆，则唯一解析解为：

$$\mathbf w^*=(X^\top X)^{-1}X^\top\mathbf y$$

最优条件也可以写成：

$$X^\top\mathbf r=0$$

它表示残差与 $X$ 的每一列都正交，而不是要求每个样本的残差都等于零。若 $X^\top X$ 不可逆，不能直接使用逆矩阵公式；实际计算通常使用最小二乘求解器，例如 torch.linalg.lstsq，而不是显式求逆。

#### 示例题

已知：

$$X=
\begin{bmatrix}
1\\3
\end{bmatrix},
\qquad
\mathbf y=
\begin{bmatrix}
2\\8
\end{bmatrix}$$

求无偏置模型 $\hat{\mathbf y}=Xw$ 的解析解。

$$X^\top X=1^2+3^2=10$$

$$X^\top\mathbf y=1\times2+3\times8=26$$

正规方程为：

$$10w^*=26$$

所以：

$$w^*=2.6$$

重新前向计算：

$$\hat{\mathbf y}=
\begin{bmatrix}
2.6\\7.8
\end{bmatrix},
\qquad
\mathbf r=
\begin{bmatrix}
0.6\\-0.2
\end{bmatrix}$$

虽然残差不为零，但：

$$X^\top\mathbf r=1\times0.6+3\times(-0.2)=0$$

因此梯度为零。最终损失为：

$$L=\frac14\left(0.6^2+(-0.2)^2\right)=0.1$$

### 5. 梯度下降与学习率

解析解是直接计算最低点，梯度下降则从初始参数出发，重复执行：

$$\mathbf r_t=X\mathbf w_t-\mathbf y$$

$$\mathbf g_t=\frac1nX^\top\mathbf r_t$$

$$\mathbf w_{t+1}=\mathbf w_t-\eta\mathbf g_t$$

每次更新参数后，都必须在新位置重新计算预测、残差和梯度。若每次使用全部样本计算梯度，这种方法称为批量梯度下降。

对于一维凸二次函数：

$$L(w)=\frac a2(w-w^*)^2,\qquad
g(w)=a(w-w^*)$$

定义参数误差 $e_t=w_t-w^*$，则：

$$e_{t+1}=(1-a\eta)e_t$$

误差倍率 $1-a\eta$ 决定训练行为：

| 学习率范围 | 训练行为 |
|---|---|
| $0<\eta<\frac1a$ | 单侧逐渐收敛 |
| $\eta=\frac1a$ | 一步到达最优点 |
| $\frac1a<\eta<\frac2a$ | 左右振荡但逐渐收敛 |
| $\eta=\frac2a$ | 左右等幅振荡，不收敛 |
| $\eta>\frac2a$ | 左右振荡且逐渐发散 |

学习率的安全范围取决于损失曲面的曲率 $a$，不存在对所有问题都通用的固定上限。

#### 示例题

已知：

$$X=
\begin{bmatrix}
1\\2
\end{bmatrix},
\qquad
\mathbf y=
\begin{bmatrix}
2\\5
\end{bmatrix},
\qquad
w_0=0,\qquad
\eta=0.1$$

使用：

$$L(w)=\frac14\lVert Xw-\mathbf y\rVert^2$$

完成两次梯度下降。

第 $0$ 步：

$$\hat{\mathbf y}_0=
\begin{bmatrix}
0\\0
\end{bmatrix},
\qquad
\mathbf r_0=
\begin{bmatrix}
-2\\-5
\end{bmatrix}$$

$$g_0=\frac12
\begin{bmatrix}
1&2
\end{bmatrix}
\begin{bmatrix}
-2\\-5
\end{bmatrix}
=-6$$

$$L(w_0)=\frac{29}{4}=7.25$$

$$w_1=0-0.1(-6)=0.6$$

第 $1$ 步必须重新计算梯度：

$$\hat{\mathbf y}_1=
\begin{bmatrix}
0.6\\1.2
\end{bmatrix},
\qquad
\mathbf r_1=
\begin{bmatrix}
-1.4\\-3.8
\end{bmatrix}$$

$$g_1=\frac12
\begin{bmatrix}
1&2
\end{bmatrix}
\begin{bmatrix}
-1.4\\-3.8
\end{bmatrix}
=-4.5$$

$$L(w_1)=4.1$$

$$w_2=0.6-0.1(-4.5)=1.05$$

本题的解析解为 $w^*=2.4$，参数轨迹：

$$0\rightarrow0.6\rightarrow1.05$$

正在单侧靠近最优点。因为：

$$g(w)=2.5(w-2.4)$$

所以：

$$e_{t+1}=(1-2.5\eta)e_t$$

当 $\eta=0.1$ 时误差倍率为 $0.75$；收敛范围是 $0<\eta<0.8$，一步到达最优点的学习率是 $\eta=0.4$。
