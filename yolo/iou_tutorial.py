"""
compute_iou 逐步拆解教学脚本
直接运行: uv run python yolo/iou_tutorial.py
配合注释和打印的中间结果一起看
"""

import torch
from eval_utils import compute_iou


# ============================================================
# 准备一对具体的框，用 YOLO 归一化格式 (cx, cy, w, h)
# pred:   中心(0.5, 0.5), 宽高(0.4, 0.4) -> 覆盖 [0.30,0.30]~[0.70,0.70]
# target: 中心(0.6, 0.6), 宽高(0.4, 0.4) -> 覆盖 [0.40,0.40]~[0.80,0.80]
# 两个框部分重叠，方便看交集
# ============================================================
pred = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
target = torch.tensor([[0.6, 0.6, 0.4, 0.4]])

print("=== 输入张量 ===")
print(f"pred   shape={tuple(pred.shape)}  value={pred}")
print(f"target shape={tuple(target.shape)}  value={target}\n")


# ============================================================
# 步骤 1: 把 (cx, cy, w, h) 转成 (x1, y1, x2, y2)
# 计算重叠区域需要左上+右下坐标，YOLO 给的是中心+宽高，要换算
# ============================================================
print("=== 步骤 1: 格式转换 (cx,cy,w,h) -> (x1,y1,x2,y2) ===")

# unbind(-1): 把最后一维拆成多个张量
# pred [1, 4] unbind(-1) -> 4 个 [1] 张量
cx, cy, w, h = pred.unbind(-1)
print(f"unbind 后: cx={cx}, cy={cy}, w={w}, h={h}")
print(f"  每个形状是 {tuple(cx.shape)}")

# 几何换算: 左上=中心-宽高/2，右下=中心+宽高/2
# torch.stack(..., dim=-1): 把 4 个 [1] 张量沿新维度堆回 [1, 4]
p = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
print(f"stack 重组后 p = {p}, shape={tuple(p.shape)}")
# stack 重组后 p = tensor([[0.3000, 0.3000, 0.7000, 0.7000]]), shape=(1, 4)

cx_t, cy_t, w_t, h_t = target.unbind(-1)
t = torch.stack(
    [cx_t - w_t / 2, cy_t - h_t / 2, cx_t + w_t / 2, cy_t + h_t / 2], dim=-1
)
print(f"target 转换后 t = {t}\n")


# ============================================================
# 步骤 2: 计算交集矩形坐标
# 交集左上 = 两框左上中"更靠右下的那个" -> max
# 交集右下 = 两框右下中"更靠左上的那个" -> min
# ============================================================
print("=== 步骤 2: 求交集矩形坐标 ===")

# p[..., 0] 是省略号索引: 取最后一维的第 0 列
# "..." 等价于 ":"，但不管前面有几个维度都能用，更通用
print(f"p[..., 0] = {p[..., 0]}，p[..., 1] = {p[..., 1]}，p[..., 2] = {p[..., 2]}，p[..., 3] = {p[..., 3]}  (所有 batch 的 x1)")

inter_x1 = torch.max(p[..., 0], t[..., 0])  # 0.30 vs 0.40 -> 0.40
inter_y1 = torch.max(p[..., 1], t[..., 1])
inter_x2 = torch.min(p[..., 2], t[..., 2])  # 0.70 vs 0.80 -> 0.70
inter_y2 = torch.min(p[..., 3], t[..., 3])
print(f"交集左上 = ({inter_x1.item():.2f}, {inter_y1.item():.2f})")
print(f"交集右下 = ({inter_x2.item():.2f}, {inter_y2.item():.2f})\n")


# ============================================================
# 步骤 3: 交集面积 (clamp 防负数)
# clamp(min=0) 把小于 0 的值截断为 0
# ============================================================
print("=== 步骤 3: 交集面积 ===")

dx = (inter_x2 - inter_x1).clamp(min=0)
dy = (inter_y2 - inter_y1).clamp(min=0)
inter_area = dx * dy
print(f"dx={dx.item():.2f}, dy={dy.item():.2f}, 交集面积={inter_area.item():.4f}\n")

# 反例: 不相交时不 clamp 会怎样
print("反例: 两个完全不相交的框")
# 这里为了突出 clamp 的作用，a/b 直接使用已经转换后的 (x1, y1, x2, y2) 坐标。
a = torch.tensor([[0.0, 0.0, 0.2, 0.2]])
b = torch.tensor([[0.5, 0.5, 0.8, 0.8]])
fake_dx = torch.min(a[..., 2], b[..., 2]) - torch.max(a[..., 0], b[..., 0])
fake_dy = torch.min(a[..., 3], b[..., 3]) - torch.max(a[..., 1], b[..., 1])
print(f"  不 clamp: dx={fake_dx.item()}, dy={fake_dy.item()}")
print(f"  不 clamp 的'面积' = {(fake_dx * fake_dy).item()} (负×负=正, 假交集!)")
print(f"  clamp 后        = {(fake_dx.clamp(min=0) * fake_dy.clamp(min=0)).item()}\n")


# ============================================================
# 步骤 4: 各自面积
# pred 要 clamp (模型可能预测负 w/h), target 来自人工标注不需要
# ============================================================
print("=== 步骤 4: 两个框各自的面积 ===")
area_p = (p[..., 2] - p[..., 0]).clamp(min=0) * (p[..., 3] - p[..., 1]).clamp(min=0)
area_t = (t[..., 2] - t[..., 0]) * (t[..., 3] - t[..., 1])
print(f"area_p = {area_p.item():.4f}  # 0.4 × 0.4")
print(f"area_t = {area_t.item():.4f}  # 0.4 × 0.4\n")


# ============================================================
# 步骤 5: 并集 = 各自面积之和 - 交集 (容斥原理)
# 直接相加会把重叠部分算两次，所以减一次交集
# ============================================================
print("=== 步骤 5: 并集面积 ===")
union = area_p + area_t - inter_area
print(f"union = {area_p.item():.4f} + {area_t.item():.4f} - {inter_area.item():.4f}"
      f" = {union.item():.4f}\n")


# ============================================================
# 步骤 6: IoU = 交集 / 并集
# clamp(min=1e-6) 防除零 (union=0 时直接除会得 nan)
# ============================================================
print("=== 步骤 6: 最终 IoU ===")
iou = inter_area / union.clamp(min=1e-6)
print(f"IoU = {inter_area.item():.4f} / {union.item():.4f} = {iou.item():.4f}\n")
print("解读: IoU ≈ 0.39 表示约 39% 的重叠度")
print("  IoU = 1.0  完美重合 | IoU ≥ 0.5  检测正确 | IoU = 0.0  不相交\n")


# ============================================================
# 步骤 7: Batch 化 (一次算多张图，所有计算自动并行)
# ============================================================
print("=== 步骤 7: Batch 化演示 (3 张图) ===")
batch_pred = torch.tensor([
    [0.5, 0.5, 0.4, 0.4],   # 图 1: 部分重叠
    [0.5, 0.5, 0.4, 0.4],   # 图 2: 完全重合
    [0.1, 0.1, 0.1, 0.1],   # 图 3: 完全不相交
])
batch_target = torch.tensor([
    [0.6, 0.6, 0.4, 0.4],
    [0.5, 0.5, 0.4, 0.4],
    [0.9, 0.9, 0.1, 0.1],
])
batch_iou = compute_iou(batch_pred, batch_target)
print(f"batch IoU = {batch_iou}")
print(f"  图 1 (部分重叠): {batch_iou[0].item():.4f}")
print(f"  图 2 (完全重合): {batch_iou[1].item():.4f}")
print(f"  图 3 (不相交)  : {batch_iou[2].item():.4f}")
