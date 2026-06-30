import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def compute_iou(pred, target):
    """
    输入: pred / target 形状 [B, 4]，YOLO 归一化格式 (cx, cy, w, h)
    输出: [B] 每张图的 IoU
    """

    def to_xyxy(box):
        cx, cy, w, h = box.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

    p = to_xyxy(pred)
    t = to_xyxy(target)

    x1 = torch.max(p[..., 0], t[..., 0])
    y1 = torch.max(p[..., 1], t[..., 1])
    x2 = torch.min(p[..., 2], t[..., 2])
    y2 = torch.min(p[..., 3], t[..., 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_p = (p[..., 2] - p[..., 0]).clamp(min=0) * (p[..., 3] - p[..., 1]).clamp(min=0)
    area_t = (t[..., 2] - t[..., 0]) * (t[..., 3] - t[..., 1])
    union = area_p + area_t - inter

    return inter / union.clamp(min=1e-6)


def _draw_box(ax, box, color, label, img_w, img_h):
    # box 是张量，tolist() 转成 Python float 才能给 matplotlib 用
    cx, cy, w, h = box.tolist()
    # 反归一化：YOLO 坐标是 0~1，画图需要乘以图片实际像素尺寸
    # matplotlib 画矩形要"左上角 + 宽高"，所以这里只算左上角
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    # facecolor="none" 让矩形透明，只画边框不挡住底图
    rect = patches.Rectangle(
        (x1, y1), w * img_w, h * img_h, linewidth=2, edgecolor=color, facecolor="none"
    )
    # add_patch 才真正把矩形贴到画布上
    ax.add_patch(rect)
    # max(y1-5, 0) 防止标签跑到图片上方负坐标导致看不见
    # bbox 给文字加半透明白底，避免和底图颜色冲突看不清
    ax.text(
        x1,
        max(y1 - 5, 0),
        label,
        color=color,
        fontsize=9,
        weight="bold",
        bbox=dict(facecolor="white", alpha=0.6, pad=1, edgecolor="none"),
    )


def visualize_batch(
    images,
    pred_bboxes,
    target_bboxes,
    pred_classes,
    target_classes,
    ious,
    classes,
    save_dir,
    batch_idx,
):
    """
    把一个 batch 的预测结果与真实标签画在原图上，每张图存一份 PNG
    images: [B, 3, H, W]
    """
    # exist_ok=True 表示目录已存在不报错
    os.makedirs(save_dir, exist_ok=True)
    # logits [B, num_classes] -> 取每行最大值的索引 [B]，得到预测类别
    pred_labels = pred_classes.argmax(dim=-1)

    for i in range(images.size(0)):
        # PyTorch 张量是 [C, H, W]，matplotlib 需要 [H, W, C]，所以要 permute
        # cpu() 把 GPU 张量搬回内存；numpy() 转成 matplotlib 认识的数组
        img = images[i].permute(1, 2, 0).cpu().numpy()
        # 取前两维 (H, W)，扔掉通道数
        h, w = img.shape[:2]

        # 创建一张 5x5 英寸的画布，ax 是真正绘图用的坐标系
        fig, ax = plt.subplots(1, figsize=(5, 5))
        # imshow 铺底图，后续的框和文字都叠在它上面
        ax.imshow(img)
        # 真值用绿色 (lime)，target_classes[i] 是张量标量，.item() 转 Python int 才能做索引
        _draw_box(
            ax,
            target_bboxes[i],
            "lime",
            f"GT: {classes[target_classes[i].item()]}",
            w,
            h,
        )
        # 预测用红色，方便和绿色真值对比
        _draw_box(
            ax, pred_bboxes[i], "red", f"Pred: {classes[pred_labels[i].item()]}", w, h
        )
        # 标题显示 IoU，:.3f 保留 3 位小数
        ax.set_title(f"IoU = {ious[i].item():.3f}")
        # 关掉坐标轴刻度，看图时更清爽
        ax.axis("off")

        # 用 batch 编号和图片序号命名，避免覆盖
        save_path = os.path.join(save_dir, f"batch{batch_idx}_img{i}.png")
        # bbox_inches="tight" 去掉多余白边；dpi=80 控制清晰度
        fig.savefig(save_path, bbox_inches="tight", dpi=80)
        # 重点！循环里画图必须 close，否则 fig 对象会堆积在内存里直到 OOM
        plt.close(fig)
