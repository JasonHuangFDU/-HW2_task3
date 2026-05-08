"""
可视化工具:
    * 把整数标签 mask 上色为 RGB 图
    * 把一个 batch 的 (image, gt, pred) 拼成网格保存
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image


# 8 类语义颜色 + 1 个 ignore 颜色 (黑)
_PALETTE: np.ndarray = np.array(
    [
        [70, 130, 180],   # sky        - 钢蓝
        [34, 139, 34],    # tree       - 森林绿
        [128, 64, 128],   # road       - 紫
        [124, 252, 0],    # grass      - 草绿
        [0, 191, 255],    # water      - 深天蓝
        [220, 20, 60],    # building   - 猩红
        [139, 69, 19],    # mountain   - 棕
        [255, 215, 0],    # foreground - 金黄
    ],
    dtype=np.uint8,
)


def colorize_mask(mask: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """把整数类别图 (H, W) 转成上色 RGB 图 (H, W, 3)。"""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c, color in enumerate(_PALETTE):
        out[mask == c] = color
    out[mask == ignore_index] = (0, 0, 0)
    return out


def _denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """把训练时 normalize 过的 tensor 还原成 0-255 的 RGB ndarray。"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.detach().cpu() * std + mean
    img = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return img


def save_prediction_grid(
    images: torch.Tensor,
    gts: torch.Tensor,
    preds: torch.Tensor,
    save_path: str | Path,
    max_samples: int = 8,
    ignore_index: int = 255,
) -> None:
    """把 (image, GT, Pred) 三列拼成网格保存成 PNG。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(images.size(0), max_samples)
    rows = []
    for i in range(n):
        rgb = _denormalize(images[i])
        gt_rgb = colorize_mask(gts[i].cpu().numpy(), ignore_index)
        pr_rgb = colorize_mask(preds[i].cpu().numpy(), ignore_index)
        # 三张图水平拼接 (高度对齐)
        row = np.concatenate([rgb, gt_rgb, pr_rgb], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid).save(save_path)
