"""
损失函数实现:
    1. CrossEntropyLoss2d  -- 标准多类交叉熵 (调用 PyTorch 内置, 加 ignore_index 处理 unknown 像素)
    2. DiceLoss            -- 手动实现的多类 Dice Loss (基本要求: 自己写)
    3. CombinedLoss        -- alpha * CE + beta * Dice

实现细节:
    * Dice Loss 使用 softmax 后的概率分布和 one-hot 标签计算, 因此对 logits 直接传入即可
    * 为了处理 unknown 像素 (label = ignore_index), 在 one-hot 时构造一个 mask, 把
      对应位置 (无论预测还是 GT) 都置零, 避免污染 Dice 的分子分母
    * 多类 Dice 采用宏平均 (macro), 即每个类先单独算 Dice 系数再求平均, 这样能缓解
      前/背景像素严重不平衡时背景类主导损失的问题
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss2d(nn.Module):
    """对 logits (B, C, H, W) 与 target (B, H, W) 计算交叉熵, 自动忽略 ignore_index。"""

    def __init__(self, ignore_index: int = 255, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, target)


class DiceLoss(nn.Module):
    """手动实现的多类 Dice Loss (macro 平均)。

    Dice 系数定义:
        Dice_c = 2 * sum(p_c * y_c) / (sum(p_c) + sum(y_c) + eps)
    Dice Loss:
        L = 1 - mean_c Dice_c
    其中 p_c 来自 softmax 概率, y_c 为 one-hot GT, 二者均屏蔽 ignore 像素。
    """

    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W);  target: (B, H, W)
        probs = F.softmax(logits, dim=1)

        valid_mask = (target != self.ignore_index)  # (B, H, W)

        # one-hot: 把 ignore 位置先临时替换成 0 防止越界
        target_clamped = torch.where(valid_mask, target, torch.zeros_like(target))
        target_oh = F.one_hot(target_clamped, num_classes=self.num_classes)  # (B, H, W, C)
        target_oh = target_oh.permute(0, 3, 1, 2).contiguous().float()  # (B, C, H, W)

        # 把 ignore 位置在通道维上整体置零, 这样不会出现在 Dice 的分子分母里
        valid_mask_c = valid_mask.unsqueeze(1).float()  # (B, 1, H, W)
        probs = probs * valid_mask_c
        target_oh = target_oh * valid_mask_c

        # 在 (B, H, W) 三维上求和, 每类得到一个 Dice 系数
        dims = (0, 2, 3)
        intersection = (probs * target_oh).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + target_oh.sum(dim=dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # macro 平均: 每个类等权
        return 1.0 - dice_per_class.mean()


class CombinedLoss(nn.Module):
    """组合损失: L = alpha * CE + beta * Dice。"""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        alpha: float = 1.0,
        beta: float = 1.0,
        ce_weight: Optional[torch.Tensor] = None,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = CrossEntropyLoss2d(ignore_index=ignore_index, weight=ce_weight)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index, smooth=smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.ce(logits, target) + self.beta * self.dice(logits, target)


def build_loss(
    name: str,
    num_classes: int,
    ignore_index: int = 255,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> nn.Module:
    """根据字符串名构造损失函数, 方便命令行切换。

    name: "ce" | "dice" | "combined"
    """
    name = name.lower()
    if name in {"ce", "cross_entropy", "crossentropy"}:
        return CrossEntropyLoss2d(ignore_index=ignore_index)
    if name == "dice":
        return DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
    if name in {"combined", "ce_dice", "ce+dice"}:
        return CombinedLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            alpha=alpha,
            beta=beta,
        )
    raise ValueError(f"未知的 loss 名称: {name}. 支持: ce / dice / combined")
