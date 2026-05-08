

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
    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W);  target: (B, H, W)
        probs = F.softmax(logits, dim=1)

        valid_mask = (target != self.ignore_index)  # (B, H, W)

        target_clamped = torch.where(valid_mask, target, torch.zeros_like(target))
        target_oh = F.one_hot(target_clamped, num_classes=self.num_classes)  # (B, H, W, C)
        target_oh = target_oh.permute(0, 3, 1, 2).contiguous().float()  # (B, C, H, W)

        valid_mask_c = valid_mask.unsqueeze(1).float()  # (B, 1, H, W)
        probs = probs * valid_mask_c
        target_oh = target_oh * valid_mask_c

        dims = (0, 2, 3)
        intersection = (probs * target_oh).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + target_oh.sum(dim=dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1.0 - dice_per_class.mean()


class CombinedLoss(nn.Module):

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
