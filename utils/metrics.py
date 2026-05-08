"""
分割任务评价指标:
    * Pixel Accuracy
    * mean IoU (mIoU)
    * 各类 IoU
通过累积一个 (num_classes, num_classes) 的混淆矩阵, 在 epoch 末一次性算出指标。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


class SegmentationMetric:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """累加一个 batch 的混淆矩阵。

        pred:    (B, H, W)  — argmax 后的预测类别
        target:  (B, H, W)  — GT, 包含 ignore_index 的位置会被过滤
        """
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
        pred = pred.detach().cpu().numpy().astype(np.int64).ravel()
        target = target.detach().cpu().numpy().astype(np.int64).ravel()

        valid = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        pred = pred[valid]
        target = target[valid]

        # 用 bincount 一次性累加混淆矩阵, 比 for 循环快非常多
        idx = self.num_classes * target + pred
        bincount = np.bincount(idx, minlength=self.num_classes ** 2)
        self.confusion += bincount.reshape(self.num_classes, self.num_classes)

    # ---- 指标计算 -----------------------------------------------------
    def pixel_accuracy(self) -> float:
        total = self.confusion.sum()
        if total == 0:
            return 0.0
        return float(np.diag(self.confusion).sum() / total)

    def per_class_iou(self) -> np.ndarray:
        diag = np.diag(self.confusion).astype(np.float64)
        gt_sum = self.confusion.sum(axis=1).astype(np.float64)
        pred_sum = self.confusion.sum(axis=0).astype(np.float64)
        denom = gt_sum + pred_sum - diag
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(denom > 0, diag / denom, np.nan)
        return iou

    def mean_iou(self) -> float:
        iou = self.per_class_iou()
        valid = iou[~np.isnan(iou)]
        if valid.size == 0:
            return 0.0
        return float(valid.mean())

    def summary(self, class_names: List[str] | None = None) -> Dict[str, float]:
        ious = self.per_class_iou()
        out: Dict[str, float] = {
            "pixel_acc": self.pixel_accuracy(),
            "mIoU": self.mean_iou(),
        }
        for i, v in enumerate(ious):
            key = f"IoU/{class_names[i]}" if class_names else f"IoU/class_{i}"
            out[key] = float(v) if not np.isnan(v) else 0.0
        return out
