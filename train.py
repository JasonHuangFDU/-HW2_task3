"""
训练脚本: 从零训练 U-Net 在 Stanford Background Dataset 上做语义分割。

支持三种损失配置 (作业要求):
    --loss ce         仅交叉熵
    --loss dice       仅 Dice Loss (手动实现)
    --loss combined   CE + Dice 组合损失

使用 swanlab 进行训练曲线可视化 (loss / mIoU / pixel acc), 离线模式可通过 --no-log 关闭。

典型用法:
    python train.py --data-root /path/to/iccv09Data --loss combined --epochs 100 \
                    --batch-size 8 --lr 1e-3 --output-dir runs/combined
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    build_dataloaders,
)
from losses import build_loss
from models import UNet
from utils import SegmentationMetric, save_prediction_grid


# ---------------------------------------------------------------------------
#  Args
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("UNet on Stanford Background Dataset")

    # 数据
    p.add_argument("--data-root", type=str, required=True,
                   help="数据集根目录 (内含 images/ 和 labels/ 子目录)")
    p.add_argument("--crop-w", type=int, default=320)
    p.add_argument("--crop-h", type=int, default=240)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=4)

    # 模型
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--bilinear", action="store_true", default=True,
                   help="解码端使用双线性插值上采样 (默认开). 关闭见 --no-bilinear")
    p.add_argument("--no-bilinear", dest="bilinear", action="store_false")

    # 训练
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adam", "adamw"])
    p.add_argument("--momentum", type=float, default=0.9, help="仅 SGD 使用")
    p.add_argument("--scheduler", type=str, default="cosine",
                   choices=["none", "cosine", "poly", "step"])
    p.add_argument("--seed", type=int, default=42)

    # 损失
    p.add_argument("--loss", type=str, default="combined",
                   choices=["ce", "dice", "combined"])
    p.add_argument("--ce-weight", type=float, default=1.0,
                   help="combined 模式下 CE 项的权重 alpha")
    p.add_argument("--dice-weight", type=float, default=1.0,
                   help="combined 模式下 Dice 项的权重 beta")

    # 输出 / 日志
    p.add_argument("--output-dir", type=str, default="runs/exp",
                   help="保存权重 / 日志 / 可视化结果")
    p.add_argument("--log-tool", type=str, default="swanlab",
                   choices=["swanlab", "wandb", "none"])
    p.add_argument("--project", type=str, default="HW2-Task3-UNet")
    p.add_argument("--exp-name", type=str, default=None)
    p.add_argument("--log-every", type=int, default=10,
                   help="每多少个 step 打印一次训练 loss")
    p.add_argument("--vis-samples", type=int, default=8,
                   help="每个 epoch 保存多少张验证集预测可视化")

    # 训练加速
    p.add_argument("--amp", action="store_true", default=False,
                   help="混合精度训练 (CUDA 上推荐打开)")
    p.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复")

    return p.parse_args()


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> optim.Optimizer:
    if args.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay, nesterov=True)
    if args.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(optimizer, args, steps_per_epoch: int):
    total_steps = steps_per_epoch * args.epochs
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    if args.scheduler == "poly":
        return optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (1 - step / max(1, total_steps)) ** 0.9,
        )
    if args.scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)
    return None


# ---------------------------------------------------------------------------
#  Logger 抽象 (兼容 swanlab / wandb / 无)
# ---------------------------------------------------------------------------
class Logger:
    def __init__(self, tool: str, project: str, name: Optional[str], config: dict, log_dir: str):
        self.tool = tool
        self.run = None
        if tool == "swanlab":
            try:
                import swanlab  # noqa: F401
                import swanlab as _sw
                self._sw = _sw
                self.run = _sw.init(project=project, experiment_name=name, config=config,
                                    logdir=log_dir)
            except Exception as e:  # 没装 swanlab / 网络问题等
                print(f"[Logger] swanlab 初始化失败 ({e}), 自动退化为本地日志模式。")
                self.tool = "none"
        elif tool == "wandb":
            try:
                import wandb
                self._wb = wandb
                self.run = wandb.init(project=project, name=name, config=config, dir=log_dir)
            except Exception as e:
                print(f"[Logger] wandb 初始化失败 ({e}), 自动退化为本地日志模式。")
                self.tool = "none"

    def log(self, data: dict, step: Optional[int] = None) -> None:
        if self.tool == "swanlab" and self.run is not None:
            self._sw.log(data, step=step)
        elif self.tool == "wandb" and self.run is not None:
            self._wb.log(data, step=step)
        # tool == "none": 仅本地控制台 (训练循环已 print)

    def finish(self) -> None:
        if self.tool == "swanlab" and self.run is not None:
            try:
                self._sw.finish()
            except Exception:
                pass
        elif self.tool == "wandb" and self.run is not None:
            self._wb.finish()


# ---------------------------------------------------------------------------
#  单个 epoch
# ---------------------------------------------------------------------------
def train_one_epoch(
    model, loader, optimizer, scheduler, criterion, device, scaler, logger, epoch, args, global_step,
):
    model.train()
    metric = SegmentationMetric(NUM_CLASSES, ignore_index=IGNORE_INDEX)

    pbar = tqdm(loader, desc=f"Train E{epoch}", ncols=100)
    loss_sum = 0.0
    n_seen = 0
    for it, (img, mask, _ids) in enumerate(pbar):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if args.amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(img)
                loss = criterion(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(img)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            metric.update(pred, mask)

        bsz = img.size(0)
        loss_sum += float(loss.item()) * bsz
        n_seen += bsz
        global_step += 1

        cur_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{cur_lr:.2e}")

        if (it + 1) % args.log_every == 0:
            logger.log(
                {"train/iter_loss": float(loss.item()), "train/lr": cur_lr},
                step=global_step,
            )

    avg_loss = loss_sum / max(1, n_seen)
    summary = metric.summary(CLASS_NAMES)
    summary["loss"] = avg_loss
    return summary, global_step


@torch.no_grad()
def evaluate(model, loader, criterion, device, save_vis_path: Optional[Path] = None,
             vis_samples: int = 8) -> dict:
    model.eval()
    metric = SegmentationMetric(NUM_CLASSES, ignore_index=IGNORE_INDEX)

    loss_sum, n_seen = 0.0, 0
    saved_vis = False
    for img, mask, _ids in tqdm(loader, desc="Val", ncols=100, leave=False):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(img)
        loss = criterion(logits, mask)
        pred = logits.argmax(dim=1)
        metric.update(pred, mask)

        bsz = img.size(0)
        loss_sum += float(loss.item()) * bsz
        n_seen += bsz

        if save_vis_path is not None and not saved_vis:
            save_prediction_grid(img, mask, pred, save_vis_path, max_samples=vis_samples,
                                 ignore_index=IGNORE_INDEX)
            saved_vis = True

    summary = metric.summary(CLASS_NAMES)
    summary["loss"] = loss_sum / max(1, n_seen)
    return summary


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vis").mkdir(exist_ok=True)
    with open(output_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}  | CUDA: {torch.cuda.is_available()}")

    # 数据
    train_loader, val_loader, test_loader, meta = build_dataloaders(
        root=args.data_root,
        crop_size=(args.crop_w, args.crop_h),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"[Data] total={meta['n_total']}, train={meta['n_train']}, "
          f"val={meta['n_val']}, test={meta['n_test']}")
    with open(output_dir / "split.json", "w", encoding="utf-8") as f:
        json.dump(
            {"train_ids": meta["train_ids"], "val_ids": meta["val_ids"],
             "test_ids": meta["test_ids"]},
            f, indent=2,
        )

    # 模型
    model = UNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base_channels=args.base_channels,
        bilinear=args.bilinear,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] UNet, params = {n_params/1e6:.2f} M, bilinear={args.bilinear}")

    # 损失
    criterion = build_loss(
        args.loss, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX,
        alpha=args.ce_weight, beta=args.dice_weight,
    ).to(device)
    print(f"[Loss] {args.loss}  (alpha={args.ce_weight}, beta={args.dice_weight})")

    # 优化器 / 调度器
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, steps_per_epoch=len(train_loader))

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    # 恢复
    start_epoch = 0
    best_miou = 0.0
    if args.resume is not None and Path(args.resume).is_file():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if scheduler is not None and "scheduler" in ck:
            scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck.get("epoch", 0) + 1
        best_miou = ck.get("best_miou", 0.0)
        print(f"[Resume] from {args.resume} @ epoch {start_epoch}, best_mIoU={best_miou:.4f}")

    # 日志
    exp_name = args.exp_name or f"unet-{args.loss}-{int(time.time())}"
    logger = Logger(
        tool=args.log_tool, project=args.project, name=exp_name,
        config=vars(args), log_dir=str(output_dir / "swanlog"),
    )

    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_summary, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            scaler, logger, epoch, args, global_step,
        )
        val_summary = evaluate(
            model, val_loader, criterion, device,
            save_vis_path=output_dir / "vis" / f"epoch_{epoch:03d}.png",
            vis_samples=args.vis_samples,
        )
        elapsed = time.time() - t0

        # 控制台
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_summary['loss']:.4f} train_mIoU={train_summary['mIoU']:.4f} | "
            f"val_loss={val_summary['loss']:.4f} val_mIoU={val_summary['mIoU']:.4f} "
            f"val_acc={val_summary['pixel_acc']:.4f} | {elapsed:.1f}s"
        )

        # 远程 log
        log_payload = {
            "epoch": epoch,
            "train/loss": train_summary["loss"],
            "train/mIoU": train_summary["mIoU"],
            "train/pixel_acc": train_summary["pixel_acc"],
            "val/loss": val_summary["loss"],
            "val/mIoU": val_summary["mIoU"],
            "val/pixel_acc": val_summary["pixel_acc"],
        }
        # 各类 IoU 也上报, 便于后续分析
        for k, v in val_summary.items():
            if k.startswith("IoU/"):
                log_payload[f"val/{k}"] = v
        logger.log(log_payload, step=global_step)

        # 保存权重
        is_best = val_summary["mIoU"] > best_miou
        if is_best:
            best_miou = val_summary["mIoU"]
        ck = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_miou": best_miou,
            "args": vars(args),
            "class_names": CLASS_NAMES,
        }
        torch.save(ck, output_dir / "last.pt")
        if is_best:
            torch.save(ck, output_dir / "best.pt")
            print(f"  -> saved best.pt  (mIoU={best_miou:.4f})")

    # 训练结束后, 用 best.pt 在测试集上做最终评估
    print("\n========== Final evaluation on test set (best.pt) ==========")
    if (output_dir / "best.pt").is_file():
        ck = torch.load(output_dir / "best.pt", map_location=device)
        model.load_state_dict(ck["model"])
    test_summary = evaluate(
        model, test_loader, criterion, device,
        save_vis_path=output_dir / "vis" / "test_final.png",
        vis_samples=args.vis_samples,
    )
    print(f"[Test] loss={test_summary['loss']:.4f} mIoU={test_summary['mIoU']:.4f} "
          f"pixel_acc={test_summary['pixel_acc']:.4f}")
    for k, v in test_summary.items():
        if k.startswith("IoU/"):
            print(f"   {k}: {v:.4f}")

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2, ensure_ascii=False)

    log_payload = {f"test/{k}": v for k, v in test_summary.items()}
    logger.log(log_payload, step=global_step)
    logger.finish()


if __name__ == "__main__":
    main()
