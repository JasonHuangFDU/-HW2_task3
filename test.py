"""
评估 / 推理脚本:
    1. 加载训练好的权重, 在测试集上计算 mIoU / Pixel Accuracy / 每类 IoU
    2. 可选: 对单张或目录下的图像进行预测并保存可视化
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from data import (
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    StanfordBackgroundDataset,
    build_dataloaders,
)
from data.dataset import JointTransform
from models import UNet
from utils import SegmentationMetric, colorize_mask, save_prediction_grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate / Predict UNet")
    p.add_argument("--checkpoint", type=str, required=True, help="训练好的 .pt 权重")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--crop-w", type=int, default=320)
    p.add_argument("--crop-h", type=int, default=240)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", type=str, default="test", choices=["val", "test"],
                   help="在哪一份 split 上评估")
    p.add_argument("--output-dir", type=str, default="runs/eval")
    p.add_argument("--save-vis", action="store_true", default=True)
    p.add_argument("--no-save-vis", dest="save_vis", action="store_false")
    p.add_argument("--predict-image", type=str, default=None,
                   help="若指定, 只对该图像 / 该目录下所有图像做预测并保存可视化")
    p.add_argument("--bilinear", action="store_true", default=True)
    p.add_argument("--no-bilinear", dest="bilinear", action="store_false")
    p.add_argument("--base-channels", type=int, default=64)
    return p.parse_args()


def _build_model_from_checkpoint(ck: dict, args) -> UNet:
    saved_args = ck.get("args", {})
    num_classes = len(ck.get("class_names", CLASS_NAMES))
    model = UNet(
        in_channels=3,
        num_classes=num_classes,
        base_channels=saved_args.get("base_channels", args.base_channels),
        bilinear=saved_args.get("bilinear", args.bilinear),
    )
    model.load_state_dict(ck["model"])
    return model


@torch.no_grad()
def evaluate_split(model, loader, device, output_dir: Path, save_vis: bool):
    metric = SegmentationMetric(NUM_CLASSES, ignore_index=IGNORE_INDEX)
    saved = False
    for img, mask, _ids in tqdm(loader, desc="Eval", ncols=100):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(img)
        pred = logits.argmax(dim=1)
        metric.update(pred, mask)

        if save_vis and not saved:
            save_prediction_grid(img, mask, pred, output_dir / "samples.png", max_samples=8,
                                 ignore_index=IGNORE_INDEX)
            saved = True

    summary = metric.summary(CLASS_NAMES)
    return summary


@torch.no_grad()
def predict_paths(model, paths, device, args, output_dir: Path) -> None:
    """对若干图像逐张预测并保存上色结果。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = JointTransform(crop_size=(args.crop_w, args.crop_h), train=False)
    for p in paths:
        img = Image.open(p).convert("RGB")
        # 用一个 dummy mask 走标准 transform, 保持 normalize 逻辑一致
        dummy = np.zeros((img.height, img.width), dtype=np.int64)
        x, _ = transform(img, dummy)
        x = x.unsqueeze(0).to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)[0].cpu().numpy()

        rgb = np.array(img.resize((args.crop_w, args.crop_h), Image.BILINEAR))
        color = colorize_mask(pred, IGNORE_INDEX)
        side = np.concatenate([rgb, color], axis=1)
        out_name = Path(p).stem + "_pred.png"
        Image.fromarray(side).save(output_dir / out_name)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device)
    model = _build_model_from_checkpoint(ck, args).to(device)
    model.eval()
    print(f"[Loaded] {args.checkpoint}  (best_mIoU={ck.get('best_miou', float('nan')):.4f})")

    if args.predict_image is not None:
        target = Path(args.predict_image)
        if target.is_dir():
            paths = sorted(
                p for p in target.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            )
        else:
            paths = [target]
        predict_paths(model, paths, device, args, output_dir / "predict")
        print(f"[Predict] saved {len(paths)} visualizations to {output_dir/'predict'}")
        return

    train_loader, val_loader, test_loader, meta = build_dataloaders(
        root=args.data_root,
        crop_size=(args.crop_w, args.crop_h),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    loader = val_loader if args.split == "val" else test_loader
    print(f"[Data] split={args.split}, n={meta['n_'+args.split]}")

    summary = evaluate_split(model, loader, device, output_dir, args.save_vis)
    print(f"\n=== Eval on {args.split} ===")
    print(f"mIoU       : {summary['mIoU']:.4f}")
    print(f"PixelAcc   : {summary['pixel_acc']:.4f}")
    for k, v in summary.items():
        if k.startswith("IoU/"):
            print(f"  {k}: {v:.4f}")

    with open(output_dir / f"{args.split}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
