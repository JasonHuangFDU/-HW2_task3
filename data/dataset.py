"""
Stanford Background Dataset 加载器

数据集说明：
    715 张室外场景图像, 像素级标注为 8 类 + 1 类 unknown (-1):
        0: sky        1: tree       2: road      3: grass
        4: water      5: building   6: mountain  7: foreground object
    标注文件 *.regions.txt 中, 每行为一行像素的整数标签, 用空格分隔, -1 表示未知。

数据集目录约定（解压后的标准结构）:
    <root>/
        images/<id>.jpg
        labels/<id>.regions.txt
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES: List[str] = [
    "sky",
    "tree",
    "road",
    "grass",
    "water",
    "building",
    "mountain",
    "foreground",
]
NUM_CLASSES: int = len(CLASS_NAMES)
IGNORE_INDEX: int = 255  # 把原始标签 -1 (unknown) 映射成 255, 训练时 ignore


# ---------------------------------------------------------------------------
#  数据增广 / 预处理
# ---------------------------------------------------------------------------
class JointTransform:
    """同时对 image 和 mask 做几何变换, 保持像素对齐。"""

    def __init__(
        self,
        crop_size: Tuple[int, int] = (320, 240),
        scale_range: Tuple[float, float] = (0.8, 1.25),
        hflip_prob: float = 0.5,
        train: bool = True,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        self.crop_size = crop_size  # (W, H)
        self.scale_range = scale_range
        self.hflip_prob = hflip_prob
        self.train = train
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

    def __call__(self, image: Image.Image, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        cw, ch = self.crop_size

        if self.train:
            # 随机缩放
            scale = random.uniform(*self.scale_range)
            new_w = max(int(image.width * scale), cw)
            new_h = max(int(image.height * scale), ch)
            image = image.resize((new_w, new_h), Image.BILINEAR)
            mask_img = Image.fromarray(mask).resize((new_w, new_h), Image.NEAREST)
            mask = np.array(mask_img)

            # 随机裁剪
            x0 = random.randint(0, image.width - cw)
            y0 = random.randint(0, image.height - ch)
            image = image.crop((x0, y0, x0 + cw, y0 + ch))
            mask = mask[y0 : y0 + ch, x0 : x0 + cw]

            # 随机水平翻转
            if random.random() < self.hflip_prob:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask[:, ::-1].copy()
        else:
            # 验证 / 测试: 直接 resize 到固定尺寸, 保证可整除 16 (UNet 4 次下采样)
            image = image.resize((cw, ch), Image.BILINEAR)
            mask_img = Image.fromarray(mask).resize((cw, ch), Image.NEAREST)
            mask = np.array(mask_img)

        # to tensor + normalize
        img_arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        img_arr = (img_arr - self.mean) / self.std

        return torch.from_numpy(img_arr.astype(np.float32)), torch.from_numpy(mask.astype(np.int64))


# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------
class StanfordBackgroundDataset(Dataset):
    def __init__(
        self,
        root: str,
        ids: Sequence[str],
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.label_dir = self.root / "labels"
        self.ids = list(ids)
        self.transform = transform

        if not self.image_dir.is_dir() or not self.label_dir.is_dir():
            raise FileNotFoundError(
                f"未找到数据集目录: {self.image_dir} 或 {self.label_dir}\n"
                "请确认已经解压 Stanford Background Dataset 并将 images/ 与 labels/ 子目录放在同一根目录下。"
            )

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def _load_label(path: Path) -> np.ndarray:
        # .regions.txt 是整数空格分隔的二维网格, -1 表示 unknown
        arr = np.loadtxt(str(path), dtype=np.int32)
        # 把 -1 (unknown) 映射为 IGNORE_INDEX, 其他保持 0..7
        arr = np.where(arr < 0, IGNORE_INDEX, arr).astype(np.uint8)
        return arr

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        img_path = self.image_dir / f"{img_id}.jpg"
        lbl_path = self.label_dir / f"{img_id}.regions.txt"

        image = Image.open(img_path).convert("RGB")
        mask = self._load_label(lbl_path)

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            image = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
            mask = torch.from_numpy(mask)

        return image, mask, img_id


# ---------------------------------------------------------------------------
#  Split & DataLoader 工厂
# ---------------------------------------------------------------------------
def _scan_ids(root: str) -> List[str]:
    image_dir = Path(root) / "images"
    label_dir = Path(root) / "labels"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"images 目录不存在: {image_dir}")
    ids: List[str] = []
    for fp in sorted(image_dir.iterdir()):
        if fp.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stem = fp.stem
        if (label_dir / f"{stem}.regions.txt").is_file():
            ids.append(stem)
    if not ids:
        raise RuntimeError(f"在 {image_dir} 下未找到任何带 regions.txt 标注的图像。")
    return ids


def split_ids(
    all_ids: Sequence[str],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """固定随机种子, 把 ID 切成 train/val/test 三份。"""
    rng = random.Random(seed)
    ids = list(all_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    test_ids = ids[:n_test]
    val_ids = ids[n_test : n_test + n_val]
    train_ids = ids[n_test + n_val :]
    return train_ids, val_ids, test_ids


def build_dataloaders(
    root: str,
    crop_size: Tuple[int, int] = (320, 240),
    batch_size: int = 8,
    num_workers: int = 4,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """构造 train / val / test 三个 DataLoader 以及一份 split 元信息。"""
    all_ids = _scan_ids(root)
    train_ids, val_ids, test_ids = split_ids(all_ids, val_ratio, test_ratio, seed)

    train_tf = JointTransform(crop_size=crop_size, train=True)
    eval_tf = JointTransform(crop_size=crop_size, train=False)

    train_set = StanfordBackgroundDataset(root, train_ids, transform=train_tf)
    val_set = StanfordBackgroundDataset(root, val_ids, transform=eval_tf)
    test_set = StanfordBackgroundDataset(root, test_ids, transform=eval_tf)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    meta = {
        "n_total": len(all_ids),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
    }
    return train_loader, val_loader, test_loader, meta
