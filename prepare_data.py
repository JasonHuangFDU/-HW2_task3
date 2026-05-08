

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path


URL = "http://dags.stanford.edu/data/iccv09Data.tar.gz"
ARCHIVE_NAME = "iccv09Data.tar.gz"


def _report_hook(blocknum: int, blocksize: int, totalsize: int) -> None:
    if totalsize <= 0:
        return
    downloaded = blocknum * blocksize
    pct = min(100.0, downloaded * 100.0 / totalsize)
    sys.stdout.write(f"\r  downloading... {pct:6.2f}% ({downloaded/1e6:.1f} / {totalsize/1e6:.1f} MB)")
    sys.stdout.flush()


def download(url: str, dst: Path) -> None:
    print(f"[Download] {url}\n        -> {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dst, _report_hook)
    print()


def extract(archive: Path, target_dir: Path) -> None:
    print(f"[Extract] {archive} -> {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(target_dir)


def verify(root: Path) -> None:
    img_dir = root / "images"
    lbl_dir = root / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise RuntimeError(f"解压后未找到 images/ 或 labels/ 子目录: {root}")
    n_img = sum(1 for _ in img_dir.glob("*.jpg"))
    n_lbl = sum(1 for _ in lbl_dir.glob("*.regions.txt"))
    print(f"[Verify] images={n_img}, labels(regions.txt)={n_lbl}")
    if n_img == 0 or n_lbl == 0:
        raise RuntimeError("数据集似乎为空, 请检查下载/解压是否成功。")


def main() -> None:
    parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=str, default="./datasets",
                        help="数据集存放根目录 (会在其下生成 iccv09Data/)")
    parser.add_argument("--archive", type=str, default=None,
                        help="若已经下载好压缩包, 直接传入路径, 跳过下载")
    parser.add_argument("--skip-download", action="store_true",
                        help="若已经解压完成, 仅做完整性校验")
    args = parser.parse_args()

    out_root = Path(args.output)
    extract_root = out_root  # 解压后会在 out_root 下生成 iccv09Data/

    if args.skip_download:
        verify(out_root / "iccv09Data")
        return

    archive_path = Path(args.archive) if args.archive else (out_root / ARCHIVE_NAME)
    if not archive_path.is_file():
        download(URL, archive_path)
    else:
        print(f"[Skip] 已存在压缩包: {archive_path}")

    extract(archive_path, extract_root)
    verify(out_root / "iccv09Data")
    print(f"\n数据集就绪: {out_root/'iccv09Data'}")
    print("训练命令示例:")
    print(f"    python train.py --data-root {out_root/'iccv09Data'} --loss combined")


if __name__ == "__main__":
    main()
