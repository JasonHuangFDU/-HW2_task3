# 任务 3 — 从零搭建 U-Net + 损失函数工程
## 1. 代码结构

```
task3/
├── README.md                # 本文件: 环境配置 / 训练 / 测试说明
├── requirements.txt         # Python 依赖
├── prepare_data.py          # 一键下载并解压 Stanford Background Dataset
├── train.py                 # 训练入口: 支持 ce / dice / combined 三种损失
├── test.py                  # 评估 / 单图推理脚本
├── models/
│   ├── __init__.py
│   └── unet.py              # 从零实现的 U-Net (DoubleConv + Down + Up + OutConv)
├── data/
│   ├── __init__.py
│   └── dataset.py           # Stanford Background Dataset Dataset / DataLoader / 增广
├── losses/
│   ├── __init__.py
│   └── losses.py            # CE / 手写 Dice / Combined 损失 + build_loss 工厂函数
├── utils/
│   ├── __init__.py
│   ├── metrics.py           # 基于混淆矩阵的 SegmentationMetric (mIoU, PixelAcc, per-class IoU)
│   └── visualize.py         # 标签上色 + 三联图 (image | GT | Pred) 网格保存
├── scripts/
│   └── run_all.sh           # 一键依次跑 CE / Dice / Combined 三组实验并打印汇总
└── .gitignore
```

各模块功能简介：

| 模块 | 关键内容 |
|---|---|
| `models/unet.py` | `DoubleConv`(两次 3×3 Conv-BN-ReLU)、`Down`(MaxPool+DoubleConv)、`Up`(上采样+Skip 拼接+DoubleConv)、`OutConv`(1×1 Conv 投影到类别数)；4 次下采样 + 4 次上采样的标准 U-Net；权重 Kaiming 初始化，**不加载任何预训练参数**。 |
| `data/dataset.py` | 解析 `*.regions.txt` 文本标签（`-1` 转为 `ignore_index=255`），`JointTransform` 同步对图像和 mask 做随机缩放 / 裁剪 / 翻转 / 归一化；`build_dataloaders` 按固定随机种子切 `train/val/test = 70/15/15`。 |
| `losses/losses.py` | `CrossEntropyLoss2d`（带 `ignore_index`）、`DiceLoss`（macro 平均，多类 one-hot，手动屏蔽 unknown 像素，**作业要求自己写**）、`CombinedLoss = α·CE + β·Dice`。 |
| `utils/metrics.py` | 用 `bincount` 累积混淆矩阵，按 `epoch` 末整体计算 mIoU / PixelAcc / 各类 IoU，避免逐 batch 估算的偏差。 |
| `utils/visualize.py` | 8 类语义 Palette + `colorize_mask` + `save_prediction_grid`（每个 epoch 自动保存一张 `image \| GT \| Pred` 三联可视化）。 |
| `train.py` | 解析 CLI 参数 → 构建数据 / 模型 / 损失 / 优化器 / 调度器 → 训练循环（支持 AMP / cosine / poly LR / 断点续训）→ 每 epoch 验证 & 保存 `last.pt` / `best.pt` → 最后用 `best.pt` 在测试集上评估并写入 `test_metrics.json`。日志通过 `swanlab` 上报，未安装时自动退化为本地控制台日志。 |
| `test.py` | 加载权重在 `val` 或 `test` 上重新评估；也可对单张图或一个目录做预测并保存可视化结果。 |
| `prepare_data.py` | 自动从 Stanford 官方地址下载 `iccv09Data.tar.gz` 并解压；也可对本地已有压缩包/已解压目录做完整性校验。 |

---

## 2. 环境配置

```bash
# 1) 创建 conda 环境 
conda create -n hw2_task3 python=3.10 -y
conda activate hw2_task3

# 2) 安装 PyTorch 
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3) 安装其它依赖
pip install -r requirements.txt
```

---

## 3. 数据准备

### 方法 1：使用脚本自动下载

```bash
python prepare_data.py --output ./datasets
```

下载完成后解压到任意位置，目录应形如：

```
iccv09Data/
├── images/             # 715 张 .jpg
├── labels/             # 每张图对应 *.regions.txt / *.surfaces.txt / *.layers.txt
├── horizons.txt
└── readme.txt
```

可用以下命令做完整性校验：

```bash
python prepare_data.py --output ./datasets --skip-download
```

**类别说明**（`*.regions.txt` 中的整数值）：

| 索引 | 类别 |
|---|---|
| 0 | sky |
| 1 | tree |
| 2 | road |
| 3 | grass |
| 4 | water |
| 5 | building |
| 6 | mountain |
| 7 | foreground object |
| -1 | unknown（训练 / 评估时被忽略） |

### 数据集划分

固定随机种子 `seed=42` 切分：`train : val : test = 70% : 15% : 15%`（约 `500 / 107 / 108` 张）。每次实验的具体 ID 列表会写入 `runs/<exp>/split.json`，保证三组损失配置使用**完全相同**的数据划分，便于公平对比。

---

## 4. 训练

### 4.1 单独训练某一种损失

```bash
# 仅交叉熵
python train.py \
    --data-root ./datasets/iccv09Data \
    --loss ce \
    --epochs 100 --batch-size 8 --lr 1e-3 \
    --output-dir runs/ce --exp-name unet-ce \
    --amp

# 仅 Dice 
python train.py \
    --data-root ./datasets/iccv09Data \
    --loss dice \
    --epochs 100 --batch-size 8 --lr 1e-3 \
    --output-dir runs/dice --exp-name unet-dice \
    --amp

# 组合损失 (CE + Dice)
python train.py \
    --data-root ./datasets/iccv09Data \
    --loss combined --ce-weight 1.0 --dice-weight 1.0 \
    --epochs 100 --batch-size 8 --lr 1e-3 \
    --output-dir runs/combined --exp-name unet-combined \
    --amp
```

### 4.2 一键跑完三组对比实验

```bash
chmod +x scripts/run_all.sh
bash scripts/run_all.sh ./datasets/iccv09Data
```

脚本结束时会打印三组实验在测试集上的 `mIoU` 与 `pixel_acc` 汇总。

### 4.3 主要 CLI 参数（`python train.py --help` 查看完整列表）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data-root` | (必填) | Stanford Background Dataset 根目录 |
| `--loss` | `combined` | `ce` / `dice` / `combined` |
| `--ce-weight`, `--dice-weight` | `1.0`, `1.0` | combined 模式下的 α, β |
| `--epochs` | `100` | 训练轮数 |
| `--batch-size` | `8` | 单卡 batch size |
| `--lr` | `1e-3` | 初始学习率 |
| `--optimizer` | `adamw` | `sgd` / `adam` / `adamw` |
| `--scheduler` | `cosine` | `none` / `cosine` / `poly` / `step` |
| `--crop-w/--crop-h` | `320` / `240` | 训练 / 验证时统一的输入分辨率 |
| `--bilinear` / `--no-bilinear` | `True` | 解码端使用双线性插值或转置卷积 |
| `--amp` | `False` | 开启混合精度训练（CUDA 推荐打开） |
| `--log-tool` | `swanlab` | `swanlab` / `wandb` / `none` |
| `--resume` | `None` | 从 `.pt` checkpoint 继续训练 |
| `--seed` | `42` | 全局随机种子，决定数据划分与权重初始化 |

### 4.4 训练输出

每次训练在 `--output-dir` 下生成：

```
runs/<exp>/
├── args.json              # 本次实验的所有 CLI 参数
├── split.json             # 用到的 train/val/test ID 列表
├── last.pt                # 最近一个 epoch 的权重
├── best.pt                # 验证集 mIoU 最高的权重
├── test_metrics.json      # 训练结束后用 best.pt 在测试集上的评估结果
├── swanlog/               # swanlab 本地日志 (--log-tool swanlab 时)
└── vis/
    ├── epoch_000.png ...  # 每个 epoch 验证集上的三联图 (image | GT | Pred)
    └── test_final.png     # 测试集最终可视化
```

---

## 5. 测试 / 推理

### 5.1 在测试集上重新评估

```bash
python test.py \
    --checkpoint runs/combined/best.pt \
    --data-root ./datasets/iccv09Data \
    --split test \
    --output-dir runs/combined/eval
```

会打印 `mIoU / pixel_acc / per-class IoU`，并把 `image | GT | Pred` 三联图写到 `runs/combined/eval/samples.png`。

### 5.2 对单张图 / 文件夹内任意图像做预测

```bash
# 单张图
python test.py \
    --checkpoint runs/combined/best.pt \
    --data-root ./datasets/iccv09Data \
    --predict-image /path/to/some_image.jpg \
    --output-dir runs/combined/predict_one

# 整个目录 (会对其中所有 jpg/png 做预测)
python test.py \
    --checkpoint runs/combined/best.pt \
    --data-root ./datasets/iccv09Data \
    --predict-image /path/to/folder \
    --output-dir runs/combined/predict_dir
```

输出位于 `--output-dir/predict/<原文件名>_pred.png`，左半为输入图，右半为上色后的语义分割结果。


---

## 6. 实验设置（默认）

| 项 | 值 |
|---|---|
| 网络 | U-Net（base_channels=64, bilinear 上采样, 全部随机初始化） |
| 输入分辨率 | 320 × 240 |
| 训练 / 验证 / 测试划分 | 70% / 15% / 15%（seed=42 固定） |
| 数据增广 | 随机缩放 [0.8, 1.25] + 随机裁剪 + 50% 水平翻转 + ImageNet 归一化 |
| Batch size | 8 |
| Optimizer | AdamW（lr=1e-3, weight_decay=1e-4） |
| LR Scheduler | CosineAnnealingLR（按 step 退火） |
| Epochs | 100 |
| 混合精度 | `--amp` |
| ignore_index | 255（来自原始 -1 的 unknown 像素） |
| Loss | (a) CE  (b) Dice (手动实现, macro 平均)  (c) CE + Dice |
| 评价指标 | mIoU（主指标）、Pixel Accuracy、per-class IoU |
| 日志 | swanlab（项目名：`HW2-Task3-UNet`） |

---

