#!/usr/bin/env bash
# 一键完成损失函数对比实验:
#   1) 仅 Cross-Entropy
#   2) 仅 Dice (手动实现)
#   3) Cross-Entropy + Dice  组合损失
#
# 用法:
#   bash scripts/run_all.sh /absolute/path/to/iccv09Data
# 默认参数与下面保持一致, 需要改请直接编辑本脚本

set -e

DATA_ROOT=${1:-"./datasets/iccv09Data"}
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-8}
LR=${LR:-1e-3}
WD=${WD:-1e-4}
NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-42}
LOG_TOOL=${LOG_TOOL:-swanlab}
PROJECT=${PROJECT:-HW2-Task3-UNet}

COMMON_ARGS=(
  --data-root "$DATA_ROOT"
  --epochs "$EPOCHS"
  --batch-size "$BATCH"
  --lr "$LR"
  --weight-decay "$WD"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --log-tool "$LOG_TOOL"
  --project "$PROJECT"
  --amp
)

echo "=============================================================="
echo "[1/3] Cross-Entropy only"
echo "=============================================================="
python train.py "${COMMON_ARGS[@]}" \
  --loss ce \
  --output-dir runs/ce \
  --exp-name unet-ce

echo "=============================================================="
echo "[2/3] Dice only (manual implementation)"
echo "=============================================================="
python train.py "${COMMON_ARGS[@]}" \
  --loss dice \
  --output-dir runs/dice \
  --exp-name unet-dice

echo "=============================================================="
echo "[3/3] Combined: Cross-Entropy + Dice"
echo "=============================================================="
python train.py "${COMMON_ARGS[@]}" \
  --loss combined \
  --ce-weight 1.0 --dice-weight 1.0 \
  --output-dir runs/combined \
  --exp-name unet-combined

echo
echo "全部三组实验完成. 结果摘要:"
for d in runs/ce runs/dice runs/combined; do
  if [ -f "$d/test_metrics.json" ]; then
    echo "----- $d -----"
    python - <<PY
import json, pathlib
m = json.load(open("$d/test_metrics.json"))
print(f"mIoU      : {m['mIoU']:.4f}")
print(f"pixel_acc : {m['pixel_acc']:.4f}")
PY
  fi
done
