#!/bin/bash
# Phase 1 完整流水线：合并 TD 数据 → 计算 TD target → 训练 value net → 报告 val_loss
# 用法: bash scripts/td_phase1_train.sh <td_pkl> <lambda> <out_tag>
# 例: bash scripts/td_phase1_train.sh output/selfplay_td_2000.pkl 0.5 v1
set -e
source /home/scroll/miniforge3/etc/profile.d/conda.sh
conda activate mahjong
export PYTHONPATH=.

TD_PKL=${1:-output/selfplay_td_2000.pkl}
LAMBDA=${2:-0.5}
TAG=${3:-v1}

TD_TARGETS_NPZ="output/td_targets_${TAG}_lam${LAMBDA}.npz"
TD_MODEL_PT="output/nn_value_model_mc_td_${TAG}_lam${LAMBDA}.pt"
TD_MODEL_CFG="output/nn_value_model_mc_td_${TAG}_lam${LAMBDA}.json"

echo "=== Step 1: Compute TD(λ=$LAMBDA) targets from $TD_PKL ==="
python scripts/compute_td_lambda_targets.py \
    "$TD_PKL" \
    "$TD_TARGETS_NPZ" \
    --lambda_ "$LAMBDA" \
    --shards 4 \
    --model output/nn_value_model_mc_best_1581.pt \
    --config output/nn_value_model_mc_config_best_1581.json

echo ""
echo "=== Step 2: Train value net (warm start from best_1581) ==="
python scripts/train_value_net_td.py \
    "$TD_TARGETS_NPZ" \
    --epochs 60 \
    --batch_size 256 \
    --lr 5e-4 \
    --patience 10 \
    --init_from output/nn_value_model_mc_best_1581.pt \
    --init_config output/nn_value_model_mc_config_best_1581.json \
    --out "$TD_MODEL_PT" \
    --out_config "$TD_MODEL_CFG"

echo ""
echo "=== Phase 1 complete ==="
echo "TD targets: $TD_TARGETS_NPZ"
echo "TD model:   $TD_MODEL_PT"
echo "TD config:  $TD_MODEL_CFG"
echo ""
echo "Reference: best_1581 val_loss on 5000-game MC data ≈ 0.84"
echo "Phase 1 acceptance: TD val_loss < 0.5 (significant improvement)"
