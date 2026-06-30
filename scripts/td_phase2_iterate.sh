#!/bin/bash
# Phase 2: TD 迭代。用上一轮 TD model 做 bootstrap 重算 target，训练新 model。
# 用法: bash scripts/td_phase2_iterate.sh <td_pkl> <prev_model_pt> <prev_config_json> <lambda> <new_tag>
# 例: bash scripts/td_phase2_iterate.sh output/selfplay_td_2000.pkl output/nn_value_model_mc_td_v3_lam0.7.pt output/nn_value_model_mc_td_v3_lam0.7.json 0.7 v4
set -e
source /home/scroll/miniforge3/etc/profile.d/conda.sh
conda activate mahjong
export PYTHONPATH=.

TD_PKL=${1:-output/selfplay_td_2000.pkl}
PREV_MODEL=${2:-output/nn_value_model_mc_td_v3_lam0.7.pt}
PREV_CONFIG=${3:-output/nn_value_model_mc_td_v3_lam0.7.json}
LAMBDA=${4:-0.7}
NEW_TAG=${5:-v4}

TD_TARGETS_NPZ="output/td_targets_${NEW_TAG}_lam${LAMBDA}.npz"
TD_MODEL_PT="output/nn_value_model_mc_td_${NEW_TAG}_lam${LAMBDA}.pt"
TD_MODEL_CFG="output/nn_value_model_mc_td_${NEW_TAG}_lam${LAMBDA}.json"

echo "=== Phase 2 iteration: $NEW_TAG (λ=$LAMBDA) ==="
echo "  data:       $TD_PKL"
echo "  bootstrap:  $PREV_MODEL"
echo "  warm start: $PREV_MODEL"
echo ""

echo "=== Step 1: Compute TD(λ=$LAMBDA) targets using $PREV_MODEL as V ==="
python scripts/compute_td_lambda_targets.py \
    "$TD_PKL" \
    "$TD_TARGETS_NPZ" \
    --lambda_ "$LAMBDA" \
    --shards 4 \
    --model "$PREV_MODEL" \
    --config "$PREV_CONFIG"

echo ""
echo "=== Step 2: Train value net (warm start from $PREV_MODEL) ==="
python scripts/train_value_net_td.py \
    "$TD_TARGETS_NPZ" \
    --epochs 60 \
    --batch_size 256 \
    --lr 5e-4 \
    --patience 10 \
    --init_from "$PREV_MODEL" \
    --init_config "$PREV_CONFIG" \
    --out "$TD_MODEL_PT" \
    --out_config "$TD_MODEL_CFG"

echo ""
echo "=== Step 3: Eval on real outcome ==="
python scripts/eval_td_vs_mc.py \
    "$TD_PKL" \
    "$TD_MODEL_PT" "$TD_MODEL_CFG" \
    "$PREV_MODEL" "$PREV_CONFIG" 2>&1 | tail -15

echo ""
echo "=== Step 4: Benchmark vs best_1581 ==="
bash scripts/benchmark_td_vs_best.sh \
    "$TD_MODEL_PT" "$TD_MODEL_CFG" 400

echo ""
echo "=== Phase 2 iteration $NEW_TAG complete ==="
