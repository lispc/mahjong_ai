#!/bin/bash
# 实验 1B：纯 policy distillation（value weight=0）+ benchmark vs base
set -e
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[1B] Training policy-only AZ model on GPU0..."
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/rl/train_alphazero.py \
    output/alphazero_trace_200.npz output/nn_full_action_data_128000.npz \
    output/nn_full_action_best.pt output/nn_full_action_az_policyonly.pt \
    --value-weight 0 --epochs 60 --batch 512 --lr 1e-4 --device cuda \
    > output/train_alphazero_policyonly.log 2>&1

echo "[1B] Benchmarking policy-only model on GPU2..."
CUDA_VISIBLE_DEVICES=2 python3 -u scripts/rl/benchmark_az_vs_base.py \
    output/nn_full_action_az_policyonly.pt 400 16 --device cuda \
    > output/benchmark_az_policyonly_400.log 2>&1

echo "[1B] Done"
