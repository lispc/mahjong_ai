#!/bin/bash
# 用 legacy eval2 + PyPy 计算 1000 局 MC value label，4 份并行。
# PyPy 对纯 Python 的 eval2 有 2-3x 加速，且不 import torch/numba。
set -e
source /home/scroll/miniforge3/etc/profile.d/conda.sh
conda activate pypy39
export PYTHONPATH=.

for PART in 0 1 2 3; do
    RAW="output/selfplay_raw_1000_part${PART}.pkl"
    OUT="output/nn_training_data_selfplay_baseline_rollout_1000_part${PART}.npz"
    echo "Starting part $PART (PyPy): $RAW -> $OUT"
    pypy3 scripts/compute_mc_values.py $RAW $OUT 4 32 180 200 500 > output/compute_mc_values_pypy_1000_part${PART}.log 2>&1 &
done

wait
echo "All 4 parts done."
