#!/bin/bash
# 把 20000 局 raw 样本分成 4 份，并行计算 MC value label
set -e
source /home/scroll/miniforge3/etc/profile.d/conda.sh
conda activate mahjong
export PYTHONPATH=.

N_ROLLOUTS=${1:-4}
N_WORKERS=${2:-32}
TIMEOUT=${3:-30}
MAX_STEPS=${4:-200}

for PART in 0 1 2 3; do
    RAW="output/selfplay_raw_20000_part${PART}.pkl"
    OUT="output/nn_training_data_selfplay_fast_rollout_20000_part${PART}.npz"
    echo "Starting part $PART: $RAW -> $OUT"
    python scripts/compute_mc_values.py $RAW $OUT $N_ROLLOUTS $N_WORKERS $TIMEOUT $MAX_STEPS 2000 > output/compute_mc_values_fast_20000_part${PART}.log 2>&1 &
done

wait
echo "All 4 parts done."
