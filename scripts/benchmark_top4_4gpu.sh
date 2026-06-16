#!/bin/bash
# Top 4 agent 4-GPU benchmark: V3-NN-PC, V3-NN-BE1, BeliefExp, Baseline
set -e
source /home/scroll/miniforge3/etc/profile.d/conda.sh
conda activate mahjong
export PYTHONPATH=.

N_GAMES=${1:-500}
N_WORKERS=${2:-4}
SEED_BASE=${3:-1000000}

PER_GPU=$((N_GAMES / 4))

for GPU in 0 1 2 3; do
    OUT_LOG="output/benchmark_splits/top4_bench_gpu${GPU}.log"
    SEED=$((SEED_BASE + GPU * PER_GPU))
    echo "Starting GPU $GPU: $PER_GPU games, seed $SEED -> $OUT_LOG"
    CUDA_VISIBLE_DEVICES=$GPU python tmp/benchmark_top4.py $PER_GPU $N_WORKERS $SEED > $OUT_LOG 2>&1 &
done

wait
echo "All 4 GPU benchmark jobs done. Results in output/benchmark_splits/"
