#!/bin/bash
# 在 4 个 GPU 上并行跑 benchmark，每 GPU 负责 1/4 局数
set -e
source /home/scroll/miniforge3/etc/profile.d/conda.sh
conda activate mahjong
export PYTHONPATH=.

N_GAMES=${1:-200}
N_WORKERS=${2:-4}
PER_GPU=$((N_GAMES / 4))

mkdir -p output/benchmark_splits
rm -f output/benchmark_splits/*.log

for GPU in 0 1 2 3; do
    OUT="output/benchmark_splits/bench_gpu${GPU}.log"
    SEED=$((GPU * PER_GPU))
    echo "Starting GPU $GPU: $PER_GPU games, seed $SEED -> $OUT"
    CUDA_VISIBLE_DEVICES=$GPU python tmp/benchmark_new_models.py $PER_GPU $N_WORKERS $SEED > "$OUT" 2>&1 &
done

wait
echo "All 4 GPU benchmark jobs done. Results in output/benchmark_splits/"
