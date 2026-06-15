#!/bin/bash
# 在 4 个 GPU 上并行生成 2000 局自对弈原始样本
set -e
source /home/scroll/miniforge3/etc/profile.d/conda.sh
conda activate mahjong
export PYTHONPATH=.

N_GAMES=2000
N_WORKERS=32
PER_GPU=$((N_GAMES / 4))
SEED_BASE=1000000

for GPU in 0 1 2 3; do
    OUT="output/selfplay_raw_2000_gpu${GPU}.pkl"
    SEED=$((SEED_BASE + GPU * PER_GPU))
    echo "Starting GPU $GPU: $PER_GPU games, seed $SEED -> $OUT"
    CUDA_VISIBLE_DEVICES=$GPU python scripts/generate_selfplay_raw.py $PER_GPU $N_WORKERS $OUT $SEED > output/selfplay_raw_2000_gpu${GPU}.log 2>&1 &
done

wait
echo "All 4 GPU jobs done."
