#!/bin/bash
# 在 4 个 GPU 上并行生成 TD 自对弈轨迹
# 用法: bash scripts/generate_selfplay_td_4gpu.sh <n_games> <n_workers_per_gpu> <seed_base> <target_seat>
set -e
source /home/scroll/miniforge3/etc/profile.d/conda.sh
conda activate mahjong
export PYTHONPATH=.

N_GAMES=${1:-2000}
N_WORKERS=${2:-32}
SEED_BASE=${3:-950000}
TARGET_SEAT=${4:-0}

# 每 GPU 生成 N_GAMES/4 局
PER_GPU=$((N_GAMES / 4))
TAG="td_${N_GAMES}"

for GPU in 0 1 2 3; do
    OUT="output/selfplay_${TAG}_gpu${GPU}.pkl"
    SEED=$((SEED_BASE + GPU * PER_GPU))
    GAME_OFFSET=$((GPU * PER_GPU))
    echo "Starting GPU $GPU: $PER_GPU games, seed_offset=$SEED, game_offset=$GAME_OFFSET -> $OUT"
    CUDA_VISIBLE_DEVICES=$GPU python scripts/generate_selfplay_td.py \
        $PER_GPU $N_WORKERS $OUT $SEED $TARGET_SEAT 50 \
        > output/selfplay_${TAG}_gpu${GPU}.log 2>&1 &
done

wait
echo "All 4 GPU jobs done."

# 合并 4 GPU 的 pkl
echo "Merging..."
python -c "
import pickle, glob
all_traj = []
for pkl in sorted(glob.glob('output/selfplay_${TAG}_gpu*.pkl')):
    with open(pkl, 'rb') as f:
        traj = pickle.load(f)
    all_traj.extend(traj)
    print(f'  {pkl}: {len(traj)} trajectories')
# 重排 game_id 避免重复
for i, t in enumerate(all_traj):
    t['game_id'] = i
    for j, s in enumerate(t['samples']):
        s['game_id'] = i
        s['step_idx'] = j
with open('output/selfplay_${TAG}.pkl', 'wb') as f:
    pickle.dump(all_traj, f)
print(f'Merged: {len(all_traj)} trajectories -> output/selfplay_${TAG}.pkl')
"
