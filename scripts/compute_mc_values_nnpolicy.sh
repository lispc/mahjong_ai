#!/bin/bash
# 用 NN policy 作为 rollout policy 计算 MC value label（CPython，4 parts 并行）。
# 用法：bash scripts/compute_mc_values_nnpolicy.sh <raw.pkl> <out_prefix> [n_rollouts] [workers_per_part] [save_every]
set -e
source /home/scroll/miniforge3/etc/profile.d/conda.sh
conda activate mahjong
export PYTHONPATH=.
export MJ_ROLLOUT_POLICY=nnpolicy

RAW_PKL=${1:-output/selfplay_raw_2000.pkl}
OUT_PREFIX=${2:-output/nn_training_data_nnpolicy}
N_ROLLOUTS=${3:-4}
N_WORKERS=${4:-32}
SAVE_EVERY=${5:-250}

# 拆分 raw pkl 为 4 parts
python -c "
import pickle, sys
raw = pickle.load(open('$RAW_PKL','rb'))
n = 4
chunk = (len(raw) + n - 1) // n
for p in range(n):
    part = raw[p*chunk:(p+1)*chunk]
    with open('${OUT_PREFIX}_part${p}.pkl','wb') as f:
        pickle.dump(part, f)
    print(f'part{p}: {len(part)}')
"

for p in 0 1 2 3; do
    python scripts/compute_mc_values.py \
        ${OUT_PREFIX}_part${p}.pkl \
        ${OUT_PREFIX}_part${p}.npz \
        $N_ROLLOUTS $N_WORKERS 600 200 $SAVE_EVERY > ${OUT_PREFIX}_part${p}.log 2>&1 &
done

wait
echo "All 4 parts done."
