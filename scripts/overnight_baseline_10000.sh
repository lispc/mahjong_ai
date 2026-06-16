#!/bin/bash
# 过夜跑 10000 局 baseline rollout MC value，训练并 benchmark
set -e
source /home/scroll/miniforge3/etc/profile.d/conda.sh
conda activate mahjong
export PYTHONPATH=.
export MJ_ROLLOUT_POLICY=baseline

ROOT="/home/scroll/personal/mahjong_ai"
cd "$ROOT"

# 备份当前 best 模型（万一新模型更差）
BACKUP_DIR="output/best_before_overnight"
mkdir -p "$BACKUP_DIR"
cp output/nn_model.pt output/nn_model_config.json \
   output/nn_value_model_mc.pt output/nn_value_model_mc_config.json "$BACKUP_DIR/" || true

echo "=== Phase 1: baseline rollout part0 + part1 (64 workers each) ==="
python scripts/compute_mc_values.py \
    output/selfplay_raw_10000_part0.pkl \
    output/nn_training_data_baseline_rollout_10000_part0.npz 4 64 600 200 1000 > output/baseline_rollout_part0.log 2>&1 &
PID0=$!
python scripts/compute_mc_values.py \
    output/selfplay_raw_10000_part1.pkl \
    output/nn_training_data_baseline_rollout_10000_part1.npz 4 64 600 200 1000 > output/baseline_rollout_part1.log 2>&1 &
PID1=$!
wait $PID0 $PID1

echo "=== Phase 2: baseline rollout part2 + part3 (64 workers each) ==="
python scripts/compute_mc_values.py \
    output/selfplay_raw_10000_part2.pkl \
    output/nn_training_data_baseline_rollout_10000_part2.npz 4 64 600 200 1000 > output/baseline_rollout_part2.log 2>&1 &
PID2=$!
python scripts/compute_mc_values.py \
    output/selfplay_raw_10000_part3.pkl \
    output/nn_training_data_baseline_rollout_10000_part3.npz 4 64 600 200 1000 > output/baseline_rollout_part3.log 2>&1 &
PID3=$!
wait $PID2 $PID3

echo "=== Phase 3: merge ==="
python scripts/merge_mc_parts.py \
    output/nn_training_data_baseline_rollout_10000_part0.npz \
    output/nn_training_data_baseline_rollout_10000_part1.npz \
    output/nn_training_data_baseline_rollout_10000_part2.npz \
    output/nn_training_data_baseline_rollout_10000_part3.npz \
    output/nn_training_data_baseline_rollout_10000.npz

echo "=== Phase 4: train policy-value net + value net ==="
python scripts/train_nn.py output/nn_training_data_baseline_rollout_10000.npz 40 256 1e-3 256 > output/train_baseline_10000_pv.log 2>&1
python scripts/train_value_net_mc.py output/nn_training_data_baseline_rollout_10000.npz 80 256 1e-3 512,256,128 > output/train_baseline_10000_v.log 2>&1

echo "=== Phase 5: benchmark 400 games ==="
rm -f output/benchmark_splits/results_seed*.pkl
bash scripts/benchmark_4gpu.sh 400 4 > output/benchmark_baseline_10000_400.log 2>&1

echo "=== Phase 6: summarize ==="
python - <<'PY' > output/benchmark_baseline_10000_summary.log 2>&1
import glob, os, pickle, sys
sys.path.insert(0, '.')
from checker.report import compute_metrics, compute_elo
files = sorted(glob.glob('output/benchmark_splits/results_seed*.pkl'), key=os.path.getmtime, reverse=True)[:4]
print('Using:', files)
all_results = []
for f in files:
    with open(f,'rb') as fh:
        all_results.extend(pickle.load(fh))
print('Total games:', len(all_results))
names = ['Baseline','BeliefExp','V3-NN','V3-NN-PC']
metrics = compute_metrics(all_results, names)
elo = compute_elo(all_results, names)
print('\nCombined results:')
print(f"{'Agent':<12} {'win':<8} {'self':<8} {'ron':<8} {'deal-in':<10} {'draw':<8} {'Elo':<8} {'avg_ms':<10}")
for n in names:
    m = metrics[n]
    print(f"{n:<12} {m['win_rate']:<8.3f} {m['self_rate']:<8.3f} "
          f"{m['ron_rate']:<8.3f} {m['deal_in_rate']:<10.3f} "
          f"{m['draw_rate']:<8.3f} {elo[n]:<8.0f} {m['avg_decision_time'] * 1000:<10.1f}")
PY

echo "=== Done ==="
cat output/benchmark_baseline_10000_summary.log
