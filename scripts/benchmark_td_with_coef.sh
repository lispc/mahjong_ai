#!/bin/bash
# 用 TD value model + 自定义 NN_VALUE_COEF 跑 benchmark
# 用法: bash scripts/benchmark_td_with_coef.sh <td_model_pt> <td_config_json> <coef> <n_games>
set -e
source /home/scroll/miniforge3/etc/profile.d/conda.sh
conda activate mahjong
export PYTHONPATH=.

TD_MODEL=${1:-output/nn_value_model_mc_td_v4_lam0.7.pt}
TD_CONFIG=${2:-output/nn_value_model_mc_td_v4_lam0.7.json}
COEF=${3:-1.0}
N_GAMES=${4:-400}

# 备份当前 default
cp output/nn_value_model_mc.pt output/nn_value_model_mc.pt.pre_td_bench.bak
cp output/nn_value_model_mc_config.json output/nn_value_model_mc_config.json.pre_td_bench.bak

# 安装 TD model
cp "$TD_MODEL" output/nn_value_model_mc.pt
cp "$TD_CONFIG" output/nn_value_model_mc_config.json

# 跑 benchmark with custom coef
echo "=== Running ${N_GAMES}-game benchmark with MJ_NN_VALUE_COEF=$COEF ==="
MJ_NN_VALUE_COEF=$COEF bash scripts/benchmark_4gpu.sh "$N_GAMES" 4

# 合并
python -c "
import pickle, glob, os
from checker.report import compute_metrics, compute_elo
all_results = []
for pkl in sorted(glob.glob('output/benchmark_splits/results_seed*.pkl')):
    with open(pkl, 'rb') as f:
        r = pickle.load(f)
    all_results.extend(r)

names = ['Baseline', 'BeliefExp', 'V3-NN', 'V3-NN-PC']
metrics = compute_metrics(all_results, names)
elo = compute_elo(all_results, names)

print(f'\\n=== Benchmark Results (coef=$COEF, {len(all_results)} games) ===')
print(f\"{'Agent':<12} {'win':<8} {'self':<8} {'ron':<8} {'deal-in':<10} {'draw':<8} {'Elo':<8} {'avg_ms':<10}\")
for n in names:
    m = metrics[n]
    print(f'{n:<12} {m[\"win_rate\"]:<8.3f} {m[\"self_rate\"]:<8.3f} '
          f'{m[\"ron_rate\"]:<8.3f} {m[\"deal_in_rate\"]:<10.3f} '
          f'{m[\"draw_rate\"]:<8.3f} {elo[n]:<8.0f} {m[\"avg_decision_time\"]*1000:<10.1f}')
"

# 恢复 best_1581
cp output/nn_value_model_mc.pt.pre_td_bench.bak output/nn_value_model_mc.pt
cp output/nn_value_model_mc_config.json.pre_td_bench.bak output/nn_value_model_mc_config.json
echo "Done. Restored best_1581."
