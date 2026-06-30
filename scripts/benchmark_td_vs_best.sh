#!/bin/bash
# 用 TD value model 替换默认 value model，跑 4-GPU benchmark vs best_1581
# 用法: bash scripts/benchmark_td_vs_best.sh <td_model_pt> <td_config_json> <n_games>
set -e
source /home/scroll/miniforge3/etc/profile.d/conda.sh
conda activate mahjong
export PYTHONPATH=.

TD_MODEL=${1:-output/nn_value_model_mc_td_v1_lam0.5.pt}
TD_CONFIG=${2:-output/nn_value_model_mc_td_v1_lam0.5.json}
N_GAMES=${3:-400}

# 备份当前 default (best_1581)
echo "=== Backing up current default value model (best_1581) ==="
cp output/nn_value_model_mc.pt output/nn_value_model_mc.pt.pre_td_bench.bak
cp output/nn_value_model_mc_config.json output/nn_value_model_mc_config.json.pre_td_bench.bak

# 安装 TD model 为 default
echo "=== Installing TD model as default ==="
cp "$TD_MODEL" output/nn_value_model_mc.pt
cp "$TD_CONFIG" output/nn_value_model_mc_config.json

# 跑 4-GPU benchmark
echo "=== Running ${N_GAMES}-game 4-GPU benchmark ==="
bash scripts/benchmark_4gpu.sh "$N_GAMES" 4

# 合并 4 GPU 结果
echo "=== Merging benchmark results ==="
python -c "
import pickle, glob, os
from checker.report import compute_metrics, compute_elo
all_results = []
for pkl in sorted(glob.glob('output/benchmark_splits/results_seed*.pkl')):
    with open(pkl, 'rb') as f:
        r = pickle.load(f)
    all_results.extend(r)
    print(f'  {pkl}: {len(r)} games')

names = ['Baseline', 'BeliefExp', 'V3-NN', 'V3-NN-PC']
metrics = compute_metrics(all_results, names)
elo = compute_elo(all_results, names)

print(f'\\n=== TD Benchmark Results ({len(all_results)} games) ===')
print(f\"{'Agent':<12} {'win':<8} {'self':<8} {'ron':<8} {'deal-in':<10} {'draw':<8} {'Elo':<8} {'avg_ms':<10}\")
for n in names:
    m = metrics[n]
    print(f'{n:<12} {m[\"win_rate\"]:<8.3f} {m[\"self_rate\"]:<8.3f} '
          f'{m[\"ron_rate\"]:<8.3f} {m[\"deal_in_rate\"]:<10.3f} '
          f'{m[\"draw_rate\"]:<8.3f} {elo[n]:<8.0f} {m[\"avg_decision_time\"]*1000:<10.1f}')

# 保存到文件
with open('output/benchmark_td_summary.txt', 'w') as f:
    f.write(f'TD model: {os.environ.get(\"TD_MODEL\", \"\")}\\n')
    f.write(f'Games: {len(all_results)}\\n\\n')
    for n in names:
        m = metrics[n]
        f.write(f'{n}: win={m[\"win_rate\"]:.3f} deal_in={m[\"deal_in_rate\"]:.3f} '
                f'Elo={elo[n]:.0f} time_ms={m[\"avg_decision_time\"]*1000:.1f}\\n')
print('\\nSummary saved to output/benchmark_td_summary.txt')
"

# 恢复 best_1581
echo "=== Restoring best_1581 as default ==="
cp output/nn_value_model_mc.pt.pre_td_bench.bak output/nn_value_model_mc.pt
cp output/nn_value_model_mc_config.json.pre_td_bench.bak output/nn_value_model_mc_config.json
echo "Done."
