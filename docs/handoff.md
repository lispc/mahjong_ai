# Handoff：换机器继续工作的指南

> 本文档记录当前项目状态、最强配置、已 push 的数据/checkpoint，以及建议的后续路径。换到新机器后先读这篇。

---

## 1. 当前最强配置

经过 2000 局 legacy eval2 baseline rollout 数据训练，**V3-NN-PC** 成为当前最强配置：

```python
BeliefExpectimaxV3Agent(
    'V3-NN-PC',
    expectimax_depth=1,
    max_candidates=5,
    leaf_evaluator='nn',
    candidate_policy='nn'
)
```

400 局 benchmark（4 GPU 并行）：

```
Agent        win      self     ron      deal-in    draw     Elo      avg_ms
Baseline     0.278    0.065    0.212    0.228      0.037    1484     342.9
BeliefExp    0.275    0.085    0.190    0.130      0.037    1543     229.2
V3-NN-BE1    0.242    0.062    0.180    0.177      0.037    1421     172.8
V3-NN-PC     0.168    0.037    0.130    0.177      0.037    1552     155.2
```

V3-NN-PC Elo 1552，超过旧 best V3-NN-BE1（Elo ~1524）约 +28。

对应模型（PyTorch `.pt`）：

- `output/nn_model.pt` + `output/nn_model_config.json`
  - Policy-Value Net，`hidden_dim=256`
- `output/nn_value_model_mc.pt` + `output/nn_value_model_mc_config.json`
  - Deep Value Net，`hidden_dims=[512,256,128]`

备份：

- `output/nn_model_best_1552.pt` / `output/nn_value_model_mc_best_1552.pt`
- `output/nn_model_best_1524.pt` / `output/nn_value_model_mc_best_1524.pt`（旧 best）

> 注意：本次 2000 局训练中，**candidate_policy='baseline_eval1' 的 V3-NN 表现较差**（1421），而 candidate_policy='nn' 的 V3-NN-PC 更强。后续建议默认使用 V3-NN-PC。

---

## 2. 已 push 的数据与 Checkpoint

以下文件已加入 git（模型权重在 `.gitignore` 里默认被忽略，用 `-f` 强制跟踪）：

| 文件 | 说明 |
|---|---|
| `output/nn_model.pt` | 当前 policy-value 网络权重（PyTorch） |
| `output/nn_model_config.json` | policy net 配置 |
| `output/nn_value_model_mc.pt` | 当前 deep value 网络权重（PyTorch） |
| `output/nn_value_model_mc_config.json` | value net 配置 |
| `output/nn_training_data_selfplay_baseline_rollout_2000.npz` | 25569 条 2000 局 baseline rollout MC value 数据 |
| `output/nn_training_data_selfplay_baseline_rollout_1000.npz` | 12835 条 1000 局 baseline rollout MC value 数据 |

---

## 3. 环境要求

- Python 3.10（主环境 `mahjong`）
- conda: `/home/scroll/miniforge3`
- PyTorch CUDA 用于 NN 训练/推理：
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  pip install numba numpy cython
  ```
- **PyPy 3.9** 用于加速 legacy eval2 MC value 计算：
  ```bash
  mamba create -n pypy39 python=3.9.18=1_73_pypy pip numpy -c conda-forge -y
  ```
- **MLX 版本已备份**：`algo/nn/model_mlx.py`、`algo/nn/value_model_mlx.py`、`scripts/train_nn_mlx.py`、`scripts/train_value_net_mc_mlx.py`。由于 MLX 与 PyTorch CUDA 包版本冲突，不要在一个环境里同时安装两者。

---

## 4. 到新机器后先验证

```bash
source /home/scroll/miniforge3/etc/profile.d/conda.sh
conda activate mahjong
PYTHONPATH=. python run_tests.py
PYTHONPATH=. python tmp/benchmark_new_models.py 100 4
```

---

## 5. 建议的后续路径

### 5.1 已验证：PyPy 能大幅加速 legacy eval2 MC value 计算

- 已实现 `mc_value.py` 在 PyPy 下跳过 Numba fast_eval import，仅使用纯 Python 的 legacy eval2。
- 通过把 `algo/eval/legacy.py` 的 `_eval0_cache` 改为 `functools.lru_cache(maxsize=1_000_000)`，解决了 PyPy 长任务内存无限增长导致的 OOM / BrokenProcessPool 问题。
- 2000 局（25569 样本 × 4 rollouts）在 128 CPU core 机器上用 4 parts × 32 workers PyPy 约 23 分钟完成，比 CPython 估计快 2-3 倍。

### 5.2 继续放大 baseline rollout 数据规模（推荐下一步）

既然 PyPy 把 MC value 计算瓶颈解除，可以跑 5000–10000 局自对弈 + legacy eval2 baseline rollout，看 V3-NN-PC 是否还能再提升。

```bash
# 1. 生成 5000 局（4 GPU 约 25 分钟）
bash scripts/generate_selfplay_4gpu.sh 5000 32

# 2. 合并 pkl
source /home/scroll/miniforge3/etc/profile.d/conda.sh
conda activate mahjong
PYTHONPATH=. python -c "
import pickle, glob
all_samples = []
for pkl in sorted(glob.glob('output/selfplay_raw_5000_gpu*.pkl')):
    with open(pkl, 'rb') as f:
        all_samples.extend(pickle.load(f))
with open('output/selfplay_raw_5000.pkl', 'wb') as f:
    pickle.dump(all_samples, f)
print(len(all_samples))
"

# 3. 拆成 4 份（PyPy 每 part 32 workers，降低单进程内存压力）
PYTHONPATH=. python -c "
import pickle
n = 4
raw = pickle.load(open('output/selfplay_raw_5000.pkl','rb'))
chunk = (len(raw) + n - 1) // n
for p in range(n):
    part = raw[p*chunk:(p+1)*chunk]
    with open(f'output/selfplay_raw_5000_part{p}.pkl','wb') as f:
        pickle.dump(part, f)
"

# 4. 用 PyPy 计算 MC value（4 parts 并行，每 part 32 workers）
source /home/scroll/miniforge3/etc/profile.d/conda.sh
conda activate pypy39
export PYTHONPATH=.
for p in 0 1 2 3; do
    pypy3 scripts/compute_mc_values.py \
        output/selfplay_raw_5000_part${p}.pkl \
        output/nn_training_data_selfplay_baseline_rollout_5000_part${p}.npz \
        4 32 240 200 250 > output/compute_mc_values_pypy_5000_part${p}.log 2>&1 &
done
wait

# 5. 合并
conda activate mahjong
PYTHONPATH=. python -c "
import numpy as np, glob
parts = sorted(glob.glob('output/nn_training_data_selfplay_baseline_rollout_5000_part*.npz'))
Xs, ys, vs, qs = [], [], [], []
for f in parts:
    d = np.load(f)
    Xs.append(d['X']); ys.append(d['y']); vs.append(d['v']); qs.append(d['q'])
X = np.concatenate(Xs); y = np.concatenate(ys); v = np.concatenate(vs); q = np.concatenate(qs)
np.savez_compressed('output/nn_training_data_selfplay_baseline_rollout_5000.npz', X=X, y=y, v=v, q=q)
print(X.shape)
"

# 6. 训练
PYTHONPATH=. python scripts/train_nn.py output/nn_training_data_selfplay_baseline_rollout_5000.npz 60 256 0.001 256
PYTHONPATH=. python scripts/train_value_net_mc.py output/nn_training_data_selfplay_baseline_rollout_5000.npz 60 256 0.001 512,256,128

# 7. benchmark（4 GPU）
bash scripts/benchmark_4gpu.sh 400 4
```

### 5.3 单独优化 candidate policy

本次训练显示 `candidate_policy='nn'` 的 V3-NN-PC 明显强于 `candidate_policy='baseline_eval1'` 的 V3-NN。可以：

- 训练一个更强的 NN policy candidate；
- 或者尝试不同的 candidate 数量（max_candidates=3/5/7）组合。

### 5.4 其他长期方向

- 用 DetMCTS / MCTS 替代 ExpectiMax；
- 加入更丰富的特征（对手报听、壁牌、筋牌）；
- 尝试 Expert Iteration / outcome 加权训练。
