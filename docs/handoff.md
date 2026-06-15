# Handoff：换机器继续工作的指南

> 本文档记录当前项目状态、最强配置、已 push 的数据/checkpoint，以及建议的后续路径。换到新机器后先读这篇。

---

## 1. 当前最强配置

默认 V3-NN agent：

```python
BeliefExpectimaxV3Agent(
    'V3-NN',
    expectimax_depth=1,
    max_candidates=5,
    leaf_evaluator='nn',
    candidate_policy='baseline_eval1'
)
```

对应模型（PyTorch `.pt`）：

- `output/nn_model.pt` + `output/nn_model_config.json`
  - Policy-Value Net，`hidden_dim=256`
- `output/nn_value_model_mc.pt` + `output/nn_value_model_mc_config.json`
  - Deep Value Net，`hidden_dims=[512,256,128]`

500 局 benchmark（4 GPU 并行，200 局初筛 + 400 局确认）：

```
Baseline    : win 27.2%, deal-in 25.6%, Elo 1578, 321ms
BeliefExp   : win 30.0%, deal-in 14.2%, Elo 1569, 214ms
V3-NN (BE1) : win 25.8%, deal-in 14.0%, Elo 1524, 170ms
V3-NN-PC    : win 15.0%, deal-in 17.6%, Elo 1329, 153ms
```

V3-NN-BE1 相对上一版 best（Elo ~1503）提升约 +21，点炮率明显降低。

> 注意：V3-NN-PC 本次训练后显著下降，因此 **current best 仅采用 V3-NN-BE1**，NN policy candidate 需要单独再优化。

---

## 2. 已 push 的数据与 Checkpoint

以下文件已加入 git（模型权重在 `.gitignore` 里默认被忽略，用 `-f` 强制跟踪）：

| 文件 | 说明 |
|---|---|
| `output/nn_model.pt` | 当前 policy-value 网络权重（PyTorch） |
| `output/nn_model_config.json` | policy net 配置（framework=pytorch, input_dim=175, hidden_dim=256） |
| `output/nn_value_model_mc.pt` | 当前 deep value 网络权重（PyTorch） |
| `output/nn_value_model_mc_config.json` | value net 配置（framework=pytorch, arch=deep, hidden_dims=[512,256,128]） |
| `output/nn_training_data_mc.npz` | 46k 条历史 MC 数据（BeliefExp 自对弈 + eval0 rollout） |
| `output/nn_training_data_selfplay.npz` | 50k 条历史 V3-NN 自对弈数据 |
| `output/nn_training_data_merged.npz` | 96k 条合并训练数据 |

到新机器后，拉下来即可直接跑 benchmark 或继续训练。

---

## 3. 环境要求

- Python 3.10
- conda: `/home/scroll/miniforge3`
- PyTorch CUDA 用于 NN 训练/推理：
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  pip install numba numpy cython
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

### 5.1 已验证：fast rollout 能加速 MC value 计算

- 已实现 `algo/eval/fast_eval.py`：用 v3 Numba shanten/ukeire/wait 做一 ply 快速评估，替代 legacy eval2 作为 MC rollout policy。
- MC value 计算速度提升约 15×（12835 样本 × 4 rollouts 从 ~1.5h 降到 ~50s）。
- 2000 局纯 fast rollout 数据训练的 V3-NN-BE1 Elo 1524，超过旧 best（~1503）+21，已替换 current best。

### 5.2 继续放大 fast rollout 数据规模（推荐下一步）

既然速度瓶颈解除，可以跑 5000–10000 局自对弈 + fast rollout + 纯 fast rollout 训练，看 V3-NN-BE1 是否还能再提升。

```bash
# 1. 生成 5000 局（4 GPU 约 25 分钟）
bash scripts/generate_selfplay_4gpu.sh 5000 32

# 2. 合并 pkl
PYTHONPATH=. python -c "
import pickle, glob
all_samples = []
for pkl in sorted(glob.glob('output/selfplay_raw_1000_gpu*.pkl')):
    with open(pkl, 'rb') as f:
        all_samples.extend(pickle.load(f))
with open('output/selfplay_raw_5000.pkl', 'wb') as f:
    pickle.dump(all_samples, f)
print(len(all_samples))
"

# 3. 计算 MC value（fast rollout, 4 rollouts, 128 workers，约 5 分钟）
PYTHONPATH=. python scripts/compute_mc_values.py output/selfplay_raw_5000.pkl output/nn_training_data_selfplay_fast_rollout_5000.npz 4 128 30 200 1000

# 4. 训练
PYTHONPATH=. python scripts/train_nn.py output/nn_training_data_selfplay_fast_rollout_5000.npz 60 256 0.001 256
PYTHONPATH=. python scripts/train_value_net_mc.py output/nn_training_data_selfplay_fast_rollout_5000.npz 60 256 0.001 512,256,128

# 5. benchmark（4 GPU）
bash scripts/benchmark_4gpu.sh 400 4
```

### 5.3 单独优化 NN policy candidate

V3-NN-PC 本次下降明显，可以：
- 用更大/更干净的 NN policy 训练数据；
- 或者训练一个专门的 fast NN policy 用于 candidate generation。

### 5.4 其他长期方向

- 用 DetMCTS / MCTS 替代 ExpectiMax；
- 加入更丰富的特征（对手报听、壁牌、筋牌）；
- 尝试 Expert Iteration / outcome 加权训练。
