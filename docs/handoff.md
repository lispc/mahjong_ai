# Handoff：换机器继续工作的指南

> 本文档记录当前项目状态、最强配置、已 push 的数据/checkpoint，以及建议的后续路径。换到新机器后先读这篇。

---

## 1. 当前最强配置

经过 **5000 局** legacy eval2 baseline rollout 数据训练，**V3-NN-PC** 成为当前最强配置：

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
Baseline     0.285    0.075    0.210    0.253      0.035    1470     341.0
BeliefExp    0.275    0.087    0.188    0.147      0.035    1495     228.6
V3-NN        0.233    0.065    0.168    0.150      0.035    1455     171.6
V3-NN-PC     0.172    0.040    0.133    0.147      0.035    1581     155.0
```

V3-NN-PC Elo **1581**，超过 2000 局版本的 1552 约 +29，超过旧 best V3-NN-BE1（Elo ~1524）约 +57。

对应模型（PyTorch `.pt`）：

- `output/nn_model.pt` + `output/nn_model_config.json`
  - Policy-Value Net，`hidden_dim=256`
- `output/nn_value_model_mc.pt` + `output/nn_value_model_mc_config.json`
  - Deep Value Net，`hidden_dims=[512,256,128]`

备份：

- `output/nn_model_best_1581.pt` / `output/nn_value_model_mc_best_1581.pt`
- `output/nn_model_best_1552.pt` / `output/nn_value_model_mc_best_1552.pt`
- `output/nn_model_best_1524.pt` / `output/nn_value_model_mc_best_1524.pt`（旧 best）

> 注意：**candidate_policy='baseline_eval1' 的 V3-NN 持续弱于 candidate_policy='nn' 的 V3-NN-PC**。5000 局中 V3-NN 仅 1455，而 V3-NN-PC 达到 1581。后续默认使用 V3-NN-PC。

---

## 2. 已 push 的数据与 Checkpoint

以下文件已加入 git（模型权重在 `.gitignore` 里默认被忽略，用 `-f` 强制跟踪）：

| 文件 | 说明 |
|---|---|
| `output/nn_model.pt` | 当前 policy-value 网络权重（PyTorch） |
| `output/nn_model_config.json` | policy net 配置 |
| `output/nn_value_model_mc.pt` | 当前 deep value 网络权重（PyTorch） |
| `output/nn_value_model_mc_config.json` | value net 配置 |
| `output/nn_training_data_selfplay_baseline_rollout_5000.npz` | 68529 条 5000 局 baseline rollout MC value 数据 |
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

### 5.1 已验证：CPython 64 workers 是 baseline rollout 的甜点

- 在 128 CPU core 机器上，单 part 64 workers 跑 100 样本约 70s，96/128 workers 收益很小。
- 4 parts × 32 workers 并发会严重抢 CPU/cache，反而比 2 parts × 64 workers 慢很多。
- 推荐并发策略：**2 parts × 64 workers**，两批跑完 4 parts。

### 5.2 已验证：nnpolicy rollout 不可行

- 尝试用训练好的 Policy Net top-1 作为 MC rollout policy，生成 10000 局（134,358 样本）。
- 训练出的 V3-NN-PC 在 400 局 benchmark 中 Elo 仅 **1386**，远低于当前 best **1581**。
- 结论：NN policy 作为 rollout policy 太弱，value label 质量差，导致模型退化。应继续使用 **baseline (`algo.select`) rollout**。

### 5.3 继续放大 baseline rollout 数据规模（进行中）

已推送一键过夜脚本：

```bash
bash scripts/overnight_baseline_10000.sh
```

它会自动完成：
1. part0 + part1 baseline rollout（各 64 workers）
2. part2 + part3 baseline rollout（各 64 workers）
3. 合并为 `output/nn_training_data_baseline_rollout_10000.npz`
4. 训练 policy-value net + deep value net
5. 跑 400 局 4-GPU benchmark 并汇总

预计总耗时约 **13 小时**。运行前会自动备份当前 best 模型到 `output/best_before_overnight/`。

如果需要分步手动执行，参考脚本内容即可。

### 5.3 单独优化 candidate policy

本次训练显示 `candidate_policy='nn'` 的 V3-NN-PC 明显强于 `candidate_policy='baseline_eval1'` 的 V3-NN。可以：

- 训练一个更强的 NN policy candidate；
- 或者尝试不同的 candidate 数量（max_candidates=3/5/7）组合。

### 5.4 其他长期方向

- 用 DetMCTS / MCTS 替代 ExpectiMax；
- 加入更丰富的特征（对手报听、壁牌、筋牌）；
- 尝试 Expert Iteration / outcome 加权训练。
