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

### 5.3 已放弃：单纯放大 baseline rollout 数据规模

10000 局 baseline rollout（134,358 样本）已完成，但训练出的 V3-NN-PC Elo 仅 **1528**（默认网络）和 **1462**（扩大网络 + weight decay），均低于当前 best **1580**。

关键原因：
- value net 在 10000 局数据上 val_loss 最低只能到 ~0.78，而 5000 局数据可达 ~0.199
- 单纯增加样本量没有降低 label 噪声，反而让网络难以拟合
- 扩大网络（1024 hidden policy + 1024/512/256 value）+ weight decay 也无法解决

结论：**数据量不是当前瓶颈，数据/标签质量和算法结构才是**。已恢复 true best_1581 模型。

### 5.4 当前主攻方向：特征工程（已验证 2000 局，效果不佳）

DetMCTS + NN value 截断已快速验证：400 局 benchmark Elo 仅 **1315**，远低于当前 best 1580，暂时放弃该路线。

#### 特征工程第一次尝试（2000 局 baseline rollout + 291 维输入）

已实现 291 维输入（原 175 维）：

- 手牌质量：向听数(1) + 有效进张(1) + 待牌分布(34)
- 自己的弃牌历史(34)
- 壁牌/筋牌安全度(34)
- 对手花色偏好(12)
- 保留原特征：手牌(34)、剩余牌山(34)、对手弃牌(3×34)、报听 flag(4)、进度(1)

同步修复了 `features.py` 中 `_suit_of_tile` 的 tile value 编码错误，并在 `nn_leaf.py` 中补上了 leaf value 推理时缺失的 36 维手牌质量特征。

生成 2000 局（25,315 样本）baseline rollout 数据：

- 使用 `DataCollectorBaseline`（eval0 + baseline_eval1）决策
- `MJ_FAST_ROLLOUT=1` 加速 MC rollout
- 输出：`output/nn_training_data_selfplay_baseline_rollout_2000_v2.npz`

训练结果：

- Policy net val_acc 仅 **0.346**，value net best val_loss **0.743**（严重过拟合）
- 400 局 benchmark：V3-NN-PC Elo **1379**，远低于 best_1581 的 **1581**
- V3-NN-PC 平均决策时间 **3.36s**（原 best 约 155ms），新 quality 特征显著拖慢 leaf evaluation

可能原因：

1. 数据量不足（25k vs 旧 best 68k），且来自较弱策略（eval0+baseline_eval1），分布与 V3-NN-PC 不匹配。
2. 手牌质量特征计算昂贵，leaf evaluation 调用 `eval_v2.shanten/ukeire/winning_tiles`，使 expectimax 极慢。
3. 新增特征（尤其是 suji/safety、opp_suit_pref）可能噪声大于信号。

当前已恢复 `output/nn_model.pt` / `output/nn_value_model_mc.pt` 为 best_1581（175 维）。

#### 已验证失败的其他方向

- **DetMCTS / MCTS 替代 ExpectiMax**
  - Flat MC（10 worlds × 6 candidates，fast rollout）：Elo ~1391，决策时间 ~211ms
  - NN rollout（3 worlds × 4 candidates）：Elo ~1353，决策时间 ~394ms
  - 结论：当前 DetMCTS 实现无法替代 BeliefExpectimax/V3-NN-PC。

- **V3-NN-PC 配置调优**
  - 测试了 max_candidates / expectimax_depth / defense_margin 的多种组合。
  - depth=2 极慢（100 局 4 workers 跑 46 分钟以上），不实用。
  - depth=1 的初步结果也未超过 best_1581。

- **Expert Iteration / outcome 加权训练**
  - 用当前 best_1581 V3-NN-PC 自对弈 500 局（6,845 样本），value label 用最终 outcome。
  - Policy net val_acc 仅 0.36，value net best val_loss 0.737，严重过拟合。
  - 200 局 benchmark：V3-NN-PC Elo **1389**，远低于 best_1581。
  - 结论：500 局 + outcome label 噪声太大，无法提升；扩大数据量或改用 MC value label 可能再试，但当前证据不乐观。

#### 当前状态

- **当前最强配置仍为 V3-NN-PC（175-dim，Elo 1581）**。
- `output/nn_model.pt` + `output/nn_value_model_mc.pt` 已恢复为 best_1581。

#### 仍开放的长期方向

- 收集/生成更高质量的训练数据（如人类对局、多 agent 混合对局）；
- 改进 value label（如用更强 rollout policy 或结合 outcome 与 MC value）；
- 扩展动作空间（吃、碰、杠、报听决策）；
- 对手建模与防守推理；
- 若继续特征工程，需先解决推理速度问题，并找到与 175-dim 模型迭代的路径。


