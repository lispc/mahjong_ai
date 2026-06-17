# 近期工作汇总：V3-NN-PC、网络训练与自对弈循环

> 本文档汇总最近引入的 NN-based agent、hybrid 算法、训练流程和实验结论，便于后续 follow。

---

## 1. 当前最强配置：V3-NN-PC

当前默认的最强 NN agent 是 **V3-NN-PC**：

```python
BeliefExpectimaxV3Agent(
    'V3-NN-PC',
    expectimax_depth=1,
    max_candidates=5,
    leaf_evaluator='nn',
    candidate_policy='nn'
)
```

与之前的 V3-NN-BE1（`candidate_policy='baseline_eval1'`）相比，V3-NN-PC 直接用训练好的 Policy Net 生成候选弃牌。实验表明这显著提升了最终 Elo。

400 局 benchmark（4 GPU 并行）：

```
Agent        win      self     ron      deal-in    draw     Elo      avg_ms
Baseline     0.285    0.075    0.210    0.253      0.035    1470     341.0
BeliefExp    0.275    0.087    0.188    0.147      0.035    1495     228.6
V3-NN        0.233    0.065    0.168    0.150      0.035    1455     171.6
V3-NN-PC     0.172    0.040    0.133    0.147      0.035    1581     155.0
```

V3-NN-PC Elo **1581**，超过 2000 局 baseline rollout 版本的 1552 约 +29，超过旧 best V3-NN-BE1（Elo ~1524）约 +57。

> 注意：**candidate_policy='baseline_eval1' 的 V3-NN 持续弱于 candidate_policy='nn' 的 V3-NN-PC**。后续默认使用 V3-NN-PC。

### 1.1 候选生成：`candidate_policy='nn'`

每次到自己回合时手牌 14 张，先用训练好的 Policy Net 给出合法弃牌的概率分布，取 top-k（默认 5）进入 expectimax。

### 1.2 叶子评估：`leaf_evaluator='nn'`

对 top-k 候选弃牌做 depth=1 expectimax，叶子节点用 Deep Value Net 评估。具体流程见下文 1.2（原 V3-NN-BE1 章节已折叠，逻辑相同）。

### 1.3 防守 tie-breaking

与 V3-NN-BE1 相同：检测到危险信号时，在进攻价值 top 候选中按危险度选最安全的弃牌。

---

## 2. 网络训练流程

### 2.1 数据

当前最强模型来自 **5000 局 baseline rollout** 数据：

- `output/nn_training_data_selfplay_baseline_rollout_5000.npz`：68,529 条；
- 由 V3-NN-PC 自对弈生成局面，再用 legacy eval2 (`algo.select`) 做 4 次 MC rollout 得到 value 标签；
- 已验证 baseline rollout 标签质量显著优于 greedy eval0 / fast rollout。

每条样本：
- `X`：175 维局面特征；
- `y`：实际弃牌的 tile index（policy 监督）；
- `v`：MC rollout 估计的期望收益 `[-1, 1]`（value 监督）；
- `q`：质量标记（0=OK，1=timeout，2=exception，3=truncated）。

### 2.2 Policy-Value Net

```bash
python scripts/train_nn.py <data.npz> <epochs> <batch> <lr> <hidden_dim>
```

输出：`output/nn_model.pt` + `output/nn_model_config.json`。

5000 局数据上：`hidden_dim=256`，40 epochs，val loss ~1.0，policy acc ~68.5%。

### 2.3 Deep Value Net

```bash
python scripts/train_value_net_mc.py <data.npz> <epochs> <batch> <lr> <hidden_dims>
```

输出：`output/nn_value_model_mc.pt` + `output/nn_value_model_mc_config.json`。

当前保留配置：`hidden_dims=[512,256,128]`。

### 2.4 自对弈 + 模型筛选门

`scripts/self_play_loop.py` 已 rarely 使用；当前主要用固定规模的自对弈 + 独立 benchmark 验证。

---

## 3. MC Rollout 标签质量实验

### 3.1 三种 rollout policy 已测试

`algo/nn/mc_value.py` 现在支持环境变量 `MJ_ROLLOUT_POLICY`：

| Rollout Policy | 说明 | 结果 |
|---|---|---|
| `baseline`（默认）| `algo.select(...)`，2-ply expectimax | ✅ 当前 best 1581 的来源 |
| `nnpolicy` | Policy Net top-1 | ❌ 训练后 Elo 仅 1386，模型退化 |
| `v3nnpc` | 完整 V3-NN-PC agent | 极慢，每 rollout ~46s，未用于大规模训练 |

### 3.2 nnpolicy rollout 为什么失败？

- Policy Net 只是“哪个弃牌最好”的模仿器，没有 value/防守/lookahead；
- 用它作为 rollout policy，生成的对局质量低，value 标签偏差大；
- 相当于用弱 teacher 蒸馏学生，导致模型性能下降。

结论：**rollout policy 必须本身是一个较强的对局 agent**，而不仅仅是 good action classifier。

### 3.3 baseline rollout 的速度与并发

在 128 CPU core 机器上：

| 配置 | 100 样本耗时 | 估算 33,590 样本/part |
|---|---|---|
| 32 workers | 92.9s | ~8.7 h |
| 64 workers | 69.6s | ~6.5 h |
| 96 workers | 63.6s | ~5.9 h |
| 4 parts × 32 workers | 严重竞争 | >20 h |
| 2 parts × 64 workers | 最优并发 | ~6.5 h/batch |

推荐：**2 parts × 64 workers**，分两批跑完 4 parts。

---

## 4. 当前最强配置（建议作为后续实验起点）

```python
BeliefExpectimaxV3Agent(
    'V3-NN-PC',
    expectimax_depth=1,
    max_candidates=5,
    leaf_evaluator='nn',
    candidate_policy='nn'
)
```

配套模型：
- `output/nn_model.pt`：policy-value net，`hidden_dim=256`；
- `output/nn_value_model_mc.pt`：deep value net，`hidden_dims=[512,256,128]`。

400 局 benchmark：

```
Agent        win      self     ron      deal-in    draw     Elo      avg_ms
Baseline     0.285    0.075    0.210    0.253      0.035    1470     341.0
BeliefExp    0.275    0.087    0.188    0.147      0.035    1495     228.6
V3-NN        0.233    0.065    0.168    0.150      0.035    1455     171.6
V3-NN-PC     0.172    0.040    0.133    0.147      0.035    1581     155.0
```

---

## 5. 10000 局 baseline rollout 实验结论

已完成的 overnight 实验：

| 配置 | V3-NN-PC Elo |
|---|---|
| 10000 局 + 256 hidden policy / 512-256-128 value | 1528 |
| 10000 局 + 1024 hidden policy / 1024-512-256 value + weight decay | 1462 |
| true best_1581（5000 局 baseline） | **1580** |

**结论**：单纯放大 baseline rollout 数据量无法提升性能。10000 局数据的 value label 质量（val_loss ~0.78）明显低于 5000 局（val_loss ~0.199）。已恢复 best_1581。

---

## 6. 下一步：DetMCTS + 特征工程

采用两阶段策略：

### 阶段一：DetMCTS + NN value 截断

在 `algo/agents/determinized_mcts.py` 已有 Flat Monte Carlo 基础上，实现 rollout 的 NN value 截断：

- rollout 不再模拟到牌局结束；
- 跑固定深度后，用 `nn_leaf` 评估叶子手牌价值；
- 目标是提升决策速度，同时保持或提升强度。

### 阶段二：特征工程

扩展当前 175 维特征到 250+ 维：

- 向听数、ukeire
- 待牌分布
- dora
- 壁牌 / 筋牌
- 对手花色偏好
- 自己的弃牌历史

扩展特征后重新生成训练数据、训练 NN，并集成到 MCTS。

### 其他备选

- 调优 V3-NN-PC 自身配置（max_candidates、depth、margin）；
- 尝试 Expert Iteration / outcome 加权训练。
