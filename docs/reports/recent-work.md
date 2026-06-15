# 近期工作汇总：V3-NN-BE1、网络训练与自对弈循环

> 本文档汇总最近引入的 NN-based agent、hybrid 算法、训练流程和实验结论，便于后续 follow。

---

## 1. V3-NN-BE1 算法详解

`V3-NN-BE1` 是当前默认的最强 NN agent，对应：

```python
BeliefExpectimaxV3Agent(
    name='V3-NN',
    expectimax_depth=1,
    max_candidates=5,
    leaf_evaluator='nn',
    candidate_policy='baseline_eval1'
)
```

它由两部分组成：候选生成 + 叶子评估 + 防守 tie-breaking。

### 1.1 候选生成：`candidate_policy='baseline_eval1'`

每次到自己回合时手牌 14 张，必须选一张弃牌。我们不直接枚举所有 unique 弃牌做完整的 NN expectimax（太慢），而是先用一个**轻量但稳健**的规则函数筛出 top-k 候选：

```python
top = algo.select(list(self.cur), False, metric_f=algo.eval1,
                  c=self.context)[:self.max_candidates]
```

- `algo.select` 会遍历当前手牌中所有 unique 弃牌；
- 对每张候选弃牌 `d`，构造 `hand13 = hand14 - {d}`；
- 用 `algo.eval1(hand13, self.context)` 打分；
- 按分数排序，取前 `max_candidates=5` 张进入下一步。

`algo.eval1` 是什么？

- `eval0` = 静态手牌价值（面子 + 对子数量，经过 `config.pair_coef` 加权）；
- `eval1` = 在 `eval0` 基础上做 **1-ply 期望**：
  ```
  eval1(hand13, ctx) = sum_{t in remaining} P(摸到 t | ctx) * eval0(hand13 + [t], ctx)
  ```
- `eval2` = 在 `eval1` 基础上再做 1-ply，是 baseline agent 的默认评分。

为什么选 `eval1` 而不是 `eval2` 或 `eval0`？

| 候选策略 | 含义 | 速度 | 质量 |
|---|---|---|---|
| `eval0` | 只算当前手牌静态价值 | 极快 | 偏低，点炮率高 |
| `baseline` / `eval2` | baseline 的完整 2-ply 评分 | 慢（~180ms/决策） | 稳健 |
| `baseline_eval1` | 1-ply 期望 | 快（~95ms/决策） | 与 `eval2` 几乎一样稳健 |

`eval1` 保留了“已见牌影响牌山概率”这一关键信息，又只比 `eval0` 多一层期望，速度接近纯 `eval0`。

### 1.2 叶子评估：`leaf_evaluator='nn'`

拿到 5 张候选弃牌后，对每张候选做一次 **depth=1 的 expectimax**，但叶子节点不再用 `eval0`，而是用训练好的深度价值网络 `MahjongValueNetDeep`。

具体流程（以 NN leaf depth=1 为例）：

1. 设当前候选弃牌为 `d`，得到 `hand13 = hand14 - {d}`；
2. 用 belief model 把全局剩余转换成**有效剩余** `effective_remaining`；
3. 对每张可能摸到的牌 `t`（按 effective 概率加权）：
   - `hand14' = hand13 + [t]`；
   - 如果 `hand14'` 胡牌，给 `WIN_VALUE=100`；
   - 否则枚举 `hand14'` 中所有合法弃牌 `x`，得到若干 13 张叶子手牌；
4. 把所有叶子手牌拼成一个 batch，调用 `nn_leaf.nn_leaf_values_batch(...)` 一次性得到 NN 价值；
5. 对每张摸牌，取“最好弃牌”的 NN 价值，按摸牌概率加权求和，得到该候选弃牌的期望进攻价值。

`nn_leaf` 的具体计算：

```python
value = algo.eval0(hand, empty_context) + 2.0 * NN_value(hand, ctx)
```

- `eval0` 提供强先验；
- `NN_value` 学习在 `eval0` 基础上的残差修正；
- 乘以 2 是因为 NN value head 输出接近 `[-1, 1]`，需要放大到与 `eval0` 同量级。

### 1.3 防守 tie-breaking

如果检测到危险信号（有对手报听，或某对手危险等级 ≥1），不会无脑选进攻价值最高的弃牌，而是：

1. 找进攻价值最高的候选 `best_offense`；
2. 在 `[best_offense - margin, best_offense]` 范围内保留若干候选；
3. 按 `self._aggregate_danger(disc)` 排序，选危险度最低的。

`margin` 会随听牌玩家数量增加而增大，危险局面下更偏防守。

### 1.4 为什么 V3-NN-BE1 强？

- `baseline_eval1` 候选池排除了大量“一眼差”的弃牌，让 NN 不容易选到危险牌；
- NN leaf 在候选池里做精细的 1-ply 期望评估，捕捉到 `eval1` 看不到的 multi-step 价值；
- 防守 tie-breaking  further 压低了点炮率。

200 局 benchmark 结果：

```
V3-NN (BE1): win 22.0%, deal-in 19.5%, Elo 1568, avg_time 95ms
Baseline   : win 26.5%, deal-in 24.5%, Elo 1526
```

胜率略低，但点炮率显著更低，Elo 也更高。

---

## 2. 网络训练流程

### 2.1 数据

- `output/nn_training_data_mc.npz`：46k 条，由 `BeliefExpectimaxAgent` 自对弈 + 8 次 MC rollout 生成；
- `output/nn_training_data_selfplay.npz`：50k 条，由 V3-NN 自对弈 + 4 次 MC rollout 生成；
- `output/nn_training_data_merged.npz`：96k 条，上面两者合并。

每条样本：
- `X`：175 维局面特征（手牌 34 + 有效剩余 34 + 三家弃牌 102 + 听牌 flag 4 + 进度 1）；
- `y`：实际弃牌的 tile index（policy 监督）；
- `v`：MC rollout 估计的期望收益 `[-1, 1]`（value 监督）。

### 2.2 Policy-Value Net

```bash
python scripts/train_nn.py <data.npz> <epochs> <batch> <lr> <hidden_dim>
```

输出：`output/nn_model.npz` + `output/nn_model_config.json`。

最近从 `hidden=128` 扩到 `hidden=256`，在合并数据上 val loss 从 ~1.46 降到 ~1.28，policy accuracy 从 ~55% 提升到 ~58%。

### 2.3 Deep Value Net

```bash
python scripts/train_value_net_mc.py <data.npz> <epochs> <batch> <lr> <hidden_dims>
```

输出：`output/nn_value_model_mc.npz` + `output/nn_value_model_mc_config.json`。

`MahjongValueNetDeep` 现在支持可配置 `hidden_dims`，例如：
- 默认：`512,256,128`（旧模型，稳定）；
- 实验：`1024,512,256`（在合并数据上训练反而让 V3-NN-BE1 点炮率上升；在纯 MC 数据上过拟合）。

**当前保留配置**：policy net `hidden=256` + value net `512,256,128`。

### 2.4 自对弈 + 模型筛选门

`scripts/self_play_loop.py`：

1. 用当前 best agent 自对弈生成数据；
2. 与历史 MC 数据合并；
3. 训练 candidate policy/value net；
4. candidate 和 current best 各跑 100 局 benchmark；
5. 只有 candidate 的 V3-NN Elo 比 current best 高 `elo_margin`（默认 20）时才替换。

最近一次 1000 局循环：candidate 没有超过 best，所以保留了旧模型。这说明筛选门有效避免了一次性能倒退。

---

## 3. MC Rollout 标签质量：要不要提升？

### 3.1 当前做法

`algo/nn/mc_value.py` 里，每个样本的 value 标签 `v` 来自快速 rollout：

1. 把未知牌随机分配给对手和牌山，得到一个与已见信息一致的“世界”；
2. 从当前玩家必须弃牌开始，让所有玩家按 **greedy eval0** 打完该局；
3. 重复 `n_rollouts` 次，取当前玩家收益 `+1/0/-1` 的平均。

问题：greedy eval0 是一个**很弱的 rollout policy**，它：
- 不看上下文（空 context）；
- 不防守；
- 做牌也偏局部最优。

导致 rollout 出来的对局质量低，label 方差大，value net 学到的“未来收益”信号不准。

### 3.2 可以提升什么？

核心思路：**用更强的 rollout policy 代替 greedy eval0**。

可选方案：

| Rollout Policy | 好处 | 代价 |
|---|---|---|
| **Baseline (`algo.select`)** | 比 eval0 强、稳定、有 context | 生成标签速度慢几倍 |
| **NN Policy** | 直接模仿训练好的 policy net，风格一致 | 每次 rollout 都要前向 NN，慢；需要把 NN 导入 worker |
| **BeliefExp** | 当前最强规则 agent，标签最接近真实对局 | 极慢，只能做少量 rollout |
| **增加 rollout 次数** | 降低方差，不改变偏差 | 线性增加时间 |

### 3.3 值得吗？

**值得尝试，但优先级低于“把当前 best 固定下来再生成新一轮数据”。**

理由：
- **好处**：更强的 rollout 会产生更真实、方差更小的 value 标签，value net 上限更高；
- **风险**：
  - rollout policy 本身有偏差（baseline 偏防守、NN policy 有它自己的盲点），标签会继承这些偏差；
  - 生成数据变慢，1000 局自对弈可能从 30 分钟变成几小时；
  - 如果 rollout 方差还是大，收益可能不明显。

### 3.4 推荐做法

下一轮自对弈时，把 MC rollout 的 `_greedy_discard` 换成 **baseline 弃牌**（`algo.select(..., False)`）：

```python
def _greedy_discard(hand14):
    return algo.select(hand14, _EMPTY_CONTEXT)[0]
```

- 这是成本最低的改进；
- baseline 比 greedy eval0 强很多，标签质量会提升；
- 不需要额外加载 MLX/NN，worker 仍然是纯 CPU；
- 速度可接受（比 eval0 rollout 慢，但比 NN/BeliefExp rollout 快得多）。

如果这轮数据训练出的 value net 明显更强，再考虑用 NN policy 或 BeliefExp rollout。

---

## 4. 当前最强配置（建议作为后续实验起点）

```python
BeliefExpectimaxV3Agent(
    'V3-NN',
    expectimax_depth=1,
    max_candidates=5,
    leaf_evaluator='nn',
    candidate_policy='baseline_eval1'
)
```

配套模型：
- `output/nn_model.npz`：policy-value net，`hidden_dim=256`；
- `output/nn_value_model_mc.npz`：deep value net，`hidden_dims=[512,256,128]`。

Benchmark（200 局）：

```
Baseline    : win 26.5%, deal-in 24.5%, Elo 1526
BeliefExp   : win 25.0%, deal-in 10.0%, Elo 1456
V3-NN (BE1) : win 22.0%, deal-in 19.5%, Elo 1568, avg_time 95ms
V3-NN-PC    : win 22.5%, deal-in 15.5%, Elo 1450, avg_time 92ms
```

---

## 5. 下一步可选方向

1. **立即启动新一轮自对弈 + 重训练**：用 V3-NN-BE1 生成 1000 局，MC rollout 改用 baseline；
2. **继续优化 value net**：尝试 residual connection、dropout、更长的训练、更小的学习率；
3. **加特征**：dora、向听数、ukeire、待牌数等；
4. **DetMCTS 升级**：用 NN value 截断 rollout、NN policy 做 prior。
