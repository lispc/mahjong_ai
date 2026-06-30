# TD(λ) 替代 MC Value Label 实施方案

> 目标：突破当前 MC rollout value label 的 teacher 上限（eval2），用 TD(λ) 让 value net 通过 bootstrap 自我提升。
>
> 关联文档：[`../handoff.md`](../handoff.md)（项目状态）、[`../reports/recent-work.md`](../reports/recent-work.md)（MC rollout 分析）、[`mahjong-ai-research-designs.md`](mahjong-ai-research-designs.md)（方案 D 现状）。

---

## 1. 动机

当前最强 V3-NN-PC（Elo 1581）来自 5000 局 baseline rollout 训练。label 管线：

```
trajectory (V3-NN-PC 自对弈)
  → 每个 decision point (s_t, a_t)
  → 对 s_t 做 8 次 MC rollout，rollout policy = algo.select (eval2)
  → label = 平均 outcome ∈ [-1, 1]
  → MSE 训练 V_net
```

**根本问题**：label ≈ `V_eval2(s_t)`。学生 NN 最多学到 eval2 的 value，无法超越。

已观察到的失败：
- 10000 局 rollout val_loss 反而比 5000 局高（0.78 vs 0.199）→ 数据量不是瓶颈
- nnpolicy rollout → Elo 1386（policy net 太弱，label 质量差）
- Expert Iteration + outcome label → Elo 1389（= λ=1 的 TD，被噪声淹没）

TD(λ) 的核心优势：label = `real outcome + NN self-bootstrap`，不依赖 eval2。

---

## 2. TD(λ) 原理

### 2.1 单玩家视角的轨迹

从某个玩家 P 的视角，一局是：

```
s_0, a_0, s_1, a_1, ..., s_{T-1}, a_{T-1}  →  outcome ∈ {+1, 0, -1}
```

- 中间 reward r_t = 0
- 终局 outcome：+1 赢 / -1 输 / 0 流局
- γ = 1（麻将单局 ≤30 步，无需折扣）

### 2.2 n-step return

```
G_t^{(n)} = V(s_{t+n})              for n < T-t
G_t^{(T-t)} = outcome               (terminal)
```

（中间 reward 为 0 时简化形式）

### 2.3 TD(λ) target

```
G_t^λ = (1-λ) * Σ_{n=1}^{T-t-1} λ^{n-1} * V(s_{t+n})  +  λ^{T-t-1} * outcome
```

- λ=0：纯 TD(0)，`G_t = V(s_{t+1})`，完全靠 bootstrap
- λ=1：纯 MC，`G_t = outcome`，等价当前 outcome label
- λ∈(0,1)：折中，bootstrap 降方差，real outcome 锚定真值

**关键性质**：`V(s')` 来自 NN 自己，没用 eval2。NN 改进 → V(s') 改进 → target 改进 → NN 进一步改进，形成自举循环。这就是 TD-Gammon / AlphaZero 突破 teacher 上限的机制。

---

## 3. 代码改动清单

### 3.1 改 `algo/agents/data_collectors.py`

`DataCollectorV3NN` 当前 buffer 项 = `{features, action, context, hand, name}`。
需新增：

- `step_idx`：本局内步序（保证顺序）
- `game_id`：局号（多局合并时区分）
- `outcome`：终局收益 +1/0/-1（game 结束后由外部回填）
- `terminal_reason`：`tsumo_win` / `ron_win` / `ron_mine` / `lose_tsumo` / `lose_ron_others` / `draw`

新增方法 `set_outcome(outcome, terminal_reason)`：把 outcome/terminal_reason 写到 buffer 每一项。

### 3.2 改 `scripts/generate_selfplay_raw.py`（生成带 outcome 的轨迹）

输出格式从「每条样本一个 tuple」改为「每局一个 trajectory dict」：

```python
{
    'game_id': int,
    'outcome': float,           # +1/0/-1，target_seat 视角
    'terminal_reason': str,
    'samples': [
        {'features', 'action', 'context', 'hand', 'name', 'step_idx'},
        ...
    ]
}
```

不再调用 `mc_value.estimate_win_rate`。生成速度比 baseline rollout 快 10×（省掉 MC rollout）。

### 3.3 新 `scripts/compute_td_lambda_targets.py`

输入：`selfplay_raw_*.pkl`（含轨迹）+ 当前 `nn_value_model_mc.pt`
输出：`nn_training_data_td.npz`，含 `X, y, v`（v 是 TD(λ) target）

核心算法（向量化版）：

```python
def compute_td_targets_for_game(samples, V_preds, lambda_):
    """
    samples: 单局样本列表，长度 T
    V_preds: np.ndarray shape (T,)，对应每个样本的 V(s_t)
    返回：np.ndarray shape (T,)，TD(λ) target
    """
    T = len(samples)
    outcome = samples[-1]['outcome']
    targets = np.zeros(T, dtype=np.float32)
    
    # 对每个 t:
    #   G_t^λ = (1-λ) * Σ_{n=1}^{T-t-1} λ^{n-1} V(s_{t+n}) + λ^{T-t-1} * outcome
    #
    # 向量化技巧：定义 W[k] = (1-λ) * λ^k for k=0..T-2
    # 则 G_t^λ = Σ_{k=0}^{T-t-2} W[k] * V(s_{t+1+k}) + λ^{T-t-1} * outcome
    
    # 预计算权重
    powers = lambda_ ** np.arange(T)  # powers[k] = λ^k
    
    for t in range(T):
        n_terms = T - t - 1  # V 项数量
        if n_terms > 0:
            # V(s_{t+1}), V(s_{t+2}), ..., V(s_{T-1}) 对应权重 (1-λ)*λ^0, (1-λ)*λ^1, ...
            v_weights = (1 - lambda_) * powers[:n_terms]
            targets[t] = np.dot(v_weights, V_preds[t+1:T])
        # terminal term
        targets[t] += powers[T - t - 1] * outcome
    
    return np.clip(targets, -1.0, 1.0)
```

V_preds 由 NN 批量推理得到（GPU，单局 batch=20~30，几毫秒）。

并行策略：每局独立，可用 `multiprocessing.Pool` 拆 4 份到 4 GPU（每 GPU 一个进程，batch inference）。但 V 推理本身已很快，单 GPU 串行 5000 局估计 < 5 分钟。

### 3.4 新 `scripts/train_value_net_td.py`

基于 `train_value_net_mc.py` 改：

1. **Warm start**：从 `nn_value_model_mc_best_1581.pt` 加载初始权重（从头训会发散）
2. **数据源**：`v` 字段读 TD(λ) target
3. **Target clipping**：`v = np.clip(v, -1.0, 1.0)`
4. **学习率降一档**：1e-3 → 5e-4（warm start 不需要太大步长）
5. **Early stopping**：val_loss 连续 5 epoch 不降就停
6. **Checkpoint**：每个 epoch 都保存 `.checkpoint.pt`，支持断点续训

输出：`output/nn_value_model_mc_td.pt` + 配置文件。**不覆盖 best_1581**。

### 3.5 后续：迭代闭环（Phase 2）

`scripts/self_play_loop.py` 改为 TD(λ) 路径：

```
Loop k=1, 2, ...:
  1. 用当前 V3-NN-PC(self_k) 自对弈 N 局 → trajectories
  2. compute_td_lambda_targets(trajectories, V_net_k, λ) → td_data_k
  3. train_value_net_td(td_data_k, init=V_net_k) → V_net_{k+1}
  4. (可选 Phase 3) policy 更新：A2C-style，policy_loss = -log π(a_t|s_t) * (G_t^λ - V(s_t))
  5. Benchmark V3-NN-PC(self_{k+1}) vs best_1581
     - Elo > best + 10（1000 局验证）→ 替换 best
     - 否则保留旧版作为对手池
```

---

## 4. 参数选择

| 参数 | 建议值 | 理由 |
|---|---|---|
| λ | 0.5 起步，扫 0.3/0.5/0.7/0.9 | 0.5 平衡 bootstrap 和 outcome |
| γ | 1.0 | 麻将单局 ≤30 步 |
| 初始 V | `best_1581` warm start | 从随机初始化 bootstrap 会发散 |
| 自对弈 policy | V3-NN-PC（当前 best） | 不能用 eval2 或更弱策略生成数据 |
| 对手池 | 保留历史 best + Baseline + BeliefExp | 防自对弈模式坍缩 |
| 单轮数据量 | 2000~5000 局 | TD label 计算无 rollout 成本，可比 MC 多 5-10× |
| 学习率 | 5e-4（warm start） | 比 from-scratch 的 1e-3 小一档 |
| Batch size | 256 | 与 MC 训练一致 |

---

## 5. 分阶段实施

### Phase 1（验证 TD target 本身可用）

目标：在不重新跑自对弈的前提下，验证 TD(λ) target 能让 val_loss 显著低于 MC 的 0.199。

步骤：
1. 改 `DataCollectorV3NN` + `generate_selfplay_raw.py` 保存 outcome
2. 4 GPU 并行跑 2000 局自对弈，每局 target_seat=0
3. 写 `compute_td_lambda_targets.py`，用 `best_1581` 算 λ=0.5 target
4. 写 `train_value_net_td.py`，warm start from `best_1581`，训 60 epochs
5. **验收**：TD val_loss < 0.199（当前 MC best）

耗时估计：
- 数据生成：4 GPU × 32 workers × 500 games，~30 分钟
- TD target 计算：~5 分钟（GPU 批量推理）
- 训练：60 epochs × ~2s = 2 分钟
- **总计：< 40 分钟**

### Phase 2（自对弈闭环，1 周）

1. 改 `self_play_loop.py` 走 TD(λ) 管线
2. 跑 3-5 轮迭代，每轮 2000 局 + benchmark
3. **验收**：V3-NN-PC Elo > 1581（1000 局验证，CI 不重叠）

### Phase 3（policy 也用 TD，2 周）

1. A2C-style policy 更新：`-log π(a_t) * (G_t^λ - V(s_t))`
2. 扫 λ，加 reward shaping（只在 deal-in 步加 -1）
3. 若 Phase 2 已突破，可跳过

---

## 6. 工程注意

### 6.1 性能

- **数据生成**：4 GPU + 32 workers/GPU，PyPy 不可用（需 torch NN 推理）。每局 V3-NN-PC ~150ms × 20 步 ≈ 3s，500 games/worker × 128 workers ≈ 12s/wall... 实际更慢，估 20-30 分钟。
- **TD target 计算**：纯 GPU 推理 + 向量化 numpy，5000 局 < 5 分钟。
- **训练**：60 epochs，单 GPU 1-2 分钟。

### 6.2 断点续跑

- `generate_selfplay_raw.py`：每 50 局 save 一次 `output/selfplay_raw_td_N_gpuX.partial.pkl`
- `compute_td_lambda_targets.py`：每 1000 局 save checkpoint `.checkpoint.npz`
- `train_value_net_td.py`：每 epoch save `.checkpoint.pt`

### 6.3 资源监控

启动长任务后 5 分钟检查：
- `nvidia-smi` 4 GPU 都 ≥80% 利用率
- `top` CPU load 接近 128 核满载
- `tail -f` 日志显示样本数稳定增长
- 内存稳定不爆

---

## 7. 已知坑与对策

1. **bootstrap 发散**：初始 V 太差 + λ 偏低 → V 往错误方向收敛。
   - 对策：必须 warm start from best_1581，首轮 λ 偏高（0.5+）

2. **distribution shift**：第 k 轮的 trajectories 是 self_k 产生的，V_net_{k+1} 训它们。
   - 对策：每轮必须重新生成数据，不能复用旧 trajectories 算新 target

3. **自对弈坍缩**：纯 self-play 容易收敛到 Nash 但偏弱。
   - 对策：对手池混合 Baseline + BeliefExp + 历史 best，每局随机抽对手

4. **outcome 稀疏性**：单局 1 个 outcome，trajectory 长 20+ 步，credit assignment 难。
   - 对策：λ ≥ 0.5，必要时加 deal-in reward shaping（deal-in 步 reward = -1）

5. **overfit to self-play distribution**：V 在 self-play 分布下拟合好，但部署到混合对手时退化。
   - 对策：benchmark 时用混合对手池，不只 self-play

---

## 8. 为什么这次有戏

回顾失败的 Expert Iteration（500 局 + outcome label → Elo 1389）：那其实就是 λ=1 的 TD，没有 bootstrap 降方差，被噪声淹没。

TD(λ) 的 λ∈(0, 1) 是关键中间地带：
- 保留 outcome 的真值锚定 → 突破 teacher 上限
- 用 bootstrap 把单局 1 个信号"摊"到 20+ 个 state 上 → 解决 outcome 稀疏性

这是当前管线还没尝试过的方向。

---

## 9. Phase 1 实验结果（2026-06-18）

### 9.1 数据生成

4 GPU × 32 workers 跑 2000 局 V3-NN-PC 自对弈，10 分钟完成，得到 26,381 个样本（13.2 step/game 平均）。轨迹完整保存 `(features, action, context, hand, step_idx, game_id, outcome, terminal_reason)`。

### 9.2 TD(λ) 训练（warm start from best_1581）

| λ | TD val_loss | 真实 outcome MSE | win/loss acc | V3-NN-PC Elo | deal-in |
|---|---|---|---|---|---|
| best_1581 (MC, 参考) | 0.84 | 0.91 | 47.5% | 1581 | 14.7% |
| 0.5 | 0.08 | 0.84 | 57.0% | 1470 | 19.0% |
| 0.7 | 0.11 | 0.79 | 61.3% | 1512 | 15.5% |
| 0.9 | 0.25 | 0.71 | 72.6% | 1411 | 18.2% |

### 9.3 Phase 2 迭代（v3→v4，bootstrap=TD v3）

| Model | 真实 outcome acc | V3-NN-PC Elo | deal-in |
|---|---|---|---|
| TD v3 (bootstrap=best_1581) | 61.3% | 1512 | 15.5% |
| TD v4 (bootstrap=TD v3) | 70.0% | 1506 | 17.0% |

迭代让 outcome 预测更好（61%→70%），但 benchmark Elo 没提升（1512→1506）。

### 9.4 关键发现

**TD value 在真实 outcome 预测上明显优于 MC value，但部署到 V3-NN-PC 后 benchmark Elo 反而下降。**

1. **TD val_loss 0.08 是虚低**：target 含 V_best1581(s') 分量（循环），val_loss 主要在测"能不能预测 best_1581 在下一状态的值"。真实 outcome 预测 MSE 才是有效指标。

2. **Elo 不与 outcome 预测准确率正相关**：λ=0.9 的 acc 72.6% 最高，但 Elo 最低（1411）。说明 V3-NN-PC 的决策质量不只取决于 value 预测准确率。

3. **`eval0 + 2.0 * nn_value` 公式是瓶颈**：best_1581 的 nn_value 是"eval2 rollout 预测"，平滑且与 eval0 互补；TD 的 nn_value 是"实际 outcome 预测"，语义不同，2.0 倍数会让 TD value 过度影响 leaf 评估，破坏攻防平衡。

4. **迭代未突破**：Phase 2 v3→v4 让 outcome acc 从 61% 升到 70%，但 Elo 没动。说明问题不在 bootstrap 质量，而在 value 的使用方式。

### 9.5 下一步：调 `MJ_NN_VALUE_COEF`

把 `nn_leaf.py` 的 `2.0 * nn_value` 改为环境变量可调，测试 `coef=1.0` 和 `coef=0.5`，看 TD value 是否能在更小权重下与 eval0 协同。

如果调 coef 也不行，则 TD(λ) 在当前 V3-NN-PC 架构下不是有效路径，需要：
- 改 leaf 公式（如纯 nn_value，无 eval0）
- 或换 policy 学习（A2C/PPO），不依赖 value 做 leaf
- 或换模型架构（ResNet 替代 MLP，能学更复杂的 value → policy 映射）

### 9.6 Coef 扫描结果

`MJ_NN_VALUE_COEF` 扫描（TD v4, λ=0.7）：

| coef | V3-NN-PC Elo | deal-in |
|---|---|---|
| 1.0 | 1400 | 17.0% |
| 2.0（默认）| 1506 | 17.0% |
| 4.0 | 1511 | 18.2% |
| best_1581 + coef=2.0（参考） | 1581 | 14.7% |

coef 在 2.0-4.0 间基本饱和，无法逼近 best_1581。**Coef 调优不能弥补语义差异**。

### 9.7 最终结论

**TD(λ) 在当前 V3-NN-PC 架构下不是有效路径。**

- TD value 在 outcome 预测上明显更强（acc 47.5% → 70%）
- 但部署到 `eval0 + coef * nn_value` 的 leaf 公式后，Elo 反而下降 70+
- 迭代（v3→v4）和 coef 扫描都不能弥补这个 gap

**根本原因**：`eval0 + coef * nn_value` 公式假设 nn_value 是 eval0 的"残差修正"。best_1581 的 MC value 满足这个假设（平滑、与 eval0 同向、量级互补）。TD value 是"实际 outcome 预测"，与 eval0 语义不同，加在一起反而互相干扰。

### 9.8 后续方向

要利用 TD value 的优势，必须避开 `eval0 + coef * nn_value` 这个 leaf 公式：

1. **纯 NN leaf**：`leaf = nn_value`，不用 eval0。需要重训 policy + value，工作量大。
2. **A2C/PPO policy 学习**：直接优化 policy，value 只做 advantage baseline。绕过 leaf 公式问题。
3. **ResNet 替代 MLP**：更深的网络可能学到 value → 候选排序的更复杂映射。
4. **接受现状，保留 best_1581**：当前 V3-NN-PC 已是项目最强，TD 路线暂停。

### 9.9 纯 NN leaf 实验（option 1）

修改 `nn_leaf.py` 支持 `MJ_NN_LEAF_MODE=pure`（`leaf = scale * nn_value`，无 eval0），扫描 scale：

| scale | V3-NN-PC Elo | deal-in | draw | 备注 |
|---|---|---|---|---|
| best_1581 residual coef=2.0（参考） | 1581 | 14.7% | 3.5% | 当前最强 |
| TD v4 residual coef=2.0 | 1506 | 17.0% | 3.5% | 上一节基线 |
| TD v4 pure scale=10 | 1402 | 18.0% | **6.8%** | draw 翻倍 |
| TD v4 pure scale=50 | 1416 | 17.7% | **7.0%** | draw 翻倍 |
| TD v4 pure scale=100 | 1386 | 19.0% | **8.5%** | draw 三倍 |

**纯 NN leaf 全军覆没**，最佳（scale=50）仍比 residual 低 90 Elo，比 best_1581 低 165 Elo。

**根本原因**：TD v4 的 nn_value 预测实际 outcome，分布偏负（mean=-0.16，因为 target_seat 在 4 个相同 agent 中胜率只有 25%）。eval0 提供的"手牌结构正信号"在 pure 模式下消失，agent 失去主动追胜的驱动，导致：
- 胜率从 17% 降到 7-9%
- 流局率从 3.5% 飙到 7-8.5%
- 点炮率仍维持 17-19%（没改善防守）

Scale 加大反而更差：放大悲观偏置，agent 更倾向"少输"而非"多赢"。

### 9.10 最终结论

**TD(λ) + 当前 V3-NN-PC 架构 = 死路。**

- Residual leaf (`eval0 + coef * nn_value`)：TD value 语义与 eval0 冲突，coef 怎么调都差 70+ Elo
- Pure leaf (`scale * nn_value`)：失去 eval0 的结构正信号，agent 不追胜，draw 翻倍
- 迭代（v3→v4）：outcome 预测更准但 Elo 不动

要突破 best_1581，必须从架构层面改：
- **A2C/PPO 直接学 policy**，value 只做 advantage baseline，绕过 leaf 公式
- **重训 value net 输出 win probability ∈ [0, 1]**（sigmoid），让 pure leaf 有正信号
- **ResNet 替代 MLP**，让 value 学更复杂的 mapping

否则，**best_1581 仍是当前架构上限**。



