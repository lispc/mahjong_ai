# 方向 D 阶段报告：默听检测可学，但接入不转化为胜率（2026-07-16）

> 结论：**离线检测问题已解决**（序列特征 + 混合数据，silent AUC 0.92），
> **在线接入未产生可复现的胜率增益**（5000-pair 复跑 +0.1% [−0.4,+0.5]），
> 按 `docs/eval-protocol.md` 不晋升、方向收尾。1000-pair 筛查的 +1.0% 是
> winner's curse——协议的多重比较防护首次实战拦截成功。

---

## 1. 离线成果（真实、可复用）

默听检测是该项目第一个被训练出来的对手建模能力：

| 数据 | 模型 | silent AUC | silent wait r@5 |
|---|---|---|---|
| v1（BeliefExp 自对弈 20k，993k 样本） | no-seq（175 维计数特征） | 0.735 | 0.110 |
| v1 | seq（GRU 序列编码） | **0.913** | 0.105 |
| v2（混合池 20k，789k 样本） | no-seq | 0.828 | 0.380 |
| v2 | seq | **0.919** | 0.376 |

- **弃牌顺序携带计数特征没有的信息**（seq 稳定 +0.09~0.18 AUC）——此前
  291 维/212 维手工特征时代的失败部分源于此（顺序被丢弃）。
- 数据分布关键：BeliefExp 自对弈默听率仅 1.4%/row，混合池（Baseline 永不报听、
  Hybrid 从不报听、BeliefExp 积极报听）10.3%/row——对手建模数据必须混合池。
- 校准（v1）：阈值 0.5 → silent recall 56% @ FPR 2.5%；0.3 → 67% @ 5.6%。
- 产物：`scripts/rl/gen_seq_opp_data.py`（含 --mix、shard 断点续跑）、
  `scripts/rl/train_seq_opp_model.py`（seq/no-seq 消融、all/silent 拆分指标）、
  `output/nn_seq_opp_v1.pt` / `output/nn_seq_opp_v2.pt`、
  `output/seq_opp_data_20000.npz` / `output/seq_opp_mixed_20000.npz`。

## 2. 在线接入：五组 paired duplicate，无可复现增益

接入方式：tenpai 概率并入 `_danger_signal`，wait 分布并入 danger
（`algo/agents/belief_silent_guard_agent.py`、`hybrid_nn_besilent_agent.py`）。

| 对比 | 阈值 | pairs（seed-offset） | paired win diff | score-proxy |
|---|---|---|---|---|
| hybridsilent vs hybrid | 0.5 | 1000（940k 段） | +0.3% [−0.5,+1.1] | −0.001 |
| hybridsilent vs hybrid | 0.3 | 1000（500k） | **+1.0% [+0.0,+2.0]** | +0.027 [+0.002,+0.052] |
| hybridsilent vs hybrid | 0.3 | **5000（0）复跑** | **+0.1% [−0.4,+0.5]** | +0.005 [−0.006,+0.015] |
| hybridsilent vs hybrid | 0.2 | 1000（600k） | −0.6% [−1.4,+0.2] | +0.000 |
| besilent vs beliefexp | 0.5 | 1000（0） | −0.6% [−2.1,+0.9] | +0.004 |

- 唯一显著项（0.3 @ 1000 pairs）在 5000-pair 独立种子复跑中消失；
  阈值趋势非单调（0.5→+0.3、0.3→+1.0、0.2→−0.6）= 噪声形态。
- 机制探针（20 局 vs Baseline 对手）：模型在启发式之外新增触发 5.6% 的决策、
  改变 7.5% 的弃牌选择——**决策确实被改变，但结局不变**。解读：safe set 内
  候选 EV 接近（改动≈中性重排），或触发状态下无安全替代（被迫进攻），
  或防守收益被进攻损失抵消（score-proxy 恒 ≈0）。

## 3. 结论与定位

1. 默听检测本身已解决且廉价（20k 局混合数据 + GRU 小模型，小时级）。
2. **检测 ≠ 可变现**：经 safe-mode 触发 + wait-danger 重排的接入路径，
   真实效应 ≤0.5pp（5000-pair CI）。与项目历史上所有防守接入失败
   （opponent model、wait_dist、danger 特征、exact solver）同型：
   防守信息不是当前 best 的瓶颈，攻防权衡已被 BeliefExp 搜索层吃透。
3. 若未来要再试，剩余可动的只有：把 tenpai/wait 概率作为**搜索叶值的连续修正项**
   （而非离散触发 + margin 重排），或接入进攻侧（对手默听时抢攻）。
   预期收益均低，不推荐优先。
4. 方向 D 关闭。序列特征管线与模型保留——若项目重启 AlphaZero 式
   自对弈迭代，默听/待牌 head 应作为**辅助任务**挂到主网络
   （multi-task 正则），而不是独立防守模块。

## 4. 数据与日志索引

- 训练日志：`output/train_seq_opp_v{1,2}(_noseq).log`
- benchmark：`output/duplicate_hybridsilent(02,03)_vs_hybrid_{1000,5000}.{pkl,log}`、
  `output/duplicate_besilent_vs_beliefexp_1000.{pkl,log}`
- 设计文档：`docs/designs/silent-tenpai-seq-model.md`（含三级门结果）
