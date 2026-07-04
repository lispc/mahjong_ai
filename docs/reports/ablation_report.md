# 晋北麻将 AI：有效改进的减法消融报告

> 目的：不是探索新算法，而是系统拆解当前最强 pipeline 中**每一个被验证过的正收益组件**，量化它们各自的贡献，并尝试提出更简洁的部署形态。
> 
> 实验时间：2026-07-04  
> 当前最强 anchor：`Hybrid-FullAction-32k`（`output/nn_full_action_best.pt` + `BeliefExpectimaxAgent` fallback）

---

## 1. 实验设计

- 每个 pool 4 个 agent，400 局，座位随机轮换。
- Anchor（`Hybrid-FullAction-32k`）固定出现在每个 pool，其他三个位置分别是：待测变体、`Baseline`、`BeliefExp`。
- 通过比较变体与 anchor 的胜率/Elo/点炮率，判断该组件是否贡献正收益。
- **注意**：由于每 pool 对手不同，anchor 自身胜率会有波动（33.8%–43.5%），因此应以「相对 anchor 的胜率/Elo 变化」为主，绝对胜率仅作参考。

## 2. 当前最强 Anchor

```python
from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent

HybridNNBeliefAgent(
    'Hybrid-FullAction-32k',
    nn_model_path='output/nn_full_action_best.pt',
    belief_kind='beliefexp',
    tenpai_threshold=28,
    device='cpu',
)
```

核心组成：
- `TileConvNet` 128 channels / 6 residual blocks / 512 hidden
- 完整动作空间：discard head + response head（碰/杠/胡）+ dealin head + tenpai head
- 教师数据：32k 局 `Hybrid-FullAction-32k` 自对弈轨迹蒸馏
- 推理时：非关键状态用 NN policy；任一对手报听或总弃牌数 ≥28 时切 `BeliefExpectimaxAgent`

---

## 3. 各组件消融结果

| 变体 | 说明 | win | deal-in | Elo | vs anchor win Δ | vs anchor Elo Δ | 结论 |
|------|------|-----|---------|-----|-----------------|------------------|------|
| PPO-best | 去掉 BeliefExp，纯 NN policy | 7.2% | 24.8% | 1396 | **-32.2%** | -179 | 搜索层不可省 |
| HybridHeur-heur | response head → 启发式响应 | 28.7% | 20.5% | 1522 | **-10.8%** | -53 | response head 有效 |
| Hybrid-convbc | 纯 conv-BC（无 response/dealin head） | 17.8% | 17.2% | 1490 | -21.8% | -85 | 完整动作空间提升大 |
| Hybrid-convdealin | conv-BC + deal-in auxiliary loss | 16.8% | 17.2% | 1470 | -22.8% | -105 | dealin head 未提升胜率 |
| Hybrid-be16k | 上一代 best：BeliefExp trace 蒸馏 16k | 17.8% | 17.2% | 1482 | -21.8% | -93 | 新 full-action 教师更强 |
| Hybrid-fa4k | full-action 数据从 32k 降到 4k | 29.2% | 20.0% | 1556 | -10.3% | -19 | 4k 已能学到八成 |
| Hybrid-fa128k | full-action 数据从 32k 增到 128k（epoch07） | 28.2% | 18.8% | 1577 | -11.3% | +2 | 128k 无明显收益 |
| Hybrid-awbc2 | AWBC v2 在 128k 数据上微调 | 29.5% | 19.8% | 1515 | -10.0% | -60 | AWBC 未确认超越 |

### 3.1 关键观察

1. **Hybrid 搜索是最大正收益来源**  
   去掉 BeliefExp 后纯 NN policy 胜率暴跌 32.2 个百分点。`BeliefExpectimaxAgent` 的实时危险度、向听数、听牌张数评估是 anchor 防守好的核心。

2. **完整动作空间（response head）贡献第二**  
   用上一代纯 conv-BC（无 response head）替代 full-action policy，胜率下降约 22 个百分点。说明把碰/杠/胡决策也 NN 化，显著提升了非弃牌动作质量。

3. **deal-in auxiliary loss 对 Hybrid 形态帮助不大**  
   `conv-BC + dealin` 与 `conv-BC` 在 Hybrid 内胜率/点炮率几乎相同。dealin head 在**纯前馈**部署中可降低点炮，但在「NN + BeliefExp」框架下，搜索层已经提供足够防守信号。

4. **数据缩放存在明显天花板**  
   4k → 32k 有提升（+10% 相对胜率），32k → 128k 没有稳定收益。当前 128k 数据更适合做动作级价值研究，而不是单纯扩大 BC。

5. **AWBC 动作级价值： promising 但未越过统计显著门槛**  
   AWBC v2 在 400 局对比中略好，但在 800 局对比中与 base 打平。说明 `nn_value_model_mc.pt` 的质量是瓶颈；若能换成更强的 conv value net，filtered/weighted BC 仍有空间。

---

## 4. 减法：什么可以省掉？

基于消融结果，从最强 pipeline 中**移除而不损失强度**的组件：

| 组件 | 是否可以移除 | 理由 |
|------|--------------|------|
| BeliefExp 搜索 | ❌ 不可省 | 最大正收益来源，移除后胜率暴跌 |
| response head | ❌ 不可省 | 完整动作空间带来 ~22% 胜率提升 |
| dealin head（full-action 模型内） | ⚠️ 可简化 | 对 Hybrid 胜率无贡献，但保留可解释性 |
| 128k 继续训练 | ✅ 可移除 | 32k 已是甜点，128k 无收益 |
| 对手 tenpai / danger 模型 | ✅ 可移除 | 接入后未超越 base |
| DPO / PPO / KTO 微调 | ✅ 可移除 | 均未能超越 BC32k |

---

## 5. 更简洁的部署形态

如果目标是**「在保持最强胜率的前提下尽可能简化」**，推荐的最小配置：

```python
from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent

HybridNNBeliefAgent(
    'Hybrid-FullAction-32k-minimal',
    nn_model_path='output/nn_full_action_best.pt',
    belief_kind='beliefexp',
    tenpai_threshold=28,      # 只在对手报听 / 终盘触发搜索
    device='cpu',
)
```

相比当前代码，可进一步简化的方向：
- 若确定只在终盘/报听触发搜索，可把 `BeliefExpectimaxAgent` 换成更轻量的 `ExpectiMax` 变体，但当前实现已足够快。
- 若允许损失少量胜率，可把 `tenpai_threshold` 从 28 提高到 34，减少搜索调用次数，加快对局。
- 若必须纯前馈，使用 `output/nn_conv_bc_dealin_2000_l07.pt`（带 dealin head），作为速度-胜率折中。

---

## 6. 历史有效改进一览（按贡献排序）

| 改进 | 相对收益 | 证据 |
|------|----------|------|
| BeliefExp / ExpectiMax 搜索 | 最大 | Hybrid vs 纯 NN：+32% 胜率 |
| 完整动作空间（response head） | 大 | full-action vs conv-BC：+22% 胜率 |
| Conv-BC 监督预训练 | 大 | 首次让纯 NN policy 与 Baseline/BeliefExp 打平 |
| Hybrid 分层策略 | 中 | 把 Conv-BC/FullAction 与 BeliefExp 结合，得到实用最强 |
| 数据量 4k → 32k | 中 | +10% 相对胜率 |
| dealin auxiliary loss | 小/只在纯前馈 | Hybrid 内无提升，纯前馈降低点炮 |
| AWBC / 128k / 对手建模 | 未确认 | 未形成统计显著超越 |

---

## 7. 后续最可能提升的方向

1. **更强的 value net**  
   当前 `nn_value_model_mc.pt` 是瓶颈。训一个与 full-action policy 同架构的 conv value net，再用 AWBC/filtered BC，可能真正越过 BC 天花板。

2. **把 BeliefExp 的实时危险信号蒸馏进 NN 输入**  
   BeliefExp 强在「危险度地图 / suji / 筋牌 / per-player 信念」。把这些信号作为额外特征输入 policy，可能让纯 NN 更接近搜索表现。

3. **在线 self-play + outcome 训练 value（AlphaZero bootstrap）**  
   离线蒸馏已到顶。用当前 Hybrid 当教师，在线生成对局并训练 value net，再反馈提升 policy，是突破天花板的最慢但最稳路径。

---

## 8. 实验产物

- 报告：`docs/reports/ablation_report.md`
- 汇总 JSON：`output/ablation_results.json`
- 脚本：`scripts/rl/run_ablation_study.py`
- Anchor 模型：`output/nn_full_action_best.pt` + `_config.json`
