# 后续三大突破方向：详细分析与历史对照

> 文档时间：2026-07-04  
> 背景：完成 `docs/reports/ablation_report.md` 后，对当前最强 `Hybrid-FullAction-32k` 做了减法消融。这里把三个最有潜力的后续方向展开，列出我们历史上已做的相关实验、踩过的坑，以及下一步具体可执行的方案。

---

## 方向一：训练更强的 value net，再试 AWBC / filtered BC

### 目标

把 AWBC 里的价值基线从 `output/nn_value_model_mc.pt`（MLP 512/256/128，eval2 rollout 训练）升级成与 full-action policy 同架构、同分布的 conv value net，让 `A = R - V(s)` 更可靠，从而真正越过 BC 天花板。

### 为什么当前 AWBC 没能超越

- `nn_value_model_mc.pt` 预测的是 `[-1,1]` 的最终 seat reward；
- 它只在 **eval2 级 rollout** 上训练，而 full-action policy 已经是 eval2 级甚至更强；
- 用它做基线只能把 outcome 噪声稍微摊薄，无法识别“动作级”优劣。

### 历史相关实验

#### 1. TD(λ) 训练 value net（`docs/designs/td-lambda-plan.md`）

| 实验 | outcome acc | V3-NN-PC Elo | 结论 |
|---|---|---|---|
| best_1581 (MC value，参考) | 47.5% | 1581 | 当前最强 baseline |
| TD λ=0.7 warm start | 61.3% | 1512 | value 预测更准，但 Elo 降了 70 |
| TD λ=0.9 | 72.6% | 1411 | 越准越差 |
| 迭代 v3→v4 | 61% → 70% | 1512 → 1506 | bootstrap 不提升 |
| `MJ_NN_VALUE_COEF` 扫描 | 1.0/2.0/4.0 → 1400/1506/1511 | | 调系数救不回来 |
| pure NN leaf（无 eval0） | scale 10/50/100 → Elo 1402/1416/1386 | | 纯 value leaf 全军覆没，draw 翻倍 |

**核心教训**：`eval0 + coef * nn_value` 这个 leaf 公式假设 nn_value 是 eval0 的**残差修正**。MC value（best_1581）满足这个假设；TD value 预测真实 outcome，分布偏负、语义不同，加进去反而干扰。

#### 2. conv-BC value head 替换 leaf（`docs/reports/rl-ppo-report.md` §12.2 / `docs/handoff.md` §6.3）

- 用 `MJ_NN_VALUE_MODEL` 把 conv-BC 自己的 value head 接入 `BeliefExpectimaxV3Agent`；
- residual 模式：胜率 18.0%（< conv-BC 单独 23.0%）；
- pure 模式：scale 失配，胜率 7.5%；
- `v3deep:1-nn`（conv-BC 候选 + conv-BC value leaf）：22.0% vs conv-BC 19.3%，互角。

**结论**：当时的 conv-BC value head 没有提供超越 policy 的新信息。

#### 3. 扩展危险/防守特征（`docs/handoff.md` §6.3）

- 从 175 维扩到 212 维（suji、危险度地图、对手危险等级）；
- `nn_conv_bc_ext.pt` val acc 0.844；
- benchmark 点炮反而从 15.2% 升到 21.2%。

**结论**：网络把 danger 特征用来“更激进地搏牌”，而不是 fold。特征给了，目标函数没告诉它点炮代价高。

### 下一步可执行方案

1. **用 full-action policy 自己的 value head 做 AWBC 基线（短期，成本最低）**  
   `TileConvNet` 本来就带 `value_head`。直接在 128k 数据上微调这个 value head（MSE on outcome），替代 `nn_value_model_mc.pt`，再跑 AWBC。  
   - 优势：同架构、同状态分布，不需要额外网络；  
   - 风险：value head 可能过拟合到 outcome。

2. **训练 conv value net，但用 search trace value 当标签**  
   不预测最终 outcome，而是预测 `BeliefExpectimaxAgent` 在叶节点的 expectimax value。这样 value net 的语义与搜索一致，未来可直接做 pure NN leaf 或 residual leaf。  
   - 需生成 `(state, search_value)` 数据；  
   - 优势：避开 outcome 噪声；  
   - 风险：若 search 本身不强，value net 只是复制搜索。

3. **value net 输出改成 win probability ∈ [0,1]**（`docs/designs/td-lambda-plan.md` §9.10 建议）  
   用 sigmoid 输出正概率，解决 TD value 分布偏负、pure leaf 不追胜的问题，再重测 pure NN leaf。

4. **用 value net 做 PPO/A2C baseline，而不是 search leaf**  
   绕过 `eval0 + coef * nn_value` 公式。直接优化 policy，value 只算 advantage。历史 PPO 失败是因为稀疏 reward，但如果有好的 value baseline，可能更稳定。

---

## 方向二：把 BeliefExp 的实时危险信号蒸馏进 NN policy 输入

### 目标

消融报告显示去掉 BeliefExp 搜索后纯 NN policy 胜率暴跌 32.2%。如果能把 BeliefExp 实时计算的 per-player 信念、危险度地图、suji、筋牌、dora 等信号作为 NN 输入特征，纯 NN 可能更接近搜索表现，甚至让 Hybrid 触发搜索的次数减少。

### 历史相关实验

#### 1. 212 维扩展特征（`docs/handoff.md` §6.3）

- 加入 suji、危险度地图、对手危险等级；
- `nn_conv_bc_ext.pt` val acc 0.844（base 0.710）；
- benchmark：**点炮 21.2%**，base 15.2%。

**原因**：BC loss 对“极稀疏但极 costly 的点炮错误”不敏感。网络学会用 danger 信号去“搏危险牌”，而不是避开。

#### 2. 危险样本加权 BC（`docs/handoff.md` §6.3）

- 对高危险状态样本加权 α=2/5；
- 点炮从 21.2% 降到 19.0%（仍远高于 base）；
- α=5 时 val acc 下降。

**结论**：加权不能解决目标函数层面的问题。

#### 3. Perfect-Info Safety Oracle 蒸馏（`docs/reports/rl-ppo-report.md` §15.3.2）

- 用完美信息直接排除会即时点炮的弃牌，选 conv-BC 分数最高的安全牌；
- 把这个 oracle 的动作蒸馏成 normal policy：

| 数据 | 400 局胜率 | 点炮 |
|---|---|---|
| Mixed-safety 2000 局 | 15.2% | 20.0% |
| Oracle → distill | 13.5% | 19.5% |
| conv-BC base | 22.5–24.5% | 15.2–18.2% |

**结论**：即使能**完全避免即时点炮**的 oracle，蒸馏出来的 policy 也没有降低点炮，反而因过度保守损失胜率。

#### 4. deal-in auxiliary loss（`docs/reports/rl-ppo-report.md` §16 / `docs/handoff.md` §6.4）

- 不扩展输入，而是在目标函数里加点炮 BCE 辅助 loss；
- 用 perfect-info 即时点炮标签训练；
- 结果：**conv-BC 点炮从 19.1% 降到 14.5–16.6%**，是首次稳健降低点炮的阳性实验。

**关键对比**：改输入失败；改目标函数成功（ modest ）。

### 下一步可执行方案

1. **full-action 模型 + danger 输入 + deal-in head**  
   之前 danger 特征是在 conv-BC 上试的，且没有 deal-in head。现在 `nn_full_action_best.pt` 已带 dealin/tenpai/value/response head。可以把 BeliefExp 的 per-player 危险度地图作为额外特征输入，同时保留 deal-in auxiliary loss。  
   - 需要重新生成 128k 数据（特征变了），工程量大。

2. **用 BeliefExp 的“估计点炮概率”作为 deal-in head 的软标签**  
   当前 deal-in head 用 perfect-info 硬标签。如果改成用 BeliefExp 在信念下的期望点炮概率（软标签），则 deal-in head 学到的是“信息不完美情况下的危险度”，更符合实际部署。

3. **对比学习 / triplet loss：dangerous tile vs safe tile**  
   对每个状态构造三元组 `(危险弃牌, 安全弃牌, margin)`，直接惩罚“存在安全替代时选危险牌”。这比 BCE 更能刻画点炮代价。

4. **接受现实：不蒸馏搜索，而是让搜索更便宜**  
   既然 -32% 的 gap 说明搜索不可省，那不如把 Hybrid 的触发阈值调得更高（例如只在终盘触发），减少搜索调用次数。

---

## 方向三：在线 self-play + outcome 训练 value（AlphaZero bootstrap）

### 目标

离线蒸馏的天花板就是教师本身。用当前 Hybrid 当教师生成对局，从真实 outcome 训练 value net，再用 value net 改进 search/policy，形成自举循环。这是 AlphaZero / TD-Gammon 的范式。

### 历史相关实验

#### 1. TD(λ) value bootstrap（`docs/designs/td-lambda-plan.md`）

本质上就是 AlphaZero 的 value 训练部分：

- 自对弈 2000 局 V3-NN-PC；
- 用 TD(λ) target 训练 value net；
- 结果：outcome 预测变准，但部署到 V3-NN-PC 后 Elo 下降。

**教训**：value 训练成功了，但**使用方式错了**。`eval0 + coef * nn_value` 公式不适合 TD value。

#### 2. 两代 Bootstrap（`docs/handoff.md` §6.4 / `docs/designs/conv-bc-roadmap.md`）

| 代 | 教师 | 产物 | 400 局胜率 | 点炮 |
|---|---|---|---|---|
| 一代 | Hybrid-dealin07 | `nn_conv_bc_hybrid_2000.pt` | 25.2% | 14.5% |
| 二代 | Hybrid-hybridBase | `nn_conv_bc_hybrid_v2.pt` | 22.8% | 17.8% |

**结论**：一代有效，二代倒退，当前框架下 bootstrap 收敛。

#### 3. PPO 自对弈微调（`docs/handoff.md` §6.5 / `docs/reports/rl-ppo-report.md`）

- 从强 BC 初始化继续 PPO；
- 128k Epoch 2 启动：entropy 从 0.06 涨到 0.595，KL 连续 early-stop，Elo 1424（比初始化低 180）；
- BE16k_t8 启动：20 iter 后 Elo 1518（低于初始化 1581）。

**结论**：vanilla PPO 在强 BC 上退化，稀疏终局 reward 不稳定。

### 下一步可执行方案

1. **AlphaZero 式 MCTS + search trace distillation（中期，工程量大）**  
   我们已经知道 `BeliefExp trace distillation` 是阳性（`Hybrid-BE16k_t8` Elo 1581）。下一步升级为迭代 MCTS：
   - 用当前 policy 做 prior，value net 做 leaf；
   - MCTS 给出 visit distribution 和 value；
   - policy 目标 = MCTS visit distribution，value 目标 = MCTS value；
   - 训练新 policy/value → 下一迭代教师更强。  
   - 风险：MCTS 在麻将上慢（每步可能秒级），大规模自对弈是小时/天级工程。

2. **A2C/PPO with GAE，value 只做 baseline**  
   历史 PPO 失败是因为 value 同时被当 leaf 用、reward 稀疏。如果：
   - 用 conv value net 做 GAE baseline；
   - policy 直接优化，不经过 search leaf；
   - 用 self-play 对手池和 entropy 退火；  
   可能更稳定。但这和之前 PPO 尝试差别不大，只是 value 质量更好。

3. **从 AWBC v2 开始迭代 bootstrap**  
   AWBC v2 的 policy 已经比 base 更保守、点炮更低。可以用它当教师生成新数据，再训 filtered BC，迭代。  
   - 优势：离线，不需要重新写 MCTS；  
   - 风险：可能像 bootstrap 二代一样收敛。

---

## 优先级建议

| 方向 | 预期收益 | 工程成本 | 历史先验 | 推荐度 |
|---|---|---|---|---|
| full-action value head 微调 + AWBC v3 | 中 | 低 | AWBC 已接近 | ⭐⭐⭐⭐ |
| conv value net + search trace value labels | 高 | 中 | TD 死路提示需换语义 | ⭐⭐⭐ |
| danger 特征 + deal-in head（full-action） | 中 | 高（需重新生成 128k 数据） | 212 维特征失败 | ⭐⭐ |
| MCTS AlphaZero 迭代 | 很高 | 很高 | search trace distillation 已阳性 | ⭐⭐⭐⭐（长期） |
| A2C/PPO with good value baseline | 中 | 中 | PPO 多次失败 | ⭐⭐ |

### 推荐执行顺序

1. **短期（今晚可出结果）**：用 `nn_full_action_best.pt` 自己的 value head，在 128k 数据上微调 value，再跑 AWBC v3 并 benchmark。这是成本最低、最可能验证“value net 瓶颈”假说的实验。
2. **中期**：如果 v3 有效，就训一个 conv value net 做 MCTS leaf，启动 AlphaZero 式迭代。
3. **长期**：如果 v3 仍无效，说明问题在目标函数/特征层面，再投入 danger 特征 + deal-in head 的大工程。
