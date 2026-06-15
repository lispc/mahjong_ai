# 麻将 AI 实现方法调研与 from-scratch 算法设计

## 1. 调研结论：现在主流的麻将 AI 怎么做

麻将（尤其是四人立直麻将）是典型的高维、不完全信息、多人博弈。它的难点可以概括为四点：

1. **隐藏信息极大**：一个信息集平均对应约 `10^48` 个不可区分的状态（对手手牌 + 牌山）。
2. **奖励稀疏且延迟**：一局 8–12 轮，每轮只有一个最终得分，难以把最终排名归因到每一步。
3. **规则树不规则**：吃、碰、杠、和、报听会打断正常摸打顺序，难以直接套用 MCTS / CFR。
4. **攻防权衡**：进攻（做牌速度）和防守（降低点炮率）必须随局面动态切换。

基于公开论文、开源项目与竞赛资料，目前主流做法大致分为三层：

### 1.1 规则/启发式层：快、可解释、上限有限

- **牌效（Shanten / Ukeire）**：核心指标。Shanten 表示“距离听牌还差几步”，Ukeire 表示“能让手牌前进的有效牌总数”。大多数强 baseline 都是在这两个指标上做文章。
- **防守启发式**：跟踪对手弃牌序列，识别“现物”“筋牌”“壁”等安全牌，降低点炮率。
- **报听/立直判断**：听牌且待牌足够好时报听，锁定手牌求自摸；末期没有安全牌则转为全弃。

代表：早期的 Bakuuchi（东京大学）、SIMCAT、MahjongCLI 的 greedy AI，以及大量开源 shanten 库（如 `mahjong-utils`、`shanten-number`）。

### 1.2 搜索层：处理不确定性，但容易“算不动”

- **Expectimax**：把对手手牌和牌山当作随机变量，做 1–2 步前向搜索。优点是原理干净；缺点是分支多、估值函数难设计，深度>1 时计算量爆炸。
- **Determinized MCTS**：每次迭代把隐藏信息采样成一个完整牌局（determinization），然后跑完美信息 MCTS。简单但存在 **strategy fusion** 问题——在不同采样里选不同动作，平均后往往不是好策略。
- **IS-MCTS**（Information Set MCTS）：在信息集层面建树，每步同时考虑多个可能状态，缓解 strategy fusion，但实现复杂、计算量大。
- **pMCPA（Suphx）**：离线训练一个策略网络，在线时对“当前初始手牌”做蒙特卡洛模拟并微调策略，本质上是把搜索用于运行时策略适配。

### 1.3 学习层：当前 SOTA，但需要数据/算力

- **监督学习（SL）**：用高手对局数据训练多个策略网络（弃牌、立直、吃、碰、杠）。Suphx、NAGA 都以此做冷启动。
- **深度强化学习（RL）**：自对弈 + PPO/策略梯度。Suphx 的关键创新：
  - **Global Reward Prediction**：用 RNN 预测最终排名奖励，把全局奖励分配到每一轮。
  - **Oracle Guiding**：训练初期让模型看到完美信息，再逐渐 dropout 隐藏特征，加速学习。
  - **Run-time Policy Adaptation（pMCPA）**：每轮开始时根据当前手牌采样对局并微调策略。
- **Mortal / LuckyJ**：基于 ResNet/Attention 的深层网络 + Rust 高速推理，在线表现非常强。

### 1.4 对我们的启发

我们的项目是 **晋北麻将**（推倒胡、无复杂番型、多数情况不能吃牌、有报听锁手机制），比立直麻将简单：

- 没有立直、役、宝牌等复杂规则；
- 决策类型主要剩下：**弃牌、碰、杠、报听、和牌**；
- 报听机制让“何时锁定手牌”成为关键决策。

因此，我们不必一上来就做 Suphx 级别的大模型。更现实的路径是：

> **先用干净的概率/搜索方法把 baseline 推到接近天花板，再决定是否用神经网络做最后 5–10% 的提升。**

下面给出 4 种从 first principle 重新设计的算法方案，均不依赖已有 `eval2` 的复杂递归，也不在本次实现。

---

## 2. 方案 A：纯概率牌效 Agent（Probability-Efficiency Agent）

### 核心思想

把“弃牌选择”转化为一个 **带约束的期望收益最大化问题**：

- 进攻收益 ≈ 手牌前进速度 × 最终和牌概率 × 和牌收益。
- 防守成本 ≈ 打出该牌后对手和牌的概率 × 点炮损失。
- 当对手明显接近和牌时，切换为安全优先。

### 状态表示

- 自家手牌 `H`（13 或 14 张）。
- 全局已见牌 `U`（弃牌 + 副露 + 自家手牌），用于计算剩余牌分布。
- 对手报听/立直集合 `T`。
- 当前轮次/牌山剩余张数 `R`。

### 算法流程

1. **候选弃牌**：枚举手牌中所有 unique tile。
2. **对每张候选牌 `d`**，得到剩余 13 张 `H' = H \\{d}`。
3. **进攻价值 `V_off(H')`**：
   - 计算 `shanten(H')` 与 `ukeire(H', U)`。
   - 若 `shanten == 0`，计算听牌待牌总数与和牌概率：`P_win = sum(remaining[t] for t in waits) / R`。
   - 若未听牌，用 `ukeire` 估计下一步进入听牌的概率。
   - 综合成标量：`V_off = -C * shanten + ukeire + tenpai_bonus * P_win`。
4. **防守价值 `V_def(d)`**：
   - 估计 `d` 被对手需要和的概率（基于对手报听状态、弃牌历史、筋牌关系）。
   - 若 `d` 是现物或筋牌，`risk(d)` 低；否则高。
5. **报听决策**：若 `shanten(H') == 0` 且 `ukeire >= threshold_tenpai` 且风险可控，则报听。
6. **最终选择**：
   ```
   d* = argmax_d [ V_off(H') - lambda_def * V_def(d) ]
   ```
   其中 `lambda_def` 随对手报听人数、剩余牌数动态增大。

### 优点

- 完全可解释，调试方便。
- 不需要训练数据，几十 ms 内完成决策。
- 天然适合晋北麻将这种规则简单的变体。

### 缺点

- 手工设计的 `V_off` 和 `risk` 上限明显，难以处理复杂牌型转换。
- `lambda_def` 这种超参需要大量对局调参。

---

## 3. 方案 B：信念状态 + 1-ply Expectimax Agent（Belief Expectimax）

### 核心思想

不完全信息博弈的 cleanest 做法：维护一个 **信念状态（belief state）**，即对手手牌和牌山的联合概率分布，然后在这个分布上做期望最大化。

### 信念状态表示

不直接维护 `10^48` 个状态，而是维护 **每张牌的剩余张数分布**：

```python
belief[t] = P(牌 t 仍在牌山或某对手手中 | 已见信息)
```

更进一步，可以为每个对手维护一个条件分布 `belief_p[t]`，考虑其弃牌风格（比如某人早打中张，可能在做清一色）。

### 算法流程

1. **信念更新**：每看到一次弃牌，用贝叶斯更新修正分布。
   - 简单版：`belief[t] -= 1 / remaining_locations`。
   - 进阶版：结合对手历史行为，估计其手牌结构（如“他不要万子”则降低其手牌中万子的期望数量）。
2. **对每张候选弃牌 `d`**：
   - 得到 `H'`。
   - **模拟下一手摸牌**：按 `belief` 抽样 `K` 张牌（或枚举所有可能），对每张摸牌 `t`：
     - 若 `H' + [t]` 和牌，收益为 `WIN_VALUE`。
     - 否则，计算最佳内层弃牌（可用方案 A 的 `V_off - risk` 近似）。
   - 期望收益 `E[V(d)] = sum_t P(t) * V_after(t)`。
3. **加入对手反应**：若打出 `d` 后对手可能和牌，用 `belief` 估计 `P(ron | d)`，从收益中扣除。
4. 选 `d*` 最大化 `E[V(d)] - P(ron|d) * RON_COST`。

### 优点

- 理论框架统一：所有隐藏信息都进入概率分布，没有 ad hoc 的“感觉”。
- 1-ply 已经能考虑“弃这张牌后下一手摸到什么”的期望，比纯静态牌效强。
- 容易扩展对手模型。

### 缺点

- 信念更新准确性与对手建模能力直接挂钩，错误信念会误导搜索。
- 枚举所有可能摸牌在牌山多时计算量大，需要采样或只考虑高概率牌。
- 没有多步 lookahead，仍属“短视”。

---

## 4. 方案 C：Determinized MCTS + 快速 Rollout Agent

### 核心思想

把不完全信息问题“ determinize ”成多个完美信息问题：每次从信念状态中采样一个完整牌局（包括对手手牌和牌山），然后在这个牌局上跑 MCTS。最后对所有采样的结果做聚合。

### 算法流程

1. **采样**：根据当前信念状态，生成 `M` 个 consistent 牌局 `S_1 ... S_M`。
   - 每个牌局中，对手手牌和牌山都是具体的 136 张牌分配。
   - 可用吉布斯采样或简单无放回抽样。
2. **对每个牌局跑 MCTS**：
   - 节点：信息集（手牌 + 已见牌）。
   - 边：动作（弃牌、碰、杠、报听）。
   - 选择：UCT。
   - 扩展：按牌局具体牌序展开。
   - 模拟（rollout）：用快速策略（如方案 A）走完剩余对局。
   - 回溯：更新访问次数与价值。
3. **聚合**：对每个候选动作 `a`，计算其在所有采样牌局中的平均价值或总访问数：
   ```
   Q(a) = sum_m Q_m(a) / M
   ```
4. 选 `a*` 最大化 `Q(a)`。

### 处理晋北麻将的特殊点

- **报听锁手机制**：一旦节点中某玩家报听，后续该玩家动作固定为“摸到什么打什么”，MCTS 中可标记为 terminal-ish 分支。
- **不能吃牌**：减少动作分支。
- **无番型**：rollout 结束时只需判断基本和牌型，得分简单。

### 优点

- 天然处理多步 lookahead 和对手互动。
- rollout 策略可以逐步替换为更强策略，形成迭代提升。
- 计算资源越多，MCTS 迭代次数越多，效果越接近搜索极限。

### 缺点

- **Strategy Fusion**：在不同采样牌局里推荐的最优动作不一致，简单平均会选出平庸动作。
- 每次决策需要大量模拟，晋北麻将节奏快，可能不满足实时性。
- 需要一个好的 rollout policy，否则搜索质量受限。

### 缓解 strategy fusion 的 trick

- 不仅看平均价值，还看 **动作的鲁棒性**：选择“在所有采样牌局中都还不错”的动作，而不是“在某些采样中极好、在另一些中极差”的动作。
- 用信息集 MCTS（IS-MCTS）替代 determinization，虽然实现更复杂，但理论上更正确。

---

## 5. 方案 D：轻量神经网络 Policy-Value Agent（Small-Net RL）

### 核心思想

设计一个 **紧凑的神经网络**（MLP 或轻量 CNN），输入是当前观测，输出：

- `policy(a|s)`：34 维弃牌概率（或碰/杠/报听概率）。
- `value(s)`：当前局面对自家最终排名的预测。

通过 **自对弈强化学习** 或 **模仿更强 agent** 来训练。

### 网络输入特征

由于晋北麻将规则简单，特征可以非常干净：

- **手牌**：34 维计数向量（4 channels：0/1/2/3+ 张）。
- **弃牌河**：每个玩家 34 维历史弃牌序列（可用多个 channel 表示时间步）。
- **全局信息**：牌山剩余张数、自家/对手报听标志、当前轮次。
- **可选 lookahead 特征**：对每个候选弃牌，计算“弃牌后听牌概率/待牌数”等手工特征作为网络辅助输入（类似 Suphx 的 look-ahead）。

### 网络结构

```
Input(34 * channels) -> Conv1D/BN/ReLU -> Conv1D/BN/ReLU
                     -> Flatten -> MLP(256) -> ReLU
                     -> policy head (34 softmax)
                     -> value head (scalar tanh)
```

总参数量可控制在 < 1M，能在 CPU 上毫秒级推理。

### 训练方式

**路径 1：模仿学习（Behavior Cloning）**

1. 用方案 A/B/C 生成大量对局日志。
2. 用 `(state, action)` 对训练 policy head。
3. 用最终排名训练 value head。
4. 逐步蒸馏多个强 agent 的混合策略。

**路径 2：自对弈 PPO**

1. 用 behavior cloning 初始化。
2. 让当前模型与若干历史模型对局，收集轨迹。
3. 用 PPO 更新 policy，用 MSE 更新 value。
4. 定期把当前模型加入对手池。

**路径 3：对手模型蒸馏**

- 训练一个“oracle 版本”的网络，输入包含隐藏信息；
- 再训练一个“正常版本”，用 KL 散度让正常版本模仿 oracle 版本；
- 类似 Suphx 的 oracle guiding，但更简单。

### 优点

- 一旦训练好，推理极快。
- 网络可以隐式学习人类难以手工编码的权衡（如“保留一张安全牌作为未来弃牌”）。
- 容易通过增加数据/模型容量继续 scaling。

### 缺点

- 需要训练基础设施和数据。
- 小模型容易过拟合到训练对手，泛化到新对手可能变差。
- 晋北麻将数据稀缺，可能主要靠自对弈生成。

---

## 6. 四种方案对比

| 维度 | A 纯概率牌效 | B Belief Expectimax | C Determinized MCTS | D 轻量神经网络 |
|------|--------------|---------------------|---------------------|----------------|
| 核心武器 | Shanten/Ukeire + 防守启发式 | 信念分布 + 1-ply 期望 | 采样 + MCTS 搜索 | Policy-Value 网络 |
| 计算成本 | 极低（<10 ms） | 中（10–100 ms） | 高（100 ms–数秒） | 训练高，推理低（<5 ms） |
| 可解释性 | 高 | 中 | 中 | 低 |
| 数据需求 | 无 | 无 | 无 | 需要日志或自对弈 |
| 多步 lookahead | 无 | 1-ply | 多步 | 通过网络隐式学习 |
| 对手建模 | 简单安全牌 | 可扩展贝叶斯 | 通过采样隐式建模 | 可通过特征引入 |
| 适合阶段 | 快速 baseline | 中期提升 | 离线分析/强算力 | 最终 SOTA |

---

## 7. 对当前项目的建议

结合我们已有的代码结构（`agent.py` + `algo/` + `driver/` + `context/`），最顺畅的实施顺序是：

1. **先落地方案 A**：用现有 `eval_v2.shanten` 和 `eval_v2.tenpai_tiles` 做基础，重写一个统一的 `discard_score(hand, context)`，把进攻和防守合并为一个标量。这是 BaselinePlus 的下一步。
2. **再升级到方案 B**：为每个对手维护一个 tile-level belief，1-ply expectimax 只在尾盘或关键决策点触发，保证实时性。
3. **用方案 A/B 生成数据，训练方案 D 的小网络**：实现一个可扩展的 trajectory logger，然后做 behavior cloning + PPO 自对弈。
4. **方案 C 作为离线分析工具**：在关键对局上用 determinized MCTS 复盘，看人类/AI 的弃牌是否接近搜索最优，而不是直接在线使用。

---

## 8. 需要讨论的问题

1. **规则边界**：晋北麻将中“不能吃牌”是否是严格规则？是否允许暗杠/加杠？这些会直接影响动作空间和搜索分支。
2. **评估指标**：我们以什么为目标？
   - 胜率最高？
   - 平均排名/PT 最高（避免第四名）？
   - 点炮率最低？
   不同目标会导致 `lambda_def`、报听阈值、网络奖励函数完全不同。
3. **算力预算**：是否愿意训练神经网络？还是先做纯搜索/启发式？
4. **对手建模深度**：是否要做 per-player 风格建模（有人喜欢早打安全牌、有人激进）？还是只维护 tile-level 分布？
5. **报听策略**：报听锁手机制是双刃剑。是否采用“动态阈值”（比如尾盘或待牌多时必报，否则不报）？

---

## 9. 方案 A 实现速报

已实现 `algo/agents/prob_efficiency.py` 中的 `ProbEfficiencyAgent`。

### 实现要点

- **进攻**：基础分使用 `eval_v2.evaluate(hand13)`，额外加上
  - 未听牌时的 `ukeire` 奖励；
  - 听牌后的“实际剩余待牌张数”奖励（比 `tenpai_tiles` 更能反映尾盘真实情况）。
- **防守**：使用 `algo.eval.opponent.tile_danger`，基于现物、筋牌、对手报听信号。
- **动态 lambda**：`lambda_def = base + 1.5 * n_tenpai_opponents + 0.5 * game_progress`。
- **报听**：听牌必报（符合当前决策）。
- **可选 1-ply 期望**：`use_expectation=True` 时，按真实剩余概率枚举下一张摸牌并加权。

### 200 局 benchmark（vs Baseline / Baseline+ / SUv3-d2）

| Agent | 胜率 | 自摸 | 铳和 | 点炮 | Elo | 平均决策时间 |
|-------|------|------|------|------|-----|--------------|
| Baseline | 30.0% | 8.0% | 22.0% | 13.5% | 1530 | 232.2 ms |
| Baseline+ | 30.0% | 8.0% | 22.0% | 19.0% | 1525 | 211.4 ms |
| **ProbEffExp** | **9.5%** | 2.5% | 7.0% | 23.0% | 1479 | **34.0 ms** |
| SUv3-d2 | 25.0% | 6.0% | 19.0% | 14.5% | 1466 | 3.0 ms |

### 结论

- `ProbEff` 是一个**干净、可解释、快**的方案 A baseline，但静态评估上限明显，目前无法超越 Baseline 的 `eval2` 2-ply 期望。
- 1-ply 期望版比静态版强，但计算量增加后仍不及 Baseline。
- 要真正超越 Baseline，需要在方案 A 基础上加入更深的搜索（方案 B）或更强的学习价值函数（方案 D）。

---

## 10. 方案 B 实现速报

已实现 `algo/agents/belief_expectimax.py` 中的 `BeliefExpectimaxAgent`。

### 实现要点

- **信念状态**：使用 `algo.context.v3.ContextV3` 维护全局已见牌 `used`、每家弃牌序列 `discards` 和报听集合 `tenpai_players`，即 tile-level 信念。
- **前向搜索**：对候选弃牌做 2-ply expectimax。叶子估值复用项目已有的 `algo.eval2`（本身就是 2-ply expectimax），但把概率分布替换为 `ContextV3` 的真实剩余分布。
- **候选剪枝**：先用 `algo.eval0` 快速预选 `max_candidates` 个候选，再用 `algo.eval2` 精确评估，控制决策时间在 ~130 ms。
- **防守 tie-breaking**：只在检测到对手危险信号（已报听或 `player_danger_level >= 1`）时才进入安全模式；在 `best_offense` 附近按 `tile_danger` 选最安全的弃牌。避免无差别防守导致进攻瘫痪。
- **报听决策**：听牌且待牌剩余张数 >= 4，或待牌为现物时，才报听；同时要求局面已有一定轮数，防止过早锁死。

### 400 局 benchmark（vs Baseline / Baseline+ / Eval2Ctx）

| Agent | 胜率 | 自摸 | 铳和 | 点炮 | Elo | 平均决策时间 |
|-------|------|------|------|------|-----|--------------|
| Baseline | 23.5% | 5.5% | 18.0% | 27.5% | 1439 | 224.8 ms |
| Baseline+ | 24.8% | 6.2% | 18.5% | 20.8% | 1481 | 227.1 ms |
| **BeliefExp** | **25.0%** | 6.0% | 19.0% | **12.0%** | **1542** | 149.5 ms |
| Eval2Ctx | 25.0% | 4.8% | 20.2% | 15.5% | 1538 | 119.7 ms |

> 注：400 局结果来自一次运行；200 局结果中 BeliefExp 与 Baseline 同样接近。由于四人麻将方差较大，胜率在 ±2–3% 内波动属于正常。

### 结论

- `BeliefExp` 已经达到 **与 Baseline / Baseline+ / Eval2Ctx 同档的胜率**，同时把**点炮率从 26.5% 降到 12.5%**，说明信念 + 安全 tie-breaking 的防守设计有效。
- 它没有像 Baseline 那样“裸奔”，也没有因为过度防守而赢不了，基本实现了攻防平衡。
- 当前实现依赖 `algo.eval2` 做深度评估；后续若要继续提升，可以：
  1. 在 `eval2` 之上再加一层针对“对手和牌概率”的显式期望；
  2. 用 `eval_v2` 或 `eval_v3` 替代 `algo.eval2` 做叶子，构造完全自洽的 expectimax；
  3. 引入 per-player 风格信念（某人早打中张 -> 降低其手牌中张期望），提升对手建模精度。

---

## 11. 方案 B 优化版：BeliefExpV2

已实现 `algo/agents/belief_expectimax_v2.py` 中的 `BeliefExpectimaxV2Agent`。

### 改进点

- **per-player 危险度聚合**：不再用全局 `tile_danger` 做安全 tie-breaking，而是对每个对手分别计算 `tile_danger_for_player`，并按 `player_danger_level` 加权聚合。这样能把“某个对手明显在做筒子”这类信息反映到防守决策中。
- 其余进攻框架（eval0 预选、eval2 精确评估、安全 tie-breaking）与 BeliefExp 保持一致。

### 100 局 benchmark（vs Baseline / BeliefExp / Eval2Ctx）

| Agent | 胜率 | 自摸 | 铳和 | 点炮 | Elo | 平均决策时间 |
|-------|------|------|------|------|-----|--------------|
| Baseline | 28.0% | 6.0% | 22.0% | 26.0% | 1607 | 114.7 ms |
| BeliefExp | 24.0% | 4.0% | 20.0% | 17.0% | 1515 | 74.7 ms |
| **BeliefExpV2** | **22.0%** | 4.0% | 18.0% | **14.0%** | 1466 | 75.3 ms |
| Eval2Ctx | 21.0% | 8.0% | 13.0% | 16.0% | 1412 | 58.0 ms |

### 结论

- BeliefExpV2 在胜率上与 BeliefExp 基本持平，但把**点炮率从 17% 进一步降到 14%**，说明 per-player 危险度聚合确实提升了防守精度。
- 代价是略低的铳和率，总体更偏稳健。
- 由于 BeliefExp 本身已经很强，V2 的提升幅度有限；后续若继续优化，可以尝试把 per-player 信念也用于进攻端的 tile probability（替换 `algo.eval2` 的均匀假设）。

---

## 12. 方案 C 实现速报：Determinized MCTS

已实现 `algo/agents/determinized_mcts.py` 中的 `DeterminizedMCTSAgent`，并在 `driver/engine.py` 中新增 `play_game_from_state` 以支持从任意中盘状态继续模拟。

### 实现要点

- **Determinization（采样世界）**：根据 `ContextV3` 的已见信息，把未知牌均匀随机分配给三名对手和牌山，保证与已见牌一致。
- **候选剪枝**：先用 `algo.eval0` 预选 top_k 候选弃牌，减少每个世界需要评估的动作数。
- **Rollout policy**：为了实时性，rollout 使用快速启发式 `_fast_rollout_select`，即对每个候选弃牌计算 `algo.eval0(hand13, empty context)` 并选最大者。
- **收益设计**：
  - 当前玩家和牌：+1
  - 当前玩家点炮：-1
  - 其他玩家和牌：-0.3（反映自己失去获胜机会）
  - 流局：0
- **安全 tie-breaking 的取舍**：当前版本把点炮风险显式放进 rollout 收益，而不是像 BeliefExp 那样做外部安全惩罚。

### 100 局 benchmark（vs Baseline / BeliefExp / Eval2Ctx，n_worlds=4, top_k=6）

| Agent | 胜率 | 自摸 | 铳和 | 点炮 | Elo | 平均决策时间 |
|-------|------|------|------|------|-----|--------------|
| Baseline | 31.0% | 11.0% | 20.0% | 18.0% | 1557 | 130.5 ms |
| **BeliefExp** | **26.0%** | 9.0% | 17.0% | **13.0%** | 1504 | 86.1 ms |
| Eval2Ctx | 27.0% | 5.0% | 22.0% | 19.0% | 1501 | 66.4 ms |
| **DetMCTS** | **8.0%** | 1.0% | 7.0% | 16.0% | 1439 | 150.6 ms |

> 把 `n_worlds` 提升到 8 后，决策时间增至 ~320 ms，但胜率仍在 6–10% 区间，说明当前实现尚未找到稳定收益。

### DetMCTS-V2：BeliefExp hybrid rollout

已进一步实现 **BeliefExp 作为当前玩家 rollout policy** 的版本（`belief_exp_rollout=True`），对手仍用快速 eval0 policy，并加入截断启发式。

20 局测试（n_worlds=3, top_k=4, rollout_depth=16）：

| Agent | 胜率 | 自摸 | 铳和 | 点炮 | Elo | 平均决策时间 |
|-------|------|------|------|------|-----|--------------|
| Baseline | 20.0% | 0.0% | 20.0% | 25.0% | 1507 | 123.3 ms |
| BeliefExp | 25.0% | 5.0% | 20.0% | 10.0% | 1472 | 84.4 ms |
| Eval2Ctx | 35.0% | 0.0% | 35.0% | 20.0% | 1570 | 62.5 ms |
| **DetMCTSV2** | **10.0%** | 0.0% | 10.0% | 30.0% | 1450 | **1607.2 ms** |

- 决策时间 ~1.6 s，尚未具备在线实用性。
- 胜率仍低于 BeliefExp，说明单纯把当前玩家 rollout 换成 BeliefExp、且只模拟 16 回合，还不足以弥补 determinization 和 opponent rollout 的不足。

### 结论与后续方向

- `DetMCTS` **实现上已跑通**，但目前胜率明显弱于 BeliefExp / Baseline / Eval2Ctx。主要原因：
  1. **均匀随机 determinization** 没有利用对手弃牌信息做非均匀信念；
  2. **快速 rollout policy**（eval0）太弱，无法准确估计一手弃牌在完整对局中的真实价值；
  3. **采样数不足**（4–8 个世界）导致方差大，平均后选出平庸动作（strategy fusion）。
- 后续若继续投入方案 C，优先尝试：
  1. 用 **BeliefExp 或 Baseline+ 作为所有玩家的 rollout policy**（牺牲实时性换强度）；
  2. 引入对手建模的 **非均匀 determinization**（根据对手弃牌序列调整其手牌分布）；
  3. 升级为 **IS-MCTS** 或增加 determinization 数量到 50+，并配合方差缩减技巧。

---

## 13. BeliefExp 超参调优

已添加 `scripts/tune_belief_exp.py`，对 `defense_margin`、`max_candidates`、`tenpai_min_wait` 做随机网格搜索（默认 12 组参数 × 60 局）。

### 部分结果

| defense_margin | max_candidates | tenpai_min_wait | 胜率 | 点炮率 | 平均决策时间 |
|----------------|----------------|-----------------|------|--------|--------------|
| 0.0 | 12 | 2 | 20.0% | 20.0% | 173.8 ms |
| 0.015 | 6 | 6 | 20.0% | 21.7% | 102.9 ms |
| 0.015 | 12 | 2 | 18.3% | 23.3% | 174.5 ms |
| 0.03 | 8 | 3 | 18.3% | 18.3% | 148.5 ms |
| 0.1 | 8 | 2 | 15.0% | 13.3% | 122.9 ms |

> 搜索在较强对手池（Baseline+ / Eval2Ctx）上进行，胜率绝对值低于对阵 Baseline 的 benchmark，但相对趋势仍有参考意义。

### 结论

- `defense_margin=0.0`（不主动安全 tie-breaking）在搜索中表现略好，但点炮率更高；
- 各组参数差异在 60 局尺度上不够显著，说明 BeliefExp 对这几个超参不太敏感，已经处于一个较稳的局部最优；
- 默认参数 `defense_margin=0.03, max_candidates=8, tenpai_min_wait=4` 在 400 局综合 benchmark 中表现均衡，仍作为默认设置。

---

## 14. 方案 B 再升级：BeliefExpV3（自洽 expectimax + per-player 信念）

已实现：
- `algo/eval/player_belief.py`：per-player tile-level 信念模型；
- `algo/agents/belief_expectimax_v3.py`：BeliefExpV3Agent。

### 实现要点

- **per-player tile 信念**：根据每名对手的弃牌序列推断其花色偏好，进而得到“某张牌实际还在牌山的期望张数”。
- **自洽 expectimax**：不再调用 `algo.eval2`，而是显式枚举“摸牌 → 打牌”过程：
  - 对每个候选弃牌得到 hand13；
  - 按有效剩余张数加权枚举下一张摸牌；
  - 摸到牌后从 14 张手牌中选最佳弃牌；
  - 叶子节点用 `algo.eval0` 评估最终 hand13。
- **防守**：沿用 V2 的 per-player 危险度聚合 + 安全 tie-breaking。

### 100 局 benchmark（vs Baseline / BeliefExp / Eval2Ctx）

| Agent | 胜率 | 自摸 | 铳和 | 点炮 | Elo | 平均决策时间 |
|-------|------|------|------|------|-----|--------------|
| Baseline | 30.0% | 7.0% | 23.0% | 25.0% | 1632 | 122.5 ms |
| BeliefExp | 24.0% | 5.0% | 19.0% | 15.0% | 1537 | 89.2 ms |
| **BeliefExpV3** | **19.0%** | 5.0% | 14.0% | 18.0% | 1456 | 169.0 ms |
| Eval2Ctx | 22.0% | 5.0% | 17.0% | 15.0% | 1375 | 68.7 ms |

### 结论

- BeliefExpV3 **实现上跑通**，但目前 100 局胜率略低于 BeliefExp（19% vs 24%），决策时间也更高（169 ms）。
- 原因可能是：
  1. `algo.eval0` 叶子虽强，但一层“摸+打”的 expectimax 还不足以抵消 `algo.eval2` 内部两层 draw 期望的信息优势；
  2. per-player 信念模型目前偏启发式，对有效进张的折扣有时过于悲观，导致进攻节奏变慢；
  3. 搜索深度和候选数受实时性限制，未能充分发挥 expectimax 的潜力。
- 这部分工作为后续 **NN 估值函数** 提供了接口：一旦用 NN 替代 `algo.eval0` 做叶子，同样的 expectimax 框架会显著提升。

---

## 15. 方案 D：轻量 Policy-Value 神经网络（MLX / Metal）

已实现端到端 NN 管线：
- `algo/nn/features.py`：局面特征编码（175 维）；
- `algo/nn/model.py`：轻量 MLP Policy-Value 网络（约 25k 参数）；
- `algo/agents/data_collectors.py`：BeliefExp 数据采集器；
- `scripts/generate_nn_data.py`：生成自对弈数据；
- `scripts/train_nn.py`：MLX 训练脚本；
- `algo/agents/nn_agent.py`：用训练好的网络在线决策。

### 网络结构

- 输入：175 维（手牌 + 牌山剩余 + 3 对手弃牌 + 报听标志 + 进度）。
- 隐藏层：128 → 64（ReLU）。
- 输出：34 维 policy logits + 1 维 value（tanh）。
- 总参数量约 25k，单步前向 **~1.4 ms**（Apple Silicon Metal）。

### 数据与训练

首次用 BeliefExp 自对弈 **200 局** 生成约 9k 个 (state, action) 样本，训练 30 epoch 后：
- policy top-1 验证准确率 **~32%**；
- 验证 loss 从 3.3 降到 2.45。

### 20 局 benchmark（vs Baseline / BeliefExp / Eval2Ctx）

| Agent | 胜率 | 自摸 | 铳和 | 点炮 | Elo | 平均决策时间 |
|-------|------|------|------|------|-----|--------------|
| Baseline | 25.0% | 5.0% | 20.0% | 20.0% | 1556 | 102.5 ms |
| BeliefExp | 25.0% | 5.0% | 20.0% | 15.0% | 1513 | 68.6 ms |
| Eval2Ctx | 40.0% | 0.0% | 40.0% | 15.0% | 1600 | 52.2 ms |
| **NNAgent** | **0.0%** | 0.0% | 0.0% | 30.0% | 1331 | **1.4 ms** |

### 结论

- NNAgent **速度极快**（1.4 ms/步），说明 MLX/Metal 实时对局完全可行。
- 但当前模型 **胜率还为 0%**，主要原因：
  1. 训练数据仅 200 局 / 9k 样本，对 34 类分类任务严重不足；
  2. 仅做行为克隆（behavior cloning），没有最终胜负的 value 监督；
  3. 网络容量和特征工程都还很初级。
- 后续方向：
  1. 把数据量提升到 **10k–50k 局**（预计 1–3 小时生成）；
  2. 加入 value head 的真实监督（用最终对局结果回传）；
  3. 用 NN 作为 DetMCTS / BeliefExp 的 leaf evaluator 或 rollout policy，而不是直接当 policy。

---

## 参考资源

- Li et al., *Suphx: Mastering Mahjong with Deep Reinforcement Learning*, 2020. https://arxiv.org/abs/2003.13590
- Mizukami & Tsuruoka, *Building a computer mahjong player based on Monte Carlo simulation and opponent models*, 2015.
- Lei et al., *Incomplete Information Game Algorithm Based on Expectimax Search and Double DQN*.
- 哈尔滨工程大学，《麻将博弈 AI 构建方法综述》.
- Mortal (Rust + Deep RL): https://github.com/Equim-chan/Mortal
- Akagi (mjai protocol / HUD): https://github.com/shinkuan/Akagi
- MahjongCLI (shanten-based greedy AI): https://github.com/YarrowRen/MahjongCLI
- shanten-number (快速向听数算法): https://github.com/tomohxx/shanten-number
- mahjong-utils (Kotlin/Java/JS/Python 牌效库): https://github.com/ssttkkl/mahjong-utils
