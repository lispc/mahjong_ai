# 更一致的弃牌算法设计方案

当前 `MCTS-Eval2` 的结构比较 ad hoc：

1. 用 `eval0` 预选候选弃牌；
2. 对每个候选随机模拟 5 次摸牌；
3. 每次模拟用 `eval0` 选内层弃牌；
4. 用 `eval2` 给最终 13 张手牌打分。

这相当于把“候选生成 / 前向搜索 / 价值估计”三件事混在了一起，而且 2 和 3 之间不一致：`eval2` 本身是精确期望，MCTS 又变成蒙特卡洛近似。

下面给出三个由浅入深、更一致、更从原理出发的方案。

---

## 方案 A：Shanten + Ukeire 贪心

### 核心思想

手牌价值由两个明确可解释的指标决定：

- **Shanten（向听数）**：距离听牌还差几张有效进张。
- **Ukeire（上张数/待牌数）**：当前手牌能胡或能推进的有效牌总数。

这是日麻牌效率理论的标准做法，也是最容易实现、最稳定的一个方案。

### 价值函数

```python
def value(hand13, context):
    s = shanten(hand13)
    if s == 0:
        # 听牌：价值 = 所有待牌的剩余张数之和
        return sum(remaining[t] for t in winning_tiles(hand13))
    else:
        # 未听牌：惩罚 shanten，并奖励能让 shanten 降 1 的进张
        return -C * s + sum(
            remaining[t] for t in all_tiles
            if shanten(hand13 + [t]) < s
        )
```

其中 `C` 是 shanten 惩罚系数（例如 10），保证“少 1 向听”优先于“多几张待牌”。

### 14 张弃牌决策

```python
def select(hand14, context):
    return max(
        unique_tiles(hand14),
        key=lambda d: value(remove_one(hand14, d), context)
    )
```

### 优点

- **单一目标函数**：没有 `eval0/eval1/eval2` 三层嵌套。
- **无随机性**：结果确定，易于调试和复现。
- **麻将意义明确**：每一步都在优化“最快听牌 + 最多待牌”。
- **对尾盘那手牌的判断**：打 `发财` 和打 `四万` 都听牌，但 `发财` 路线通常待牌更多，会稳定选 `发财`，不会点到 24% 概率的 `四万`。

### 局限

只看弃牌后 13 张手牌的静态价值，没有 multi-step lookahead。

---

## 方案 B：统一 Expectimax + Ukeire Leaf

保留当前 `eval_rec` 的递归结构，但把 leaf 换成干净的 ukeire 函数，不再依赖 handcrafted 的 `eval_naive`。

### 单一价值函数

```python
def V(hand13, context, depth=2):
    # Terminal：14 张胡牌
    if is_win(hand13 + [some_drawn_tile]):
        return WIN_PAYOUT

    # Leaf：用 ukeire/shanten 估计
    if depth == 0:
        return ukeire_value(hand13, context)

    # Expectimax：对每一种可能的摸牌，选最优内层弃牌
    total = 0
    for tile, prob in context.tile_prob(hand13).items():
        hand14 = hand13 + [tile]
        best_inner = max(
            V(remove_one(hand14, d), context, depth - 1)
            for d in unique_tiles(hand14)
        )
        total += prob * best_inner
    return total
```

### 14 张弃牌决策

```python
def select(hand14, context):
    return max(
        unique_tiles(hand14),
        key=lambda d: V(remove_one(hand14, d), context, depth=2)
    )
```

### 优点

| 方面 | 当前 MCTS-Eval2 | 统一 Expectimax |
|------|----------------|-----------------|
| 价值定义 | `eval0/eval1/eval2` 三层，意义不明 | 单一 `V`，递归定义清晰 |
| 随机性 | `samples=5`，方差大 | 精确期望（或可控采样） |
| 候选生成 | `eval0` 预选，可能漏好弃牌 | 所有候选统一用 `V` 评估 |
| 扩展性 | 难加防守 | 容易加：`U = V - λ·risk` |

### 计算量与优化

递归分支约为 `34 种摸牌 × ~10 种弃牌 × depth`。

优化手段：

- `functools.lru_cache` 缓存 `(sorted_hand_tuple, depth)`。
- 非 unique 弃牌剪枝。
- `depth=1` 几乎无压力；`depth=2` 与当前 `eval2` 同量级。

---

## 方案 C：带防御的统一效用函数

把进攻和防守合成一个标量：

```python
def utility(discard, hand14, context, self_name):
    hand13 = remove_one(hand14, discard)
    offense = V(hand13, context)                     # 方案 B 的进攻价值
    risk    = tile_risk(discard, context, self_name) # 估计的点炮概率
    return offense - lambda_defense * risk
```

`tile_risk` 可以从简单到复杂：

- **简单版**：该牌是否已被多家弃过、是否为现物。
- **中等级**：从对手 discard 序列推断其听牌花色倾向。
- **复杂版**：贝叶斯推断对手待牌分布。

整个 agent 就一句话：

```python
return argmax_d utility(d, hand14, context, self.name)
```

非常干净，而且防守强度可以只通过 `lambda_defense` 一个旋钮调节。

---

## BaselinePlus：对原 Baseline 的最小侵入式增强

在保持 `algo.eval2` 作为主干的前提下，实现了三个低成本增强（`algo/agents/baseline_plus.py`）：

1. **报听机制**：当手牌听牌且剩余待牌总数 ≥ `tenpai_min_wait`（默认 3）时报听，锁定手牌求自摸。
2. **尾盘 1-ply 期望**：当牌山剩余 ≤ `endgame_threshold`（默认 16）时，用 `Context` 的真实剩余概率调用 `eval2(hand13, context)` 选牌；`eval2` 内部已经是 1-ply 期望，比默认空 Context 更贴合尾盘。
3. **eval2 缓存**：用 `functools.lru_cache` 缓存 `eval2(hand13, 空 Context)`，决策速度略快于原 Baseline。

### 250 局 4 人赛 benchmark（CPython，8 workers）

| Agent | 胜率 | 自摸 | 铳和 | 点炮 | Elo | 平均决策时间 |
|-------|------|------|------|------|-----|--------------|
| Baseline | 26.0% | 7.2% | 18.8% | 20.4% | 1479 | 181.7 ms |
| Baseline+noTnoE（仅缓存） | 22.0% | 3.6% | 18.4% | 20.4% | 1518 | 174.8 ms |
| Baseline+noTen（缓存+尾盘） | 22.4% | 7.2% | 15.2% | **14.8%** | 1479 | 172.2 ms |
| **Baseline+（缓存+尾盘+报听）** | **28.0%** | 6.4% | 21.6% | 18.4% | **1524** | 172.6 ms |

结论：
- 单独的缓存/noTnoE 对强度帮助不大，只带来 3–5% 加速。
- 尾盘 Context-`eval2` 显著降低点炮率（14.8% vs 20.4%），但胜率提升有限。
- 加上报听后，`Baseline+` 在胜率、Elo 和点炮率上均优于或持平原 Baseline，且决策时间略快（≈5%）。
- 报听参数敏感：`tenpai_min_wait=3` 是当前默认；`tw8` 在初步测试中反而变弱，说明锁死手牌需要足够好的待牌。

## 实验更新

方案 A 和方案 3（复用 v3 CEM 权重 + 防守）已实现并测试，详细结果见 [`../reports/shanten-ukeire-experiment.md`](../reports/shanten-ukeire-experiment.md)。关键结论：

- **方案 3 的防守项非常有效**：`ShantenUkeireV3Agent` 在 `defense_weight=2.0` 时，1.4–2.0 ms/决策，胜率与 Eval2Ctx 同级，点炮率更低。
- **直接套用 expectimax（方案 B 的思想）收益不明显**：depth=1 全枚举反而变弱；depth=1 + top-k 限制有潜力但耗时 583 ms；depth=2 更慢更不稳定。
- **根本问题：leaf value 太粗糙**。在 leaf 不够精细前，加深搜索会放大误差。

## 更新后的推荐实施路径

1. ✅ **已实现方案 A**（`algo/agents/shanten_ukeire.py`）。
2. ✅ **已实现方案 3**（`ShantenUkeireV3Agent`），当前最实用。
3. ✅ **已实现 BaselinePlus**（`algo/agents/baseline_plus.py`），作为 Baseline 的稳妥升级。
4. **暂不推进方案 B 的纯 expectimax 扩展**，除非先改进 leaf value。
5. **下一步调参**：微调 `BaselinePlus.tenpai_min_wait` / `endgame_threshold`，并与 `ShantenUkeireV3Agent` 的 `defense_weight` 做 A/B。
