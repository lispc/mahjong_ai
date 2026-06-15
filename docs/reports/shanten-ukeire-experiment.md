# Shanten + Ukeire Agent 实验记录

## 实验目的

验证在方案 A（Shanten + Ukeire 贪心）基础上加入 Expectimax 搜索深度是否能稳定提升强度。

## 实现版本

| 版本 | 说明 |
|------|------|
| SU-d0 | 原始方案 A：`leaf_value(hand14 - d)` 直接选最大 |
| SU-d1 | depth=1 精确 expectimax，内层弃牌全枚举 |
| SU-d1k4 | depth=1 精确 expectimax，内层只保留 leaf top-4 弃牌 |
| SU-d2k2 | depth=2 采样 expectimax，每层抽 5 张，内层 top-2 |

为了加速 depth>=1 的搜索，在 CPython 下使用 Numba 加速的 `algo.eval.v3` 计算向听数与有效进张；PyPy 下退回到纯 Python 的 `algo.eval.v2`。

## 关键结论

### 1. depth=1 全枚举反而变弱

不加 top-k 限制的 depth=1 表现比 depth=0 还差：

- **SU-d1**：胜率 3%，点炮率 8%，耗时 218 ms
- **SU-d0**：胜率 9%，点炮率 7%，耗时 2.7 ms

原因：leaf value 只考虑 shanten + ukeire，本身不够精细；expectimax 内层选择过多时，会放大 leaf 的评估误差，选到一些“leaf 上好看、实际很糟”的弃牌。

### 2. top-k 限制能救回 depth=1

限制内层只考虑 leaf top-4 弃牌后：

- **SU-d1k4**：胜率 13%，点炮率 5%，耗时 583 ms

胜率最高、点炮率最低，但耗时是 depth=0 的 200 倍以上。

### 3. depth=2 收益不明显

- **SU-d2k2**：胜率 0%，点炮率 4%，耗时 793 ms

2-ply 采样方差大，且 leaf 误差随深度传播，100 局样本中表现最差。

### 4. 速度对比

在复盘那手牌上（CPython，Numba 已 warmup）：

```text
depth=0:            1.2 ms
depth=1 top_k=4:  476 ms
depth=2 n=5 k=2:  630 ms
depth=2 n=10 k=2: 1.96 s
```

### 5. 100 局 benchmark 总览

```text
Baseline:    win 12%, deal-in 6%,  Elo 1515, time 124ms
Eval2Ctx:    win  8%, deal-in 6%,  Elo 1525, time  60ms
MCTS:        win  8%, deal-in 12%, Elo 1486, time  64ms
SU-d0:       win  9%, deal-in 7%,  Elo 1494, time 2.7ms
SU-d1:       win  3%, deal-in 8%,  Elo 1494, time 218ms
SU-d1k4:     win 13%, deal-in 5%,  Elo 1486, time 583ms
SU-d2k2:     win  0%, deal-in 4%,  Elo 1489, time 793ms
```

> 注：该 100 局流局率高达 82%，有胜负的局只有 18 局，统计噪声较大。

## 结论

1. **直接把 Shanten+Ukeire 套进 expectimax 不是银弹**。Leaf value 太粗糙时，加深搜索会放大误差。
2. **depth=1 + top-k 限制有潜力**，但 583 ms 的耗时让它无法作为主力 agent。
3. **depth=2 目前不值得**：更慢、更不稳定、收益未验证。
4. **当前最实用的仍是 SU-d0**：2.7 ms、胜率 9%、点炮率 7%，性价比最高。

## 方案 3 实验：复用 eval_v3 的 CEM 权重 + 防守

实现：`algo/agents/shanten_ukeire.py` 中的 `ShantenUkeireV3Agent`。

对每个候选弃牌直接调用 `eval_v3.discard_score(hand14, d, context, defense_weight=w)`，它内部使用 CEM 调优后的权重：

```python
DEFAULT_WEIGHTS = {'shanten': 7.5, 'ukeire': 0.001, 'wait': 0.46, 'algo_eval0': 13.5}
```

并叠加 `eval_v3.discard_safety` 作为防守项。

### 4 人 benchmark（CPython + Numba，100 局）

**第一组：Baseline / Eval2Ctx / MCTS / SUv3-d2**

```text
Baseline:  win 33%, self 8%, ron 25%, deal-in 14%, draw 2%, Elo 1610, time 113ms
Eval2Ctx:  win 21%, self 5%, ron 16%, deal-in 19%, draw 2%, Elo 1453, time  56ms
MCTS:      win 15%, self 5%, ron 10%, deal-in 25%, draw 2%, Elo 1425, time  61ms
SUv3-d2:   win 29%, self 12%, ron 17%, deal-in 10%, draw 2%, Elo 1512, time 1.8ms
```

**第二组：Baseline / Eval2Ctx / SUv3-d0 / SUv3-d2**

```text
Baseline:  win 34%, self 8%, ron 26%, deal-in 16%, draw 5%, Elo 1641, time 110ms
Eval2Ctx:  win 25%, self 5%, ron 20%, deal-in 13%, draw 5%, Elo 1508, time  54ms
SUv3-d0:   win 12%, self 7%, ron  5%, deal-in 25%, draw 5%, Elo 1398, time 1.5ms
SUv3-d2:   win 24%, self 8%, ron 16%, deal-in 13%, draw 5%, Elo 1453, time 1.4ms
```

**第三组：Baseline / Eval2Ctx / SUv3-d2 / SUv3-d3**

```text
Baseline:  win 39%, self 7%, ron 32%, deal-in 13%, draw 3%, Elo 1678, time 120ms
Eval2Ctx:  win 17%, self 3%, ron 14%, deal-in 20%, draw 3%, Elo 1440, time  59ms
SUv3-d2:   win 19%, self 8%, ron 11%, deal-in 19%, draw 3%, Elo 1465, time 2.0ms
SUv3-d3:   win 22%, self 11%, ron 11%, deal-in 16%, draw 3%, Elo 1416, time 2.0ms
```

### 结论

1. **防守权重非常关键**：SUv3-d0（defense_weight=0）胜率仅 12%、点炮率 25%；加到 defense_weight=2 后胜率翻倍到 24%、点炮率降到 13%。
2. **SUv3-d2 是性价比甜点**：1.4–2.0 ms/决策，胜率与 Eval2Ctx 同级，点炮率更低。
3. **defense_weight=3 点炮率更低（16%），但 Elo 没有继续提升**，可能因为过于保守错过和牌。
4. **Baseline 在这组对手里很强**：Elo 1600+，说明 SUv3 还没有超越现有最强 agent。

## Baseline-plus 实验

实现：`algo/agents/baseline_plus.py`。

尝试了两种增强方式：

### 版本 1：eval2 + 已见牌 + 防守

- eval2 使用 context.used 修正牌山概率。
- 在 eval2 top-K 候选中用 `tile_danger` 做防守权衡。
- 尾盘/报听时增强防守。

结果：**明显变弱**。

```text
Baseline:    win 35%, self 8%, ron 27%, deal-in 10%, draw 12%, Elo 1626, time 116ms
Baseline+:   win  1%, self 0%, ron  1%, deal-in  9%, draw 12%, Elo 1264, time 106ms  ← 变弱
Eval2Ctx:    win 24%, self 7%, ron 17%, deal-in 20%, draw 12%, Elo 1558, time  55ms
SUv3-d2:     win 28%, self 16%, ron 12%, deal-in 18%, draw 12%, Elo 1626, time 2.1ms
```

原因：改变 eval2 的牌山分布后，原 Baseline 的 tuned 权重不再适用；外加的防守惩罚又太重。

### 版本 2：保留原 eval2，只在接近候选间做防守 tie-break

- 不改 eval2 分布。
- 仅在 eval2 分数差距 <= threshold 的候选里，选 danger 最小的。

测试结果：

```text
Baseline:      win 30%, self 6%, ron 24%, deal-in  8%, draw 8%, Elo 1485, time 110ms
Baseline+0.05: win 20%, self 4%, ron 16%, deal-in 16%, draw 8%, Elo 1417, time 116ms
Baseline+0.08: win  6%, self 0%, ron  6%, deal-in 18%, draw 8%, Elo 1376, time 112ms
```

即使 threshold 很小，Baseline+ 仍然明显弱于原 Baseline。

### 结论

1. **原 Baseline 已经很强，且其 eval2 内部可能已经隐式编码了安全偏好**。
2. **外部叠加 `tile_danger` 会与 Baseline 的内部安全机制冲突**，导致过度保守、胜率下降。
3. **防守技巧对 SUv3 有效，但对 Baseline 无效**——这是本次实验最重要的负向发现。

## 后续方向（按优先级）

1. ✅ **加入防守项已验证有效**（SUv3-d2）。
2. **进一步调 defense_weight**：尝试 1.0、1.5、2.5、4.0 找更优值。
3. **优化 SUv3 的进攻 leaf**：v3 CEM 权重里 `algo_eval0` 占比很大，本质还是 handcrafted；可以尝试替换/削弱 eval0，只用 shanten+ukeire+wait+taatsu。
4. **暂不推进 expectimax depth>=1 和 Baseline-plus（context+defense）**。
5. **把 SUv3 集成进 `record_game.py` 和更多 benchmark 做长期观察**。
