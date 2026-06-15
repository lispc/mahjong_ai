# eval_v3 实现与 benchmark 报告

## 实现内容

针对 `eval_v2` 评估函数“只看速度、不看防守/待牌质量”的问题，实现了 `algo/eval_v3.py` 与 `algo/expectimax_v3_agent.py`，引入三项改进：

1. **Ukeire（有效进张数）**
   - 对 13 张手牌，计算弃某张后还有多少张牌能减少向听数，并按牌山实际剩余张数加权。
   - 使用 Numba 批量计算，核心 `_ukeire_nb` 全循环在 JIT 内完成。

2. **听牌质量（Wait Quality）**
   - 听牌时给不同待型打分：两面 > 坎张/边张 > 单骑，并考虑待牌剩余张数。
   - 同样用 Numba 批量实现。

3. **基础防守（Defense / 弃牌安全度）**
   - 扩展 `ContextV2` 为 `ContextV3`，记录每家弃牌序列与全局已见牌。
   - `discard_safety(tile, context)` 基于：
     - 现物（genbutsu，自己或对手已打出）加分
     - 筋牌/邻居已出牌加分
     - 字牌、幺九相对安全，中张相对危险
   - 在 ExpectiMax 的弃牌层直接加入 `defense_weight * safety`。

4. **Numba 加速**
   - 安装 `numba==0.65.1` / `numpy==2.4.6` 成功。
   - 编写了 34 数组表示的贪心向听数 `_shanten_fast_nb`，pair 枚举版比 Python `shanten_fast` 快约 **5×**，精度 98%+；更粗略的贪心版快约 **35×**，精度 99%+。

## 新增/修改文件

- `algo/eval_v3.py`
- `algo/context_v3.py`
- `algo/expectimax_v3_agent.py`
- `scripts/compare_v3.py`
- `scripts/compare_v2v3.py`
- `docs/eval-improvement-plan.md`
- `tmp/test_numba.py`、`tmp/test_numba2.py`（实验脚本）

## Benchmark

### 最终版 V2 vs V3 专场 100 局（CPython，8 workers）

| AI | 胜率 | 自摸率 | 点和率 | 点炮率 | Elo | 平均决策耗时 |
|---|---|---|---|---|---|---|
| V2 | 24.0% | 7.0% | 17.0% | 18.5% | 1492 | 74.5 ms |
| **V3** | **25.0%** | 6.5% | 18.5% | **17.0%** | **1508** | **71.8 ms** |

- V3 在专场中 **略胜 V2**（Elo +16），且决策耗时更短。
- 点炮率 17% vs 18.5%，防守有微弱优势。

### 混合对手 100 局（CPython，8 workers）

| AI | 胜率 | 自摸率 | 点和率 | 点炮率 | Elo | 平均决策耗时 |
|---|---|---|---|---|---|---|
| Baseline（原 eval2） | 30.0% | 7.0% | 23.0% | 18.0% | 1587 | 310 ms |
| V2 | 17.0% | 1.0% | 16.0% | **14.0%** | **1426** | 76.9 ms |
| V3 | 17.0% | 7.0% | 10.0% | 21.0% | 1393 | 74.3 ms |
| MCTS | 29.0% | 5.0% | 24.0% | 20.0% | 1595 | 92.3 ms |

- 在混合环境中，V3 不如 V2（Elo 1393 vs 1426），主要表现为 **点和率大幅下降**（10% vs 16%），说明当前权重下 V3 偏向防守，错失进攻机会。
- 面对 Baseline/MCTS 时，V3 的点炮率反而更高（21% vs 14%），基础防守对非 ExpectiMax 对手的针对性不足。

## 关键发现

1. **ukeire 权重必须极小**：第一次尝试 `ukeire=0.8` 时，评估值被 ukeire 绝对值（可达 90+）主导，导致 V3 胜率仅 1%、点炮 22%。将权重降到 `0.001` 后，评估尺度才与 V2 接近。
2. **融合原项目 `algo.py` 评估非常有效**：加入 `algo_eval0` 后，V3 的手牌结构理解明显提升，最终能反超 V2。
3. **内层 expectimax 可以大幅简化**：内层递归只保留 `shanten + algo_eval0`，顶层才使用 `ukeire + wait + defense + algo_eval0`，这样在不损失太多强度的情况下把决策耗时从 240 ms 降到 72 ms。
4. **CEM 调权噪声大**：小样本（8–10 局）CEM 结果大量并列，调出的权重对混合对手不一定稳健。

## 当前默认权重

```python
DEFAULT_WEIGHTS = {
    'shanten': 7.5,
    'ukeire': 0.001,
    'wait': 0.46,
    'algo_eval0': 13.5,
}
defense_weight = 2.2
```

## 后续优化方向

1. **专门化权重/策略**：
   - 对 V2 专场权重 和 混合对手权重 分别调优；或根据对手类型动态选择。
2. **加强防守建模**：
   - 检测对手是否听牌/报听，听牌后切换 betaori（全弃）。
   - 根据每家弃牌花色推断其待牌范围。
3. **提升进攻**：
   - 调整 wait/uukeire 权重，避免过度防守导致点和率过低。
   - 尝试加入 `algo_eval1` 作为特征（比 eval0 强但较慢，可只用于顶层）。
4. **尝试 depth=2 + eval_v3**：
   - 现在 V3 顶层评估更准、内层更快，depth=2 的剪枝搜索可能带来进一步提升。
5. **更稳健的调权**：
   - 用更大的 n_games（30–50）做 CEM，或改用网格搜索 + 贝叶斯优化。
