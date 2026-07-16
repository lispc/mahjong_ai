# Fable-5 下一步建议执行报告（2026-07-06）

按 `docs/fable-5-review-0706.md` 的优先级，依次落实方向 1–4 的初步实验与代码实现。

---

## 1. 复式（duplicate）赛制评测

### 实现
- 在 `driver/tournament.py` 新增 `run_duplicate_tournament(candidate_a_factory, candidate_b_factory, opponent_factories, n_seeds=400, mirror_positions=False, ...)`。
- 新增 `scripts/rl/benchmark_duplicate.py`，输出候选胜率、paired A−B 差值及 bootstrap 95% CI。
- 固定标准考场：候选 A/B 轮换座位，其余三席固定为 Baseline / BeliefExp / 上代 best（视比较对象而定）。

### 结果（1000-seed 配对赛）
| 配对 | A−B 胜率差 | 95% CI | 结论 |
|---|---|---|---|
| Hybrid-Best vs Baseline | −15.4% | [−19.2%, −11.6%] | Baseline 显著更强 |
| Hybrid-Best vs BeliefExp | −12.7% | [−16.3%, −9.1%] | BeliefExp 显著更强 |
| Baseline vs BeliefExp | +2.7% | [−1.6%, +7.0%] | 不显著 |

### 结论
- 在 duplicate 固定座位考场中，**当前 best（Hybrid-FullAction-SoupDistilled）显著弱于 Baseline 和 BeliefExp**，与常规 pool benchmark 中「Hybrid 最强」的结论矛盾。
- 这强烈提示：常规 pool 的 Elo/胜率受座位/发牌方差影响很大，当前 best 的某些优势可能是噪声。
- **下一步**：用更大的 duplicate 样本（如 5000 seed）复核；若复现，需要重新评估「当前 best」链条。

---

## 2. 把搜索做便宜：Cython 化 eval0

### 实现
- 新增 `algo/eval/_fast_eval0.pyx`：用 Cython/C++ 实现 34 维手牌 counts 到 eval0 metric 的精确计算（面子/搭子分解 + Pareto frontier merge）。
- 新增 `setup_fast_eval0.py` 编译脚本。
- 在 `algo/eval/legacy.py` 的 `eval0()` 中优先调用 Cython 实现，失败时回退原 Python。

### 性能
- 单 call eval0：legacy ≈ 0.001 ms，Cython ≈ 0.001 ms，基本打平（legacy 已有 lru_cache）。
- `legacy_test.py` 平均耗时从 0.212 s 降到 0.088 s，说明批量场景下仍有收益。

### 结论
- eval0 本身已足够快，Cython 化 eval0 的收益有限。
- **真正的瓶颈是 eval2 / expectimax 的递归结构**。下一步应直接 Cython 化 `eval_rec` / `_expectimax_cached`，而非停留在 eval0。
- 但此工作验证了 Cython 工具链可用，为后续深度优化打下基础。

---

## 3. 报听终局精确求解 + defensive head

### 实现
- 新增 `algo/eval/endgame_solver.py`：
  - `exact_tenpai_ron_prob`：精确计算防守方弃牌后报听者通过 ron/self-draw 和牌的概率。
  - `best_defensive_discard`：在 hand14 中选择 EV 最高的弃牌。
- 新增 `scripts/rl/generate_exact_endgame_labels.py`：在 BeliefExp 自对弈中捕获「对手报听后、当前玩家出牌前」状态，计算每个合法弃牌的 exact EV。
- 新增 `scripts/rl/train_defensive_head.py`：在 current best backbone 上新增 34 维 `defensive_head`，用 MSE 监督预测 exact EV，只训练 head。
- 新增 `algo/agents/exact_defensive_agent.py` 与 `algo/agents/hybrid_nn_belief_filter_agent.py`：把 defensive EV 作为 safety filter 接入 discard 决策。

### 数据
- 1000 局：13,843 样本。
- 5000 局：69,611 样本。
- 合并 6000 局：83,454 样本。

### 训练结果
| 数据量 | val MSE | best tile accuracy | Spearman |
|---|---|---|---|
| 1k | 0.0258 | ~8.3% | 0.20 |
| 6k | 0.0270 | ~8.3% | — |

### benchmark 结果
- `ExactDefensiveAgent` / `HybridFilter(def)` 在小/中规模 pool 中未显著优于 Hybrid baseline。
- 主要原因：defensive head 需要从当前玩家视角特征预测精确 EV，但特征中看不到对手手牌/待牌，任务本身信噪比低。

### 结论
- 独立训练 defensive head 预测 exact EV **效果不明显**。
- 更合理的用法可能是：把 exact endgame solver 直接作为 BeliefExp 在终盘 critical 状态的决策模块（已知/推断对手待牌分布），而不是让 NN 从特征中学。
- 当前 `ExactEndgameAgent` 是朝这个方向的一个 wrapper，但对战中需要信念推断对手手牌，尚未闭环。

---

## 4. 对手建模从「是否听牌」改为「待牌分布」

### 实现
- 在 `algo/nn/model.py` 的 `TileConvNet` 新增 `wait_dist_head`，输出 34 维待牌分布 logit。
- 新增 `scripts/rl/generate_wait_dist_labels.py`：在 BeliefExp 自对弈中捕获当前玩家视角特征 + 下家真实待牌 one-hot。
- 新增 `scripts/rl/train_wait_dist.py`：只训练 `wait_dist_head`（backbone frozen）。
- 新增 `algo/agents/wait_dist_defensive_agent.py` 与 `algo/agents/hybrid_nn_belief_waitdist_agent.py`：
  - 用 wait_dist 作为 safety filter 调整 NN policy。
  - 或用最大待牌概率触发 BeliefExp 搜索。
- 在 `scripts/rl/benchmark_pool.py` 注册 `waitdef` / `hybridwait` / `hybridfilter` token。

### 数据与训练
- 生成 10,000 局，过滤后保留 66,960 个下家听牌样本。
- 原始 fc-only head：recall@1/2/3/5 = 7.4% / 13.5% / 19.6% / 30.1%。
- 改用 conv+fc head（与 policy/dealin 头一致）：recall@1/2/3/5 = 9.5% / 16.9% / 23.8% / 36.0%。
- 300 局小数据上 recall@5 ≈ 50%，但泛化到 10k 分布后明显下降，说明任务难度高/大数据更真实。

### benchmark 结果
- `HybridFilter(wait)` 在 400 局 pool 中：32.5% 胜率 / Elo 1545，略高于 Hybrid 30.0% / Elo 1516，但差异在噪声范围内。
- `HybridWait`（用 wait 触发 BeliefExp）未显著优于 Hybrid。

### 结论
- wait_dist_head 学到了一定信号，但 recall@5=36% 仍偏低，单独作为 policy filter 提升有限。
- 原因：当前只预测下家待牌，而点炮风险来自三家；且误报率较高（precision@5≈12%）。
- **下一步**：同时预测三家待牌（旋转座位训练三个 head 或一个 head + seat 嵌入）；或把 wait_dist 接入 BeliefExp 的信念更新而非直接改 policy。

---

## 5. 其他代码改动

- `algo/agents/ppo_agent.py`：修复 `_response_action` 在模型同时含 `response_head` 与 `wait_dist_head` / `defensive_head` 时误取最后一个输出的 bug，改为按输出维度（4）查找 response logits。
- `scripts/rl/benchmark_pool.py`：注册 `waitdef`、`exactdef`、`hybridwait`、`hybridfilter` token 及对应环境变量说明。
- `.gitignore`：追加 `*.cpp`（Cython 生成文件）。

---

## 6. 综合结论与下一步建议

1. **测量问题最大**：duplicate 结果与常规 pool 结论冲突，当前 best 链条需要复核。在继续算法实验前，应先把 duplicate 标准考场跑大到 5000 seed，并统一晋升标准（CI 不含 0 才晋升）。
2. **Cython 化应深入 eval2/expectimax**：eval0 已够快，真正的瓶颈是递归搜索。建议用 Cython 重写 `eval_rec` / `_expectimax_cached` 的热路径，目标让 V3 depth=2 可行。
3. **exact endgame 应直接接入搜索**：独立 NN head 学习困难，应把 exact solver 作为 BeliefExp 终盘决策模块，结合信念推断的待牌分布。
4. **wait_dist 应预测三家并接入信念**：当前只预测下家且质量有限；扩展为三家待牌分布，并用于 BeliefExp 的 `_effective_remaining`，可能更有价值。
5. **不建议继续**：单独训练更多 defensive/wait_dist 数据、在现有架构下继续 soup/蒸馏 bootstrap。

---

## 7. 产出文件清单

```
algo/eval/_fast_eval0.pyx
algo/eval/endgame_solver.py
algo/agents/wait_dist_defensive_agent.py
algo/agents/hybrid_nn_belief_waitdist_agent.py
algo/agents/hybrid_nn_belief_filter_agent.py
algo/agents/exact_defensive_agent.py
scripts/rl/benchmark_duplicate.py
scripts/rl/generate_wait_dist_labels.py
scripts/rl/generate_exact_endgame_labels.py
scripts/rl/train_wait_dist.py
scripts/rl/train_defensive_head.py
setup_fast_eval0.py
output/wait_dist_labels_10000_tenpai.npz
output/nn_wait_dist_10k_tenpai_conv.pt
output/exact_endgame_labels_6000.npz
output/nn_defensive_6000.pt
```

---

## 更新（2026-07-16）：§1 duplicate 结论为 harness bug 产物

本报告 §1 的 paired 差（Hybrid-Best −15.4% vs Baseline、−12.7% vs BeliefExp）由
`benchmark_duplicate.py` 旧版同名前缀匹配 bug 造成。修正后符号反转：**Hybrid-Best +7.0%
[+3.9,+10.1] vs Baseline、+7.1% [+4.1,+10.1] vs BeliefExp**（1000 arena pairs）。
Baseline vs BeliefExp +0.1% [−2.3,+2.5]（结论不变：不显著）。§2–§5（Cython、exact
endgame、wait_dist）不受影响。完整重算见 `docs/reports/duplicate-reanalysis-0716.md`。
