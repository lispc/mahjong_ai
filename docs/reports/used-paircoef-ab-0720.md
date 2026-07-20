# used 修复 + pair_coef A/B 判决（2026-07-20）

> 预登记批次：2 候选，1000-pair duplicate 筛查，门 = +1.0% 进 5000p。
> 背景：AGENTS.md §7.18（BeliefExp 的 eval2 实为「空 Context」，used 被 Cython
> 快路径静默忽略）与「Cython eval0/eval2 实际 pair_coef=1.0 ≠ config 0.6」。
> 结论：**两个候选均不晋升，方向关闭；默认行为维持不变。**

---

## 1. 实现（默认行为零改动）

- `context.py` 新增 `UsedAwareContext`（带 `all_tiles_as_dict()` 返回 used），
  默认 `Context` 不动 → 现有 agent 行为逐值不变。
- `BeliefExpectimaxAgent(used_aware_eval2=True)` 时 `_legacy_context()` 换用
  `UsedAwareContext`；默认 False = 历史行为。
- `algo.eval2(..., pair_coef=None)` 新增可选参数（None = Cython 默认 1.0，
  与历史 arena 语义一致）。
- 新 agent：`BaselinePairCoefAgent`（与 Baseline 仅 pair_coef 不同）。
- token：`beliefexpused`、`baselinepc06`（benchmark_pool / benchmark_duplicate 通用）。
- 测试 `tests/test_eval2_used_paircoef.py`（已并入 `run_tests.py`，全绿）：
  Cython used-aware 与纯 Python 路径 150/150 逐值一致（<1e-9）；used 改变
  147/150 随机状态的结果；pair_coef 0.6 vs 1.0 改变 60/60；默认路径回归一致。

附带考古发现：route-a 的 `ExpectiMaxEval2Agent`（Eval2Ctx）同样被 Cython 迁移
静默回退为空 Context——「used-aware eval2」在 Cython 时代从未真正上过场，
本次是其首次功率充分的测量。

## 2. 考场与判决

考场：duplicate arena（对手三件套 `baseline, beliefexp,
hybridnm:Base:output/nn_full_action_best.pt`），pos0 镜像，各 1000 pairs。
本 meta 下三家对手均不副露 → used 信息完整无缺口，测试条件对修复方最有利。

| 候选 | paired win diff | score-proxy | 点炮 guardrail | 判定 |
|---|---|---|---|---|
| BeliefExpUsed − Baseline | **−2.6% [−5.0, −0.2]** | +0.012 [−0.044, +0.068] | 17.8% vs 20.4%（−2.6pp） | **放弃**（CI 不含 0，方向为负） |
| BaselinePC06 − Baseline | −1.0% [−2.5, +0.5] | +0.004 [−0.030, +0.038] | 19.6% vs 20.4% | **维持现状**（噪声内，维持 de-facto 1.0） |

pkl：`output/dup_beused_vs_baseline_1000.pkl`、`output/dup_baselinepc06_vs_baseline_1000.pkl`。

## 3. 解读

1. **used 修复是第四次「扰动 eval2 攻守平衡」阴性**：点炮 −2.6pp（防守确实
   变好）但胜率 −2.6%（显著），score-proxy 不显著——按 eval-protocol §1
   「更守但不更强不算晋升」的明文裁决，关闭。与 route-a BaseDef、Baseline+
   tie-break、Eval2Ctx+BD 三次历史阴性同构；差异在于本次扰动来自「更准的
   剩余分布」而非外挂防守项，说明 eval2 的空 Context 分布本身就是一种
   已调优的隐含先验，精确化不等于更强。
2. **「Baseline ≈ BeliefExp 是因为进攻同构」的假说被削弱而非证实**：修复后
   BeliefExpUsed 反而显著弱于 Baseline——顶层五和局的成因不是「used 被
   忽略」，used 通道本身就不是强度杠杆（与 god-mode 上界 +1.2% 一致）。
3. **pair_coef 1.0 vs 0.6 无实际差异**（94.2% pairs 平局，CI 半宽 ±1.5pp）：
   config 与 Cython 的历史不一致**不需要修**，维持 de-facto 1.0；
   历史 PyPy MC 标签（0.6 语义）与 arena（1.0 语义）的口径差影响可忽略。
4. **jax 侧无需同步**：修复方案未晋升，beliefjax 按「空 Context」移植的
   行为继续与 arena 一致（§7.18 的描述保持为「现状语义」）。

## 4. 资产与后续

- 可复用：`UsedAwareContext`、`used_aware_eval2` 开关、`pair_coef` 参数、
  两个 token、parity 测试（Cython↔Python eval2 的首个逐值对齐套件）。
- 可选（低先验，未做）：hybridnm 搜索层接 used_aware（S1 先验为负，且顶层
  噪声地板 ±1.3pp，预期不可分辨）；pair_coef 中间值扫描（同上）。
- 本批 2 候选零晋升，按多重比较控制不消耗 5000-pair。
