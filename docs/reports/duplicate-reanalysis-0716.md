# Duplicate 评测复核：配对统计 bug 与修正结果（2026-07-16）

> 结论先行：fable-5 评审（`docs/fable-5-review-0706.md` 方向 1 与 `docs/reports/fable-5-execution-0706.md`）
> 依据的 duplicate 负数结果由 **benchmark 脚本配对统计 bug** 造成。用正确方法重算全部历史
> pkl 后，结论反转：**Hybrid-Best（SoupDistilled）在 duplicate 赛制下显著强于 Baseline 与
> BeliefExp（+7~10pp，95% CI 不含 0），当前 best 链条有效**。fable-5 评审的系统性批评
> （统计功效、多重比较、协议不统一）依然成立，已据此制定 `docs/eval-protocol.md`。

---

## 1. bug 是什么

2026-07-06 的 duplicate 运行使用的 `benchmark_duplicate.py` 旧版本，在配对统计与候选胜率
统计中用**名字前缀匹配**识别候选。当候选与某个固定对手同名（如候选 `Baseline` 对阵 3 个
`Baseline` 对手，或 arena 对手三件套中恰好包含 `baseline`/`beliefexp`）时，同名对手赢的局
也被计入候选，导致：

- 同名对手越多，候选"胜率"越虚高 → 配对差被系统性拉向负值或荒谬值（−62.3%、+72.7%）。

实锤证据（`output/duplicate_hybrid_vs_baseline_5000.pkl` 原始结果重算）：

- B 方候选 `Baseline@0_b` 实际胜 **1351/5000（27.0%）**，旧日志却报 B wins 3140/5000（62.8%）；
- A 方 `Hybrid-Best@0_a` 实际胜 1822/5000，旧日志报 A-only wins 23/5000。

无名称碰撞的运行（`hybrid_vs_beliefexp_5000`、`beend3_*`、`v3d2nn_*`）旧结果与重算一致，
进一步确认 bug 的触发条件就是名称碰撞。

现行脚本（commit `2ceae7d` 起）配对比较已改为带席位后缀的精确匹配，本次又把候选胜率统计
改为按 `players_order[候选席位]` 识别，并新增计分代理配对差（见 §4）。修复后脚本已用
40 seeds 小跑验证输出合理。

## 2. 重算方法

对每份 pkl 的原始 `results`：按存储顺序每两条为一组（A 局、B 局），候选席位由
`players_order[pos]` 直接给出（非镜像 pos=0，镜像按 (pair index) mod 4 轮换），
候选胜利 = `winner == players_order[pos]`。配对差与 95% CI 公式不变
（±1/0/∓1 配对差的正态近似）。全部 13 份 pkl 重算，bad pair（席位名与候选名不符）为 0。

## 3. 修正后的全部结果

| pkl | 对手 | pairs | 旧存储（bug） | **修正后** |
|---|---|---|---|---|
| arena_hybrid_vs_baseline_1000 | 混合三件套 | 1000 | −15.4% [−19.2,−11.6] | **+7.0% [+3.9,+10.1]** |
| arena_hybrid_vs_beliefexp_1000 | 混合三件套 | 1000 | −12.7% [−16.3,−9.1] | **+7.1% [+4.1,+10.1]** |
| arena_baseline_vs_beliefexp_1000 | 混合三件套 | 1000 | +2.7% [−1.6,+7.0] | +0.1% [−2.3,+2.5] |
| hybrid_vs_baseline_5000 | 3×Baseline | 5000 | −62.3% [−63.7,−61.0] | **+9.4% [+8.0,+10.9]** |
| hybrid_vs_beliefexp_5000 | 3×Baseline | 5000 | +10.4% [+9.0,+11.8] | +10.4% [+9.0,+11.8]（本就正确）|
| baseline_vs_beliefexp_5000 | 3×Baseline | 5000 | +72.7% [+71.5,+74.0] | +1.0% [+0.1,+1.8] |
| best_vs_baseline_400 | 混合三件套 | 400 | −20.2% [−26.2,−14.3] | +4.5% [−0.3,+9.3] |
| best_vs_baseline_mirror_100 | 混合三件套（4 席位镜像） | 400 | −22.8% [−28.7,−16.8] | +6.2% [+1.5,+11.0] |
| best_vs_baseline_100 | 混合三件套 | 100 | −30.0% [−41.9,−18.1] | −3.0% [−13.2,+7.2] |
| beend3(_tuned/_tuned2)_vs_beliefexp_400 | — | 400×3 | 与重算一致 | （无碰撞，未受影响）|
| v3d2nn_vs_v3d1nn_1000 | — | 1000 | 与重算一致 | （无碰撞，未受影响）|

（"混合三件套" = `baseline,beliefexp,hybrid:Base:output/nn_full_action_best.pt`。）

另有本次新跑：newbest vs oldbest 5000 seeds（`output/duplicate_newbest_vs_oldbest_5000.log`），
结果见该日志与本文件末尾更新段落。

## 4. 修正后的结论

1. **当前 best 有效**：Hybrid-Best（SoupDistilled）对 Baseline 的 paired 差在 4 个独立
   种子集/赛制下分别为 +9.4%（5000，3×Baseline）、+7.0%（1000 arena）、+6.2%（镜像 100×4）、
   +4.5%（400 arena，CI 擦 0）。对 BeliefExp 为 +10.4%（5000）与 +7.1%（1000）。
   「best 链条部分由运气选出」的危机判断**不成立**。
2. **BeliefExp ≈ Baseline**（5000 pairs 仅 +1.0%）：在 3×Baseline 考场中两者几乎同强，
   Hybrid 的优势（+9~10pp）来自 NN + 搜索的混合结构本身，这是对消融结论
   （BeliefExp 搜索 +32pp）的独立印证。
3. 统计功效：配对后 ties 占 57–72%，5000 pairs 的 95% CI ≈ ±1.3pp，1000 pairs ≈ ±3pp。
   未来的晋升决策应以 5000-pair duplicate 为准（见 `docs/eval-protocol.md`）。
4. 教训已转化为工程改动：`benchmark_duplicate.py` 候选统计改为席位识别；
   新增 **score-proxy 配对差**（自摸 +3 / 点和 +1 / 放炮 −1，推倒胡计分代理）作为
   主指标之外的第二指标，全部历史 pkl 因含 `dealer`/`win_type` 字段可直接复算。

## 5. 对既有文档的修正

- `docs/fable-5-review-0706.md`：方向 1 的「当前 best 存疑」不成立；其余系统性批评
  （测量精度优先、多重比较、统一考场、晋升门槛成文）仍然正确并已采纳。
- `docs/reports/fable-5-execution-0706.md`：§1 的 duplicate 结论（−15.4%/−12.7%）为 bug 产物。
- `AGENTS.md` §4 的「2026-07-06 重要更新」段落：已按本报告修正。

## 更新记录

- 2026-07-16：初版（bug 发现 + 13 份 pkl 重算）。
- 2026-07-16：补充新跑结果——NewBest vs OldBest 5000 pairs（arena 对手）：paired win
  diff **+0.2% [−0.5,+0.9]**，score-proxy +0.003 [−0.013,+0.020]。soup→蒸馏这最后一环
  配对增益为零，winner's curse 实锤；两模型同强，anchor 维持 `nn_full_action_best.pt`。
  结果文件：`output/duplicate_newbest_vs_oldbest_5000.{pkl,log}`。
