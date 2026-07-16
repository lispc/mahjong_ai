# 评测协议（Evaluation Protocol）

> 2026-07-16 制定。背景与依据见 `docs/reports/duplicate-reanalysis-0716.md`
> （duplicate 配对统计 bug 的修正）与 `docs/fable-5-review-0706.md`（统计功效批评）。
> 本协议是**晋升/放弃决策的唯一有效依据**；此前的普通 pool 胜率、Elo 数字只作历史参考。

---

## 1. 目标函数结论（先回答"以什么为目标"）

晋北推倒胡无番型计分，引擎也不记点数，单局结果只有 `win_type ∈ {self, ron, draw}`、
`winner`、`dealer`（放炮者）。真实对局的 stakes 结构是：自摸三家付、点和放炮者一家付。

因此：

- **主指标：paired win diff**（同一副牌下 A、B 两候选的胜率配对差，95% CI）。
  在推倒胡规则下，赢牌就是得分的前提，胜率与期望得分高度相关，且二元指标方差最小、
  最稳健。
- **辅指标：score-proxy paired diff**，计分代理 = 自摸 +3 / 点和 +1 / 放炮 −1 / 其他 0。
  它比胜率多利用"赢的成色"和"点炮代价"信息；`benchmark_duplicate.py` 已内置输出。
  当两个候选胜率差在噪声内、但 score-proxy 差 CI 不含 0 时，可作为补充证据。
- **guardrail：点炮率**。从普通 pool tournament（含 event_log）获取，不进晋升公式，
  但点炮率显著恶化（> +3pp）的候选即使胜率达标也应暂缓晋升。
- **不用于决策**：Elo（4 人非对称池中已证明漂移失真）、vs-frozen 自对弈胜率、
  小样本（<400 局）胜率。

「更守但不更强（点炮降、胜率不变）算不算改进」的裁决：**不算晋升**，除非 score-proxy
差显著为正。防守收益应体现在少放炮转化为得分差上。

> 目标分歧备注（2026-07-16）：引擎不计分（赢即赢，无报听/自摸加成），且当前 best
> （HybridNNBeliefAgent）因无 `.context` 属性在引擎对局中**从不报听**——在引擎目标下
> 报听只有代价没有收益，这不影响本协议内部比较的有效性，但与真实晋北计分对局存在
> 系统性分歧。若未来接入计分（报听加分/自摸三家付），报听相关策略与终盘求解器
> 需重新评估（`docs/reports/endgame-solver-ab-0716.md` §2.3）。

## 2. 标准考场

### 2.1 Duplicate arena（主考场）

```
对手三件套（固定）：baseline, beliefexp, hybrid:Base:<anchor_model>
候选席位：position 0 镜像（默认 2 局/seed）
anchor_model：做决策时的当前 best（当前为 output/nn_full_action_best.pt）
```

命令：

```bash
PYTHONPATH=. python3 scripts/rl/benchmark_duplicate.py \
    --a <candidateA token> --b <candidateB token> \
    --opponents baseline,beliefexp,hybrid:Base:output/nn_full_action_best.pt \
    --n-seeds 5000 --workers 32 \
    --output output/duplicate_<A>_vs_<B>_5000.pkl
```

样本量档位（paired win diff 的 95% CI 半宽经验值）：

| n-seeds | CI 半宽 | 用途 |
|---|---|---|
| 400 | ±5pp | 只用于 sanity check，不做决策 |
| 1000 | ±3pp | 筛查，效应 >5pp 可初判 |
| 5000 | ±1.3pp | **晋升/放弃决策标准** |

### 2.2 普通 pool tournament（辅助）

`scripts/rl/benchmark_pool.py` 同一 pool 400+ 局，用于观察绝对胜率、自摸/点和构成、
点炮率 guardrail 与决策耗时。Elo 列打印仅作记录，不作依据。

## 3. 晋升与放弃规则

1. **晋升**：5000-pair duplicate 中 paired win diff 的 95% CI 不含 0，且效应量 ≥ 2×SE；
   随后用不同 `--seed-offset` 独立复跑一次，符号一致才生效。
2. **放弃**：同样证据强度下 CI 不含 0 且方向为负；或 CI 宽度已小于候选先验收益而含 0
   （即"小到不值得"）。**否决一条大方向所需的证据强度不低于晋升一个模型**——
   禁止用 80–100 局的否定结果判死刑（历史教训：depth-2 教师曾被 80 局否决）。
3. **多重比较控制**：一批候选只挑一个晋升；对"这批里最好的那个"必须用独立种子复跑
   （winner's curse 防护）。每批实验在报告中登记候选总数。
4. **名称碰撞检查**：candidate label 不得与任何对手 base name 相同（脚本已改为席位识别，
   但报告阅读时仍需确认）。
5. 所有 duplicate 原始 pkl 保留在 `output/duplicate_*.pkl`，报告中引用文件名与 paired
   统计，便于事后用正确方法重算。

## 4. 报告模板（每次 benchmark 后附在实验报告里）

```
- 考场：duplicate arena（对手：baseline, beliefexp, hybrid:Base:<anchor>），pos0 镜像
- pairs：N；paired win diff A−B = +x.x% [lo, hi]；score-proxy diff = +x.xxx [lo, hi]
- 独立复跑（seed-offset M）：符号一致 / 不一致
- guardrail：pool 400 局点炮率 A x.x% vs B y.y%；平均决策耗时
- 决策：晋升 / 不晋升 / 加样本复测
```

## 5. 已确认的 best 链条基线（2026-07-16 修正后）

| 对比 | pairs | paired win diff | 出处 |
|---|---|---|---|
| Hybrid-Best − Baseline | 5000（3×Baseline） | **+9.4% [+8.0,+10.9]** | duplicate_hybrid_vs_baseline_5000.pkl 重算 |
| Hybrid-Best − Baseline | 1000（arena） | +7.0% [+3.9,+10.1] | duplicate_arena_hybrid_vs_baseline_1000.pkl 重算 |
| Hybrid-Best − BeliefExp | 5000（3×Baseline） | **+10.4% [+9.0,+11.8]** | duplicate_hybrid_vs_beliefexp_5000.pkl |
| Hybrid-Best − BeliefExp | 1000（arena） | +7.1% [+4.1,+10.1] | duplicate_arena_hybrid_vs_beliefexp_1000.pkl 重算 |
| Baseline − BeliefExp | 5000（3×Baseline） | +1.0% [+0.1,+1.8] | duplicate_baseline_vs_beliefexp_5000.pkl 重算 |
| Hybrid-NewBest − Hybrid-OldBest | 5000（arena） | +0.2% [−0.5,+0.9]（score-proxy +0.003 [−0.013,+0.020]，同强） | duplicate_newbest_vs_oldbest_5000.pkl |

> 注：soup→蒸馏一环 paired 增益为零（winner's curse 实锤），NewBest/OldBest 同强；
> 当前 anchor 保持不变（NewBest 不差于 OldBest 的下界 −0.5%）。
