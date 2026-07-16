# 方向 A/B 阶段报告：终盘精确求解路线判死（2026-07-16）

> 结论：**「报听终局精确求解接入 BeliefExp/Hybrid」在四个独立证据层面全部阴性**，
> 按 `docs/eval-protocol.md` 止损。防守的真实前沿在**中盘默听检测**（82% 点炮来源），
> 已由方向 D 接手（`docs/designs/silent-tenpai-seq-model.md`）。

---

## 1. 证据链

### 1.1 结局级：paired duplicate（协议标准考场）

| 对比 | pairs | paired win diff | 判读 |
|---|---|---|---|
| hybridend vs hybrid（当前 best 接 exact solver 搜索层） | 5000（arena） | +0.0%（99.9% ties；score-proxy −0.000 [−0.002,+0.002]） | 统计无差异 |
| oracle-beend vs beliefexp（**完美待牌**上界） | 2000 | **−0.1% [−0.1%,+0.0]**（A-only 0 / B-only 1） | 上界为零 |
| beend vs beliefexp（NN wait_dist3 待牌） | 5000 | +0.0% [−0.1,+0.1]（4998/5000 ties；score-proxy +0.001） | null（确认） |

- oracle gate（`scripts/rl/oracle_endgame_gate.py`）：给 exact solver 换上完美待牌
  （同进程读报听对手真实手牌，仅限已报听者=部署信息边界）。即使信息完美，
  相对 BeliefExp 2000 局只改变 1 局结果。**待牌预测质量不是瓶颈，方向 B
  （训练更好 wait 模型喂 solver）的前提不成立，止损。**
- 历史对照：standalone exact defensive head（11% win）、`HybridFilter(def)`、
  beend 400-pair null——当时的"未显著"实为功效不足，本次 5000-pair + oracle
  上界给出决定性证据。

### 1.2 机制级：solver 的决策增量≈0

在触发状态（终盘 + 存在报听威胁）逐决策探针（`belief_endgame_agent` 逻辑）：

- 触发率 **0.16 次/agent-game**（120 agent-games 共 19 次）；
- BeliefExp 首选落入**真实**待牌集合仅 **1/120 agent-games**——
  即 solver 的 veto 最大决策影响 ≈ 0.8% 的局；
- 用 NN wait_dist3 时 solver 与 danger 启发式 **0/9 分歧**；
  用完美待牌时 7/13「分歧」全部是 safety 并列后的 offense tie-break 噪声
  （exact EV 模型对非待牌候选输出相同值：弃牌不点炮时不影响报听者自摸概率，
  评分天然二元化，无梯度区分力）。

### 1.3 误差分解：点炮 82% 送给默听，solver 域只有 18%

200 局 arena（Hybrid-Best/Baseline×2/BeliefExp）event_log 分解：

- 点炮 153 次：**82.4% 和牌者未报听（默听）**，17.6% 已报听；
- BeliefExp 对已报听者的防守样本内 **0 失误**（42 次点炮全部送给默听）；
- 时段分布：中盘（28–55 张弃牌）64%，尾盘 20%，早盘 16%。

### 1.4 结论

对已报听对手的终盘防守，BeliefExp 的 danger 启发式（现物/筋/可见数）
**已经接近信息可达的最优**；精确求解器在这个状态类没有增量可拿。
历史上所有终盘防守改进尝试（safety oracle、危险特征、加权 BC、defensive head、
exact solver）失败的共同根因即此：**目标状态类选错了**——剩余误差在中盘默听。

## 2. 附带发现

1. **Baseline 永不报听**（`agent.Agent.declare_tenpai` 基类返回 False）；
   **HybridNNBeliefAgent 在引擎对局中也从不报听**——引擎用
   `getattr(agent, 'context', None)` 传 context，Hybrid 无 `.context` 属性，
   PPOAgent 的 tenpai head 拿到 None 后回退基类。当前 best 全程默听。
   在引擎"赢即得分、无报听加分"的目标下这未必是坏事（报听只有锁手+暴露的代价），
   但它暴露了**评测目标与真实晋北计分的分歧**（真实对局报听有计分意义），
   记入 `docs/eval-protocol.md` §1 的目标函数备注。
2. 混合agent池（BeliefExp/Baseline/Hybrid）默听分布：Baseline 听牌 100% 默听、
   BeliefExp 约半数报听、Hybrid 全默听——对手建模数据必须用混合池
   （BeliefExp 自对弈的默听率仅 1.4%/row，混合池 10.9%/row）。
3. 产物保留：`algo/agents/hybrid_nn_belief_endgame_agent.py`（`hybridend:` token）、
   `scripts/rl/oracle_endgame_gate.py`——若未来评测目标改为含计分，
   终盘求解器可重新评估。

## 3. 对执行计划的影响

- 方向 A：止损（本报告）。
- 方向 B（3 家 wait_dist 接入 belief 更新喂 solver）：前提不成立，止损；
  wait 预测研究并入方向 D（默听检测）。
- 「教师更强则 trace 蒸馏」：A/B 未产出更强教师，跳过。
- 方向 C（外部数据）：判死——公开牌谱仅天凤（日麻）/MCR，无晋北同规则数据，
  跨规则迁移目标函数冲突。
- 方向 D（默听检测 + 序列特征）：唯一存活分支，数据生成中
  （v1 BeliefExp 自对弈 20k + v2 混合池 20k），离线门 AUC/recall 见后续报告。
