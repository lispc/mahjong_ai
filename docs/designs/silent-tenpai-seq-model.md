# 设计：默听检测与待牌预测的序列特征模型（方向 D）

> 2026-07-16。动机来自方向 A/B 的误差分解（`docs/handoff.md` §4）：
> **82% 的点炮送给默听（未报听）玩家**，而 exact endgame solver 只覆盖已报听者
> （18%），且 BeliefExp 对已报听者的防守已接近最优。防守的剩余前沿是
> **中盘默听检测**：谁已经听牌、待什么。

---

## 1. 为什么现有模型不够

- `nn_wait_dist3_10k.pt`（3 家待牌）：per-opp recall@5 仅 21%（`output/train_wait_dist3_10000.log`）。
- 对手听牌二分类（`output/opponent_model.pt`，16k 局 68.5 万 snapshot）：
  val_acc 0.840 vs 常数基线 0.826，信号仅 +1.5pp。
- 共同瓶颈被诊断为**特征**：175 维输入只含每名对手的弃牌**计数向量**
  （顺序丢失）+ 报听 flag。默听判断的关键证据在**顺序与时机**里：
  早巡打出的中张、某花色连切、速度突变、宣言牌邻域（对已报听者）。

## 2. 数据（复用 gen_opponent_data.py 框架，新格式）

每个决策 snapshot 记录：

- 静态特征：现有 175 维（自己手牌 + 牌山 + 弃牌计数 + flags）。
- 每名对手（下家/对家/上家）：
  - 弃牌**序列**（tile id 列表，保留顺序，≤40 步）——`ContextV3.discards` 本就是
    按时间 append 的；
  - 是否已报听 + 报听发生在序列第几步（−1 = 未报听）；引擎的 `tenpai` 事件含时机；
  - 副露列表（类型 + tile）。
- 标签（免费完美标签，自对弈可见真实手牌）：
  - `opp_tenpai`（3，shanten==0 即听，**含默听**——`gen_wait_dist3_labels.py` 已按此口径）；
  - `opp_wait`（3×34 one-hot，听牌者的 winning_tiles）。

规模：先 20k 局（BeliefExp 自对弈，~30-60 min @ 64 workers）。checkpoint 分段保存
（AGENTS.md §9 守则）。

## 3. 模型（小规模，避免重蹈"大网络从零训练失败"）

- 每对手序列编码器：tile embedding(32) + 位置/巡目 embedding + "报听后"步标记 →
  GRU(64) 或 2 层 1D conv → 64 维对手向量。
- 3 对手向量 + 175 维静态 → MLP(256) → 两个 head：
  - tenpai：3 维 BCE（样本不均衡，pos_weight 按占比设）；
  - wait：3×34 sigmoid（仅对 tenpai 样本计 loss）。
- 参数量 < 500k，单 GPU 训练分钟级。

## 4. 验收门（止损线，按 `docs/eval-protocol.md` 精神）

依次检查，任一不过即止损：

1. **离线门**：tenpai 检测 AUC ≥ 0.75（现状特征基线 ~0.58 等效）；或 tenpai 玩家
   wait recall@5 ≥ 40%（现状 21%/36%）。若序列特征不能带来量级提升 →
   说明该信息在弃牌序列中也不可得，方向 D 关闭。
2. **接入门**：tenpai 概率接入 BeliefExp `_danger_signal`（替代/补充启发式
   danger level），wait 分布接入 beend 安全分；1000-pair duplicate vs 对应 base
   （beliefexp / hybrid）CI 不含 0 才算过。
3. **晋升门**：5000-pair duplicate arena vs 当前 best，paired win diff CI 不含 0
   + 独立种子复跑。

### 离线门结果（2026-07-16，已过）

| 数据 | 模型 | silent AUC | silent wait r@5 |
|---|---|---|---|
| v1（BeliefExp 自对弈 20k，993k 样本） | no-seq（仅 175 维计数） | 0.735 | 0.110 |
| v1 | seq（GRU 序列编码） | **0.913** | 0.105 |
| v2（混合池 20k，789k 样本，默听率 10.3%） | no-seq | 0.828 | 0.380 |
| v2 | seq | **0.915**（epoch 6，仍在升） | 0.371 |

- 检测门 **PASS**：seq 在两个数据集上 silent AUC ≥ 0.91，且 seq 稳定优于
  no-seq（v1 +0.18 / v2 +0.09）——**弃牌顺序里确实有计数特征没有的信息**。
- wait 内容预测仍未到 40% 门（v2 37%）：接入以 tenpai 概率为主（`_danger_signal`
  扩展），wait 分布只作弱 danger 修正。
- 校准（v1 模型）：阈值 0.5 时 silent-tenpai recall 56% / FPR 2.5%；
  阈值 0.3 时 recall 67% / FPR 5.6%。
- 模型：`output/nn_seq_opp_v1.pt`（v1）、`output/nn_seq_opp_v2.pt`（v2，部署匹配）。

### 接入门/晋升门结果（2026-07-16，未过，方向关闭）

接入：tenpai 概率并入 `_danger_signal`，wait 分布并入 danger
（`algo/agents/belief_silent_guard_agent.py`、`hybrid_nn_besilent_agent.py`）。

| 对比 | 阈值 | pairs | paired win diff |
|---|---|---|---|
| hybridsilent vs hybrid | 0.5 | 1000 | +0.3% [−0.5,+1.1] |
| hybridsilent vs hybrid | 0.3 | 1000 | +1.0% [+0.0,+2.0]（筛查命中） |
| hybridsilent vs hybrid | 0.3 | **5000 复跑** | **+0.1% [−0.4,+0.5]**（不复现） |
| hybridsilent vs hybrid | 0.2 | 1000 | −0.6% [−1.4,+0.2] |
| besilent vs beliefexp | 0.5 | 1000 | −0.6% [−2.1,+0.9] |

机制探针：模型新增触发 5.6%、改变 7.5% 弃牌选择，但结局不变。
**检测可学、接入不变现**——筛查的 +1.0% 为 winner's curse（协议拦截成功）。
方向关闭，详见 `docs/reports/silent-tenpai-d-0716.md`。

## 5. 与已证伪方向的区别（为什么这次可能不同）

- 不是「改输入特征做 BC」：212 维 danger 特征失败是让 policy 网络用 danger 去搏牌；
  这里 danger 只进**搜索层的安全 tie-breaking**（BeliefExp 已验证的 +32pp 机制），
  不改 policy 学习目标。
- 不是「standalone defensive head」：不独立决策，只在 safe set 内重排。
- 不是「已报听终盘」：目标是默听（82% 误差所在），前人模型训练目标是已报听者
  的待牌（部署错配）。
- 序列信息从未进过任何模型（291 维/212 维都是计数快照）。

## 6. 风险

- 默听的可预测性本身有上限（信息论角度，高手默听就是设计来不可读的）；
  离线门不过就认。
- 接入后过度防守（历史：CEM 调权混合场 V3 过度防守点和率 10%）；用 score-proxy
  配对差守门，不只看点炮率。
