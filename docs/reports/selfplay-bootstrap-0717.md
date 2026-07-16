# 自对弈 Bootstrap + AWBC：突破 BC 天花板的又一次严格尝试

> 日期：2026-07-16/17（进行中）
> 依据：`docs/reports/ablation_report.md` §3.1/§7 ——旧 AWBC 未确认超越 BC，瓶颈被归因为
> value net 质量（旧 MLP `nn_value_model_mc.pt`）；消融推荐的下一步即「conv value net +
> online self-play outcome bootstrap」。本轮把这个 identified blocker 修掉并按
> `docs/eval-protocol.md` 严格评测。
> 与历史失败尝试的差异：PPO（在线、坍缩）/DPO（跨状态配对噪声）/KTO（未超越）/
> 旧 AWBC（旧 value net + BC 数据无动作对比）。本轮：on-policy 温度采样自对弈
> （动作对比）+ 同架构 conv critic（outcome 监督）+ advantage-weighted BC（离线、稳定）。

---

## 1. 设计（预登记，先于任何评测结果）

### 1.1 数据

- 演员：`output/nn_full_action_best.pt`（当前 best 的 NN 部分），4 座位自对弈，
  采样温度按局轮转 T ∈ {0.3, 0.5, 0.7}（试点：T=0 自对弈流局率 56%，T 越高流局越多；
  低温贴近部署 argmax 且保留动作对比）。
- 响应/报听决策用 response head / tenpai head（与部署一致，见
  `BootstrapActorAgent`）；只记录弃牌决策（feat/action/mask/old_value）。
- 标签：终局 score-proxy（自摸 +3 / 点和 +1 / 放炮 −1 / 其他 0，÷3 缩放以适配 tanh），
  按座位广播到该轨迹每一步；`reason_code` 落盘支持事后换 reward 重推。
- 规模：24,000 局 ≈ 168 万决策样本；按 game_id 切 2% val。

### 1.2 Critic

- 冻结主干，只训 `value_fc`/`value_head`（135k 参数），MSE 拟合 score-proxy/3。
- 验收：val corr(V, R) 显著 > 0 且优于模型自带 value head（old_value）的相关性。

### 1.3 候选（本批登记 3 个；多重比较控制见 eval-protocol §3.3）

| # | 候选 | 构造 |
|---|---|---|
| C1 | AWBC β=1.0 | w = clip(exp(A_std/1.0), 0, 20)，train split 归一化 |
| C2 | AWBC β=0.5 | w = clip(exp(A_std/0.5), 0, 20)（更激进） |
| C3 | filtered | w = 1[A_std > 0]（只学高于平均优势的样本） |

A = R − V(s)，按 train split 标准化。三者均从 `nn_full_action_best.pt` 微调
（masked 加权 CE，lr 5e-5，2 epochs，只动 discard policy 头及主干；
response/dealin/tenpai 头不直接训练）。lr/epochs 按 val 指标（加权 CE、
argmax 一致率）在筛查前调好，不产生额外候选。

### 1.4 评测（严格按 eval-protocol）

- 考场：duplicate arena，对手 `baseline,beliefexp,hybrid:Base:output/nn_full_action_best.pt`，
  pos0 镜像；`--b hybrid:Best:output/nn_full_action_best.pt`（当前 anchor）。
- 筛查：每候选 1000 pairs。晋级线（预登记）：point estimate ≥ +1.0% 取最优 1 个进 5000-pair；
  若全部 < +1.0% 直接判本批无晋升，不再消耗 5000-pair。
- 确认：5000 pairs（独立 seeds，seed-offset 与筛查不重叠），CI 不含 0 且效应 ≥ 2×SE
  → 再换 seed-offset 复跑一次符号一致才晋升。
- guardrail：晋级者补 pool 400 局看点炮率（恶化 > +3pp 暂缓）。
- 全部 pkl 留存 `output/duplicate_bootstrap_*.pkl`。

### 1.5 已知结构性限制（诚实记录）

- 纯 NN 自对弈流局率 ~50-56%（部署时 Hybrid vs 强对手 ~0.5%）：draw 局步 R=0 稀释信号，
  且自对弈对手分布 ≠ 部署对手分布（arena 1/3 Baseline）。若本轮无效，分布失配是
  首要候选解释。
- Hybrid 中 NN 实际职责 = 开局弃牌（总弃牌 <28）+ 副露后全部弃牌 + 响应 + 报听；
  本管线只微调 discard policy 头，响应头不动。
- 部署时 NN 的输入状态分布受搜索层影响（搜索层接管后产生的状态 NN 看不到）。

---

## 2. 结果（待填）

### 2.1 数据（collect，2026-07-16 23:48 完成）

- 24,000 局 / 1,651,675 决策步，96 workers 6.0 min（69 g/s）。
- 结局分布（步级）：draw 63.4%，ron 6.8%，deal_in 7.2%，tsumo 2.1%，
  lose_ron_others 13.8%，lose_tsumo 6.8%。局级流局率 ~50%+（与试点一致）。
- 试点发现：T=0 argmax 自对弈流局率同为 56% → 流局是 NN 策略自身完场慢所致，
  非温度噪声；故正式采集用低温 {0.3,0.5,0.7} 贴近部署。

### 2.2 Critic

- 基线（模型自带 value head，old_value）：val corr **0.084**（全步）/ 0.154（decisive 步），
  MSE 0.420（baseline var 0.036）——基本未校准，证实消融报告的瓶颈诊断。
- 本论 critic（冻主干，只训 value_fc/value_head，4 epochs lr 1e-3）：
  val MSE 0.0338 < 0.0357，corr **0.231**（≈3×）；校准桶近 1:1 无偏
  （V=−0.038→R=−0.022 … V=+0.105→R=+0.117）。
- 全干微调变体（lr 1e-4×2ep）corr 0.218，更差，弃用。
  产物：`output/nn_bootstrap_v1_critic.pt`（`..._critic_ft.pt` 弃用）。

### 2.3 微调诊断（recipe：lr 5e-5 × 2 epochs，全批 3 候选共用）

| 候选 | val CE（init 0.1667） | argmax 一致率 |
|---|---|---|
| C1 AWBC β=1.0 | 0.1437 | 0.983 |
| C2 AWBC β=0.5 | 0.1447 | 0.983 |
| C3 filtered | 0.1424 | 0.984 |

- 优势统计：A mean +0.007 / std 0.184，frac A>0 = 51.5%（中心化良好）。
- 更激进 recipe（lr 1e-4×3ep）val CE 0.146 更高，弃用（hygiene 调参，不产生新候选）。
- 三候选 argmax 漂移均 ~1.7%，集中在高 |A| 状态。

### 2.4 Duplicate 筛查（1000 pairs ×3，arena 标准考场）

| 候选 | paired win diff (A−B) | score-proxy diff | 决策 |
|---|---|---|---|
| C1 AWBC β=1.0 | −0.6% [−1.6,+0.4]（ties 97.4%） | −0.011 [−0.031,+0.009] | 未过线 |
| C2 AWBC β=0.5 | +0.0% [−1.0,+1.0]（ties 97.2%） | +0.004 [−0.017,+0.025] | 未过线 |
| C3 filtered | −0.3% [−1.3,+0.7]（ties 97.3%） | +0.001 [−0.023,+0.025] | 未过线 |

全部低于预登记晋级线 +1.0%，**本批无晋升，不消耗 5000-pair**。
pkl：`duplicate_bootstrap_{awbcb10,awbcb05,filtered}_vs_best_1000.pkl`。

### 2.5 v1 机制分析（为什么没动）

1. **探索对比太小**：记录动作 ≠ 模型 argmax 的比例仅 **4.8%**
   （T=0.3: 3.0%，T=0.5: 4.7%，T=0.7: 6.9%）。AWBC 的信号只能流经非 argmax
   样本（argmax 动作的 CE 已 ≈0，调权无效）→ 有效学习样本 ~7.9 万，且优势的
   噪声（70 步牌运）远大于动作效应。
2. **弃牌信用分配先天低 SNR**：终局 R 与单步早期弃牌之间的因果链被几十步
   牌运稀释；critic corr 0.231 已接近该数据规模下的合理上限。
3. 结论：outcome 级 RL（PPO/DPO/KTO/AWBC 第 4 次独立失败）的瓶颈不是算法，
   是**信用分配 SNR vs 可行数据规模**。提升探索温度会同时恶化对局质量与
   分布保真（T≥1 流局率 58%+），不是出路。

### 2.6 v2 设计：响应头 AWR

- 动机：响应头（碰/杠/胡）是消融第二大贡献组件（+22%），部署时**完全由 NN 负责**
  （无搜索接管），此前从未被 RL 触过（PPO 用启发式响应）；响应决策因果链短、
  杠杆大（碰改变手牌结构、胡立即结束）。
- 数据：v2 collect 24k 局（24h 内 1.8 min），响应决策 feat 含 offered tile
  （与部署一致），响应采样温度 max(T,1.0) 保证对比；共 140,923 条响应决策
  （pass 8.4k / peng 109k / gang 13.8k / hu 9.3k；响应头碰 take 率 ~82%）。
- 候选（本批登记 2 个）：R1 响应头 AWR β=1.0；R2 β=0.5。**冻结主干**，
  只训 response_fc/response_head（68k 参数）；critic 复用 v1。

### 2.7 v2 结果：响应头 AWR 筛查（1000 pairs ×2）

| 候选 | paired win diff | score-proxy diff | 决策 |
|---|---|---|---|
| R1 resp β=1.0 | −0.3% [−0.7,+0.1] | −0.005 [−0.012,+0.002] | 未过线（趋势负） |
| R2 resp β=0.5 | −0.2% [−0.7,+0.3] | −0.004 [−0.011,+0.003] | 未过线（趋势负） |

pkl：`duplicate_bootstrap_{respb10,respb05}_vs_best_1000.pkl`。

**每动作优势诊断**（v2 数据，n=141k，critic 基线）：
hu meanA=+0.249（显然）、**peng meanA=−0.058（n=109k，显著为负——
响应头确实过碰）**、gang −0.014、pass ≈0。但 AWR 下调碰后实战略负，
候选解释（结构性）：**碰 → 副露 → `len(cur)!=14` → 本局搜索层永久关闭**；
纯 NN 自对弈数据里没有搜索层，AWR 估计的碰代价不含「失去搜索」这一项，
部署后方向可能反转。这是自对弈分布 ↔ 部署结构的本质失配，普通 AWR 修不了。

### 2.8 附带探针：tenpai_threshold 扫描（5 档 × 1000 pairs）

t12 −0.4% [−2.0,+1.2]；t16 +0.2%；t20 +0.3%；t24 +0.2%；t32 +0.2%——
全部无显著差异、无单调趋势。**t28 维持**，边界位置不是免费收益。
pkl：`duplicate_hybridt{12,16,20,24,32}_vs_best_1000.pkl`。
（本次新增 `hybridt:LABEL:PATH:THRESHOLD` token 到 benchmark_pool。）

### 2.9 F5：成对 rollout 碰效应（结果）

12,000 个状态 × M=8 × 2 分支（18.5 min on 96 workers）：

- **mean Δ = +0.1173 ± 0.009**（SE of mean 0.0046）——**碰在平均意义上因果地为正**，
  响应头 82-93% 的高碰 take 率是**正确的**，BC 教师被平反；
  v2 AWR 的「peng meanA=−0.058」确系混杂（选择偏差），不是因果。
- 头部交叉表：head-peng 状态（93.4%）mean Δ=+0.121；head-pass 状态 mean Δ=+0.061
  ——头部的 pass 并非集中在「坏碰」上，其区分度弱（sign(Δ) 一致率 54%）。
- 表面可收割量：head-peng 且 Δ<−0.5 有 875 态（mean −0.85）。

训练与筛查（本批登记 4 个：strict/loose/pfa/pfa2）：

| 候选 | 构造 | 1000-pair paired diff | 结论 |
|---|---|---|---|
| strict（τ=2SE，双向 1338 标签） | 全显著态重标定 | −0.6% [−1.4,+0.2] | 负 |
| loose（τ=1SE，双向 4236 标签） | 同上放宽 | **−1.8% [−3.5,−0.1]** | 显著负 |
| pfa（pass 修错+peng 锚，lr 5e-4） | 单边修复 | **−2.0% [−3.8,−0.2]** | 显著负（take rate 崩到 0.56） |
| pfa2（同上 lr 1e-4 ×2ep） | 手术式修复（take 0.917） | −0.2% [−0.7,+0.3] | 零 |

pkl：`duplicate_bootstrap_{pengstrict,pengloose,pfa,pfa2}_vs_best_1000.pkl`。

机制结论：
1. 「少碰」（loose/pfa）**剂量-响应式地显著变负**——反向验证了 mean Δ>0 的因果结论；
2. 手术式修复（pfa2）为零：875 个「坏碰」状态在**当前 175 维特征上不可分**
   （label-acc 上限 ~0.7，修复梯度必然误伤好碰）——响应头的可压榨空间
   在特征层面不存在，要修必须先扩特征（如 belief 信号入特征，消融建议 #2）。

### 2.10 附带：tenpai 触发死代码修复探针

`HybridNNBeliefAgent._is_critical` 误用 `getattr(ctx,'tenpai',set())`
（ContextV3 实为 `tenpai_players`）→「对手报听触发搜索」从未生效。
修复变体（`hybridfix` token）1000 pairs：**+0.1% [−0.1,+0.3]，仅 1/1000 pairs 有差**——
BeliefExp 报听几乎都发生在总弃牌 ≥28 之后（计数阈值已触发），死代码几乎无害。
**不改原类**，记录备查。pkl：`duplicate_hybridfix_vs_best_1000.pkl`。

---

## 3. 总结论（2026-07-17 02:15）

**本批共筛查 15 个候选/变体（v1×3、v2×2、阈值×5、F5×4、bugfix×1），
无一达到 +1.0% 预登记晋级线；2 个显著为负（均为「少碰」方向）。零晋升。**

对「RL / NN / 自对弈 bootstrap」的证据强度，今晚之后可以定级为**判死**：

1. **outcome 级 RL 第 4-5 次独立失败**（v1 AWBC、v2 AWR）。根因不是算法：
   弃牌信用分配 SNR 先天不足（70 步牌运稀释），温度探索提供的有效对比样本
   仅 4.8%；响应决策的 AWR 信号被选择偏差混杂（peng meanA<0 是假象）。
2. **配对因果标签（当前技术下 SNR 上限）也无效**（F5×4）。配对 rollout 证明
   响应头平均策略已近最优（mean Δ=+0.117 方向与之一致），仅存的误差状态
   在现有特征空间不可分。
3. **运行点/死代码层面无免费收益**（阈值×5、tenpai fix）。

剩余未证伪但有代价的方向（优先级排序，均非「再试一次 RL」）：
- **特征扩容**：把 BeliefExp 的 danger/belief 信号作为额外输入特征重训全模型
  （消融建议 #2；F5 的 12k 配对 Δ 数据集可直接当高质量标签源复用，
  尤其用于检验「新特征能否分开坏碰」这一可证伪命题）；
- 引擎接入真实计分（报听/自摸加成）后重估一批与报听相关的结论；
- 修复 legacy `test_select` 断言（集合比较，勿动 eval 语义）。

**资产**（可复用）：
- `scripts/rl/selfplay_bootstrap.py`（collect/train_value/finetune/finetune_resp）；
- `scripts/rl/peng_paired_eval.py`（god-state 快照 + 成对 rollout + 三种训练模式）；
- `output/bootstrap_v1_merged.npz`（1.65M 弃牌步 + 结局）；
- `output/bootstrap_v2_merged.npz`（+141k 响应决策）；
- `output/peng_states_v1_merged.pkl`（14,210 god-state 快照）；
- `output/peng_eval_v1.npz`（12k 配对 Δ，含 state_idx）；
- benchmark_pool 新 token：`hybridt:`（阈值）、`hybridfix:`（tenpai 修复变体）；
- critic `output/nn_bootstrap_v1_critic.pt`（corr 0.231，校准 1:1）。




