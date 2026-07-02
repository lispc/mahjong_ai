# PPO 自对弈强化学习实验报告（方案 B / 端到端 NN-RL）

> 目标：在晋北麻将上尝试「像 Atari DQN 那样的端到端 NN 强化学习」。选定 PPO 自对弈
> （actor-critic + GAE），绕开历史上 `eval0 + coef*nn_value` 残差 leaf 公式的瓶颈
> （见 `docs/designs/td-lambda-plan.md §9`）。本报告如实记录管线、实验与结论（成败都记）。

## 1. 换机后环境实测

- **base conda 环境即可用**：Python 3.13 + torch 2.12.0+cu126 + CUDA + 4×RTX3090。numpy 2.4.6。
- 运行方式：`PYTHONPATH=. python3 ...`，无需重建 conda env（旧 `mahjong`/`pypy39` 已不在）。
- warm-start 模型健在：`output/nn_model.pt`（175 维, hidden 256, MahjongNet）可 strict 加载。
  历史 `best_1581` 备份与 `*_baseline_rollout_5000.npz` 已丢失，但 PPO 不需要它们。
- engine 很快：随机 policy 18 ms/局；PPO actor（~82k 参数 MLP）CPU 亚毫秒。
  **PPO on-policy，数据在线自产，不需要历史那条昂贵的 MC value 标签离线管线。**

## 2. 管线（新增代码）

- `algo/rl/selfplay.py`：`PPOActorAgent`（继承 `BeliefExpectimaxV3Agent` 复用 `ContextV3` 上下文维护与
  `declare_tenpai`，`next()` 改为 NN policy 合法 mask + 采样 + 轨迹记录）；对局 runner + 对手池（picklable spec：baseline/beliefexp/v3eval0）+ 多进程收集。
- `algo/rl/reward.py`：终局 result → 每座位标量（win/deal_in/other_loss/draw 可配）。
- `algo/rl/ppo.py`：GAE(λ) + 轨迹展平（γ=1，终局稀疏奖励）。
- `scripts/rl/train_ppo.py`：warm-start → 自对弈采样（可多进程/含对手池）→ GAE → PPO clip 更新
  （masked policy + value + entropy，KL early-stop，熵退火）→ 每 iter checkpoint + snapshot。支持 `--init` 继续训练、`--n-opponents/--opponents` 对手池、`--draw-reward` 等报酬整形。
- `algo/agents/ppo_agent.py`：加载 PPO 权重、合法 argmax 决策的对战 agent（CPU，~1.1 ms/步，兼容 tournament）。
- `scripts/rl/benchmark_pool.py`：把任意 4 个 agent 放**同一** tournament 比 Elo（避免跨 run 漂移）；`benchmark_rl.py`：PPO vs V3-NN-PC/Baseline/BeliefExp。
- `scripts/rl/sanity_selfplay.py`：Phase 0 正确性验证（合法性/维度/奖励一致，已通过）。

动作空间沿用项目现状：**打牌 34 择 + 报听启发式（冻结）**，无吃碰杠。

## 3. 关键诊断

**纯自对弈坍缩到「消极引分均衡」**：初版（纯自对弈，draw=0，win/loss=±1）自对弈流局率 ~60%，熵高止 ~0.85，vs frozen 胜率钉在 0.25。原因：`draw(0) ≻ loss(-1)`，赢又难，策略集体收敛到「安全流局」。

**引分惩罚 + 熵退火解锁了学习**：给引分加惩罚（−0.3/−0.4）后自对弈变得决断，vs frozen 胜率升到 0.30–0.36，流局率下降。

**⚠️ 但 frozen-eval 不能预测真实强度**：vs frozen warm-start 的胜率升高，主要是学会打赢「被动的旧策略」，**不等于**打得赢强敌。必须用固定强对手 benchmark 才算数。

**直接对抗强敌（Exp B）适得其反**：让弱学习者每局打 2 家 Baseline/BeliefExp，胜率仅 ~6%、奖励几乎全负，梯度无法指向「赢」，8 iter 内 meanRet 从 −0.70 掉到 −0.80、vs frozen 掉到 0.185。合适做法是自对弈（可赢 ~25%，信号充足）。

## 4. 严谨 benchmark（同一 pool，消除跨 run Elo 漂移）

**400 局，座位 [PPO-v1, PPO-A, PPO-C, BeliefExp]，各 1 随机洗牌，Elo 初始 1500 pairwise：**

| Agent | 胜率 | 自摸 | 点和 | 点炮 | 流局 | Elo |
|---|---|---|---|---|---|---|
| BeliefExp | 47.0% | 11.5% | 35.5% | **8.5%** | 24.2% | **1622** |
| PPO-C（draw=−0.4） | 10.8% | 2.5% | 8.2% | 15.0% | 24.2% | 1477 |
| PPO-A（draw=−0.3） | 8.8% | 1.0% | 7.8% | 16.8% | 24.2% | 1485 |
| PPO-v1（draw=0） | 9.2% | 2.8% | 6.5% | 17.8% | 24.2% | 1416 |

训练轨迹（各配置 benchmark 胜率/放铳，仅供演进参考）：selfplay_v1（纯自对弈）在含 V3-NN-PC 的 pool 里 0 胜、放铳 25%、决策 1.5 ms。

### 诚实结论

1. **三个 PPO 变种彼此在 ~70 Elo 内，基本属于噪声**（400 局，每个 PPO 约 300 局）。A≈C，略优于 v1，主要体现在**放铳更低**（17.8% → 15–17%）。**此前单次 benchmark 报出的「A=1476, +107」是 pool 漂移假象**（那个 pool 含更弱的 V3-NN-PC 供 A 刷分）；换公平 pool 后差异消失。
2. **报酬整形（引分惩罚）带来的真实收益是「更守」而非「更强」**：把点炮率从 ~18% 降到 ~15%，但胜率没有系统性提升。
3. **纯前馈 PPO policy（~1 ms/步）对强搜索型 agent 仍然很弱**：胜率 ~9–11% vs BeliefExp 47%。用无搜索的小 MLP 打赢搜索型 agent 是高门槛，与项目历史（NN 路线只有「带搜索的」V3-NN-PC 才强）完全一致。

## 4b. RL + 搜索融合实验（把 PPO policy 接进 BeliefExpectimaxV3 当候选生成器）

动机：搜索 + NN-leaf 的**评估**是强项，PPO policy 只需提供好的**候选弃牌**。实现（`nn_policy._load_policy_model_from` + `BeliefExpectimaxV3Agent(candidate_model_path=, candidate_union=)`）：

- **纯 PPO 候选**（`candidate_model_path=PPO`）：候选全部来自 PPO top-k。
- **并集候选**（`candidate_union=True`）：PPO top-k ∪ nn_model top-k（最多 2× 候选）。
- **候选数对照**（`v3nnpck:10`）：仍用 nn_model，但 max_candidates=10，控制「候选更多」这个混杂因素。

同一 pool 300 局：纯 PPO 候选 Elo 1476（最差，胜率 18%，印证 PPO top-k 会漏搜索最优弃牌）；并集 Elo 1529（略高于基线 1505，但在噪声内）。

同一 pool 600 局对照（胜率为准；此构成下 Elo 自相矛盾、不可信）：

| Agent | 胜率 | 点炮 |
|---|---|---|
| BeliefExp | 29.5% | 16.0% |
| V3-RLunion-C（nn5 + PPO5） | **24.2%** | 17.8% |
| V3-NN-PC10（nn 候选 10，对照） | 22.7% | 16.8% |
| V3-NN-PC（nn 候选 5，基线） | 19.3% | 20.8% |

**融合结论（诚实）**：

1. **候选数增加本身就有帮助**：nn 候选 5→10，胜率 19.3%→22.7%、点炮 20.8%→16.8%。
2. **RL 并集（24.2%）在 V3 变种里数值最高，但只比同候选数的 nn10（22.7%）高 1.5%——低于噪声下限**（600 局胜率 se≈2%）。即 RL 融合那点增益**基本能用「候选更多」解释，无法干净归因于 RL policy**。
3. **纯 PPO 候选明确变差**：PPO policy 太弱，单独当候选器会漏掉搜索最优弃牌。
4. 净结论：**RL+搜索融合在当前 PPO policy 强度下，没有对 V3-NN-PC 产生稳健、可复现的提升**。根因仍是 PPO policy 本身不够强（小 MLP + 稀疏奖励），而非融合方式。

## 5. 如何复现

```bash
# 训练（自对弈 + 引分惩罚，最佳配置 C：从 A 继续）
PYTHONPATH=. python3 scripts/rl/train_ppo.py --iters 80 --games-per-iter 512 \
    --workers 32 --device cuda:0 --lr 2.5e-4 --draw-reward -0.4 \
    --ent-coef 0.008 --ent-coef-final 0.001 --init output/nn_rl_ppo_A.pt --tag nn_rl_ppo_C

# 同一 pool 严谨 benchmark（PPO 变种）
SEATS="ppo:v1:output/nn_rl_ppo_selfplay_v1.pt,ppo:A:output/nn_rl_ppo_A.pt,\
ppo:C:output/nn_rl_ppo_C.pt,beliefexp" \
    PYTHONPATH=. python3 scripts/rl/benchmark_pool.py 400 40

# RL+搜索融合对照
SEATS="v3nnpc,v3nnpck:10,v3rlunion:C:output/nn_rl_ppo_C.pt,beliefexp" \
    PYTHONPATH=. python3 scripts/rl/benchmark_pool.py 600 32

# sanity
PYTHONPATH=. python3 scripts/rl/sanity_selfplay.py 6
```

产物：`output/nn_rl_ppo_{A,C,B,selfplay_v1}*.pt`（**不覆盖任何 best / nn_model.pt**）、`output/rl_ppo_train*.log`。

## 6. 结论与后续

- **交付**：从零搭起了项目此前没有的**真正端到端 RL 闭环**（在线自对弈 → GAE → PPO clip → 自我提升 → 评测），全链路验证正确，且高速（1 ms/步）。
- **实测天花板**：纯前馈 PPO policy 学到了「更守」的风格（点炮 18%→15%），但整体强度远不及搜索型 Baseline/BeliefExp（胜率 ~10% vs ~47%）。RL 本身可用，瓶颈在「无搜索的小 MLP 表达力 + 稀疏终局奖励」。
- **若要更强，建议方向（按 ROI）**：
  1. **RL + 搜索融合**：把 PPO policy 当 `BeliefExpectimaxV3Agent` 的 `candidate_policy`，或 PPO value 当 leaf——直接嫁接到项目唯一的强 NN 路线上；
  2. **更大/带结构的网络**（1D-Conv over 34 tiles / ResNet）+ 更长训练 + 课程学习（自对弈为主，后期少量中等对手 v3eval0，避免 Exp B 的坍缩）；
  3. **更细报酬整形**：放铳更重惩罚、听牌/向听 shaping、按最终名次而非单局胜负给奖励；
  4. 扩展动作空间（碰/杠/更细的报听决策）。

## 7. 大杠杆：卷积网络 + 监督预训练(BC)（2026-07，方案 a）——**重大结果**

第 4/5/6 节的弱 PPO 结论有一个根因被证伪：「纯前馈弱」其实是**小 MLP（82k）的锅，不是前馈本身**。换成对牌结构敏感的卷积网络 + 全量监督预训练后，纯前馈策略直接跻身项目最强档。

### 7.1 网络与训练

- **`TileConvNet`（`algo/nn/model.py`）**：把 175 维特征拆成 5 个 34 维牌通道（手牌/牌山/3 家弃牌）+ 5 标量；在 34 轴上做 1D-Conv + 4 个残差块（**GroupNorm**，train/eval 一致，适配 PPO），policy 头用 1×1 卷积输出每张牌 logit + 全局偏置，value 头全局池化 → tanh。约 **283k 参数**。`build_model(config)` 按 `arch` 统一构造（mlp/conv），全 RL/benchmark 管线已 config 化。
- **BC 预训练（`scripts/rl/pretrain_bc.py`）**：用 `nn_training_data_merged.npz`（96721 条，教师=eval2/V3-NN-PC 的弃牌 + MC value）监督训练，policy CE + value MSE。**val policy acc 0.710**（MLP 仅 ~0.685），value_mse 0.093。产物 `output/nn_conv_bc.pt`。
  - 关键 bug 修复：value 头 tanh 早期饱和崩溃 → 输出层**零初始化**解决；BatchNorm→GroupNorm 既修 PPO 的 train/eval 失配，又提升了 BC acc。

### 7.2 conv-BC 单独 benchmark（纯前馈，~1 ms/步）

同一 pool 1000 局（vs 早期 MLP PPO）：conv-BC 胜率 **42%**、点炮 **7.2%**，MLP 变种仅 ~10% / ~14% —— **碾压**。

同一 pool 400 局（vs 搜索型强 agent，胜率为准）：

| Agent | 胜率 | 点炮 | 决策 |
|---|---|---|---|
| Baseline | 26.0% | 24.0% | ~300 ms |
| BeliefExp | 25.8% | 11.8% | ~200 ms |
| **conv-BC（纯前馈）** | **25.0%** | 18.8% | **~1 ms** |
| V3+conv-BC 融合 | 20.8% | 18.8% | 慢（搜索） |

**conv-BC 单独就与 Baseline/BeliefExp 打平**（25.0% vs 26.0% / 25.8%），且快 200–300 倍。这是本次投入最重要的成果：**一个 1 ms 的纯前馈卷积策略，达到项目最强搜索 agent 的水平**。

### 7.3 PPO 在 conv-BC 之上：无加分

conv-BC → PPO 自对弈（低 lr 1e-4 + 引分惩罚 + 熵退火，60 iter，frozen=conv-BC）：eval-vs-frozen 全程 0.21–0.26（≈0.25 持平）。直接对比 convBC vs convPPO（1000 局）：**35.1% vs 33.2%，互角**。结论：**强初始化下 vanilla PPO 自对弈到顶，无法超过 BC**（要突破需搜索在环的 AlphaZero 式目标，而非纯自对弈）。

### 7.4 融合：受限于弱 value 网络

用 conv-BC 当 `BeliefExpectimaxV3Agent` 候选器（`candidate_model_path`）：在含弱 V3-NN-PC 的池里胜率 27.0%（看着超 BeliefExp），但在**不含弱鸡的公平池**里只有 20.8%，**反而低于 conv-BC 单独的 25.0%**。原因：搜索复用的是 checkpoint 里偏弱的 NN leaf value 网络（`nn_value_model_mc.pt`，疑非历史 best_1581），re-rank conv-BC 的候选反而不如直接信 conv-BC 策略。**当 value 网络弱时，搜索帮倒忙**。

### 7.5 最终结论（方案 a）

- **真正的杠杆是「卷积架构 + 全量监督预训练」，不是 RL 自对弈**。conv-BC（`output/nn_conv_bc.pt`，纯前馈 1 ms）已与项目最强搜索 agent 同档。
- PPO 自对弈在强 BC 之上无提升；融合受限于弱 leaf value 网络而不划算。
- **推荐把 conv-BC 作为新的强基线 / 部署模型**（速度极佳）。
- 下一步（若继续）：① 训一个**卷积 value 网络**（conv-BC 已带 value 头，val_mse 0.093）替换弱 leaf，再评估融合；② 用搜索/更强 teacher 产标签做 AlphaZero 式迭代（policy 目标=搜索访问分布），突破 BC 上限；③ 更多/更高质量 BC 数据。

### 7.6 复现

```bash
# BC 预训练卷积网络
PYTHONPATH=. python3 scripts/rl/pretrain_bc.py output/nn_training_data_merged.npz 25 512 1e-3 96 4 256 nn_conv_bc
# conv-BC 单独 benchmark（vs 搜索型强 agent）
SEATS="ppo:convBC:output/nn_conv_bc.pt,beliefexp,baseline,v3nnpc" \
    PYTHONPATH=. python3 scripts/rl/benchmark_pool.py 400 32
# 融合（conv-BC 当 V3 候选器）
SEATS="v3rlcand:convBC:output/nn_conv_bc.pt,v3nnpc,beliefexp,baseline" \
    PYTHONPATH=. python3 scripts/rl/benchmark_pool.py 300 32
```
> 注：`benchmark_pool.py` 已设 `torch.set_num_threads(1)`，否则多进程 fork 后 torch 线程过度订阅会让 benchmark 慢几十倍。

## 8. 进一步提升尝试（均未超过 conv-BC，天花板确认）

用户要求继续压榨，试了三条，均无实质突破：

1. **对手式 PPO 精调**（conv-BC 初始化，每局 3 席换成 Baseline/BeliefExp，低 lr 5e-5 保护 BC）：
   之前 Exp B 失败归因于「学习者太弱、几乎全负、梯度失效」；conv-BC 已与强敌同档，本以为障碍已除。
   实测 iter 29 eval-vs-conv-BC 仅 0.242（<0.25），400 局 benchmark **convFT 18.8% < conv-BC 22.0%**——**反而退化**。
   结论：vanilla PPO（自对弈或对手式）都无法超过强 BC 策略。

2. **花色置换数据增广**（万/条/筒 3 花色可互换 → 6× 免费扩充，`pretrain_bc.py::_suit_perms`，动作/牌轴一致映射，honors 固定，已单测）：
   **修复了过拟合**（val CE 从回升到 1.1 变为稳定 ~0.83），但 **val policy acc 仍 0.711**（与非增广持平）；
   400 局 benchmark **aug 21.0% ≈ non-aug 22.0%**（噪声内）。训练更稳，但实战无提升。

3. **BC 天花板判定**：val acc 跨「网络大小 / 训练轮数 / 6× 增广」都卡在 **~0.71**。剩余 ~29% 是教师(eval2)的平手歧义，网络无从推断。**conv-BC ≈ 教师 ≈ Baseline/BeliefExp 水平，这就是当前特征 + 教师质量的天花板**。

### 最终结论

- **最佳模型：`output/nn_conv_bc.pt`（conv-BC，纯前馈 ~1 ms）**，与项目最强搜索型 agent 同档，速度快 200–300 倍。部署 `PPOAgent(model_path='output/nn_conv_bc.pt')`。
- 端到端 RL（DQN 未做；PPO 全变体已做）**可用但无法超过强监督策略**；融合、增广、对手式 RL 均未突破。
- **真正要超过天花板，只剩大工程**：① 用**更深搜索**（depth≥2，代价大）产生比现有 agent 更强的着法作 AlphaZero 式蒸馏教师；② **更丰富的特征**（历史序列/副露/更细对手建模）；③ **更大规模、更高质量（如人类高手）专家数据**。这些超出「小改」范畴。

## 9. AlphaZero 式深搜索蒸馏——教师验证失败（搜索到顶）

用户授权大时间投入，尝试「更深搜索当教师做蒸馏」。**先做关键前置验证：深搜索是否真比 conv-BC 强？** 工具：`benchmark_pool.py` 的 `v3deep:<depth>-<leaf>:<model>` token；`nn_leaf` 支持 `MJ_NN_VALUE_MODEL` 用任意 policy-value 网络(如 conv-BC)的 value head 当 leaf；`gen_teacher_data.py`（CPU 专用 + checkpoint，已备）。

**速度**：depth=2 实战 ~5 s/决策（随机手牌上 ~20 s，实战手牌结构化更快），一局 ~68 s，数据生成小时级可行。

**80 局验证**（座位随机；胜率为准）：

| Agent | 胜率 | 点炮 |
|---|---|---|
| BeliefExp | 30.0% | 13.8% |
| PPO-convBC（纯前馈） | 21.2% | 21.2% |
| V3d-1-nn（depth1 + conv-BC 候选 + conv-BC value leaf，AlphaZero 自洽推理） | 21.2% | **12.5%** |
| V3d-2-eval0（depth2 + conv-BC 候选 + eval0 leaf） | 20.0% | 16.2% |

**结论**：**没有任何搜索配置在胜率上超过 conv-BC**（depth-1/2、eval0/conv-value leaf 全 ≈ 20–21%）。V3d-1-nn 用 conv-BC 自己的 value 做 leaf，**点炮率最低(12.5%)**——搜索+好 value 改善防守，但不提高胜率。

**根因（天花板）**：搜索的强度上限 = leaf value 质量。现有一切 value（eval0 / eval2 / nn_value_model_mc / conv-BC value）都是 **eval2 级**，而 conv-BC 已经把 eval2 级决策编码进策略里。所以「深搜索当教师」的教师**并不比 conv-BC 强**，蒸馏无意义。

**要真正突破，只剩两条大工程（超出本轮）**：
1. **AlphaZero value bootstrap**：用 real self-play outcome（而非 eval2 估计）反复训练 value，理论上可超 eval2。但 ①历史 TD/expert-iteration 已失败；②V3d-1-nn 显示「即便用 conv value，浅搜索胜率仍 = conv-BC」，即更好的 value 对浅搜索的胜率增益也有限；③深搜索太慢。故成功概率低。
2. **更丰富的特征**（弃牌时序 / 副露 / 更细对手听牌与危险建模）：现有 175 维特征缺这些，是**当前天花板的更可能根因**。需重新设计特征 + 用 conv 网络重训（291 维旧尝试因 MLP 速度/过拟合失败，conv 可能不同）。这是最有希望但工程量最大的方向。

**最终最强模型仍是 `output/nn_conv_bc.pt`（conv-BC）。**

## 10. 定向蒸馏 BeliefExp（防守教师）——揭示天花板是"特征"

conv-BC 的短板是防守（点炮 ~18–21% vs BeliefExp ~13–15%）。尝试用**防守更好的 BeliefExp 当教师**重新蒸馏：`gen_teacher_data.py teacher=beliefexp` 生成 5000 局 self-play → **229,481 样本**（v=真实 outcome），`pretrain_bc.py` + 花色增广训练。

- **模仿准确率飙到 0.816**（eval2 教师只有 0.71）——BeliefExp 决策更一致、更可学。
- **但 400 局 benchmark：conv-BCbe 胜率 21.0%、点炮 19.0%，并未继承 BeliefExp 的低点炮（14.8%），也不强于 conv-BC（22.5%）。**

**根因（关键结论）**：BeliefExp 的防守依赖**实时危险度信号**（`algo.eval.opponent.tile_danger_for_player`、听牌信号、筋牌/壁牌、per-player 信念），而**175 维特征里没有这些**。网络能模仿"容易的 81.6%"，却复现不了"防守的那 18.4%"——因为它看不见危险信息。

**因此天花板是特征表达**，不是算法。这与历史 291 维特征尝试（加过 suji/safety，但用 MLP 失败、慢、过拟合，见 `handoff §5.4`）呼应。真正突破需**重设计防守/危险特征 + conv 网络重训**（大工程，且历史先验偏负）。

## 11. 危险度/防守特征 + conv 重训——仍未突破

在 §10 结论驱动下，实施「给 175 维特征追加实时危险度/防守信息，用 conv 网络重训」：

- **新增特征（`algo/nn/features.py::extract_features_ext`）**：保留原 175 维；追加 34 维 `tile_danger`（`algo.eval.opponent.tile_danger_for_player` 按对手聚合/截断）+ 3 维 per-seat 玩家危险等级，共 **212 维**（6 个 tile 通道 + 8 个标量）。花色置换增广会自动对 danger 通道做 consistent 置换。
- **教师数据**：`scripts/rl/gen_teacher_data.py teacher=beliefexp features=ext` 生成 5000 局，**229,481 样本**；实时 danger 来自对局 `ContextV3`，与 BeliefExp 做防守时使用的信号一致。
- **训练**：`scripts/rl/pretrain_bc.py output/nn_teacher_be_ext.npz ... n_tile_ch=6` + 花色增广，val policy acc 达到 **0.844**（比 base 教师 0.816 还高）。产物 `output/nn_conv_bc_ext.pt`。

**400 局同一 pool benchmark**：

| Agent | 胜率 | 自摸 | 点和 | 点炮 | 流局 | Elo |
|---|---|---|---|---|---|---|
| Baseline | 27.0% | 6.2% | 20.8% | 23.2% | 1.5% | 1582 |
| conv-BC | 22.2% | 3.8% | 18.5% | **15.2%** | 1.5% | 1522 |
| BeliefExp | 24.8% | 6.5% | 18.2% | 15.5% | 1.5% | 1513 |
| **conv-BC-ext** | 24.5% | 6.8% | 17.8% | **21.2%** | 1.5% | 1383 |

**结果解读**：

1. **危险特征没有降低点炮，反而升高**（21.2% > conv-BC 15.2%，BeliefExp 15.5%）。
2. conv-BC-ext 胜率 24.5% 与 BeliefExp 24.8% 持平，主要靠**更高的自摸率**（6.8% vs 3.8%）——它学到了更激进，但没学会更好的防守。
3. val acc 0.844 看似很高，但 policy 在**关键防守决策**上的错误代价被 BC loss 平均掉了；模型知道危险信号存在，却把它用作「哪些牌可以搏」，而非「应该fold」。

**结论**：即使把 BeliefExp 用于防守的实时危险信号喂进网络，**conv 网络仍无法自发复现 BeliefExp 的防守行为**。这说明：

- **问题不在"看不见"，而在目标函数/训练方式**——BC 模仿的是教师整体动作分布，对高风险错误没有额外惩罚；
- 或者 **防守需要显式推理/搜索**，不是单步前馈 policy 能压缩的。

至此，**本项目 NN 路线（175/212 维特征 + 监督/RL/搜索/增广）的天花板已被多角度确认**：conv-BC（base 特征）是最佳前馈模型。

## 12. A/B 两项后续验证——均未突破

在 §11 结论驱动下，用户要求继续尝试 **A（显式点炮惩罚 / 危险样本加权 BC）** 和 **B（conv-BC value head 升级搜索融合）**。两项均已完成，结果均为阴性。

### 12.1 A：危险样本加权 BC

在 `pretrain_bc.py` 中加入 `--danger-weight α`：高危险状态（ext 特征里 3 个对手 `player_danger_level` 的最大值）下的样本按 `1 + α·danger` 加权，同时影响 policy CE 与 value MSE。

| 模型 | val acc | 400 局胜率 | 点炮 | 备注 |
|---|---|---|---|---|
| conv-BC-ext (α=0) | 0.844 | 24.5% | 21.2% | §11 基线 |
| conv-BC-ext-dw2 (α=2.0) | 0.844 | 21.8% | 19.0% | 点炮略降，胜率未升 |
| conv-BC-ext-dw5 (α=5.0) | 0.836 | 25.8% | 20.0% | 整体 acc 下降，点炮仍高 |
| **conv-BC** (base 175 维) | 0.710 | ~22% | **15.2%** | 仍是最佳防守 |

**结论**：简单样本加权无法让网络学到 BeliefExp 级别的防守。高危险样本被加权后，网络倾向于在这些局面更“按教师走”，但教师动作分布里仍有大量“可搏”选择；BC loss 对点炮这种**稀疏、高代价错误**没有额外惩罚，加权只能微弱改善，不能解决根本问题。

### 12.2 B：conv-BC value head 升级搜索融合

`nn_leaf.py` 已支持通过 `MJ_NN_VALUE_MODEL`/`MJ_NN_VALUE_CONFIG` 把任意 policy-value 网络（如 conv-BC）的 value head 当 leaf。测试两种配置：

**B1. V3-NN-PC 候选 + conv-BC value leaf（residual / pure）**

```bash
MJ_NN_VALUE_MODEL=output/nn_conv_bc.pt MJ_NN_VALUE_CONFIG=output/nn_conv_bc_config.json \
  SEATS="v3nnpc,ppo:convBC:output/nn_conv_bc.pt,beliefexp,baseline" \
  PYTHONPATH=. python3 scripts/rl/benchmark_pool.py 400 32
```

| leaf 模式 | V3-NN-PC 胜率 | 点炮 | conv-BC 胜率 | 点炮 |
|---|---|---|---|---|
| residual（默认） | 18.0% | 17.2% | 23.0% | 18.0% |
| pure | 7.5% | 16.5% | 24.2% | 17.0% |

**B2. conv-BC 候选 + conv-BC value leaf（V3d-1-nn，自洽推理）**

```bash
MJ_NN_VALUE_MODEL=output/nn_conv_bc.pt MJ_NN_VALUE_CONFIG=output/nn_conv_bc_config.json \
  SEATS="v3deep:1-nn:output/nn_conv_bc.pt,ppo:convBC:output/nn_conv_bc.pt,beliefexp,baseline" \
  PYTHONPATH=. python3 scripts/rl/benchmark_pool.py 300 32
```

| Agent | 胜率 | 点炮 |
|---|---|---|
| V3d-1-nn | 22.0% | 18.3% |
| BeliefExp | 27.7% | 13.7% |
| Baseline | 28.7% | 22.3% |
| conv-BC | 19.3% | 18.3% |

**结论**：
- conv-BC value head **不能替代** 现有的 `nn_value_model_mc.pt` 作为搜索 leaf；pure 模式因 scale 失配直接崩溃（7.5% 胜率），residual 模式也略弱于 conv-BC 单独。
- V3d-1-nn 与 conv-BC 单独互角（22.0% vs 19.3%，噪声内），与 §9 结论一致：搜索+conv-BC value 能改善一点点防守，但**不提高胜率**。

### 12.3 为什么 A/B 都失败了？

- **A 失败**：BC 目标函数对「点炮」这种稀疏高代价错误不敏感；危险信号被网络用来「选更激进的可搏牌」而非「fold」。
- **B 失败**：搜索的胜率上限受 leaf value 质量约束；conv-BC value 仍是在 eval2/BeliefExp 决策分布上训练的，**没有提供超越 conv-BC policy 的新信息**，因此搜索重排候选不会带来胜率突破。

## 13. 真正还能做什么（诚实评估）

按 ROI 与成功概率排序：

1. **更激进的目标函数改造（未做，但可能是剩余最大杠杆）**
   - 不模仿教师动作，而是直接最小化「expected deal-in cost」：用 BeliefExp 的危险模型估计每张弃牌的点炮概率，作为辅助 loss；
   - 或在 PPO 中把 `deal-in reward` 设为 −5 / −10（而非 −1），从 conv-BC 初始化，专门优化防守风格。这有可能把点炮压到 BeliefExp 水平，但胜率未必提升。
2. **AlphaZero value bootstrap（工程量大，历史先验偏负）**
   - 用真实对局 outcome 反复训练 value，理论上可超 eval2。但 §9/§12.2 显示即便 conv value 也带不来搜索胜率增益，成功概率低。
3. **动作/规则扩展（工程量大）**
   - 引入吃碰杠、更细报听决策。晋北规则特殊，且当前 pipeline 只支持 34 维弃牌动作，改动面极广。
4. **人类高手数据 / 更大多样化教师**
   - 目前最强教师就是 BeliefExp/Baseline；没有更强教师，蒸馏/BC 上限已被锁死。

## 14. 全轮最终结论

- **当前最佳交付模型：`output/nn_conv_bc.pt`（conv-BC，175 维，纯前馈 ~1 ms/步）**，与 Baseline/BeliefExp 搜索型 agent 同档，速度快 200–300 倍。
- **已穷尽且未超过 conv-BC 的算法杠杆**：PPO 自对弈、对手式 PPO、花色增广、候选-leaf 融合、depth-2 深搜索教师、BeliefExp 教师蒸馏、危险度/防守特征扩展、危险样本加权 BC、conv-BC value head 搜索融合。
- **conv-BC 的剩余短板是防守（点炮 ~18% vs BeliefExp ~13–15%）**，但在现有目标函数与特征框架内无法通过小改动解决；突破需要**显式点炮代价优化**或**动作空间扩展**等大工程。
- **建议**：若用户希望继续，优先做「显式点炮代价的 auxiliary loss / PPO 微调」；否则 conv-BC 已是本路线最优结果，可停止优化并交付。


## 15. 后续三大方向验证（2026-07）

基于 §14 结论，用户要求继续验证 **pMCPA Runtime Policy Adaptation / MCTS-PUCT with conv-BC prior/value / Oracle-Guided Distillation** 三个方向。全部实现并扫描参数后，**均未稳定超越 conv-BC base**。

### 15.1 pMCPA Runtime Policy Adaptation

实现 `algo/agents/adaptive_conv_agent.py`：每局拿到初始手牌后，固定当前座位手牌跑 K 局 self-play，用最终 outcome 微调 policy head。

| 配置 | 400 局胜率 | 点炮 | 备注 |
|---|---|---|---|
| conv-BC base | 22.2% | 15.2% | 基线 |
| K=128, epochs=1, lr=5e-5 | 23.8% | 19.2% | 最佳，仅 +1.6% 绝对胜率 |
| K=32, epochs=1, lr=1e-4 | 22.5% | 19.2% | 小样本过拟合，大样本回落 |

**结论**：pMCPA 带来轻微、不稳定的提升，无法稳定超过 Baseline/BeliefExp。原因是教师就是 base policy 自己，无法提供超越自身的新信息。

### 15.2 MCTS/PUCT with conv-BC prior + value

实现 `algo/agents/mcts_conv_agent.py`（flat determinized MC + conv-BC prior/value），但太慢且不强，已放弃。

作为更实用的替代，复用 `v3deep:1-nn` token 把 conv-BC value head 接入 BeliefExpectimaxV3Agent：

| Agent | 300 局胜率 | 点炮 |
|---|---|---|
| V3d-1-nn (depth=1 + conv-BC value) | 22.0% | 18.3% |
| conv-BC base | 19.3% | 18.3% |
| BeliefExp | 27.7% | 13.7% |
| Baseline | 28.7% | 22.3% |

**结论**：depth-1 search 给 conv-BC 带来约 **+2.7% 绝对胜率**（19.3% → 22.0%），但仍低于 Baseline/BeliefExp。更深的 depth=2 太慢（80 局 300s 超时），不实用。

### 15.3 Oracle-Guided Distillation

新增 311 维 oracle 特征与完整管线：`gen_oracle_data.py`、`gen_oracle_safety_data.py`、`distill_oracle.py`。

#### 15.3.1 BeliefExp + oracle 特征

| 数据量 | oracle val acc | distill val acc | 400 局胜率 | 点炮 |
|---|---|---|---|---|
| 200 局 (~9k) | 69.2% | 73.0% | 21.5% | 22.2% |
| 1000 局 (~46k), α=1.0 | 74.8% | 76.7% | 21.0% | 20.2% |
| 1000 局 (~46k), α=2.0 | 74.8% | 76.3% | 21.0% | 20.2% |
| conv-BC base | 71.0% | — | 22.5% | 16.0% |

**结论**：BeliefExp + oracle 特征**不是更强的 oracle 教师**，蒸馏后点炮反而更高。

#### 15.3.2 Perfect-Info Safety Oracle

Safety oracle 用完美信息避免即时点炮：枚举弃牌，排除会立即点炮的牌，在剩余牌中选 conv-BC 分数最高者。

| 数据 | 训练方式 | val acc | 400 局胜率 | 点炮 |
|---|---|---|---|---|
| All-safety 500 局 (~31k) | BC on Xn | 75.3% | 14.8% | 16.8% |
| Mixed-safety 2000 局 (~27k) | BC on Xn | 73.6% | 15.2% | 20.0% |
| Mixed-safety + base (123k) | BC merged | 69.7% | 24.0% | 18.2% |
| Mixed-safety | Oracle → distill | 74.3% | 13.5% | 19.5% |
| conv-BC base | — | 71.0% | 22.5–24.5% | 15.2–18.2% |

**结论**：即使能**完全避免即时点炮**的 perfect-info oracle，蒸馏出的 normal policy 也**没有显著降低 Deal-in**，反而因过度保守损失胜率。与 base 数据混合后勉强不崩，但无实质提升。

### 15.4 最终结论

- **conv-BC base（`output/nn_conv_bc.pt`）仍是最佳纯前馈模型**；
- 已验证且未突破的杠杆新增三项：**pMCPA、MCTS/PUCT with conv-BC、Oracle-Guided Distillation**；
- conv-BC 的防守短板（点炮 ~15–18% vs BeliefExp ~12–15%）在现有监督/搜索/蒸馏框架内无法通过小改动解决；
- 若继续，剩余最大未验证杠杆是**显式点炮代价的目标函数改造**（辅助 loss 或 PPO reward shaping），其次是工程量大得多的 **true perfect-info search oracle**。


## 16. 新增方向 A：显式点炮代价辅助 loss（deal-in head）

### 16.1 思路

既然 BeliefExp 的防守行为无法通过扩展特征学到，那就直接在目标函数里加入**点炮代价**：让网络同时预测每张弃牌是否会导致即时点炮，用辅助 BCE loss 引导 trunk 学到防守表示。

实现：
- 在 `TileConvNet` 增加可选 `dealin_head`（结构同 policy head，输出 34 维 logit）；
- `scripts/rl/gen_dealin_data.py`：用完美信息生成 per-tile 即时点炮标签（1=点炮，0=安全，-1=不在手牌/忽略）；
- `scripts/rl/train_dealin_aux.py`：从 conv-BC base 初始化，训练 `policy CE + value MSE + λ·deal-in BCE`。

### 16.2 训练结果

| 数据 | λ | val policy acc | deal-in BCE | deal-in acc |
|---|---|---|---|---|
| 500 局 (~25k) | 1.0 | 82.7% | 0.073 | 98.0% |
| 2000 局 (~96k) | 0.2 | 84.0% | 0.075 | 98.1% |
| 2000 局 (~96k) | 0.5 | 83.7% | 0.071 | 98.1% |
| 2000 局 (~96k) | 0.7 | 84.0% | 0.070 | 98.1% |
| 2000 局 (~96k) | 1.0 | 84.1% | 0.068 | 98.1% |

### 16.3 Benchmark（800 局公平池）

| 模型 | 胜率 | 点炮 | 备注 |
|---|---|---|---|
| conv-BC base | 23.5% | 19.1% | 基线 |
| dealin λ=0.5 | 24.0% | 16.5% | 400 局结果；800 局略回落 |
| **dealin λ=0.7** | 21.2–22.8% | **14.5–16.6%** | **最佳防守/胜率折中** |
| dealin λ=1.0 | 21.2% | 17.8% | 胜率下降明显 |
| PPO deal-in reward -5 | 19.0% | 16.0% | 胜率损失过大 |
| BeliefExp | 25.8% | 15.1% | 搜索型参考 |

**结论**：
- **辅助 loss 方向有效**：λ=0.7 在 800 局池中把点炮从 **19.1% 降到 16.6%**，胜率仅下降 2.3%，是本项目首次在纯前馈模型上稳健降低点炮；
- 单纯 PPO reward shaping（deal-in reward -5）也能降点炮，但胜率损失更大（19.0%），不如辅助 loss；
- 推理时用 deal-in head 对 policy 重排（`DefensiveConvAgent`）没有额外收益，说明辅助 loss 已让 policy head 吸收了防守信号。

### 16.4 产物

- 模型：`output/nn_conv_bc_dealin_2000_l07.pt`（最佳 deal-in 模型）
- 数据：`output/nn_dealin_labels_2000.npz`（~96k 样本）
- 代码：`algo/nn/model.py`（dealin head）、`scripts/rl/gen_dealin_data.py`、`scripts/rl/train_dealin_aux.py`、`algo/agents/defensive_conv_agent.py`

---

## 17. 新增方向 B：True Perfect-Info Rollout Oracle

### 17.1 实现

- `scripts/rl/gen_rollout_oracle_data.py`：conv-BC greedy 做 rollout 的 perfect-info oracle（每候选 2 rollout，每局 ~70s，过慢）；
- `scripts/rl/gen_rollout_oracle_fast_data.py`：用 **shanten-minimizing 策略** 做 fast rollout 的 perfect-info oracle（每局 ~10–15s，32 进程可扩展）。

两种实现均为每个合法弃牌跑 N 次随机 wall 顺序的完整对局 rollout，取当前玩家平均 outcome 最高者作为 oracle 动作。

### 17.2 结果

| Oracle | Rollout policy | 数据量 | normal val acc | 400 局胜率 | 点炮 |
|---|---|---|---|---|---|
| conv-BC rollout | conv-BC greedy | 50 局（未完成，太慢） | — | — | — |
| fast rollout | shanten-minimizing | 200 局 (~16k) | 43.1% | **1.5%** | 17.2% |
| conv-BC base | — | — | 71.0% | 25.5% | 19.5% |

**结论**：
- **conv-BC greedy rollout oracle 单局成本过高**（~70s），50 局生成在 8 线程下运行超过 20 分钟仍未完成，无法规模化；
- **shanten-minimizing rollout oracle 虽然快，但 oracle 本身太弱**：蒸馏出的 normal policy val acc 仅 43%，实战中胜率仅 1.5%，完全不可用；
- **True perfect-info rollout oracle 在本项目资源约束下被证伪**：要么太慢（conv-BC rollout），要么太弱（shanten rollout）。

### 17.3 最终结论

- 完美信息 rollout oracle **工程成本极高且收益不确定**；
- 在当前框架内，**deal-in auxiliary loss（方向 4）是唯一有效的防守改进**；
- 若继续追求超越 dealin07，需要更强的 rollout policy 或真正的 perfect-info MCTS，成本远超本次探索。


## 18. NN 模型与 BeliefExp 搜索的结合

在 deal-in auxiliary loss 取得进展后，继续探索把最近 NN 模型（conv-BC / dealin07）与 BeliefExpectimax 搜索结合。

### 18.1 搜索内部结合（V3 deep）

`BeliefExpectimaxV3Agent` 已支持 `candidate_policy='nn'` 与 `leaf_evaluator='nn'`，通过 `MJ_NN_VALUE_MODEL` 可指定 leaf 模型。测试组合（400 局公平池，`MJ_NN_VALUE_MODEL=dealin07`）：

| Agent | 候选策略 | leaf | 胜率 | 点炮 |
|---|---|---|---|---|
| BeliefExp | eval2 | eval2 | 25.5% | 16.0% |
| PPO-convBC | — | — | 22.2% | 15.2% |
| V3d-1-nn | conv-BC | dealin07 | 22.0% | 18.5% |
| V3-RLunion-convBC | conv-BC ∪ nn_model | dealin07 | 25.0% | 18.5% |

**结论**：把 NN 作为 V3 搜索的候选/叶子，并未显著超过 BeliefExp；V3-RLunion 接近但胜率仍略低。

### 18.2 Hybrid Agent（NN + BeliefExp 分层决策）——**成功**

实现 `algo/agents/hybrid_nn_belief_agent.py`：
- 平时用快速 NN policy；
- 当任一对手报听或总弃牌数 ≥ 28 时，切换到 BeliefExpectimax 搜索。

400 局公平池结果：

| Agent | 胜率 | 点炮 | 备注 |
|---|---|---|---|
| BeliefExp | 35.2–36.2% | 21.0–23.8% | 搜索基线，慢 |
| PPO-convBC | 30.8% | 21.0% | 纯前馈基线 |
| **Hybrid-convBC** | **32.8%** | **17.0%** | 比 convBC 胜率高 2%，点炮低 4% |
| PPO-dealin07 | 29.8% | 21.8% | 纯前馈 dealin |
| **Hybrid-dealin07** | **34.8%** | **18.8%** | 胜率接近 BeliefExp，点炮更低 |

**结论**：
- **Hybrid-dealin07 是最佳结合点**：胜率接近 BeliefExp（34.8% vs 35.2%），点炮更低（18.8% vs 21.0%），且平均速度远快于全程 BeliefExp；
- Hybrid-convBC 也有效，但 dealin07 作为 NN 基座让 hybrid 更强；
- 这是目前**最推荐的部署形态**：`hybrid:dealin07:output/nn_conv_bc_dealin_2000_l07.pt:beliefexp`。

### 18.3 产物与用法

- 代码：`algo/agents/hybrid_nn_belief_agent.py`、`algo/nn/nn_policy.py`（兼容 dealin head 3 输出）
- benchmark token：
  ```bash
  SEATS="hybrid:dealin07:output/nn_conv_bc_dealin_2000_l07.pt:beliefexp,beliefexp,..."
  ```
- 关键模型：`output/nn_conv_bc_dealin_2000_l07.pt` + 旧 `BeliefExpectimaxAgent`。

### 18.4 总体结论

- 纯 NN 路线：`dealin07` 是最佳前馈模型（防守强）。
- 搜索融合路线：`Hybrid-dealin07` 是最佳 NN+BeliefExp 结合（胜率接近 BeliefExp，点炮更低，速度更快）。
- 若接受全程搜索的开销，BeliefExp 仍是胜率上限；若需要实用部署，**Hybrid-dealin07 是更优折中**。


## 19. Bootstrap：用 Hybrid-dealin07 当教师迭代提升

### 19.1 数据生成

实现 `scripts/rl/gen_hybrid_dealin_data.py`：4 个座位全是 Hybrid-dealin07，记录每个决策的 `(X, dealin, y, v)`。用 `spawn` + `ProcessPoolExecutor` 并行，1000 局约 230s，2000 局约 475s。

产物：
- `output/nn_teacher_hybrid_dealin_1000.npz`（46.7k 样本）
- `output/nn_teacher_hybrid_dealin_2000.npz`（95.4k 样本）

### 19.2 迭代训练结果

#### 第一代：deal-in auxiliary loss（ warm init from dealin07）

| 模型 | 训练数据 | val acc | 400 局 Hybrid 胜率 | 点炮 | 备注 |
|---|---|---|---|---|---|
| dealin07 | base 2000 | 84.0% | 33.2–33.5% | 18.5% | 上一代 |
| dealinV2 | hybrid 1000 | 82.5% | 33.0% | 21.0% | 未提升 |
| dealinV2m | hybrid 1000 + base 2000 | 83.2% | 34.8% | 19.0% | 略好但不稳定 |
| dealinV3 | hybrid 2000 | 82.8% | 21.5% | 15.0% | 更差 |

**结论**：直接把 Hybrid 动作蒸馏进带 deal-in head 的 NN，没有稳定超过上一代。

#### 第一代：纯 BC on Hybrid 数据（无 deal-in head）

| 模型 | 训练数据 | val acc | 纯 NN 400 局胜率 | 点炮 | Hybrid 组合 400 局胜率 | 点炮 |
|---|---|---|---|---|---|---|
| hybridBase | hybrid 2000 | 81.9% | 21.0% | 19.8% | **23.8%** | **12.5%** |

**关键发现**：
- 纯 BC 模型本身不如 dealin07（21.0% vs 24.5% 胜率，点炮更高）；
- 但把它放进 **Hybrid** 后，表现优于 Hybrid-dealin07（23.8% vs 21.5% 胜率，点炮 12.5% vs 14.8%）。

原因：deal-in auxiliary loss 让 fast NN 过于保守，牺牲了进攻；而 Hybrid 中 BeliefExp 已经负责关键防守，fast NN 只需要尽量模仿 Hybrid 的**常规进攻决策**，纯 BC 更合适。

### 19.3 当前最佳候选

- **Hybrid-hybridBase**：`hybrid:hybridBase:output/nn_conv_bc_hybrid_2000.pt:beliefexp`
- 在 400 局池中：胜率 23.8%，点炮 12.5%；
- 优于 Hybrid-dealin07（21.5% / 14.8%），也更接近 BeliefExp（27.0% / 19.0%）的胜率，但点炮显著更低。

### 19.4 Bootstrap 结论

- **一轮 bootstrap 有效**：用 Hybrid 教师数据训练的纯 BC 作为新 fast policy，再组装 Hybrid，得到了比上一代更好的组合 agent；
- 但提升幅度有限（胜率 +2–3%，点炮 -2%），且需要继续迭代才可能稳定超越 BeliefExp；
- deal-in auxiliary loss 对**纯前馈**模型仍是最佳，但对 **Hybrid 框架** 内的 fast policy 可能过度约束；
- 后续若继续 bootstrap，建议：
  1. 用 `Hybrid-hybridBase` 当教师再生成 2000 局数据；
  2. 尝试更大网络容量或更长训练；
  3. 在 Hybrid 中调整切换阈值（tenpai_threshold）看能否减少 BeliefExp 调用比例同时保持强度。


### 19.5 第二代 bootstrap（用 Hybrid-hybridBase 当教师）

用 Hybrid-hybridBase 再生成 2000 局数据（94.2k 样本），训练纯 BC 得到 `nn_conv_bc_hybrid_v2.pt`（val acc 82.8%）。

400 局直接对比：

| Agent | 胜率 | 点炮 | 备注 |
|---|---|---|---|
| BeliefExp | 27.8% | 17.2% | 搜索基线 |
| Hybrid-dealin07 | 21.2% | 17.8% | 初代 Hybrid |
| **Hybrid-hybridBase** | **25.2%** | **14.5%** | 一代 bootstrap 最佳 |
| Hybrid-hybridV2 | 22.8% | 17.8% | 二代 bootstrap |

**结论**：
- 第二代 bootstrap **没有继续提升**，Hybrid-hybridV2 不如 Hybrid-hybridBase；
- 一代 bootstrap 的结果基本触及当前框架下的收敛点；
- 继续同方向迭代收益有限，需要改变教师结构、网络容量或训练目标才能突破。

### 19.6 当前最终候选

| 场景 | 推荐 |
|---|---|
| 最低 Deal-in、稳健 | `hybrid:hybridBase:output/nn_conv_bc_hybrid_2000.pt:beliefexp` |
| 胜率优先的 Hybrid | `hybrid:dealin07:output/nn_conv_bc_dealin_2000_l07.pt:beliefexp` |
| 纯前馈 | `output/nn_conv_bc_dealin_2000_l07.pt` |
| 胜率上限 | `BeliefExpectimaxAgent` |

### 19.7 后续若继续

- 增大网络容量（channels/n_blocks）看能否更好蒸馏 Hybrid；
- 训练时同时拟合 Hybrid 的 value 和 policy，或加入 BeliefExp 搜索轨迹作为 soft target；
- 尝试调整 Hybrid 切换阈值（tenpai_threshold），在速度和强度间找更优点；
- 引入动作空间/规则层面的改造（如坎/杠决策优化）。
