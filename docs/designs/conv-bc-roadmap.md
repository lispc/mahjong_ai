# conv-BC 后续方向路线图

> 文档记录基于 `output/nn_conv_bc.pt`（纯前馈卷积策略，~1 ms/步，与 Baseline/BeliefExp 同档）的后续探索方向。该模型速度优势巨大，可解锁传统搜索型 agent 难以承受的 online adaptation / 大规模 simulation / oracle distillation 等 design space。

---

## 当前状态

- **当前最佳候选**：`hybrid:BE16k_t8:output/nn_conv_bc_beliefexp_trace_16000_big_t8.pt:beliefexp`
- **性能**：2000 局最终确认胜率 **25.8%**，点炮 **16.3%**，Elo **1581**
- **上一版本候选**：`hybrid:BE8k_t8:output/nn_conv_bc_beliefexp_trace_8000_big_t8.pt:beliefexp`（2000 局确认 Elo 1495）
- **上一代稳健候选**：`hybrid:BE4k_big:output/nn_conv_bc_beliefexp_trace_4000_big.pt:beliefexp`
- **Search-distill 成功记录**：`hybrid:HTbase:output/nn_conv_bc_hybridtrace_1000_base.pt:beliefexp`（1000 局胜率 25.3%、点炮 15.2%）
- **纯前馈基线**：`output/nn_conv_bc.pt`，~1 ms/步，胜率 ~22–25%，点炮 ~15–18%
- **对比**：Baseline/BeliefExp 胜率 ~25–28%，决策 150–350 ms/步
- **速度优势**：conv-BC NN 部分比搜索 agent 快 **200–300 倍**

这一速度优势意味着：
- 每局开头可以跑 **上百局 self-play** 做在线适应；
- 每步可以跑 **数十到数百次 MCTS simulation**；
- 可以承受 **oracle policy 蒸馏** 所需的大规模数据生成。

---

## 方向 1：Runtime Policy Adaptation（pMCPA）

### 动机

人类玩家拿到不同初始手牌时风格完全不同（好牌激进、烂牌防守）。离线 policy 是“全局平均最优”，但每局开始后的具体牌形可以用少量 online simulation 专门微调。

### 方法

1. **开局触发**：每局开始时，拿到初始手牌后启动 adaptation；
2. **快速 self-play**：用当前 conv-BC policy 和自己打 **K 局**（如 64/128/256 局），利用 ~1 ms/步的速度，耗时仅数秒到数十秒；
3. **轨迹收集**：记录这些 adaptation 局的 `(state, action, final outcome)`；
4. **小步微调**：只更新 policy 头或一个轻量 adapter（低 lr，1–5 epochs），使 policy 偏向在本局牌形下获胜更高的动作；
5. **本局使用**：后续所有决策使用微调后的 policy；对局结束后丢弃 adapter。

### 实现要点

- 在 `algo/agents/ppo_agent.py` 或新增 `algo/agents/adaptive_conv_agent.py` 中实现；
- adapter 可以是 policy 头前几层的小 MLP 或 tile-embedding 的 scale/shift；
- 为避免过拟合，训练时加权重衰减 + early stopping + 只训最顶层；
- 每局 adaptation 时间预算可配置（如 5s / 20s / 60s）。

### 验证结果（2026-07）

已实现 `algo/agents/adaptive_conv_agent.py`（固定当前座位初始手牌的 per-game adaptation），并在 `benchmark_pool.py` 增加 `adapt:` token。参数通过环境变量控制：`ADAPT_N_GAMES`, `ADAPT_EPOCHS`, `ADAPT_LR`, `ADAPT_WIN_WEIGHT`。

**关键结果（同一 pool 400 局）**：

| 配置 | 胜率 | 点炮 | 备注 |
|---|---|---|---|
| conv-BC base | 22.2% | 15.2% | 基线 |
| K=32, epochs=1, lr=1e-4 | 30.0%（120 局） / 22.5%（400 局） | 19.2% / 19.2% | 小样本看好，大样本回落 |
| K=128, epochs=1, lr=5e-5 | 26.5%（200 局） / 23.8%（400 局） | 18.0% / 19.2% | 最佳配置，但仅 +1.6% 绝对值 |

**结论**：pMCPA 能带来 **轻微、不稳定的提升**（~1–2% 绝对胜率），无法稳定超过 Baseline/BeliefExp。原因可能是：
- 小样本微调容易过拟合到 self-play 风格；
- 教师就是 base policy 自己，无法提供超越自身的新信息；
- 真正的 round-specific adaptation 需要更强的对手模型或价值信号。

---

## 方向 2：MCTS/PUCT with conv-BC Prior + Value

### 动机

历史上 DetMCTS/FlatMC 弱是因为 prior policy 和 leaf value 都弱。现在 conv-BC 可同时提供强 policy prior 和强 value estimate，这正是 AlphaZero/ReBel 类搜索的核心配方。

### 已实现方案

1. **Flat Determinized MC with conv-BC**（`algo/agents/mcts_conv_agent.py`）
   - 用 conv-BC prior 选 top-k 候选；
   - 每个 (world, candidate) 跑 eval0 fast rollout + conv-BC value 截断；
   - 太慢且不强，已放弃。
2. **depth-1 ExpectiMax with conv-BC value leaf**（复用 `v3deep:1-nn` token + `MJ_NN_VALUE_MODEL`）
   - 直接利用现有 BeliefExpectimaxV3Agent，把 leaf value 换成 conv-BC value head。

### 验证结果

**V3d-1-nn（depth=1 + conv-BC 候选 + conv-BC value）300 局**：

| Agent | 胜率 | 点炮 |
|---|---|---|
| Baseline | 28.7% | 22.3% |
| BeliefExp | 27.7% | 13.7% |
| **V3d-1-nn** | **22.0%** | 18.3% |
| conv-BC | 19.3% | 18.3% |

**结论**：depth-1 search + conv-BC value 给 conv-BC 带来 **~2.7% 绝对胜率提升**（19.3% → 22.0%），但仍低于 Baseline/BeliefExp。更深的 depth=2 搜索太慢（80 局 300s 超时），不实用。

---

## 方向 3：Oracle-Guided Distillation

### 动机

本项目所有历史教师（BeliefExp、Baseline、eval2、conv-BC）都是受限信息 agent。Suphx 的突破来自先训练一个**能看到所有人手牌和牌山的作弊 oracle**，再把它蒸馏到 normal policy。这是唯一可能**大幅突破当前天花板**的监督学习路线。

### 已实现管线

- `extract_features_oracle()`（311 维 = 175 基础 + 3 对手手牌 + 完整牌山）
- `scripts/rl/gen_oracle_data.py`：用 BeliefExp 当教师生成 (Xn, Xo, y, v)
- `scripts/rl/pretrain_bc.py`：支持 oracle 特征训练（`n_tile_ch=9`）与普通特征训练（`Xn`）
- `scripts/rl/distill_oracle.py`：normal policy 同时拟合 hard target + KL(oracle policy)
- `scripts/rl/gen_oracle_safety_data.py`：用 perfect-info safety oracle 生成数据；支持 `SAFETY_MIXED=1`（仅 1/4 座位为 safety oracle，其余为 conv-BC greedy）

### 实验结果

#### 3.1 BeliefExp-Oracle 蒸馏

| 数据量 | oracle val acc | distill val acc | 400 局胜率 | 点炮 |
|---|---|---|---|---|
| 200 局 (~9k) | 69.2% | 73.0% | 21.5% | 22.2% |
| 1000 局 (~46k), α=1.0 | 74.8% | 76.7% | 21.0% | 20.2% |
| 1000 局 (~46k), α=2.0 | 74.8% | 76.3% | 21.0% | 20.2% |
| conv-BC base | 71.0% | — | 22.5% | 16.0% |

**结论**：BeliefExp + oracle features **没有产生更强的 oracle 教师**。蒸馏后的 normal policy 与 conv-BC base 互角，但点炮更高。根因：BeliefExp 的动作是基于信念而非完美信息，oracle 特征虽然能看见私有信息，但学习目标仍然是受限信息下的动作，无法教会 normal policy 利用这些私有信息。

#### 3.2 Perfect-Info Safety Oracle

Safety oracle 的决策：枚举合法弃牌，排除会立即点炮的牌，在剩余牌中选 conv-BC 分数最高者。

| 教师/数据 | 训练方式 | val acc | 400 局胜率 | 点炮 | 备注 |
|---|---|---|---|---|---|
| All-safety, 500 局 (~31k) | BC on Xn | 75.3% | 14.8% | 16.8% | 全桌防守，过保守 |
| Mixed-safety, 2000 局 (~27k) | BC on Xn | 73.6% | 15.2% | 20.0% | 1/4 座位防守 |
| Mixed-safety + base data | BC on merged 123k | 69.7% | 24.0% | 18.2% | 与 base 互角 |
| Mixed-safety | Oracle policy → distill | 74.3% | 13.5% | 19.5% | soft target 也无益 |
| conv-BC base | — | 71.0% | 22.5–24.5% | 15.2–18.2% | 仍是最佳前馈 |

**结论**：
- 即便能看到对手手牌、能**完全避免即时点炮**的 safety oracle，蒸馏出的 normal policy 也**没有更低 Deal-in**，反而因为过度保守损失了胜率；
- 与 base 数据混合后勉强不崩，但**没有统计学意义上的防守提升**；
- **问题不在信息不足，而在 oracle 本身不够强**：safety oracle 只解决“这一张牌是否立即点炮”，没有解决“如何在避免点炮的同时保持进攻/听牌效率”。

### 最终结论

**方向 3 在现有实现下被证伪**：
- BeliefExp + oracle 特征不是更强的 oracle；
- Perfect-info safety oracle 虽然能避免即时点炮，但无法蒸馏出更强的 normal policy；
- 真正有效的 oracle 需要**利用完美信息做全局更强的决策**（如 perfect-info MCTS / 终局求解），而不仅仅是给现有受限 agent 加上私有信息特征。

---

## 方向 4：显式点炮代价辅助 loss（deal-in head）——**成功**

### 动机

BeliefExp 的防守行为无法通过特征扩展学到，说明问题在**目标函数**。与其让网络模仿教师动作，不如直接让网络预测「弃这张牌是否点炮」，用辅助 loss 引导 trunk 学到防守表示。

### 方法

1. 在 `TileConvNet` 增加可选 `dealin_head`（34 维 logit）；
2. `gen_dealin_data.py`：用完美信息生成 per-tile 即时点炮标签；
3. `train_dealin_aux.py`：从 conv-BC base 初始化，联合训练 `policy CE + value MSE + λ·deal-in BCE`。

### 结果

| 模型 | 800 局胜率 | 点炮 | 备注 |
|---|---|---|---|
| conv-BC base | 23.5% | 19.1% | 基线 |
| **dealin λ=0.7** | 22.8% / 21.2% | **14.5% / 16.6%** | 最佳防守/胜率折中 |
| PPO deal-in reward -5 | 19.0% | 16.0% | 胜率损失过大 |
| BeliefExp | 25.8% | 15.1% | 搜索型参考 |

**结论**：辅助 loss 是**首个能稳健降低 conv-BC 点炮**的方法。λ=0.7 把点炮降到接近 BeliefExp 水平，胜率损失在噪声范围内。

### 产物

- `output/nn_conv_bc_dealin_2000_l07.pt`（推荐候选）
- `output/nn_dealin_labels_2000.npz`
- `algo/nn/model.py`、`scripts/rl/gen_dealin_data.py`、`scripts/rl/train_dealin_aux.py`、`algo/agents/defensive_conv_agent.py`

---

## 方向 5：True Perfect-Info Rollout Oracle——进行中/成本高

### 动机

Safety oracle 只看即时点炮，不够强。真正的 perfect-info oracle 应该通过 rollout 评估每个弃牌的未来期望 outcome。

### 实现

- `scripts/rl/gen_rollout_oracle_data.py`：conv-BC greedy rollout（每局 ~70s，太慢）；
- `scripts/rl/gen_rollout_oracle_fast_data.py`：shanten-minimizing rollout（每局 ~10–15s，32 进程可扩展）。

### 结果

| Oracle | Rollout policy | 数据量 | normal val acc | 400 局胜率 | 点炮 |
|---|---|---|---|---|---|
| conv-BC rollout | conv-BC greedy | 50 局（未完成） | — | — | —，太慢 |
| fast rollout | shanten-minimizing | 200 局 (~16k) | 43.1% | 1.5% | 17.2% |
| conv-BC base | — | — | 71.0% | 25.5% | 19.5% |

**结论**：
- conv-BC rollout oracle 单局 ~70s，无法规模化；
- shanten-minimizing rollout oracle 速度可接受，但 oracle 太弱，蒸馏出的 normal policy 胜率仅 1.5%；
- **True perfect-info rollout oracle 在当前资源下被证伪**：要么太慢，要么太弱。

### 最终结论

- 完美信息 rollout oracle 成本极高且收益不确定；
- **deal-in auxiliary loss 是纯前馈路线唯一有效的防守改进**；
- 继续追求更强 oracle 需要 conv-BC 级 rollout policy + 大量计算，工程上不现实。

---

## 方向 6：NN 模型与 BeliefExp 搜索结合——**成功**

### 动机

BeliefExpectimax 仍是胜率上限，但速度慢；NN（尤其是 dealin07）速度快但绝对胜率稍低。把两者结合，可在速度和强度之间取得更好折中。

### 实现

1. **搜索内部结合**：用 `BeliefExpectimaxV3Agent` 的 `candidate_policy='nn'` / `leaf_evaluator='nn'`，把 conv-BC/dealin 接入搜索。
2. **Hybrid Agent**：实现 `algo/agents/hybrid_nn_belief_agent.py`：
   - 平时用 NN 前馈决策；
   - 当任一对手报听或总弃牌数 ≥ 28 时，切换到 `BeliefExpectimaxAgent` 搜索。

### 结果（400 局公平池）

| Agent | 胜率 | 点炮 | 备注 |
|---|---|---|---|
| BeliefExp | 35.2% | 21.0% | 搜索基线 |
| PPO-convBC | 30.8% | 21.0% | 纯前馈基线 |
| V3-RLunion-convBC | 25.0% | 18.5% | NN 候选 + dealin leaf，未超 BeliefExp |
| **Hybrid-convBC** | 32.8% | **17.0%** | 比 convBC 胜率高 2%，点炮低 4% |
| PPO-dealin07 | 29.8% | 21.8% | 纯前馈 dealin |
| Hybrid-dealin07 | 34.8% | 18.8% | 胜率接近 BeliefExp，点炮更低 |
| **Hybrid-hybridBase** | **23.8%** | **12.5%** | bootstrap 后新候选，点炮显著更低 |

**结论**：
- **Hybrid-dealin07 是初代最佳结合点**：胜率几乎持平 BeliefExp，点炮更低；
- **一轮 bootstrap 后得到 Hybrid-hybridBase**：虽然纯 NN 胜率不如 dealin07，但放进 Hybrid 后点炮更低（12.5%），整体更稳健；
- 搜索内部替换 candidate/leaf 效果不如 Hybrid 分层策略。

### 产物

- `algo/agents/hybrid_nn_belief_agent.py`
- `algo/nn/nn_policy.py`（兼容 dealin head 3 输出）
- `scripts/rl/gen_hybrid_dealin_data.py`（Hybrid 教师数据生成）
- benchmark token：`hybrid:<label>:<nn_model_path>:beliefexp`

---

## 方向 7：Bootstrap 迭代提升（两代）——**一代成功、二代收敛**

### 方法

用当前最强 Hybrid 当教师自对弈，生成 1000/2000 局数据，重新训练纯 NN，再组装新 Hybrid。迭代两代：
- 一代教师：Hybrid-dealin07；
- 二代教师：Hybrid-hybridBase。

### 结果

| 模型 | 训练数据 | 纯 NN 400 局胜率 | 点炮 | Hybrid 组合 400 局胜率 | 点炮 |
|---|---|---|---|---|---|
| dealin07 | base 2000 | 24.5% | 17.5% | 21.5–33.5% | 14.8–18.5% |
| dealinV2/V2m/V3 | hybrid + base / hybrid 2000 | ~21–24% | 15–19% | 21.5–34.8% | 15.0–21.0% |
| **hybridBase** | hybrid 2000（纯 BC，无 deal-in head） | 21.0% | 19.8% | **25.2%** | **14.5%** |
| hybridV2 | hybridBase 2000（纯 BC） | 未测 | — | 22.8% | 17.8% |

**结论**：
- 直接把 Hybrid 蒸馏进 deal-in NN 没有稳定提升；
- **用纯 BC 在 Hybrid 数据上训练出的 `nn_conv_bc_hybrid_2000.pt`，放进 Hybrid 后点炮显著下降**，是一代 bootstrap 最佳结果；
- **二代 bootstrap 没有继续提升**，当前框架下基本收敛。

---

## 总体结论（最终）

| 方向 | 最佳结果 | 是否超越 base |
|---|---|---|
| pMCPA | +1.6% 胜率，不稳定 | 否 |
| MCTS/PUCT | V3d-1-nn +2.7% 胜率 | 否（仍低于搜索型） |
| Oracle Distillation | 全部阴性 | 否 |
| Deal-in auxiliary loss | 点炮 19.1% → 14.5–16.6%，胜率基本持平 | 部分超越（防守指标） |
| True perfect-info rollout oracle | 太慢/太弱 | 否 |
| NN + BeliefExp Hybrid | Hybrid-dealin07：胜率接近 BeliefExp，点炮更低 | 是（实用超越） |
| **Bootstrap 两代** | **Hybrid-hybridBase：胜率 25.2%，点炮 14.5%** | **是（历史最稳健）** |
| Tenpai head | 1000 局 Elo 最高，但点炮高于 hybridBase，未稳健超越 | 否（需更强教师） |
| **Search trace distillation** | **Hybrid-BE16k_t8（16000局 BeliefExp教师，128/6/512，T=8）：2000局确认胜率 25.8%、点炮 16.3%、Elo 1581** | **是（当前最佳）** |
| **Safety-aware 报听** | 多阈值均降低胜率、升高点炮 | **否** |
| **8× 数据缩放** | 128000局 BeliefExp教师 + 128/6/512 或 192/8/768：val acc↑，但 1000 局 Elo 1479/1456，未超越 16000 局模型 | **否（边际递减）** |
| **PPO 自对弈微调** | 从 BE16k_t8 热启，20 iter 后 Elo 1518（低于初始化 1527–1581），policy 退化 | **否（稀疏 reward 不稳定）** |

**当前最佳候选**：
- **当前最佳**：`hybrid:BE16k_t8:output/nn_conv_bc_beliefexp_trace_16000_big_t8.pt:beliefexp`
- **上一版本候选**：`hybrid:BE8k_t8:output/nn_conv_bc_beliefexp_trace_8000_big_t8.pt:beliefexp`
- **上一代稳健候选**：`hybrid:Base:output/nn_conv_bc_hybrid_2000.pt:beliefexp`
- **Search-distill 成功记录**：`hybrid:HTbase:output/nn_conv_bc_hybridtrace_1000_base.pt:beliefexp`
- **Elo 取向实验**：`hybrid:A05:output/nn_conv_bc_searchdistill_1000_a05_t2.pt:beliefexp`
- **胜率优先的 Hybrid**：`hybrid:dealin07:output/nn_conv_bc_dealin_2000_l07.pt:beliefexp`
- **纯前馈首选**：`output/nn_conv_bc_dealin_2000_l07.pt`
- **胜率上限（不计较速度）**：`BeliefExpectimaxAgent`

**结论**：
- 前馈 conv-BC  alone 触及天花板；
- NN + BeliefExp Hybrid 是当前最佳实用框架；
- **Search trace distillation 是有效提升路线，纯 BeliefExp 教师 + 大网络 + 更 soft target（最终 T=8）+ 充足数据是当前最强组合**：`Hybrid-BE16k_t8` 在 2000 局确认中胜率 25.8%、点炮 16.3%、Elo 1581；
- **Safety-aware 报听（当前实现）无效**：限制报听反而降低胜率、升高点炮；
- **数据缩放超过 16000 局后边际递减**：128000 局 + 更大网络 val acc 提升，但 Elo 下降，说明当前教师/蒸馏目标已饱和；
- **PPO 自对弈微调从强 BC 初始化易退化**：稀疏终局 reward 把 policy 拉偏，20 iter 后 Elo 低于初始化。

---

## 方向 8：报听决策学习（tenpai head）—— 已实现、效果中性

### 动机

当前 agent 的报听（declare_tenpai）由硬启发式控制（剩余待牌 ≥4 或有已现待牌）。报听是与弃牌并列的关键动作，但从未被 NN 直接学习。把报听决策纳入网络是最小的动作空间扩展，也能为后续「报听 or 继续改良」的端到端优化提供基础。

### 实现

1. **网络**：在 `TileConvNet` 增加可选 `tenpai_head`（全局特征 → MLP → 1 logit），并复用 `_trunk` 提供 `tenpai_logit(x)` 接口；
2. **Agent**：`PPOAgent.declare_tenpai()` 在 `tenpai_head=True` 时用网络决策，否则回退原启发式；
3. **数据**：`scripts/rl/gen_hybrid_tenpai_data.py` 用 Hybrid 教师自对弈，记录每个听牌状态的 `(X_tenpai, t)`，其中 `t` 是教师真实报听决策；
4. **训练**：`scripts/rl/train_tenpai.py` 从 `nn_conv_bc_dealin_2000_l07.pt` 初始化，联合训练 `policy CE + value MSE + dealin BCE + tenpai BCE`，tenpai 正样本用 pos_weight 平衡。

### 产物

- `output/nn_conv_bc_tenpai_1000_l1.pt` / `..._config.json`
- `output/nn_teacher_hybrid_tenpai_1000.npz`（47k 弃牌 + 3.2k 报听样本）
- `algo/nn/model.py`、`algo/agents/ppo_agent.py` 改动

### 结果

| Agent | 500 局胜率 | 点炮 | Elo | 备注 |
|---|---|---|---|---|
| Baseline | 27.4% | 21.0% | 1456 | — |
| Hybrid-Base | 24.2% | 14.8% | 1505 | 当前最佳 Hybrid |
| **Hybrid-Tenpai** | 22.6% | 19.8% | **1552** | 报听 NN 化 |
| PPO-Tenpai | 21.8% | 16.8% | 1488 | 纯前馈 |

1000 局同 pool 验证：

| Agent | 胜率 | 点炮 | Elo |
|---|---|---|---|
| **Hybrid-Tenpai** | 24.0% | 18.9% | **1578** |
| Baseline | 25.9% | 20.3% | 1548 |
| PPO-Tenpai | 22.5% | 16.8% | 1474 |
| Hybrid-Base | 23.7% | 16.1% | 1399 |

- tenpai head 在验证集上报听分类 acc ≈ 97%，说明网络成功拟合了教师启发式；
- 1000 局中 Hybrid-Tenpai Elo 最高（1578），但点炮 18.9% 高于 Hybrid-Base 的 16.1%；
- **未产生稳健的全面超越**：胜率/点炮与 Hybrid-Base 互有胜负，差异在统计噪声范围；
- 教师本身仍在使用同一启发式，head 只是把启发式 NN 化，这是效果中性的主因。

### 结论

- 报听 NN 化实现成功，可作为后续强化学习 / 搜索蒸馏报听策略的基础；
- 仅靠模仿现有启发式无法突破天花板；
- 下一步需要让 tenpai 教师更强（如 outcome-weighted、MC 搜索报听价值、或 safety-aware 报听），head 才有提升空间。

---

## 推荐执行顺序（已全部完成）

| 顺序 | 方向 | 状态 |
|---|---|---|
| 1 | **pMCPA** | 完成，阴性 |
| 2 | **MCTS/PUCT with conv-BC** | 完成，部分增益但不超过搜索型 |
| 3 | **Oracle-Guided Distillation** | 完成，阴性 |
| 4 | **Deal-in auxiliary loss** | 完成，阳性 |
| 5 | **True perfect-info rollout oracle** | 完成，阴性（成本/强度不可行） |
| 6 | **NN + BeliefExp Hybrid** | 完成，阳性 |
| 7 | **Bootstrap 两代** | 完成，一代阳性、二代收敛 |
| 8 | **Tenpai head** | 完成，效果中性（需更强教师） |
| 9 | **Search trace distillation** | 完成，部分阳性（Elo 提升，需更多数据） |

---

## 方向 9：搜索轨迹蒸馏（search trace distillation）—— 已实现、部分阳性

### 动机

之前的 BC / bootstrap 只用教师最终动作的 hard label 和最终 outcome（±1/0），丢失了教师内部对候选的偏好结构。V3-NN-PC 在决策时会评估每个候选的 expectimax value，这些分数是天然平滑的 target。把 policy 从 hard label 升级到搜索轨迹的 soft target，有望让 conv-BC 学到更强的策略结构。

### 实现

1. **教师轨迹暴露**：修改 `BeliefExpectimaxV3Agent.next()` 为 `next_with_trace()`，返回 `(chosen_tile, trace)`，其中 `trace` 包含每个候选的 expectimax offense score、danger score 和被选中动作的 score；
2. **数据生成**：`scripts/rl/gen_v3_trace_data.py` 用 4 座位 V3-NN-PC 自对弈，记录每个决策的 `X, y, scores(34 维), selected_value, v`；
3. **训练**：`scripts/rl/train_search_distill.py` 从 `nn_conv_bc_dealin_2000_l07.pt` 初始化，联合训练：
   - `policy CE`（hard label）
   - `α * KL(student || softmax(teacher_scores / T))`（soft policy target）
   - `value MSE(v)`（最终 outcome）
   - `β * value MSE(tanh(selected_value / τ))`（dense value target）
   - `λ * dealin BCE`

### 产物

- `output/nn_teacher_v3_trace_500.npz`（26k 样本，500 局 V3-NN-PC，教师过慢）
- `output/nn_teacher_v3_trace_eval0_1000.npz`（53k 样本，1000 局 V3-eval0，5 min 生成）
- `output/nn_teacher_hybrid_trace_1000.npz`（47k 样本，19760 条带 trace，1000 局 Hybrid-hybridBase critical 状态）
- `output/nn_conv_bc_searchdistill_1000_a{05,10,03}_t2.pt`、`..._a05_t4.pt`、`..._a05_t2_b0.pt`
- `output/nn_conv_bc_hybridtrace_1000_base.pt`、`..._big.pt`（本轮最佳）
- `algo/agents/belief_expectimax.py`、`algo/agents/belief_expectimax_v3.py`、`algo/agents/hybrid_nn_belief_agent.py`
- `scripts/rl/gen_v3_trace_data.py`、`scripts/rl/gen_hybrid_trace_data.py`、`scripts/rl/train_search_distill.py`

### 结果

#### 第一轮：V3-NN-PC 教师（500 局）

| Agent | 局数 | 胜率 | 点炮 | Elo | 备注 |
|---|---|---|---|---|---|
| Hybrid-Search (α=0.5) | 200 | 26.0% | 17.5% | **1621** | 小样本看好 |
| Hybrid-Search (α=0.5) | 1000 | 24.7% | 18.8% | **1556** | Elo 最高 |
| Hybrid-Search (α=1.0) | 500 | 22.6% | 19.0% | 1553 | 与 α=0.5 互角 |
| Hybrid-Base | 1000 | **26.6%** | **16.1%** | 1506 | 低炮稳健 |
| Baseline | 1000 | 28.1% | 19.4% | 1527 | — |
| PPO-Search | 1000 | 16.6% | 16.9% | 1412 | 纯 NN 较弱 |

#### 第二轮：V3-eval0 教师 + 1000 局 + 超参扫描

教师改用 `leaf_evaluator='eval0'` 以加速规模化（1000 局 5 min）。训练 5 个变体（α/T/β），在 4-agent pool 中对比：

**Pool A（500 局）**：Baseline / Hybrid-Base / Hybrid-A05 / Hybrid-A10

| Agent | 胜率 | 点炮 | Elo |
|---|---|---|---|
| Hybrid-A05 | 21.4% | 18.4% | **1561** |
| Baseline | 29.2% | 19.8% | 1512 |
| Hybrid-A10 | 21.2% | 17.2% | 1467 |
| Hybrid-Base | 25.0% | **16.6%** | 1461 |

**Pool B（500 局）**：Baseline / Hybrid-Base / Hybrid-A03 / Hybrid-B0（β=0）

| Agent | 胜率 | 点炮 | Elo |
|---|---|---|---|
| Hybrid-B0 | 22.8% | **16.0%** | **1530** |
| Baseline | 26.4% | 19.0% | 1502 |
| Hybrid-A03 | 22.8% | 19.8% | 1486 |
| Hybrid-Base | 24.6% | 17.2% | 1483 |

**Final 1000 局**：Baseline / Hybrid-Base / Hybrid-A05 / Hybrid-B0

| Agent | 胜率 | 点炮 | Elo |
|---|---|---|---|
| Baseline | **26.6%** | 19.6% | **1579** |
| Hybrid-A05 | 22.4% | 18.1% | 1547 |
| Hybrid-Base | 25.7% | **17.1%** | 1510 |
| Hybrid-B0 | 22.4% | 17.7% | 1365 |

#### 第三轮：Hybrid-hybridBase critical trace + 1000 局

关键改进：

1. **教师改为当前最强 Hybrid-hybridBase**，且只在它切换到 BeliefExp 的 critical 状态记录搜索轨迹；
2. **数据分布更接近实战**：正常状态用 hard label，critical 状态用 soft target；
3. 训练了 base-size（96/4/256）和 big-size（128/6/512）两个网络。

**1000 局 benchmark**：Baseline / Hybrid-Base / Hybrid-HTbase / Hybrid-HTbig

| Agent | 胜率 | 自摸 | 铳 | 点炮 | Elo |
|---|---|---|---|---|---|
| Baseline | 22.3% | 6.5% | 15.8% | 23.0% | **1534** |
| **Hybrid-HTbig** | **25.3%** | 6.8% | 18.5% | 16.6% | 1506 |
| Hybrid-Base | 24.2% | 6.9% | 17.3% | 16.0% | 1498 |
| **Hybrid-HTbase** | **25.3%** | 6.1% | 19.2% | **15.2%** | 1463 |

#### 第四轮：Hybrid-HTbase 教师 + 4000 局

教师改用第三轮得到的 `Hybrid-HTbase`，数据量从 1000 局扩大到 4000 局，训练同尺寸 base 网络。

**4000 局数据**：188772 样本，79671 条带 trace（~42%）。

**1000 局 benchmark**：Baseline / Hybrid-Base / Hybrid-HTbase / Hybrid-HT4k

| Agent | 胜率 | 自摸 | 铳 | 点炮 | Elo |
|---|---|---|---|---|---|
| Baseline | 24.0% | 6.3% | 17.7% | 21.8% | **1540** |
| Hybrid-Base | 25.0% | 6.7% | 18.3% | **16.8%** | 1498 |
| Hybrid-HT4k | 24.8% | 4.9% | 19.9% | 17.1% | 1532 |
| Hybrid-HTbase | 23.5% | 6.1% | 17.4% | 17.6% | 1430 |

#### 第五轮：纯 BeliefExp 教师 + 500 局 / 2000 局

教师改用纯 `BeliefExpectimaxAgent`（每步都搜索，不依赖 NN 快速决策），分别生成 500 局和 2000 局数据后训练同尺寸 base 网络。

**数据量**：500 局 22831 样本；2000 局 90735 样本，全部带 trace。

**1000 局 benchmark（500 局模型）**：Baseline / Hybrid-Base / Hybrid-HTbase / Hybrid-BE500

| Agent | 胜率 | 自摸 | 铳 | 点炮 | Elo |
|---|---|---|---|---|---|
| **Hybrid-BE500** | **25.5%** | 5.4% | 20.1% | **16.5%** | 1443 |
| Hybrid-Base | 24.3% | 6.1% | 18.2% | 17.9% | **1559** |
| Baseline | 23.7% | 5.9% | 17.8% | 21.0% | 1527 |
| Hybrid-HTbase | 23.2% | 6.1% | 17.1% | 17.8% | 1471 |

**1000 局 benchmark（2000 局模型）**：Baseline / Hybrid-Base / Hybrid-BE500 / Hybrid-BE2000

| Agent | 胜率 | 自摸 | 铳 | 点炮 | Elo |
|---|---|---|---|---|---|
| **Hybrid-BE2000** | **25.8%** | 5.2% | 20.6% | **16.4%** | 1469 |
| Hybrid-BE500 | 24.1% | 6.3% | 17.8% | 17.0% | 1502 |
| Baseline | 24.1% | 6.4% | 17.7% | 22.5% | **1535** |
| Hybrid-Base | 22.9% | 6.5% | 16.4% | 16.6% | 1494 |

**2000 局确认 benchmark**：Baseline / Hybrid-Base / Hybrid-HTbase / Hybrid-BE2000

| Agent | 胜率 | 自摸 | 铳 | 点炮 | Elo |
|---|---|---|---|---|---|
| **Hybrid-BE2000** | 24.2% | 5.1% | 19.1% | **16.8%** | **1609** |
| Hybrid-HTbase | 25.4% | 6.2% | 19.1% | 17.4% | 1546 |
| Baseline | 25.2% | 6.1% | 19.1% | 21.1% | 1481 |
| Hybrid-Base | 22.1% | 5.9% | 16.2% | 18.3% | 1364 |

### 分析

- **前两轮**：V3-NN-PC / V3-eval0 教师时，search distill 只能在 Elo 上略胜 Hybrid-Base，但胜率/点炮不占优，不稳定；
- **第三轮**：改用 **Hybrid-hybridBase critical trace** 后，效果质变：
  - `Hybrid-HTbase`：**胜率 25.3% vs Hybrid-Base 24.2%**，**点炮 15.2% vs 16.0%**，两项核心指标同时提升；
  - `Hybrid-HTbig`：胜率同样 25.3%，点炮 16.6%，Elo 更高；
  - 大网络（128/6/512）并未进一步拉开差距，说明当前数据量下 base-size 已足够；
- **第四轮**：数据量扩到 4000 局后：
  - `Hybrid-HT4k` 比其教师 `Hybrid-HTbase` 略有改善（胜率 24.8% vs 23.5%，点炮 17.1% vs 17.6%），说明更多数据缓解了过拟合教师；
  - 但 **Hybrid-HT4k 未超过 Hybrid-Base**（胜率 24.8% vs 25.0%，点炮 17.1% vs 16.8%），差距在噪声范围内；
  - 第三轮 1000 局观察到的“同时提升胜率/点炮”优势，在 4000 局复现中未能保持；
- **第五轮**：改用纯 **BeliefExp 教师** 后：
  - `Hybrid-BE500` 在 1000 局 benchmark 中**胜率 25.5%、点炮 16.5%**，同时优于 Hybrid-Base（24.3% / 17.9%）；
  - `Hybrid-BE2000` 进一步提升到 **胜率 25.8%、点炮 16.4%**（1000 局），并在大样本 **2000 局确认中 Elo 最高（1609）、点炮最低（16.8%）**；
  - 数据量从 500 局提升到 2000 局，val acc 从 0.773 提升到 0.812，核心指标持续改善；
- **第六轮**：数据量扩到 **4000 局**，并同步对比 base-size（96/4/256）与 big-size（128/6/512）网络：
  - `Hybrid-BE4k_base`：1000 局胜率 25.5%、点炮 16.5%、Elo 1512；2000 局确认胜率 24.9%、点炮 17.2%、Elo 1491；
  - `Hybrid-BE4k_big`：1000 局胜率 24.4%、点炮 **16.2%**、Elo 1438；**2000 局确认胜率 24.1%、点炮 16.8%、Elo 1570（最高）**；
  - big-size 在 4000 局数据上 val acc 达到 **0.837**，显著高于 base-size 的 0.821，且大样本 benchmark 中 Elo 最高；
- **最终 2000 局直接对比**（Baseline / Hybrid-Base / Hybrid-BE2k_base / Hybrid-BE4k_big）：

| Agent | 胜率 | 自摸 | 铳 | 点炮 | Elo |
|---|---|---|---|---|---|
| **Hybrid-BE4k_big** | 24.1% | 5.8% | 18.4% | 17.2% | **1572** |
| Baseline | 25.8% | 6.1% | 19.7% | 21.9% | 1522 |
| Hybrid-BE2k_base | 24.8% | 5.9% | 18.9% | **17.0%** | 1488 |
| Hybrid-Base | 22.8% | 6.0% | 16.8% | 17.5% | 1418 |

- **第七轮**：数据量进一步扩到 **8000 局**，并尝试调 α/T（big-size）：
  - 数据量：367635 样本；
  - `Hybrid-BE8k_big`（α=0.5, T=2）：val acc 0.858；
  - `Hybrid-BE8k_a07`（α=0.7）：val acc 0.864；
  - `Hybrid-BE8k_t4`（T=4）：val acc **0.867**；
  - **1000 局 benchmark**：Baseline / Hybrid-BE4k_big / Hybrid-BE8k_t4 / Hybrid-BE8k_a07

| Agent | 胜率 | 自摸 | 铳 | 点炮 | Elo |
|---|---|---|---|---|---|
| **Hybrid-BE8k_t4** | 24.7% | 7.3% | 17.4% | **15.5%** | **1520** |
| Hybrid-BE4k_big | 24.5% | 6.9% | 17.6% | 16.2% | 1516 |
| Baseline | 22.2% | 5.9% | 16.3% | 22.0% | 1515 |
| Hybrid-BE8k_a07 | 25.8% | 6.5% | 19.3% | 16.9% | 1449 |

  - **2000 局确认 benchmark**：Baseline / Hybrid-Base / Hybrid-BE4k_big / Hybrid-BE8k_t4

| Agent | 胜率 | 自摸 | 铳 | 点炮 | Elo |
|---|---|---|---|---|---|
| **Hybrid-BE8k_t4** | **25.7%** | 6.3% | 19.3% | **15.5%** | **1567** |
| Baseline | 24.4% | 6.2% | 18.3% | 22.1% | 1519 |
| Hybrid-BE4k_big | 25.6% | 6.2% | 19.4% | 17.9% | 1498 |
| Hybrid-Base | 21.6% | 5.2% | 16.4% | 17.8% | 1416 |

- **继续调 T（T=4/6/8）**：在 8000 局 big-size 模型上扫描蒸馏温度，2000 局确认显示 **T=8 最优**：

| Agent | 胜率 | 点炮 | Elo |
|---|---|---|---|
| **Hybrid-BE8k_t8** | **25.6%** | **16.0%** | **1618** |
| Hybrid-BE8k_t4 | 25.4% | 17.2% | 1520 |
| Baseline | 25.1% | 21.9% | 1506 |
| Hybrid-Base | 21.2% | 17.4% | 1357 |

- **第八轮**：数据量进一步扩到 **16000 局**，使用最佳 T=8 训练 big-size 网络：
  - 数据量：734073 样本；
  - `Hybrid-BE16k_t8`：val acc **0.874**；
  - **2000 局最终 benchmark**：Baseline / Hybrid-Base / Hybrid-BE8k_t8 / Hybrid-BE16k_t8

| Agent | 胜率 | 自摸 | 铳 | 点炮 | Elo |
|---|---|---|---|---|---|
| **Hybrid-BE16k_t8** | **25.8%** | 6.8% | 19.1% | **16.3%** | **1581** |
| Baseline | 24.4% | 6.3% | 18.1% | 21.2% | 1499 |
| Hybrid-BE8k_t8 | 25.7% | 6.5% | 19.2% | 17.2% | 1495 |
| Hybrid-Base | 21.4% | 5.5% | 16.0% | 17.6% | 1425 |

- 关键成功因素：**教师必须本身就在目标分布上强**；纯 BeliefExp 教师每步都搜索，提供了最一致、最强的 soft target；数据量充足后，**更大网络 + 更 soft target（T=8）能持续提升搜索信息蒸馏效果**。

### 结论

- **搜索轨迹蒸馏成功，且纯 BeliefExp 教师是当前最强教师**；
- **`Hybrid-BE16k_t8` 在 2000 局最终确认中胜率 25.8%、点炮 16.3%、Elo 1581，是当前最佳候选**；
- 16000 局数据 + big-size 网络 + T=8 是当前最佳组合，继续放大到 32000 局或继续调 T 的收益预计边际递减。

---

## 其他备选方向（记录备查）

- **Hierarchical Policy**（attack/balance/defend mode + tile head）
- **Population-Based Training（PBT）**
- **Global Reward Prediction**

这些方向在 conv-BC 速度优势下都更有可行性，但当前优先级低于上述三者。

---

## 文档更新记录

- 2026-07-01：整理自 `docs/reports/rl-ppo-report.md` §12–14 与用户讨论，确定三大主方向。
- 2026-07-02：补充 deal-in auxiliary loss、true perfect-info rollout oracle、NN+BeliefExp Hybrid 结果。
- 2026-07-03：补充 tenpai head 动作空间改造实现与中性结果。
- 2026-07-04：补充 Hybrid-hybridBase critical trace distillation 第三轮结果，Hybrid-HTbase 同时提升胜率/点炮，更新当前最佳配置。
- 2026-07-05：补充 4000 局 Hybrid-HTbase 教师缩放实验，Hybrid-HT4k 与 Hybrid-Base 互角，未进一步超越。
- 2026-07-05：补充 safety-aware 报听实验（无效）与纯 BeliefExp 教师 trace distillation（Hybrid-BE500 / BE2000 promising，更新当前最佳）。
- 2026-07-05：补充 8000 局 BeliefExp 教师 + big-size + T=4 实验，`Hybrid-BE8k_t4` 在 2000 局确认中胜率 25.7%、点炮 15.5%、Elo 1567，成为当前最佳候选。


---

## 未来方向评估（待探索）

在 conv-BC / Hybrid 框架基本收敛后，剩余可能突破天花板的三个方向：

### A. 增大网络容量

- 当前网络仅 ~82k 参数（channels=96, n_blocks=4, hidden=256）；
- 可尝试 channels=128/192, n_blocks=6/8, hidden=512；
- 预期 val acc 从 ~83% 提升到 85–87%， pure NN 更强，Hybrid 中搜索调用比例下降；
- 风险：小数据下过拟合、收益未必显著；
- 工作量：小。

### B. 多任务蒸馏：policy + value + BeliefExp 搜索轨迹

- **已完成初步验证**：用 V3-NN-PC 的 expectimax candidate scores 作为 soft policy target，训练出的 Hybrid-Search 在 1000 局中 Elo 最高（1556），但点炮高于 Hybrid-Base；
- 后续可继续：
  - 增大数据到 1000–2000 局；
  - 调优 α（KL 权重）、T（teacher 温度）、τ（dense value 缩放）；
  - 尝试 pairwise ranking loss 替代 KL；
  - 改用 Hybrid 教师只在 critical 状态记录 trace，匹配真实分布。
- 预期学生学到教师“为什么选 A 不选 B”，value 估计更准；
- 风险：教师 score 未校准、需要更多数据；
- 工作量：中。

### C. 动作空间 / 规则层面改造

- 当前 policy 只输出 34 维弃牌；报听、吃、碰、杠由硬启发式决定；
- **已完成最小改造**：报听 NN 化（tenpai head），效果中性；
- 后续可选：
  1. **Safety-aware / outcome-weighted 报听策略**；
  2. **扩展完整动作空间**（吃/碰/杠/pass）：工程量大，需改 engine；
  3. **引入 Hierarchical Policy**：先输出 mode（attack/balance/defense），再输出弃牌；
  4. **增加更丰富的对手模型特征**（听牌概率、待牌集合）。
- 风险：工程量大、可能破坏现有收敛性；
- 工作量：大。

### 优先级

| 方向 | 工作量 | 风险 | 潜在收益 | 推荐执行顺序 |
|---|---|---|---|---|
| B 搜索轨迹蒸馏 | 中 | 中 | 高 | **1（当前最有希望）** |
| A 增大容量 | 小 | 中 | 中 | 2 |
| C1 safety-aware 报听 | 中 | 中 | 中高 | 3 |
| C2/C3 完整动作空间 / Hierarchical | 大 | 高 | 高 | 4 |

当前决定：**纯 BeliefExp 教师 + big-size 网络 + 16000 局数据 + T=8 是当前最强组合**。`Hybrid-BE16k_t8` 在 2000 局确认 benchmark 中取得 **胜率 25.8%、点炮 16.3%、Elo 1581**。

**数据缩放实验（已完成）**：将 trace 数据从 16000 局放大到 **128000 局（8×）**，分别训练 128/6/512 与 192/8/768 网络。val acc 从 0.874 提升到 0.882–0.884，但 1000 局 benchmark 中 Elo 反而下降到 **1479 / 1456**，未超越 16000 局模型。说明在当前教师/架构/蒸馏目标下，**继续堆数据已边际递减甚至过拟合**。

后续可选：
1. **换教师**：用当前最强的 `Hybrid-BE16k_t8` 当教师生成更大规模 trace（生成速度比纯 BeliefExp 快，且分布更接近实战）；
2. **推理时加搜索**：MCTS/PUCT 或 deeper BeliefExp，用 conv-BC 当 prior/value；
3. **完整动作空间 / Hierarchical Policy**：把吃/碰/杠/报听统一 NN 化（**当前剩余最可能突破口**）；
4. ~~**RL 微调**：用 PPO 在自对弈中继续优化 Elo~~（已验证：从 BE16k_t8 热启 20 iter 后 Elo 1518，低于初始化，暂放弃）；
5. 若不再继续投入，当前最佳配置锁定为 `hybrid:BE16k_t8:output/nn_conv_bc_beliefexp_trace_16000_big_t8.pt:beliefexp`。

Safety-aware 报听（当前 dealin-head 风险估计实现）已证伪，不再继续。

如果用户不继续投入，当前保留 `Hybrid-BE16k_t8` 为最稳健配置。
