# 方向 S1 调研报告：π'（Gumbel 1-ply 搜索）arena 部署 + NN 叶 vs eval2 叶 A/B

> 日期：2026-07-19
> 状态：**两个实验均完成，结论均为否定（有价值的负结果）**
> 关联：`docs/eval-protocol.md`、`docs/reports/ablation_report.md`、`docs/plan-0718.md`

---

## 0. 一句话总结

1. **NN 叶不优于 eval2 叶（arena 直接证据）**：同一 V3 搜索骨架下只换叶值，
   JAXG value 头最好成绩是打平（scale×100），×300 显著变差 −4.6%。
   「搜索强度上限 = 叶值质量」铁律在新一代 NN 上依然成立。
2. **π' 直接部署（S1）在 arena 被证伪**：每步 Gumbel 1-ply 搜索的 GumbelSearchAgent
   在 pool 400 仅 0.5% 胜率（纯 prior 对照 7.2–14.7%），消融显示 Q 重排单调有害。
   根因：**训练版 π' 的防守信息（真实手牌胡牌掩码）在部署时不可用**，
   部署版 Q 退化为纯进攻 1-ply 重排，把 prior 里蒸馏好的防守知识覆盖掉。

---

## 1. BeliefExp 叶 A/B（JAXG 叶 vs eval2 叶）

设置：`BeliefExpectimaxV3Agent`（depth=1，NN 候选 = JAXG），仅叶值不同；
`MJ_NN_LEAF_MODE=pure`，JAXG value（score/3 tanh）线性放大到搜索的 WIN_VALUE=100 尺度。
duplicate 1000-pair，对手三件套 baseline/beliefexp/hybrid:Base。

| 臂 | paired win diff | score-proxy | 判定 |
|---|---|---|---|
| JAXG 叶 ×300 | **−4.6% [−7.4, −1.8]** | −0.109 [−0.175, −0.043] | eval0 叶显著更强 |
| JAXG 叶 ×100 | −2.0% [−4.8, +0.8] | −0.066 [−0.131, −0.001] | win diff 平，proxy 弱显著偏负 |

- 原始数据：`output/ab_leaf_jaxg300_vs_eval0_1000.pkl`、`output/ab_leaf_jaxg100_vs_eval0_1000.pkl`
- scale 敏感本身说明 tanh 饱和区到搜索尺度的线性映射很粗糙；eval0 的向听/进张
  结构信号在「评估自家手牌赢面」岗位上仍然最合身。
- **后果**：S4（eval2-free Hybrid：BeliefExp 搜索层换 gumbel 搜索层）正式判死；
  任何「NN 叶替换 eval2 叶」的变体都不应再开。

## 2. S1：GumbelSearchAgent（π' 每步部署）

### 2.1 实现（已完成，可复用）

`algo/agents/gumbel_search_agent.py`：`PPOAgent` 子类只覆盖 `next()`。
每决策：确定性 top-k(k=8) → belief 世界(S=4)估 9 声明位概率（response head，
`sigmoid(logit_X − logit_pass)`）→ 7 分支（杠×3/碰×3/全pass）摸牌(n_draws=2)
跳值截断（value head；自摸 +1/放炮 −1/3，score/3 尺度）→ Q walk →
`argmax softmax(logits + β(Q−V))`，β=32。
防点炮通道：dealin head 聚合 hu 门（`MJ_GUMBEL_DEALIN=1`）。
token：`gumbel:LABEL:PATH[:K:DRAWS]`；环境变量 `MJ_GUMBEL_K/_DRAWS/_BETA/_S/_DEALIN/_CLAIMS/_DEBUG`。
成本 ~92ms/决策（CPU，批量前向：prior 1 行 + response ~3 行 + value 112 行）。

机制验证：确定性场景下 Q 与 Q walk 理论值精确吻合（p_hu=0.997 → Q=−0.332≈−1/3）。

### 2.2 Arena 成绩（证伪）

pool 400（gumbel / hybrid(best) / baseline / beliefexp）：

| Agent | win | deal-in | Elo |
|---|---|---|---|
| Hybrid-JAXG (best) | 43.8% | 15.0% | 1693 |
| BeliefExp | 27.3% | 15.5% | 1668 |
| Baseline | 26.5% | 18.2% | 1577 |
| **Gumbel-G** | **0.5%** | **24.2%** | **1063** |

通道消融（150 局 vs 3×Baseline；β=0 臂 = 同一 harness 的纯 prior argmax）：

| 配置 | win | deal-in | 含义 |
|---|---|---|---|
| β=0（纯 prior） | 14.7% | 20.7% | 基线 |
| β=8, dealin off | 10.0% | 22.7% | 搜索即有害 |
| β=32, dealin off | 8.7% | 26.0% | 单调恶化 |
| β=32, dealin on, claims off（纯 jumpv） | 7.3% | 27.3% | **进攻性 jumpv 本身有毒** |
| β=32, dealin on, draws=8, S=8（4× 采样） | 2.0% | 27.3% | 非方差，是偏差 |
| β=−32（反向对照） | 0.0% | 20.7% | Q 排序有正信号但不可用 |

### 2.3 根因分析

**A. 防守通道三元死亡**——训练版 π' 的防守信息部署时不可用：
1. *胡牌可行性掩码*：训练版用真实对手手牌精确判定；部署版均匀 belief 世界里
   随机 13 张成和概率 ~0.1%，S=4 下 p_hu 恒 0。
2. *dealin 头已死*：JAX 系训练（PPO/Gumbel 闭环）从未监督 dealin 头
   （`jaxenv/ppo.py` 只有 dealin 日志没有 loss），trunk 在闭环里持续更新而
   头权重冻结 → 特征漂移使输出退化为常数：arena 实测 sigmoid 恒 ≈0.31
   （p10=0.297, max=0.340，无任何区分度）。
3. *response head 的 hu 位*：训练时以真实手牌为条件；部署只能用采样手牌，
   同问题 1。

于是部署版 Q ≈ 纯进攻（1-ply 手牌价值最大化）。β=32 让它覆盖 prior 中
蒸馏自 π'（含防守）的排序 → 点炮率 27%，胜率随之崩塌。

**B. 进攻性 jumpv 重排本身有害**（纯 jumpv 7.3% < 纯 prior 14.7%）：
- 与 §1 叶 A/B 互为印证：JAXG value 头作为 1-ply 局面评估不优于既有信号，
  而 prior 本身已蒸馏了更深的搜索知识，1-ply 重排只增加 winner's-curse 噪声。
- 高采样更差（2.0%）证明是偏差而非方差；β=−32 为 0% 证明 Q 排序并非倒置，
  而是「有正相关但相对 prior 是降维」。

**C. 训练-部署不对称**（为什么训练环里 π' 有效）：
1b 闭环里 π' 目标用真实手牌掩码（防守精确），且蒸馏在数百万样本上平均了
单点噪声；部署每决策只取一个高噪声（且防守缺失）的样本。

### 2.4 结论与影响

- **S1 作为强度方向证伪**，不再进 duplicate 链（pool 0.5% 远低于筛查门槛）。
- 光谱菜单更新：「NN + 每步轻搜索」象限在缺少条件化防守通道时不成立；
  给它防守需要 belief 条件化（= BeliefExp 核心机器）或重训防守头，
  两者都会把「优雅」抵消掉。**Hybrid（NN + 触发式 BeliefExp）仍是光谱上
  强度/简洁的最优折中。**
- 若要复活（仅记录，不推荐现在做）：
  a. 冻结 JAXG trunk，在 arena 自对弈数据上重训 dealin/response 头（几小时）；
  b. belief 采样按对手弃牌/报听条件化（拒绝采样到 shanten 一致），
      实质是重做 BeliefExp 的 belief 层；
  c. 即便 a+b 都做好，上限也只是「prior + 防守微调」，进攻重排必须用小 β
     压住——预期收益远不抵复杂度。

## 3. 平台 accounting 规则（本次踩坑实录，任何 belief/采样类 agent 必读）

arena 引擎（`driver/engine.py`）与 ContextV3 的可见性规则：

1. **被碰/杠/胡的牌不广播 'put'**（engine.py:331-337 仅全 pass 才广播）：
   整组副露物理牌（碰 3/杠 4，含被 claim 的那张）对所有玩家的 context 不可见。
   belief 采样前必须用 'meld' 消息自行跟踪并剔除，否则牌池含幻影牌，
   某种牌计数 >4 时 `remaining_wall` 断言崩溃（本次 pool 首崩）。
2. **自己打出的牌被 claim 时 used 不撤回**：agent 在 `next()` 里已 `see_tile`，
   被 claim 后无 'put' 也不撤回 → 与副露物理牌重复计数（本次二崩：对手 gang-31
   而 used 还记着我们打的那张 31）。需在 'meld' 消息匹配时手动撤回，
   并用 'put' 关闭 claim 窗口（残留稀有巧合误撤回，已注释）。
3. **对手闭手数 = 13 − 3×副露数**（post-discard 口径；碰/杠均 −3）。
4. **`full_hand()` 每组副露只记 1 张**（`agent.py:19-24`），因此
   `algo.is_succ` 对有副露的手牌永远判负、且七对子不算胡——
   **arena 与 jaxenv（`rules.is_win_counts` 支持副露胡与七对）规则不一致**，
   影响所有 agent 与跨环境对比，属既有平台 quirk（非本次引入）。
5. 调试：`MJ_GUMBEL_DEBUG=1` 崩溃时落盘现场（cur/melds/meld_map/used/discards）；
   复现脚本 `tmp/gumbel_repro.py`。

## 4. 产物清单

| 产物 | 位置 |
|---|---|
| GumbelSearchAgent（可用，已修复 accounting） | `algo/agents/gumbel_search_agent.py` |
| `gumbel:` benchmark token | `scripts/rl/benchmark_pool.py` |
| 叶 A/B 原始数据 | `output/ab_leaf_jaxg{100,300}_vs_eval0_1000.pkl` |
| S1 pool 日志 | `output/pool400_gumbel_s1.log` |
| 诊断/复现脚本 | `tmp/gumbel_diag.py` `tmp/gumbel_repro.py` `tmp/gumbel_dealin_check.py` |
