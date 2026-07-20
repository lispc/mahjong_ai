# 计划：BeliefExp→JAX 与防守知识蒸馏（2026-07-20）

> 状态：待批准。前置：eval2jax 已入环（`5c96c78`），gen4（eval2 对手池 AZ）训练中。
> 目标：把「带防守的搜索知识」变成可大规模生产的训练信号，打破顶层四和局
> （Baseline ≈ BeliefExp ≈ NM(old) ≈ Full(old)）。

## 0. 为什么做、做什么

S1 尸检：π' 部署死于**防守通道不可用**（胡可行性要真手牌、dealin 头 trunk 漂移死亡）。
BeliefExp 的决策里天然带防守（danger 表 + 安全让步规则）。JAX 化后它从「慢教师」
变成「批量知识生产者」。

**BeliefExp-lite 的精确定义**（勘察结论，比预想小）：
`BeliefExpectimaxAgent.next_with_trace` 的决策结构 = 
**eval2（context 感知版）进攻分 + tile_danger 危险分 + 安全 margin 规则**：
- 候选：unique 弃牌，eval0 预选 top-8；
- 每候选：eval2(hand13)（剩余分布含 `context.used`，与 Baseline 的空 Context 版不同！）；
- 有危险信号（对手报听 或 任一对手 danger_level≥1）时：在 offense ≥ best−margin 的
  候选中取 danger 最小者；否则 argmax offense；
- 响应镜像：不碰不杠、能胡必胡、报听按 `declare_tenpai` 启发式。
- **没有** belief 粒子、没有深递归——danger 表（`algo/eval/opponent.py`，146 行）
  是纯查表 + 弃牌序列统计，JAX 友好。

## 1. Phase 0：BeliefExp-lite JAX 移植（与 gen4 无关，立即开工，~0.5-1 天）

- **0a. context 感知 eval2 变体**：eval2jax 已验证的整数分子路径复用；
  剩余分布从「只看手牌」改为「4 − used − hand」（used = 全场已见，含弃牌/副露）。
  jaxenv State 的 discards/meld_counts 直接可得。
- **0b. danger 表 → JAX**：`_tile_base_safety`（字 0.35/幺九 0.25/二八 0.10/中 −0.10）、
  现物安全 1.0+0.1×seen、邻牌加成、花色权重（弃牌序列统计）、
  danger_level 0/1/2（近 6 巡安全度漂移 + 中张数）、`tile_danger` = 对手取 max。
- **0c. margin 规则 + 选择**（含 `defense_margin=0.03 + 0.02×报听对手数`）。
- **parity 门（必须过）**：从真实对局（BeliefExp 自打）采样 ≥500 个中间状态，
  JAX 版与 `BeliefExpectimaxAgent` 的 top-1 弃牌选择**完全一致**（eval2jax 同级标准）；
  danger 值逐值一致（容差 1e-9）。
- 产物：`jaxenv/beliefjax.py` + `jaxenv/test_beliefjax.py`；
  进 ppo.py 对手池 `OPP_BELIEF`（备用）。

## 2. Phase 1：数据生产 + 双头训练（~0.5 天，gen4 结论前后都可做）

- **1a. 状态+标签数据**：jaxenv 里用现 best 权重新开自对弈（或复用 gen4 rollout
  状态），对每条 DISCARD 状态记录：obs(175) + BeliefExp-lite 全标签
  （chosen tile、top-8 offense 分、danger 分、margin 选择、是否有危险信号）。
  产能目标 ≥1M 状态/小时。
- **1b. 双头训练**（GPU 0，小时级）：
  - **policy 蒸馏**（sanity 组）：CE(BeliefExp 选择)——预期落在 BeliefExp 强度档，
    只作管线验证，不作候选；
  - **danger head（核心交付物）**：frozen JAXG/old trunk 上重训 dealin 头
    （label = tile_danger 连续值 + 「该牌实际被和」二元标签双任务），
    修 S1 发现的 trunk 漂移死亡。

## 3. Phase 2：部署验证（裁决链，数小时）

- **2a. S1 复活测试**：`GumbelSearchAgent` 换 1b 的 danger head（替代死亡的 dealin 头）
  → pool 400 校准 → duplicate 1000p vs NM(old)。判据：点炮率回到 ≤15% 且胜率
  显著超纯 prior——回答「π' 部署缺的到底是不是真的只有防守」。
- **2b. 新网候选**：以 NM/full 形态 duplicate vs baseline / NM(old)，
  按 eval-protocol（1000p ≥+1.0% → 5000p → 独立种子复跑）。

## 4. Phase 3（仅当 gen4 证明 AZ 闭环有效）：gen5

BeliefExp-lite 作 in-loop improvement operator：gumbel π' 的 Q 接入 belief 防守项
（或直接用 BeliefExp-lite 的选择当 CE 目标），在同一 AZ 闭环里训练。

## 5. 分支逻辑（gen4 结果 → 路线调整）

| gen4 结果 | 调整 |
|---|---|
| 显著超 old（分布假说成立） | P0→P1 照走，**优先 P3**（环内加防守目标 gen5） |
| 仍不超（分布假说判死） | P0→P1→**P2 为主**（闭环死，走蒸馏+部署形态）；AZ 谱系彻底收官 |

## 6. 诚实边界

- god-mode 实测隐藏信息上界 +1.2%；顶层四和局 headroom 本来就小——**成功判据
  定为显著打破四和局（1000p ≥+1.0% 且 5000p CI 不含 0），不接受 pool 数字**；
- BeliefExp 在新 meta 已输 Baseline（+2.5% @5000）：我们提取的是它的**防守组件**
  （danger 表/让步规则），不是它的策略整体；若 P2 显示防守组件在新 meta 也不值钱，
  立即停手归档；
- 防守知识的替代品已有证据：NM 点炮 12.2% 已是全场最低——若新 danger head 超不过
  NM 的防守水平，同样不值得做。

## 7. 时间表

| 阶段 | 内容 | 预计 |
|---|---|---|
| P0 | 移植 + parity | 0.5-1 天（subagent 主力） |
| P1 | 数据 + 双头训练 | 0.5 天（GPU 0） |
| P2 | 部署裁决 | 数小时 |
| 首个判决 | 2a/2b duplicate | **~2 天** |
