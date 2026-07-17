# 下一步方向评估：文档通读 + 网络调研（2026-07-17）

> 结论先行：项目的「想法枯竭」是表象。所有判死方向收敛到同一根因——**信用分配 SNR**
> （70 步牌运稀释单步决策因果），而它只在「小算力 + 不完美信息 critic + 无 KL 锚」的
> 配置下被证伪。2024–2026 年外界恰好在这个根因上产出了三套经过验证的工具
> （GPU 向量化环境 + KL 锚定 PPO、完美信息 critic/PTIE、分层决策架构），本项目一个都没
> 真正试过。本文档给出评估与执行计划；随后启动方向 0/1/2。

---

## 1. 项目自己证明了的「定理」（新方向必须绕开）

通读全部 25 份文档（`docs/handoff.md`、`docs/eval-protocol.md`、`docs/reports/*`、
`docs/designs/*`）后，判死方向的根因只有四条：

1. **搜索强度 = 叶值质量**。depth-2 教师、MCTS/DetMCTS 全家、AlphaZero bootstrap 三轮
   全死，因为 leaf 永远是 eval2 级（`search_distillation_report.md`、`td-lambda-plan.md`）。
2. **信用分配 SNR 不足**。PPO/DPO/KTO/AWBC/filtered BC 五次独立失败；critic corr 仅
   0.231；875 个「坏碰」误差状态在 175 维特征上不可分（AUC 0.638）
   （`selfplay-bootstrap-0717.md`）。
3. **防守信息不是瓶颈**。默听检测 AUC 0.92 但接入 +0.1% [−0.4,+0.5]；终盘 oracle 上界
   为零（`endgame-solver-ab-0716.md`、`silent-tenpai-d-0716.md`）。
4. **在位者近最优 + 评测噪声地板 ±1.3pp**。15 候选零晋升；soup→蒸馏最后一环是
   winner's curse（`duplicate-reanalysis-0716.md`）。

另一个未被足够重视的工程事实：**纯 Python 引擎 ~69 局/秒**（96 workers 自对弈），
所有 RL 实验都在比外界主流低 ~4 个数量级的数据规模下进行。

## 2. 网络调研关键发现

### 2.1 Mahjax（2026-05，东京大学/RIKEN）：麻将 RL 的吞吐革命

- JAX 全向量化日麻环境，8×A100 达 100–200 万步/秒，比 Rust 版 Libriichi 快 >10×。
- RL 配方是 RLHF 式：**BC 初始化 + PPO + 对冻结 BC policy 的 KL 惩罚（c=0.2）**，
  γ=1，GAE 0.95；单卡 GH200 跑 1 亿步（5.8h）稳定提升 rank。
- 对照本项目 PPO 判死配置：~300 万步、无 KL 锚、entropy 坍缩/发散——**失败配置与
  成功配置每一项都不同**，判死结论不迁移。
- 来源：https://arxiv.org/html/2605.20577v1 （代码开源 github.com/nissymori/mahjax）

### 2.2 Mortal 实证定律（最强开源日麻 AI）

- value-based offline RL（MC Q-learning，刻意不用 TD bootstrap）→ online 微调仅
  +0.6PT；**转 policy-based online 后提升「超出预期」**；配 GRP（Suphx 全局奖励预测）
  做信用分配。
- 与本项目 TD(λ) 判死自洽（value 学准了但 leaf 公式没法用），但「policy-based
  online + GRP + 大算力」这后半条路从未走过。
- 来源：https://github.com/Equim-chan/Mortal/discussions/91 、https://mortal.ekyu.moe/

### 2.3 PerfectDou（NeurIPS 2022）：完美信息蒸馏的正确形态（PTIE）

- 设计：**训练时 critic 看全部隐藏手牌，policy 只看不完美信息**，PPO+GAE 直接训。
  不需要「强 oracle 策略」——与项目已判死的策略级 oracle 蒸馏（把完美信息行为蒸成
  policy）机制完全不同；PTIE 只让 critic 看完美信息来**压低 advantage 方差**。
  AlphaStar 同一招（value net 看全局）。
- 直接对治本项目诊断：不完美信息 critic corr 0.231 → 完美信息 critic 预期 0.7+。
- 来源：https://ar5iv.labs.arxiv.org/html/2203.16406

### 2.4 Tjong（CAAI 2024）：分层决策 + fan backward

- 15M 参数 transformer，决策解耦为「动作类型 → 具体牌」两级；仅 0.5M 数据、2 卡
  7 天 SL，Botzone 前 1%。分层结构（conv-bc-roadmap C3，曾标注「剩余最可能突破口」）
  在麻将上有公开成功先例。
- 来源：https://www.semanticscholar.org/paper/a59c6fc2834fe252b5af6d9639d9dcb33fe98421

### 2.5 备选工具

- Gumbel AlphaZero（ICLR 2022 spotlight）：少至 2×动作数 次模拟即有策略改进保证，
  改变推理时搜索的经济学。https://iclr.cc/virtual/2022/spotlight/6419
- PQN（ICLR 2025）：去 replay buffer/target net 的极简 Q-learning，配向量化环境
  50× 于 DQN。https://arxiv.org/abs/2407.04811
- Nash Policy Gradient（2025，Mahjax 引用）：迭代精化正则项。
  https://arxiv.org/abs/2510.18183
- CFR 系（ESCHER/RL-CFR）对 4 人局仍不实用，不推荐。

## 3. 有潜力方向（按优先级）

### 方向 0：全局 god-mode 上界测量（1–2 天，最先做）

项目测过终盘 oracle 上界（=0）与碰的配对因果效应（+0.117），但**从未测过全程完美
信息对胜率的提升幅度**。用 god-state + 配对 rollout 基建，让完美信息 agent 与 Hybrid
打配对 duplicate。上界 +2~3pp → 在位者真到顶，其余方向不值得投；+10pp+ → 头部空间
真实存在，值得下重注。成本最低，直接决定后续一切的期望值。

### 方向 1：JAX 重写晋北引擎 + KL 锚定 PPO（2–4 周，核心投入）

- 晋北规则比日麻简单一个数量级：推倒胡无番符（胡牌判定 = 向听==−1）、不能吃
  （分支少）、报听锁手（自动播放）——向量化难度远低于 Mahjax，可改其开源代码。
- 4×3090 保守估计 20–50 万步/秒，**单卡一天可跑 1 亿步**（项目历史全部 RL 实验
  加起来不到这个量级）。
- 配方照抄 Mahjax：BC（`nn_full_action_best.pt` 强初始化）→ PPO + KL-to-BC 锚。
- 即使 RL 再失败，**高速引擎本身是资产**：5000-pair duplicate、god-mode 实验、
  教师数据生成全部提速 10 倍以上。

### 方向 2：完美信息 critic（PTIE）+ AWBC 重测（3–5 天，便宜且是方向 1 配套）

在现有 `selfplay_bootstrap.py` 管线上，采集时多存「其他三家手牌」（god-state 基建
已有），训看全信息的 critic，用其 advantage 重跑 AWBC/policy gradient。这是对
「信用分配 SNR」判词的最直接反驳实验：corr 0.7+ 下 AWBC 仍不动 → 根因坐实；
动了 → RL 线翻案。成本极低，信息价值极高；critic 还可给方向 3 供叶值。

### 方向 3（中期）：Gumbel-top-k 推理时搜索

n_sims 极小也有策略改进保证；配方向 2 的 critic 当叶值、现有 NN prior，可在 NN 每个
决策点做毫秒级有保证改进，替代/增强「28 弃牌触发 BeliefExp」的粗触发。先验中等，
依赖方向 2 的叶值质量。

### 方向 4（架构升级，可随时离线并行验证）

tokenized 事件序列 transformer + Tjong 式分层头。方向 D 已证明弃牌顺序携带计数特征
没有的信息（seq AUC +0.09~0.18），失败的是「危险信号挂钩」接入方式而非信息本身。
第一步在现有 32k/128k BC 数据上纯离线验证（val acc、点炮校准），不进 arena。

### 方向 5（产品决策，非算法）：接入真实计分（报听/自摸加成）

引擎目前不计分、Hybrid 从不报听。接入后终盘求解器、报听决策、对手建模等一批已判死
结论需全部重估——唯一能「复活」死方向的元改动。服务真实晋北对局目标，与当前 arena
目标不同，需先定项目定位。

## 4. 明确不建议再投的

- 无 KL 锚 outcome 级 RL 第 6 次尝试（PPO/DPO/KTO/AWBC 原样重来）；
- 继续 soup / BC 缩放 / bootstrap 蒸馏（两轮边际递减 + winner's curse 实锤）；
- 防守侧对手建模接入（终盘 oracle 上界为零 + 默听接入不复现）；
- MCTS/AZ 式大模拟数搜索（叶值问题不解决前免谈）；
- 跨规则外部数据（无晋北牌谱，方向 C 已判死）。

## 5. 执行计划（已获批准，2026-07-17 启动）

| 方向 | 内容 | 预期产出 | 状态 |
|---|---|---|---|
| 0 | god-mode 完美信息上界测量 | 上界数字（paired diff ± CI） | 启动 |
| 2 | PTIE critic + AWBC 重测 | critic corr、AWBC 1000-pair 筛查 | 启动 |
| 1 | JAX 引擎 + KL 锚定 PPO | 引擎吞吐、PPO 训练曲线 | 启动（长工程） |

顺序：方向 0 与 2 并行（CPU 自对弈资源错峰），方向 1 作为长工程后台推进。
方向 3/4 视方向 0/2 结果决定是否启动；方向 5 待项目定位决策。

## 6. 调研参考（上文已内联）

Mahjax（arXiv 2605.20577）、Mortal（Equim-chan/Mortal）、PerfectDou（NeurIPS 2022,
arXiv 2203.16406）、Tjong（CAAI T-INT 2024）、Gumbel AlphaZero（ICLR 2022）、
PQN（ICLR 2025, arXiv 2407.04811）、Nash Policy Gradient（arXiv 2510.18183）、
Evo-Sparrow（arXiv 2508.07522，CMA-ES 路线，先验低）、AlphaZe**（Frontiers 2023，
determinization 路线，项目 DetMCTS 已判死）。
