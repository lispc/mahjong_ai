# Plan Scratch 0718：From-Scratch 优雅管线（目标 = 简单漂亮，非性能）

> 2026-07-18。目标变更（用户拍板）：**不再追求更强性能，追求最简单漂亮的算法**——
> 理想是 from-scratch（无教师数据）且不依赖 eval2 启发式评估函数。
> 本文是目标定义、可行性论证与里程碑门（预登记）。

---

## 1. 目标形态（优雅账本）

**部署时**：一个 TileConvNet，一次前向，出弃牌/响应决策。
没有：搜索层、eval2、BeliefExp、dealin/tenpai 头、手工阈值（tenpai_threshold 等）。

**训练时**：一个闭环脚本——JAX 环境 + 网络 + gumbel 搜索目标，
**随机初始化进，模型出**。没有：教师、数据生成阶段、BC、soup、蒸馏。

**三个简化决策（依据充分）**：
1. **hu 自动化**：训练环境中「能胡必胡」（自摸本就自动）。实测部署网络 hu take 率
   ~100%——这是删决策维度而非加规则。
2. **报听决策删除**：当前引擎报听只有代价没有收益（AGENTS.md #11：Hybrid 从不
   报听），训练环境恒否——删 tenpai 头。
3. **保留的辩白**：规则层查找表（shanten/win 是规则非启发式）；obs 175 维
   （原始计数+flag，无手工推导）；gumbel 1-ply 搜索（**仅训练期组件，部署不需要**）。

## 2. 可行性论证（为什么这次 from-scratch 可能成）

冷启动历来死于「无信号」（outcome 稀疏 + 随机 play ≈ 全流局）。gumbel 目标
有两个**与网络质量无关的即时真值源**：

- **精确声明判定**：弃对手能胡的牌 → Q=−1/3（从 iter 1 起 π' 就在教「别点炮」）；
- **自摸检测**：jump 摸到胡牌 → +1。

即**防守信号从第一步免费**；进攻靠 V 从结局学（12M 步 ≈ 100k 局结局，足够）。
这是 from-scratch PPO（纯 outcome）从未有过的待遇。

**锚**：无 BC 可锚 → NPG 式移动锚（`--anchor-refresh 16`，KL 锚上一代，coef 0.2）。
**奖励**：score-proxy + draw −0.25（`score_dd`；预注册：防冷启动流局停滞——
项目史上 draw=0 坍缩教训）。

## 3. 里程碑门（预登记，逐级放行）

| 门 | 标准 | 预算 |
|---|---|---|
| M1 | 12M 步后：eval 流局率 <50% 且 vs shanten-greedy margin > 0 | ~4h |
| M2 | 50M 步后：arena pool 400 胜率 ≥ Baseline（**eval2 系首次被无 eval2 模型追平**） | ~17h |
| M3 | （可选放大 100-200M）duplicate 挑战 BeliefExp / Hybrid-JAXG | 1-3 天 |
| 放弃门 | 12M 步后流局率仍 >70% 或对 greedy 无显著优势 → 冷启动失败，退回分析；次优解 = 5% 教师数据 1 代 bootstrap 再放手（记录，不算达成目标） | — |

预期 odds（诚实版）：M1 ~70%；M2 ~50%（规则简化使状态空间比日麻小一个量级）；
M3 不打包票（纯 NN 在本项目从未达到搜索层强度）。

## 4. 实现清单

- `jaxenv/env.py`：新增 reward kind `score_dd`（流局全员 −0.25，score/3 尺度）；
- `jaxenv/ppo.py`：`--reward-kind`、`--anchor-refresh N`（移动锚）、
  rollout 强制动作（CLAIM-hu 阶段强制 hu、TENPAI 阶段强制 no）、
  强制决策不进训练样本（GAE keep 排除）、默认 eval 增加 vs greedy；
- `jaxenv/random_init.py`：随机初始化 msgpack 生成；
- `algo/agents/auto_hu_ppo_agent.py`：部署 wrapper（respond_hu=能胡必胡、
  declare_tenpai=False），arena 评测用（M2+）；
- 训练命令（M1 pilot）：
  ```bash
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python3 jaxenv/ppo.py \
    --target-mode gumbel --search-beta 32.0 --reward-kind score_dd \
    --anchor-refresh 16 --iters 92 --n-envs 512 --t-steps 256 \
    --init output/jax_scratch_init.msgpack \
    --out-dir output/jax_scratch_gen1 --eval-every 8 --eval-games 512 \
    --save-every 20 --seed 10
  ```

## 5. 结果（进行中）

### M1（2026-07-18 晚，**通过**）

- 92 iters / 12.06M decisions / 3.7h（913 dec/s，GPU1）。
- **流局率 97.9% → 1.8%**（S 曲线拐点在 iter 16-24，门 <50% 富余两个量级）；
- **vs shanten-greedy cur_win 1.8% → 37.9%**（显著高于公平份额 25%，
  1v3 win_diff 收敛到 −0.12）——从零学会牌效+防守仅用 ~9M 决策；
- 移动锚（anchor-refresh 16）全程正常；kept ~108k/iter；无坍缩。
- 结论：冷启动可行性的核心论证（精确声明/自摸真值从第一步免费）成立。
- 产物：`output/jax_scratch_gen1/iter92.msgpack`（M2 起点）。

### M2（2026-07-19 凌晨启动，进行中）

- 同配置续训 380 iters（50M decisions，~14h），init=M1 iter92，seed 11。
- 门：arena pool 400 胜率 ≥ Baseline（eval2 系首次被无 eval2 模型追平）。
- 部署评测 token 已备：`autohu:LABEL:PATH`（benchmark_pool，AutoHuPPOAgent）。

### M2（2026-07-19 下午，**失败——不迁移到 eval2 对手世界**）

- 训练完成：380 iters / 49.8M decisions / 13.9h（987 dec/s）。in-env 曲线健康：
  vs greedy cur_win 37.9% → 50.8%（反超三家合计，win_diff +0.08）。
- **arena pool 400**（autohu 部署形态，vs Baseline/BeliefExp/Hybrid-JAXG2）：
  **胜率 2.5%、点炮 30.2%**（Baseline 26.0%）——M2 门 FAIL。
- 分解诊断（各 400 局）：vs 3×Baseline **4.8%**（点炮 24.2%）、
  vs 3×BeliefExp **4.8%**（点炮 27.8%）——不是 pool hostile 假象。
- **根因（按贡献排序）**：
  1. **训练分布不含 eval2 系对手**：自对弈全是 NN 风格（+greedy 牌效启发式），
     防守是针对 NN 式进攻学的；eval2 系的报听压迫/弃牌分布从未见过——
     点炮率从 in-env ~11-16% 爆到 arena 24-30%。
  2. **特征 OOD**：训练环 `no_tenpai` 强制 → 报听 flag 恒 0；arena 里 BeliefExp
     报听后 flag=1，网络从未见过该输入模式。
  3. 规则语义差（JAX 七对子胡、arena `is_succ` 不含；较小）。
- 结论修正：冷启动学习**在自对弈分布内完全成功**（M1 成立），但
  「in-env 强度 ≠ arena 强度」第二次实证（第一次：plan-0718 任务 B）。
  部署分布必须进训练分布——这是 from-scratch 管线要补的下一课。

### 后续可选（M2'：修分布再战，成本半天工程 + ~10h 训练）

1. 训练环允许对手报听（greedy 恒 yes + NN 对手继承 tenpai 决策）→ 特征覆盖
   flag=1 状态；
2. 保持/加强对手多样性（多代 NN + greedy）；
3. 从 iter380 续训 30-50M，arena 复测。
风险：差距 4.8%→26% 很大，eval2 风格差距不只在 tenpai flag；更强的修补需要
eval2 系对手进环（JAX 移植 eval2，工程贵）。建议：若做，只做这一次便宜迭代，
不追加。

{{M2_PRIME_RESULT}}
