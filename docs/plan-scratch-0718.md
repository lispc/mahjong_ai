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

{{SCRATCH_RESULT}}
