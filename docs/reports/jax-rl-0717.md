# 方向 1：JAX 高速引擎 + KL 锚定 PPO（2026-07-17，已关闭：不晋升）

> 状态：M1–M5 全部交付并验证；M6（PPO pilot 50M 步）完成——1000-pair 筛查
> +1.2% 过线、5000-pair 独立种子确认 +0.2% [−0.5,+0.9]（winner's curse），
> **不晋升，方向 1 关闭**；`jaxenv/` 作为基础设施资产留存。
> 前置：`docs/reports/web-research-directions-0717.md`（方向评估）、
> `docs/reports/godmode-ptie-0717.md`（方向 0/2 结果与证据门由来）。

---

## 1. 动机与证据门（预登记）

- Mahjax（arXiv 2605.20577）证明：麻将 RL 在「BC 初始化 + PPO + 对冻结 BC 的
  KL 惩罚 + 100M 步」配置下稳定提升；本项目历史 PPO 判死配置（~3M 步、无 KL 锚、
  entropy 坍缩）与其每一项都不同，判死结论不迁移。
- 方向 0/2 把信息通道与价值通道上界压到 ~1-2pp，因此 PPO 段按**证据门**执行：
  **pilot 50M 步，半程（iter ~190/380）eval margin vs ref < +1% → 提前停并关闭**；
  过线则放大，最终晋升仍走 `docs/eval-protocol.md`（1000-pair 筛查 → 5000-pair +
  独立种子复跑，晋级线 +1.0%）。

## 2. M1–M3：JAX 晋北引擎（`jaxenv/env.py`，已验证）

- Pgx 风格函数式环境：`init(rng)/legal_mask(state)/step(state,action)`，flax struct
  State，全 `lax.cond/switch`，可 jit/vmap。动作 40 维（弃牌 34 + pass/碰/杠/胡 +
  报听 yes/no）。
- 规则逐条对齐 `driver/engine.py`：杠后牌山**尾**补牌、碰后不摸牌、自摸自动胡、
  声明按 胡→杠→碰 × 逆时针 offset 1→3、锁手强制打出摸牌、锁手可胡不可碰杠、
  摸空流局；**刻意对齐 arena quirk**：仅无副露且弃牌后向听==0 提供报听决策点。
- 胡牌/向听：5⁹/5⁷ 计数向量预计算查找表（`jaxenv/tables.npz` 14.2MB）。
- **验证全绿**：is_win 100k 例 0 失配、shanten 100k 例 0 失配（vs `algo/eval/v2.py`）；
  场景测试 9/9；不变量 260 局（牌守恒等逐步断言）；分布对比 500 局/侧与 Python
  引擎一致（局长 +0.17%、流局 +0.2pp，容差内）。
- **吞吐：547k steps/s @batch4096（单 3090）**，约为 Python 自对弈（~8k steps/s）
  的 ~68×；jit 编译一次性 ~100-230s。
- 已知语义差（训练 env 内公平，晋升决策在 Python arena 不受影响）：JAX 胡牌含
  七对子（v2 语义），Python 引擎 `is_succ` 不含；JAX 实现正确副露规则，未复制
  Python 基类 agent 的副露记账 quirk。

## 3. M4：Flax 移植（`jaxenv/model_flax.py`，已验证）

- linen 版 TileConvNet（128ch/6blocks/512hidden，policy/value/dealin/tenpai/response
  五头）；`output/nn_full_action_best_flax.msgpack`。
- **数值对齐：1024 条真实特征全 head max abs diff < 1e-4**（policy 8.5e-05）；
  推理 200k samples/s（单 3090）。坑：Ampere 默认 TF32，parity 需
  `jax_default_matmul_precision='highest'`（训练不需要）。

## 4. M5：观测对齐 + PPO 脚本（`jaxenv/obs.py`、`jaxenv/ppo.py`，已验证）

- `observe(state)`：175 维特征与部署 `PPOAgent` 严格对齐，**1200 状态 max abs
  diff 5.96e-08**。复刻两个部署 quirk：副露牌种 ×3（Hybrid 共享 melds 列表）、
  报听决策时 pending 已入自己 used。
  - 已知残余角落：锁手玩家自己被问胡那一刻的特征偏差（引擎锁手弃牌不经
    see_tile；env State 无「报听时弃牌数」字段），影响可忽略。
- PPO（Mahjax 配方）：N_ENVS 场 vmap 自对弈，4 座位共享 policy，冻结 ref 做
  KL 锚（coef 0.2，三头 masked KL）；γ=1、GAE λ=0.95（按玩家决策子序列）、
  clip 0.2、vf 0.5、ent 0.01、Adam 3e-4；score-proxy 奖励（env 内置）。
- 吞吐：512 envs × 256 steps ≈ **6100 decisions/s**（含训练与 host 侧 GAE）；
  50M 步 ≈ 2.3h。convert_back roundtrip 72 tensors 0 diff，PPOAgent arena smoke 通过。

## 5. M6：PPO pilot（训练中，2026-07-17 15:40 启动）

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python3 jaxenv/ppo.py \
  --iters 380 --n-envs 512 --t-steps 256 --lr 3e-4 --kl-coef 0.2 --ent-coef 0.01 \
  --init output/nn_full_action_best_flax.msgpack \
  --out-dir output/jax_ppo_pilot --eval-every 10 --eval-games 512 --save-every 40
```

- 规模：380 iters × 131k decisions ≈ 50M 步；每 10 iters eval（当前 argmax vs ref
  argmax，1v3/3v1 各 512 局），指标写 `output/jax_ppo_pilot/metrics.jsonl`。
- 监控点：eval win_diff 趋势（证据门）、流局率 >55%（坍缩信号）、kl/entropy 异常。
- 结束后：`jaxenv/convert_back.py` 导出 .pt → 1000-pair duplicate 筛查
  （候选 vs anchor，标准三件套）。

## 6. 结果（2026-07-17 晚，方向 1 关闭，不晋升）

### 6.1 pilot 训练（全程健康，首个未坍缩的在线 RL）

- 380/380 iters 完成，**49.8M decisions / 2.16h / 6392 dec/s**（单 3090，
  与其他租户共享无影响）。
- 训练健康度：KL 稳定 0.034-0.039（锚工作正常）、entropy ~0.185 平坦、
  流局率 ~1%（v1 自对弈曾 50%+，KL 锚 + 强 BC 初始化彻底避免坍缩）、
  loss/value 平稳。历史 PPO 的两种失败模式（坍缩到消极均衡、entropy 发散）
  均未出现——**Mahjax 配方（BC 初始化 + KL 锚 + 大规模）机制验证通过**。
- in-env eval（当前 argmax vs ref argmax，各 512 局/10 iters）：margin vs 公平
  份额稳定 **+2~+4pp**（最终 10 次均值 1v3 +3.6pp / 3v1 +3.0pp），
  但 iter 10 → 380 **无增长**（快速 plateau）。

### 6.2 arena 裁决（eval-protocol 全流程）

| 阶段 | pairs | paired win diff | score-proxy | 判定 |
|---|---|---|---|---|
| 筛查（seed-offset 0） | 1000 | **+1.2% [−0.3, +2.7]** | +0.018 [−0.019, +0.055] | 过 +1.0% 线 |
| 确认（seed-offset 10000） | 5000 | **+0.2% [−0.5, +0.9]** | +0.005 [−0.012, +0.022] | **不晋升** |

- 筛查命中在独立种子 5000-pair 下蒸发——**winner's curse**（本项目第三次：
  默听接入、soup→蒸馏、本次；协议的多重比较防护第三次实战拦截成功）。
- pkl：`output/duplicate_jaxppo380_vs_best_1000.pkl`、`output/duplicate_jaxppo380_vs_best_5000.pkl`。

### 6.3 结论与归因

**方向 1 关闭，不晋升。** 机制上首次跑通了稳定的在线 RL（50M 步无坍缩），
in-env 对纯 NN ref 有 +3pp 优势；但该优势**不穿透部署结构**——Hybrid 的 NN
只决策非关键状态，PPO 改进的状态类与 BeliefExp 搜索层覆盖的状态类高度重叠。
与方向 0（信息通道 ≤1.2pp）、方向 2（价值通道 SNR=内在随机性）共同构成
完整证据链：**当前 best 在其结构内已接近可达上界，公开信息策略空间内的
增量改进通道全部 ≤1-2pp 且被搜索层覆盖。**

### 6.4 资产（基础设施，长期有效）

- `jaxenv/`：JAX 晋北引擎（547k steps/s）、Flax 模型移植、obs 对齐、
  PPO+KL 锚训练器、双向权重转换——未来任何大规模 RL/自对弈/评测实验的底座；
- `output/jax_ppo_pilot/`（全部 msgpack + metrics）、
  `output/jax_ppo_pilot_iter380.pt`（未晋升，留存备查）。

### 6.5 未做的事（诚实记录）

- 未做 100M+ 放大：margin 自 iter 10 即 plateau 且无增长趋势，放大无依据；
- 未做 checkpoint 挑选（iter360 等）：协议禁止多重比较；
- 未接入 GRP/fan-backward 式中间奖励：与 score-proxy 设计冲突，且方向 2
  已证价值通道 SNR 受限；
- 训练 env 与 arena 的已知语义差（七对子、副露胡牌 quirk）未对齐——
  晋升决策在 Python arena 做出，不受影响。

## 更新记录

- 2026-07-17：M1–M5 交付记录 + pilot 启动。
- 2026-07-17 晚：pilot 完成，5000-pair 确认 +0.2%，方向 1 关闭（不晋升）。

---

# 附录：方向 1b —— Gumbel-in-the-loop（2026-07-17 晚启动，预登记）

## 动机

方向 1 pilot 的 plateau 根因 = 目标只有终局结果（SNR 受内在随机性限制，方向 2 已证）。
AlphaZero 的核心机制不是自对弈本身，而是**用搜索产出每步的密集目标**。Gumbel
AlphaZero（Danihelka et al., ICLR 2022）保证极少模拟（≥2/候选）即有策略改进。
本方向在 jaxenv 自对弈环里加入**每个弃牌决策的 Gumbel-top-k 1-ply 搜索目标 π'**，
把 PPO 的 outcome 目标替换/增强为搜索蒸馏目标。

## 设计（预登记）

- 搜索（仅 DISCARD 决策点；CLAIM/TENPAI 沿用 net heads）：
  prior logits + Gumbel 噪声取 top-k=8 候选；每候选 Q = 1-ply 期望：
  **声明判定用真实状态精确计算**（env 本来就拿着全信息裁决——这不是 god-mode
  play，是真实动力学）；可声明对手的响应用 response head 概率期望；
  之后摸牌按真实余牌分布采样 D=2 次，V(下一状态)（value head，score/3 tanh）。
  改进策略 π' = masked softmax(logits + β(Q − V))。
- 训练：CE(π') 替换 PPO clipped surrogate；value MSE（outcome）与 KL 锚（0.2）
  不变；**自对弈用搜索后策略出牌**（AZ 闭环，targets 随策略共同进化）。
- 关键合法性说明：搜索只用「部署时也会发生的真实动力学 + net 策略」，不引入
  god-mode 最优行为；训练期可见全信息（PTIE 式目标生成）不影响部署 obs。

## 证据门（预登记）

- G0（工程门）：smoke 通过且吞吐 ≥500 decisions/s，否则降 k/D。
- G1（早期中止门）：iter 1-5 的 prior-vs-π' argmax 一致率 >95% → 搜索无增量，中止。
- G2（pilot 门）：~10-15M searched decisions（或 12h 上限）。final in-env margin
  需 ≥ +5pp（pilot 的 +3pp 基线之上才说明密集目标有增量）；随后 arena 1000-pair
  ≥ +1.0% → 5000-pair 独立种子确认（协议全流程）。

## 结果（2026-07-18 凌晨，**晋升**——项目史上首个 RL 来源晋升）

### 训练（全部健康）

- 92/92 iters 完成，**12.06M searched decisions / 3.68h / 911 dec/s**（单 3090）。
- agree 0.58→0.65 缓升（搜索信号持续未枯竭）、KL 稳定 ~0.45（锚定新均衡）、
  entropy ~0.85、流局 <1%——AZ 自改进签名全部符合。
- in-env eval：final-10 avg margin **1v3 +5.7pp / 3v1 +4.8pp**（G2 合并 +5.3pp
  达标；对照：方向 1 outcome pilot 同口径 +3.0pp）。

### arena 裁决（eval-protocol 全流程，**晋升链完整**）

| 阶段 | seed-offset | pairs | paired win diff | score-proxy |
|---|---|---|---|---|
| 筛查 | 0 | 1000 | +1.6% [−1.2,+4.4] | +0.003 [−0.061,+0.067] |
| 确认 | 10000 | 5000 | **+2.8% [+1.5,+4.2]**（≥2×SE ✓） | +0.054 [+0.023,+0.084] |
| 复跑 | 20000 | 5000 | +1.2% [−0.1,+2.5]（符号一致 ✓） | +0.022 [−0.008,+0.052] |
| **合并** | — | 11000 | **+2.0% [+1.1,+2.9]** | — |

- **晋升**：§3.1 两条件全满足（5000-pair CI 不含 0 且 ≥2×SE；独立复跑符号一致）。
  guardrail：pool 400 点炮 20.5% vs 18.2%（+2.3pp < +3pp 暂缓线，黄旗记录）。
- pkl：`output/duplicate_jaxgumbel_vs_best_{1000,5000,5000b}.pkl`。
- **新 best**：`output/jax_gumbel_iter92.pt`（Hybrid-FullAction-Gumbel92，
  部署形态同前：NN + BeliefExp 搜索层，`hybrid:JAXG:output/jax_gumbel_iter92.pt`）。

### 结论

1. **搜索目标突破了 outcome 天花板的机制被判活**：方向 1（outcome PPO）+0.2%
   → 方向 1b（Gumbel 搜索目标 AZ 闭环）+2.0%（11000 pairs）。密集目标确实
   提供了 outcome 信号里没有的信息，方向 2 的「SNR 判词」对搜索目标形态
   不成立（搜索把未来动力学折算进每步标签，绕开纯结果噪声）。
2. agree 仍 0.65、KL 平衡未上移——**自改进空间未尽**：下一步 = 移动锚第二代
   （anchor←iter92 继续训），AZ 迭代正式开环。
3. 训练/arena 语义差（七对子、副露 quirk）存在但未阻断晋升——晋升在 Python
   arena 实测定论。

{{NEXT_ROUND_RESULT}}
