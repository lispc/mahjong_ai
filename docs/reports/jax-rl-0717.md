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
