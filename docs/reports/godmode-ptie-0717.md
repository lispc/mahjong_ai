# 方向 0 + 方向 2：god-mode 上界与 PTIE 完美信息 critic（2026-07-17）

> 结论先行：
> 1. **方向 0（god-mode 上界）**：完美隐藏手牌信息在 BeliefExp 结构内只值
>    **+1.2% [+0.3, +2.1]** 胜率（2000 pairs）；且 god-mode BeliefExp 仍比当前
>    best **低 7.5%**——信息通道已榨干，剩余空间不在「知道更多」。
> 2. **方向 2（PTIE）**：完美信息 critic val corr **0.2525** vs v1 不完美信息
>    critic 0.231——看见全部对手手牌几乎不改善 outcome 预测。**信用分配 SNR 的
>    瓶颈是游戏内在随机性（未来 ~70 步牌山），而非隐藏信息**；PerfectDou 的
>    PTIE 机制不迁移到晋北推倒胡。确认性 AWBC β=1.0 筛查结果为
>    **+0.3% [−0.8, +1.4]**（1000 pairs，score-proxy +0.002 [−0.021, +0.025]），
>    低于 +1.0% 预登记线，方向 2 关闭。
> 3. 对方向 1（JAX + KL 锚定 PPO）的含义：信息/价值两条通道的上界都被压到
>    ~1-2pp，PPO 训练按**证据门**执行（半程未见 >1% 改善即停），高速引擎本身
>    作为基础设施资产完成。

---

## 1. 方向 0：god-mode 上界测量

### 1.1 设计（预登记思路，见 web-research-directions-0717 §3 方向 0）

`scripts/rl/god_mode_upper_bound.py`。GodBeliefAgent = BeliefExpectimaxAgent +
两项完美信息升级（同进程 `_table` 注入，与 `oracle_endgame_gate.py` 同模式）：

- **精确剩余分布**：三家对手的闭手与副露计入 eval0/eval2 的 `used`，
  进攻分的摸牌概率从信念均匀变精确；
- **精确点炮规避**：每张候选弃牌直接调用各对手自己的 `respond_hu`
  （与引擎裁决完全一致，含 Hybrid 对手的 NN 响应头），非点和候选中取进攻
  最大；全部点和（被迫）时取进攻最大。

考场：duplicate 格式，同 seed 三路配对（pos 0 = God / BeliefExp / Hybrid-Best），
对手 = 标准三件套 `baseline,beliefexp,hybrid:Base:output/nn_full_action_best.pt`，
2000 seeds × 3 局 = 6000 局，32 workers 167s。
pkl：`output/god_mode_ub_2000.pkl`。

### 1.2 结果

| 候选（pos 0） | 胜率 | 点炮率 | score-proxy |
|---|---|---|---|
| GodBelief | 26.7% | **5.1%** | +0.346 |
| BeliefExp | 25.5% | 20.3% | +0.178 |
| Hybrid-Best | **34.2%** | 19.2% | +0.320 |

paired win diff（95% CI）：

| 对比 | diff | CI |
|---|---|---|
| God − BeliefExp | **+1.2%** | [+0.3, +2.1] |
| God − Hybrid | **−7.5%** | [−9.8, −5.2] |
| BeliefExp − Hybrid（sanity） | −8.7% | [−11.0, −6.4] |

paired score-proxy diff：God − BeliefExp **+0.168** [+0.143, +0.193]；
God − Hybrid +0.027 [−0.024, +0.078]（≈0）。

sanity 行与 eval-protocol §5 已知链条（arena 1000-pair Hybrid−BeliefExp=+7.1%）
一致，考场有效。

### 1.3 解读

- **防守信息真实但小**：完美信息把点炮 20.3%→5.1%、score-proxy +0.168，
  但胜率只 +1.2pp。与历史证据链完全一致：默听检测接入 +0.1%、终盘 oracle
  上界 0、防守特征全部失败——**「知道对手手牌」通道的总价值 ≤ ~1-2pp**。
- **God − Hybrid = −7.5pp**：god-mode 的 BeliefExp 仍远弱于当前 best。
  Hybrid 的 NN+搜索结构优势（+8.7pp over BeliefExp）远大于完美信息的增益。
  剩余空间只能在公开信息策略/结构里——而那里 15 候选刚零晋升
  （`selfplay-bootstrap-0717.md`）。
- GodBelief 是信息价值的**下界**估计（候选 top-8 由 eval0 预选、响应侧未用
  god、无喂碰规避），但 +1.2pp 的量级即使放大数倍也不改变结论。

## 2. 方向 2：PTIE 完美信息 critic + AWBC 确认

### 2.1 假设与预登记判读门

PerfectDou（NeurIPS 2022）的 PTIE：训练时 critic 看全部隐藏手牌、policy 部署
只看公开信息，从而压低 advantage 方差。本项目的对应假说：v1 AWBC 失败根因
是不完美信息 critic（corr 0.231）。判读门（预登记于
`scripts/rl/ptie_critic.py` docstring）：god critic val corr 应显著超过 0.231
（预期 0.5+）；否则 PTIE 前提不成立。

### 2.2 数据与 critic

- 采集：`scripts/rl/ptie_critic.py collect`，24,000 局（T∈{0.3,0.5,0.7} 轮转，
  与 v1 一致），96 workers 1.9 min，**1,656,830 弃牌步**，每步额外记录
  god(102,)=三家对手闭手 34 维计数（next/face/prev 顺序）。
  落盘：`output/ptie_v1_merged.npz`。
- critic：冻结 best 主干 → gfeat(261) + god(102) → MLP(512) → tanh，
  186,881 可训参数，MSE 拟合 score-proxy/3，4 epochs。

### 2.3 结果：判读门 FAIL

| critic | val corr | corr_decisive | 备注 |
|---|---|---|---|
| 模型自带 value head | 0.084 | 0.154 | v1 同一数字 |
| v1 critic（无 god） | 0.231 | — | `nn_bootstrap_v1_critic.pt` |
| **PTIE god critic** | **0.2525** | 0.340 | `nn_ptie_v1_critic.pt` |

校准桶单调（V=−0.101→R=−0.020 … V=+0.164→R=+0.107），但预测方差被压缩在
很窄区间——**即使看见全部隐藏手牌，outcome 的绝大部分方差来自未来牌山，
不可预测**。斗地主（PerfectDou 有效场景）~20-30 手结束、隐藏牌主导结局；
晋北推倒胡 ~70 步、未来摸牌主导结局——PTIE 机制不迁移。

### 2.4 确认性 AWBC（β=1.0，单候选）

A 统计与 v1 几乎相同（mean 0.0076 / std 0.178 / frac>0 50.8% vs v1 的
0.007/0.184/51.5%）；微调 val CE 0.164→0.141，argmax drift 1.8%。

1000-pair duplicate 筛查（vs anchor，标准三件套）：
**+0.3% [−0.8, +1.4]**（A 18 / B 15 / ties 96.7%），score-proxy +0.002
[−0.021, +0.025]——不显著，低于 +1.0% 预登记晋级线。与 v1 三候选
（−0.6%/+0.0%/−0.3%）同一形态。
pkl：`output/duplicate_ptieb10_vs_best_1000.pkl`。

### 2.5 结论

方向 2 关闭。PTIE/完美信息 critic 路径在晋北推倒胡不成立，证据双保险：
方向 0（实战信息价值 +1.2pp）+ 方向 2 critic 门（corr 0.253≈0.231）。
「信用分配 SNR」判词从「隐藏信息不足」**修正为「游戏内在随机性」**——
任何 outcome 级 advantage 估计的方差下限都由未来牌山决定，与信息量无关。
这也解释了 outcome 级 RL 在本项目五连失败的共同根因，且该根因不可消除。

## 3. 对方向 1 的证据门

方向 0+2 把信息通道与价值通道的上界压到 ~1-2pp。方向 1（JAX 引擎 +
KL 锚定 PPO）继续执行，但分两段：

1. **基础设施段（无条件完成）**：JAX 引擎 + 验证 + 吞吐 + Flax 权重移植。
   引擎本身是资产（评测/数据/未来研究提速 ≥10×）。
2. **PPO 段（证据门）**：先跑中小规模（≤50M steps），半程检查点若 vs BC 锚
   改善 <1% 即停并关闭方向 1；过线才放大。KL-to-BC 锚 + γ=1 + GAE 0.95
   （Mahjax 配方）。

## 4. 资产清单

- `scripts/rl/god_mode_upper_bound.py`（god-mode 配对实验框架，可复用于其他
  oracle 测量）；`output/god_mode_ub_2000.pkl`。
- `scripts/rl/ptie_critic.py`（collect 含 god 特征 / train_critic / finetune）；
  `output/ptie_v1_merged.npz`（1.65M 步 + god 特征）、
  `output/nn_ptie_v1_critic.pt`（god critic，corr 0.2525，可作方向 3 的叶值候选）、
  `output/nn_ptie_v1_awbc_b10.pt`。
- 结论更新：「信用分配 SNR」根因 = 内在随机性（非信息量）；
  信息通道上界 ≈ +1.2pp（胜率）/ +0.17（score-proxy）。

## 更新记录

- 2026-07-17：初版（方向 0 结果 + 方向 2 critic 门 + AWBC 筛查）。
