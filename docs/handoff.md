# Handoff：换机器继续工作的指南

> 本文档是项目状态的**精简版快速入口**。完整实验历史见 `docs/reports/project_history.md`。

---

## 1. 当前最强配置

```python
# benchmark token: hybrid:JAXG:output/jax_gumbel_iter92.pt
# 对应类：algo.agents.hybrid_nn_belief_agent.HybridNNBeliefAgent
from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent

HybridNNBeliefAgent(
    'Hybrid-FullAction-Gumbel92',
    nn_model_path='output/jax_gumbel_iter92.pt',
    belief_kind='beliefexp',
    tenpai_threshold=28,
    device='cpu',
)
```

对应模型：`output/jax_gumbel_iter92.pt` + `output/jax_gumbel_iter92_config.json`
- `TileConvNet`，128 channels / 6 residual blocks / 512 hidden
- 带 dealin / value / tenpai / response head
- 来源：**JAX 引擎自对弈 + Gumbel-top-k 1-ply 搜索目标 AZ 闭环（12M decisions）**，
  从 `nn_full_action_best.pt` 出发经 KL 锚 PPO 训练（`jaxenv/ppo.py --target-mode gumbel`）
- 晋升证据（11000 pairs 合并）：**vs 旧 best +2.0% [+1.1,+2.9]**（score-proxy
  +0.054 [+0.023,+0.084] @5000），协议全流程通过（`docs/reports/jax-rl-0717.md` 附录）

旧 best（2026-07-17 前）：`output/nn_full_action_best.pt`（model soup + 蒸馏，
`docs/reports/project_history.md` §6.11–§6.12）。

400 局同一 pool 参考结果：

| Agent | win | self | ron | deal-in | draw | Elo |
|---|---|---|---|---|---|---|
| Hybrid-newbest (SoupDist.) | 34.2% | 7.5% | 26.8% | 15.5% | 0.5% | 1629 |
| Hybrid-oldbest | 28.2% | 8.2% | 20.0% | 19.5% | 0.5% | 1519 |
| BeliefExp | 19.2% | 6.0% | 13.2% | 16.0% | 0.5% | 1502 |
| Baseline | 17.8% | 5.8% | 12.0% | 21.0% | 0.5% | 1350 |

---

## 2. 环境与依赖

当前机器 base 环境即可：
- Python 3.13 + torch 2.12+cu126 + CUDA + 4×RTX3090
- 旧 conda 环境 `mahjong`/`pypy39` 已不存在

```bash
PYTHONPATH=. python3 run_tests.py
PYTHONPATH=. python3 tmp/benchmark_new_models.py 100 4
```

关键依赖：`torch`, `numba`, `numpy`, `cython`。

---

## 3. 常用命令速查

```bash
# 测试
PYTHONPATH=. python3 run_tests.py

# 4 GPU benchmark
bash scripts/benchmark_4gpu.sh 400 4

# 任意 4 agent 同一 pool
SEATS="hybrid:newbest:output/nn_full_action_best.pt:beliefexp,beliefexp,baseline,v3nnpc" \
    PYTHONPATH=. python3 scripts/rl/benchmark_pool.py 400 16

# Duplicate（复式）评测
PYTHONPATH=. python3 scripts/rl/benchmark_duplicate.py \
    --a hybrid:Best:output/nn_full_action_best.pt \
    --b baseline \
    --opponents baseline,beliefexp,hybrid:Base:output/nn_full_action_best.pt \
    --n-seeds 400 --workers 32
```

---

## 4. 项目状态（2026-07-18 凌晨）——**方向 1b 晋升：Gumbel AZ 闭环产出新 best**

### 方向 1b（Gumbel 搜索目标 AZ 闭环）：**晋升**（项目史上首个 RL 来源晋升）

- `jaxenv/` 全管线（JAX 引擎 547k steps/s + Flax 移植 + obs 对齐 + PPO+KL 锚 +
  Gumbel-top-k 1-ply 搜索目标）。β=32 校准（子任务默认 β=8 翻不动 prior）。
- pilot 12M searched decisions / 3.7h：训练健康（agree 0.58→0.65、KL ~0.45、
  流局 <1%）；in-env margin +5.3pp（outcome pilot 同口径 +3.0pp）。
- **晋升链**：1000-pair +1.6% → 5000-pair +2.8% [+1.5,+4.2]（4×SE）→ 独立复跑
  +1.2%（符号一致）；**合并 11000 pairs +2.0% [+1.1,+2.9]**，score-proxy
  +0.054 [+0.023,+0.084]。guardrail 点炮 +2.3pp（<+3pp 线）。
- **新 best：`output/jax_gumbel_iter92.pt`**（arena anchor 已切换，见
  `docs/eval-protocol.md` §2.1）。机制结论：密集搜索目标绕开了 outcome 信号的
  内在 SNR 天花板（方向 2 判词对该形态不成立）。
- **二代（移动锚）收敛判死（07-18）**：in-env +1~2pp 看似正，5000-pair 独立种子
  **−2.9% [−4.2,−1.7]** 显著倒退——自蒸馏回声室，与历史「二代递减」同构。
  AZ 迭代在当前形态下一代收敛；in-env eval 对二代是误导信号，arena paired 才是
  唯一可靠裁决。
- **plan-0718 批次（2-ply 搜索目标 + 对手池，07-18 晚，零晋升）**：
  A（2-ply）+0.9% [−1.7,+3.5]、B（对手池）−0.2% [−2.9,+2.5]，均未过 +1.0% 线。
  一代的 gumbel 改进是「一次性收割」，后续旋钮（更深搜索/多样化/自迭代）全部
  落在噪声带内。详见 `docs/plan-0718.md` §4。
- 详见 `docs/reports/jax-rl-0717.md` 附录。

### π' 部署（S1）+ NN 叶 A/B（2026-07-19，双证伪）——详见 `docs/reports/gumbel-deploy-0719.md`

- **NN 叶 A/B**：同 V3 骨架只换叶，JAXG 叶 ≤ eval0 叶（×100 平；×300 **−4.6%
  [−7.4,−1.8]** 显著）。「搜索强度=叶值质量」铁律在 JAXG 上依然成立；
  **S4（eval2-free Hybrid）判死**，「NN 叶换 eval2」变体勿再开。
- **S1（GumbelSearchAgent，π' 每步部署）**：pool 400 仅 **0.5%**（纯 prior 对照
  7.2–14.7%）。消融：β=0→8→32 单调恶化；4× 采样更差（偏差非方差）；β=−32 为 0
  （Q 非倒置）。根因=防守通道部署不可用：真实手牌胡牌掩码缺失（均匀 belief
  胡概率恒 0）+ **dealin 头因 JAX 训练从未监督、trunk 漂移已死（输出恒 ≈0.31）**，
  Q 退化为纯进攻 1-ply 重排，覆盖 prior 蒸馏好的防守 → 点炮 27%。
  **「NN+每步轻搜索」象限证伪；Hybrid 仍是强度/简洁最优折中。**
- 复用资产：`algo/agents/gumbel_search_agent.py`（accounting 已修，可作 belief
  类 agent 模板）+ `gumbel:` token；平台 accounting 五条规则（被 claim 牌不广播
  'put' 等）见报告 §3 / AGENTS.md §7.15–7.17。

### arena `is_succ` quirk 量化（2026-07-19）

- 120 局 / 27,962 次和牌判定：**漏判 64 起（0.229%），全部为有副露手牌物理成和
  判负**（47 铳和 + 17 自摸）；七对子 0 起（无 agent 追七对，该 quirk 无害）。
- 副露判负 quirk **不可忽略**（meld-heavy 对局 ~每 2 局 1 起），是 arena
  「无人敢碰」元游戏的规则级成因；NN response 头在 jaxenv（副露可胡）训练，
  arena 判负 = 系统性惩罚 NN 的碰/杠。修复引擎裁决可对齐训练环与真实规则，
  但改变所有历史 benchmark 可比性——**待用户拍板**（探针：`tmp/is_succ_quirk_probe.py`）。

### Gumbel gen3（2026-07-19，**不晋升**——固定锚 AZ 迭代到头）

- gen1 配方复刻：固定 KL 锚 0.2，init=`jax_gumbel_pilot/iter92.msgpack`，
  k=8/D=2/β=32，184 iters = **24.1M decisions / 6.3h**（gen1 的 2 倍算力），seed 20。
- 训练健康（agree 0.61、KL 0.09 锚定、流局 <1%）；in-env 末段温和正
  （final 1v3 每席位 +1.4pp / 3v1 +0.7pp）——再次印证 in-env 不可作裁决依据。
- **arena 筛查（1000-pair duplicate vs iter92 best）：−2.4% [−5.0,+0.2]**，
  中心负侧 → 按协议不晋升、不进 5000-pair。
  pkl：`output/duplicate_g3_vs_jaxg_1000.pkl`；权重 `output/jax_gumbel_gen3_iter184.pt`（保留）。
- **结论**：1b 的 +2.0% 是「换目标形态」（outcome→搜索蒸馏）的一次性收益，
  不是可重复的迭代杠杆；固定锚 AZ 自迭代在 gen1 即达定点。gen2（移动锚 −2.9%）
  与 gen3（固定锚 2× 算力 −2.4% 噪声）共同判死 AZ 迭代路线。
  强度路线的剩余选项见 `docs/reports/gumbel-deploy-0719.md` 与本文件 §4 各判词。

### From-scratch 优雅管线（plan-scratch-0718，2026-07-19，M1 过 / M2 败）

- **M1（通过）**：随机初始化 + gumbel 搜索目标 + NPG 移动锚 + draw −0.25，
  12M 决策从零学会牌效+防守（流局率 97.9%→1.8%，vs shanten-greedy 37.9%）。
  冷启动可行性论证（精确声明/自摸真值从第一步免费）成立。
- **M2（失败）**：50M 续训后 in-env vs greedy 50.8%，但 arena pool 400 仅
  **2.5% 胜率 / 30.2% 点炮**（vs 3×Baseline/3×BeliefExp 各 4.8%）。
  根因：训练分布不含 eval2 系对手 + 报听特征 OOD（训练恒否、arena 报听常见）。
  **in-env 强度 ≠ arena 强度**第二次实证。M2'（修分布再战）列为可选，见
  `docs/plan-scratch-0718.md` §5。

### 方向 0/2（god-mode 上界 + PTIE critic，2026-07-17 下午，双关闭）

- **方向 0**：完美隐藏手牌信息（精确剩余分布 + 精确点炮规避）在 BeliefExp 结构内
  仅值 **+1.2% [+0.3,+2.1]** 胜率（2000 pairs）；god-mode BeliefExp 仍比 best 低
  7.5pp——信息通道已榨干，剩余空间不在「知道更多」。
- **方向 2**：完美信息 critic（PTIE）val corr 0.2525 ≈ v1 的 0.231——**信用分配
  SNR 根因 = 游戏内在随机性（未来牌山），非隐藏信息**；确认性 AWBC +0.3%
  [−0.8,+1.4] 未过线，关闭。
- 详见 `docs/reports/godmode-ptie-0717.md`。方向 1（JAX 引擎 + KL 锚 PPO）按
  证据门执行中，见 `docs/reports/web-research-directions-0717.md` §5。

### 方向 1（JAX 引擎 + KL 锚 PPO，2026-07-17 晚，已关闭：不晋升）

- `jaxenv/` 管线交付：JAX 晋北引擎（**547k steps/s**@batch4096 单 3090 ≈ Python
  自对弈 68×；is_win/shanten 各 100k 例 0 失配、场景 9/9、不变量 260 局、
  分布对比容差内）、Flax 移植（全 head 对齐 <1e-4）、obs 对齐（1200 状态
  <1e-6）、PPO+KL 锚脚本（Mahjax 配方）。
- pilot（49.8M decisions，2.16h）**全程无坍缩**（KL 0.038、流局 ~1%）——
  项目史上首个稳定在线 RL；in-env margin vs 纯 NN ref +3pp（plateau）。
- **arena 裁决**：1000-pair 筛查 +1.2% 过线 → 5000-pair 独立种子确认
  **+0.2% [−0.5,+0.9]**，winner's curse（协议第三次实战拦截），**不晋升，关闭**。
- 归因：PPO 改进的状态类与搜索层高度重叠，不穿透部署结构；与方向 0/2 共同
  构成「在位者近可达上界」完整证据链。`jaxenv/` 留存为基础设施。
- 详见 `docs/reports/jax-rl-0717.md`。

### 方向 E（RL/NN/bootstrap 冲刺）：15 候选零晋升，证据定级判死

- 一晚上按 eval-protocol 筛查 **15 个候选**（1000-pair duplicate arena 各一次），
  无一达到 +1.0% 预登记线：弃牌 AWBC×3、响应头 AWR×2、tenpai 阈值×5、
  成对 rollout 标签×4、tenpai 死代码修复×1。两个「少碰」候选显著为负。
- **新增铁证**：12,000 状态 god-mode 成对 rollout（同牌山洗牌，Hybrid 续打）
  测得碰的配对因果效应 mean Δ=+0.117——响应头的高碰 take 率是对的；
  此前 AWR 的「碰 advantage 为负」是选择偏差混杂。outcome 级 RL 与
  配对因果标签 RL 均无法改进当前 best：信用分配 SNR + 特征不可分 + 在位者近最优。
- 剩余未证伪方向：~~特征扩容~~（2026-07-17 探针判死：belief 特征对坏碰可分性
  AUC 仅 0.638，<0.75 门槛，`scripts/rl/belief_feature_probe.py`）；
  只剩引擎接入真实计分后的报听类结论重估（产品向）。
- 详见 `docs/reports/selfplay-bootstrap-0717.md`（含全部候选登记与资产清单）。

### 方向 0：评测校准（完成，结论重大）

- **fable-5 的「duplicate 下 Baseline 强于 best」是 benchmark bug**（同名前缀匹配误计），
  非事实。重算全部历史 pkl：**Hybrid-Best − Baseline = +9.4% [+8.0,+10.9]（5000 pairs）、
  − BeliefExp = +10.4% [+9.0,+11.8]**，best 链条有效。
- soup→蒸馏最后一环证伪：NewBest − OldBest = +0.2% [−0.5,+0.9]（5000 pairs），同强。
- **晋升/放弃决策一律按 `docs/eval-protocol.md`**（5000-pair duplicate arena、paired win
  diff CI 不含 0 + 独立种子复跑、score-proxy 辅指标；Elo 不作依据）。
- 详见 `docs/reports/duplicate-reanalysis-0716.md`。

### 方向 A/B：终盘精确求解 + 待牌分布（已判死，证据链完整）

- hybridend（Hybrid 接 exact solver 搜索层）vs hybrid：5000 pairs 统计无差异（99.9% ties）。
- **oracle gate（完美待牌上界）：2000 pairs −0.1%，仅 1 局差异**——信息完美也无增量，
  待牌预测质量不是瓶颈，方向 B 前提不成立。
- 机制分析：exact solver 触发 ~0.16 次/agent-game，BeliefExp 首选落入真实待牌集合
  仅 1/120 agent-games——**对已报听者的防守 BeliefExp 已接近最优**。
- 误差分解（200 局 event_log）：**82% 点炮送给默听（未报听）玩家**；
  BeliefExp 对已报听者 0 失误。防守前沿在**中盘默听检测**（方向 D 接手）。
- 详见 `docs/reports/endgame-solver-ab-0716.md`。

### 方向 C/D：探索收尾（2026-07-16）

- **C（外部数据）判死**：公开牌谱仅天凤（日麻）/MCR 自对弈，无晋北同规则数据，
  跨规则迁移目标函数冲突。
- **D（默听检测 + 序列特征）**：离线检测已解决（GRU 序列编码 + 混合池数据，
  silent AUC 0.919，seq 稳定优于纯计数特征 +0.09~0.18），但在线接入经
  `_danger_signal` + wait-danger 不转化为胜率：唯一 1000-pair 筛查命中
  （+1.0%）在 5000-pair 独立种子复跑中消失（+0.1% [−0.4,+0.5]），
  阈值趋势非单调。**协议的多重比较防护首次实战拦截 winner's curse**。
  方向关闭，详见 `docs/reports/silent-tenpai-d-0716.md`。
- 「教师更强则 trace 蒸馏」：A/B/D 均未产出更强教师，跳过。

### 此前状态（2026-07-08，暂停时记录）

已验证成功：NN + BeliefExp Hybrid（当前最强框架）；Model Soup + 蒸馏回单一模型。
已验证失败：Path A（nnpolicy MC rollout value labels）、Path B（exact depth-2 search
distillation）、Cython 化 eval2/expectimax（提速 6.8× 但 teacher 不强）、exact endgame
defensive head（standalone 弱，val MSE 0.054 但 100 局仅 11% win）。

详细历史记录、所有实验数据与产物见 `docs/reports/project_history.md`。

---

## 5. 文档索引

| 文件 | 内容 |
|---|---|
| `docs/handoff.md` | 本文件：当前状态与快速入口 |
| `docs/reports/project_history.md` | 按时间线的完整实验日志 |
| `docs/reports/search_distillation_report.md` | Path A/B 详细结果 |
| `docs/reports/rl-ppo-report.md` | PPO 端到端 RL 实验报告 |
| `docs/reports/ablation_report.md` | Hybrid-FullAction 减法消融 |
| `docs/reports/future_directions_analysis.md` | 未来方向分析 |
| `docs/reports/gumbel-deploy-0719.md` | π' 部署（S1）+ NN 叶 A/B 双证伪；平台 accounting 规则 |
| `docs/expectimax-todos.md` | ExpectiMax 相关 TODO |
| `docs/rules.md` | 晋北麻将规则 |
| `AGENTS.md` | Agent 工作守则与项目约定 |

---

## 6. 已 push 的数据与 Checkpoint

以下文件已加入 git（模型权重 `.pt`/`.npz` 默认被 `.gitignore` 忽略，config json 被跟踪）：

| 文件 | 说明 |
|---|---|
| `output/nn_model_config.json` | policy net 配置 |
| `output/nn_value_model_mc_config.json` | value net 配置 |
| `output/nn_full_action_best_config.json` | 当前 best 配置 |

模型权重与数据文件较大，未入 git，详见 `docs/reports/project_history.md` §2。
