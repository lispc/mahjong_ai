# Handoff：换机器继续工作的指南

> 本文档是项目状态的**精简版快速入口**。完整实验历史见 `docs/reports/project_history.md`。

---

## 1. 当前最强配置

```python
# benchmark token: hybrid:newbest:output/nn_full_action_best.pt:beliefexp
# 对应类：algo.agents.hybrid_nn_belief_agent.HybridNNBeliefAgent
from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent

HybridNNBeliefAgent(
    'Hybrid-FullAction-SoupDistilled',
    nn_model_path='output/nn_full_action_best.pt',
    belief_kind='beliefexp',
    tenpai_threshold=28,
    device='cpu',
)
```

对应模型：`output/nn_full_action_best.pt` + `output/nn_full_action_best_config.json`
- `TileConvNet`，128 channels / 6 residual blocks / 512 hidden
- 带 dealin / value / tenpai / response head
- 来源：model soup + 蒸馏，详见 `docs/reports/project_history.md` §6.11–§6.12

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

## 4. 项目状态（2026-07-17）——**RL/自对弈 bootstrap 冲刺完成，判死**

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
