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

## 4. 项目状态（2026-07-08）

### 已验证成功
- **NN + BeliefExp Hybrid**（当前最强框架）
- **Model Soup + 蒸馏回单一模型**（产出 `output/nn_full_action_best.pt`）

### 已验证失败（近期重点）
- **Path A：nnpolicy MC rollout value labels** —— 4-rollouts label 噪声太大，value net 弱于 baseline。
- **Path B：exact depth-2 search distillation** —— depth-2 expectimax（leaf=eval0 或 leaf=nn）均未能产生强于 Hybrid-Best 的 teacher，三种蒸馏方法（BC policy/value、DPO）全部阴性。
  - 详见 `docs/reports/search_distillation_report.md`

### 正在执行 / 下一步
1. **Cython 化 eval2 / expectimax**：把搜索热路径 Cython 化，复活深度 search 并尝试蒸馏。
2. **Exact endgame defensive head**：用已有的 13,843 精确终局标签训练 defensive decision head。

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
