# 晋北麻将 AI 研究与对战平台

本项目是一个用于研究和 benchmark 晋北麻将 AI 的 Python 代码库。核心围绕**带概率修正的 ExpectiMax / MCTS 决策**展开，支持多进程对战、Elo 评分和历史实验报告。

## 当前最强配置

截至最新实验，最强实用配置为 **Hybrid-BE8k_t4**：

- **Agent 类型**：`HybridNNBeliefAgent`（NN + BeliefExp 混合）
- **NN 模型**：`output/nn_conv_bc_beliefexp_trace_8000_big_t4.pt`
- **模型架构**：conv-BC Policy-Value Net，128 channels / 6 residual blocks / 512 hidden，带 dealin head 与 value head
- **训练方式**：search trace distillation，教师为纯 `BeliefExpectimaxAgent`（每步搜索），8000 局数据，soft target 温度 T=4
- **性能**（2000 局公平 pool）：胜率 **25.7%**、点炮 **15.5%**、Elo **1567**
- **benchmark token**：
  ```
  hybrid:BE8k_t4:output/nn_conv_bc_beliefexp_trace_8000_big_t4.pt:beliefexp
  ```

使用示例：

```bash
SEATS="baseline,hybrid:BE8k_t4:output/nn_conv_bc_beliefexp_trace_8000_big_t4.pt:beliefexp,hybrid:Base:output/nn_conv_bc_hybrid_2000.pt:beliefexp,hybrid:BE4k_big:output/nn_conv_bc_beliefexp_trace_4000_big.pt:beliefexp" \
PYTHONPATH=. python3 scripts/rl/benchmark_pool.py 1000 32
```

详细实验记录与路线图见 [`docs/designs/conv-bc-roadmap.md`](docs/designs/conv-bc-roadmap.md)。

## 目录结构

```
.
├── agent.py                 # Agent 基类与消息机制
├── context.py               # 牌山概率上下文（原 type.py，已重命名避免遮蔽内置 type）
├── tile.py                  # 牌 ID、显示、随机生成
├── tile_pool.py             # 洗牌与发牌
├── config.py                # 全局配置（如 pair_coef）
├── utils.py                 # 通用工具函数
├── run_tests.py             # 统一测试入口
├── requirements.txt         # Python 依赖
│
├── algo/                    # AI 算法包
│   ├── eval/                # 评估后端
│   │   ├── legacy.py        # 原项目 eval0/eval1/eval2（当前主力）
│   │   ├── v2.py            # shanten + taatsu 快速评估
│   │   ├── v3.py            # ukeire + wait quality + 基础防守
│   │   └── opponent.py      # 对手建模（花色偏好 + 听牌信号）
│   ├── context/             # 上下文实现
│   │   ├── v2.py
│   │   └── v3.py
│   └── agents/              # Agent 实现
│       ├── expectimax.py
│       ├── expectimax_v3.py
│       ├── expectimax_eval2.py      # Eval2Ctx
│       ├── expectimax_baseline.py
│       ├── mcts.py
│       ├── mcts_eval2.py
│       └── shanten_ukeire.py      # 方案 A / 方案 3：Shanten+Ukeire + v3 CEM
│
├── driver/                  # 对战驱动
│   ├── engine.py            # 单局游戏
│   └── tournament.py        # 多局 + 多进程
│
├── checker/                 # 统计与报告
│   └── report.py
│
├── scripts/                 # benchmark / 调参脚本
│   ├── compare_mcts_eval2.py
│   ├── compare_v3.py
│   ├── tune_weights_cem.py
│   └── ...
│
├── tests/                   # 测试
│   ├── test_eval_v2.py
│   └── legacy_test.py
│
├── docs/                    # 实验报告与规划
│   ├── algo-proposals.md      # 更一致的弃牌算法设计方案
│   ├── shanten-ukeire-experiment.md  # Shanten+Ukeire 与 expectimax 实验
│   ├── mcts-eval2-report.md
│   ├── expectimax-todos.md
│   └── ...
│
└── output/                  # 运行时产物（已被 .gitignore 忽略）
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

> 推荐使用 PyPy 7.3+ 运行 benchmark，可获得约 2× 速度提升。

### 运行测试

```bash
python run_tests.py
```

### 运行 benchmark

最常用的对比脚本：

```bash
# PyPy 下跑 100 局、8 进程
pypy3 scripts/compare_mcts_eval2.py 100 8
```

其他脚本：

```bash
pypy3 scripts/compare_v3.py 100 8
pypy3 scripts/compare_baseline_defense.py 100 8
pypy3 scripts/compare_shanten_ukeire.py 100 8
pypy3 scripts/tune_weights_cem.py
```

### 复盘 GUI

1. 先用 `record_game.py` 录制一局：
   ```bash
   python3 scripts/record_game.py -s 42 -o output/replay_001.json
   ```
2. 再用 `replay_gui.py` 回放：
   ```bash
   python3 scripts/replay_gui.py output/replay_001.json
   ```
   在 GUI 中按 **← / →** 逐步前后查看，Home/End 跳转到首步/末步。

## Agent 列表

| Agent | 文件 | 评估后端 | 特点 |
|---|---|---|---|
| Baseline | `agent.py` | `algo.eval.legacy` | 原作者基线，2-ply ExpectiMax |
| Eval2Ctx | `algo/agents/expectimax_eval2.py` | `algo.eval.legacy` | 带入已见牌信息，支持报听 |
| ExpectiMaxAgent | `algo/agents/expectimax.py` | `algo.eval.v2` | shanten + taatsu，支持 depth≥2 剪枝 |
| ExpectiMaxV3Agent | `algo/agents/expectimax_v3.py` | `algo.eval.v3` | ukeire + wait + 防守 |
| ExpectiMaxBaselineAgent | `algo/agents/expectimax_baseline.py` | `algo.eval.legacy` + `v3` 防守 | eval2 + 基础防守 |
| MCTSAgent | `algo/agents/mcts.py` | `algo.eval.v2` | 采样版 ExpectiMax |
| MCTSEval2Agent | `algo/agents/mcts_eval2.py` | `algo.eval.legacy` | MCTS + eval2 叶子评估 |
| Eval2Ctx+BD | `algo/agents/expectimax_eval2.py` | `algo.eval.legacy` + `opponent` | 实验性对手建模防守 |
| V3-NN | `algo/agents/belief_expectimax_v3.py` | `algo.eval.v3` + `algo.nn` | `baseline_eval1` 候选 + NN leaf |
| V3-NN-PC | `algo/agents/belief_expectimax_v3.py` | `algo.eval.v3` + `algo.nn` | NN policy 候选 + NN leaf（历史最强） |
| DeterminizedMCTS | `algo/agents/determinized_mcts.py` | `algo.eval.v2` + rollout | 支持 NN/BeliefExp rollout |
| **Hybrid-BE8k_t4** | `algo/agents/hybrid_nn_belief_agent.py` | `algo.nn` + `BeliefExpectimaxAgent` | **当前最强**：conv-BC NN 快速决策 + BeliefExp critical 搜索 |
| Hybrid-Base | `algo/agents/hybrid_nn_belief_agent.py` | `algo.nn` + `BeliefExpectimaxAgent` | 上一代稳健 Hybrid |
| SafetyAwarePPOAgent | `algo/agents/safety_aware_ppo_agent.py` | `algo.nn` | 实验性 safety-aware 报听（已证伪） |

## NN Agent 与自对弈

NN 后端已切换到 **PyTorch**。核心网络为基于 `TileConvNet` 的 conv-BC Policy-Value Net（支持 dealin head、value head、tenpai head），用于 Hybrid agent 的快速 NN 决策。

当前主要模型：

| 模型 | 文件 | 说明 |
|---|---|---|
| `nn_conv_bc.pt` | `output/nn_conv_bc.pt` | 早期纯行为克隆 baseline |
| `nn_conv_bc_hybrid_2000.pt` | `output/nn_conv_bc_hybrid_2000.pt` | Hybrid-Base 的 NN 部分 |
| `nn_conv_bc_beliefexp_trace_4000_big.pt` | `output/...` | 4000 局 BeliefExp trace + big 网络 |
| `nn_conv_bc_beliefexp_trace_8000_big_t4.pt` | `output/...` | **当前最佳**：8000 局 BeliefExp trace + big 网络 + T=4 |

训练脚本示例：

```bash
# 普通 conv-BC（可指定 hidden、channels、n_blocks）
PYTHONPATH=. python3 scripts/train_nn.py output/nn_training_data_merged.npz 60 256 0.001 256

# Search trace distillation（支持 --init、--alpha、--temp、--channels、--n_blocks、--hidden）
PYTHONPATH=. python3 scripts/rl/train_search_distill.py \
    output/nn_teacher_beliefexp_trace_8000.npz \
    nn_conv_bc_beliefexp_trace_8000_big_t4 \
    --init output/nn_conv_bc_dealin_2000_l07.pt \
    --alpha 0.5 --temp 4.0 --beta 0.3 --lambda_dealin 0.5 \
    --epochs 40 --bs 512 --lr 1e-3 \
    --channels 128 --n_blocks 6 --hidden 512

# 生成 BeliefExp 教师 trace 数据
PYTHONPATH=. python3 scripts/rl/gen_beliefexp_trace_data.py \
    output/nn_teacher_beliefexp_trace_8000.npz 8000 32 19000000
```

benchmark token 规则见 `scripts/rl/benchmark_pool.py`，例如：

```
hybrid:BE8k_t4:output/nn_conv_bc_beliefexp_trace_8000_big_t4.pt:beliefexp
```

详细介绍与实验结论见 [`docs/designs/conv-bc-roadmap.md`](docs/designs/conv-bc-roadmap.md)。

## 实验报告索引

- [`docs/designs/conv-bc-roadmap.md`](docs/designs/conv-bc-roadmap.md)：conv-BC / Hybrid / Search trace distillation 实验路线图与最新结果（**当前最佳 Hybrid-BE8k_t4 来源**）。
- [`docs/reports/recent-work.md`](docs/reports/recent-work.md)：V3-NN-BE1 算法详解、网络训练、自对弈循环、MC rollout 标签质量分析。
- [`docs/reports/mcts-eval2-report.md`](docs/reports/mcts-eval2-report.md)：Eval2Ctx 超越 Baseline、MCTS-Eval2 失败、B+D 对手建模、去掉 deepcopy 加速。
- [`docs/expectimax-todos.md`](docs/expectimax-todos.md)：ExpectiMax 潜在改进清单。
- [`docs/reports/route-a-report.md`](docs/reports/route-a-report.md)：路线 A 实验记录。
- [`docs/designs/eval-improvement-plan.md`](docs/designs/eval-improvement-plan.md)：评估函数改进计划。

## 注意事项

- 根目录原 `algo.py` 已被 `algo/` 包取代，已删除。
- 原 `type.py` 已重命名为 `context.py`。
- 废弃的 Cython 扩展、旧 demo 驱动、临时 scratch 文件已清理。
