# 晋北麻将 AI 研究与对战平台

本项目是一个用于研究和 benchmark 晋北麻将 AI 的 Python 代码库。核心围绕**带概率修正的 ExpectiMax / MCTS 决策**展开，支持多进程对战、Elo 评分和历史实验报告。

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
| Eval2Ctx | `algo/agents/expectimax_eval2.py` | `algo.eval.legacy` | 带入已见牌信息，支持报听，当前最强 |
| ExpectiMaxAgent | `algo/agents/expectimax.py` | `algo.eval.v2` | shanten + taatsu，支持 depth≥2 剪枝 |
| ExpectiMaxV3Agent | `algo/agents/expectimax_v3.py` | `algo.eval.v3` | ukeire + wait + 防守 |
| ExpectiMaxBaselineAgent | `algo/agents/expectimax_baseline.py` | `algo.eval.legacy` + `v3` 防守 | eval2 + 基础防守 |
| MCTSAgent | `algo/agents/mcts.py` | `algo.eval.v2` | 采样版 ExpectiMax |
| MCTSEval2Agent | `algo/agents/mcts_eval2.py` | `algo.eval.legacy` | MCTS + eval2 叶子评估 |
| Eval2Ctx+BD | `algo/agents/expectimax_eval2.py` | `algo.eval.legacy` + `opponent` | 实验性对手建模防守 |
| **V3-NN** | `algo/agents/belief_expectimax_v3.py` | `algo.eval.v3` + `algo.nn` | 默认配置：`baseline_eval1` 候选 + NN leaf |
| V3-NN-PC | `algo/agents/belief_expectimax_v3.py` | `algo.eval.v3` + `algo.nn` | NN policy 候选 + NN leaf |
| DeterminizedMCTS | `algo/agents/determinized_mcts.py` | `algo.eval.v2` + rollout | 支持 NN/BeliefExp rollout |

## NN Agent 与自对弈

最近引入了基于 MLX 的 Policy-Value Net 与 Deep Value Net，以及“自对弈 + 模型筛选门”循环。

训练脚本：

```bash
# Policy-Value Net（支持 hidden_dim 参数）
python scripts/train_nn.py output/nn_training_data_merged.npz 60 256 0.001 256

# Deep Value Net（支持 hidden_dims 参数，逗号分隔）
python scripts/train_value_net_mc.py output/nn_training_data_merged.npz 60 256 0.001 512,256,128
```

自对弈 + 筛选门：

```bash
# 1000 局、6 worker、每个样本 4 次 MC rollout、100 局评估、Elo 门限 20
python scripts/self_play_loop.py 1000 6 4 1 100 20
```

详细介绍与实验结论见 [`docs/recent-work.md`](docs/recent-work.md)。

## 实验报告索引

- [`docs/recent-work.md`](docs/recent-work.md)：V3-NN-BE1 算法详解、网络训练、自对弈循环、MC rollout 标签质量分析（**最新**）。
- [`docs/mcts-eval2-report.md`](docs/mcts-eval2-report.md)：Eval2Ctx 超越 Baseline、MCTS-Eval2 失败、B+D 对手建模、去掉 deepcopy 加速。
- [`docs/expectimax-todos.md`](docs/expectimax-todos.md)：ExpectiMax 潜在改进清单。
- [`docs/route-a-report.md`](docs/route-a-report.md)：路线 A 实验记录。
- [`docs/eval-improvement-plan.md`](docs/eval-improvement-plan.md)：评估函数改进计划。

## 注意事项

- 根目录原 `algo.py` 已被 `algo/` 包取代，已删除。
- 原 `type.py` 已重命名为 `context.py`。
- 废弃的 Cython 扩展、旧 demo 驱动、临时 scratch 文件已清理。
