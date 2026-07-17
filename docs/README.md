# 文档索引

> 本文档是 `docs/` 目录的总索引。新来的协作者建议按以下顺序阅读。

## 必读（从这里开始）

| 文档 | 内容 | 更新频率 |
|---|---|---|
| [`handoff.md`](handoff.md) | **当前项目状态、最强配置、下一步建议**。换机器或换人时先看这篇。 | 每次重大迭代 |
| [`eval-protocol.md`](eval-protocol.md) | **评测协议：标准考场、晋升/放弃门槛、目标函数结论**。任何 benchmark 决策前必读。 | 协议变更时 |

## 实验报告（按时间倒序）

 located in [`reports/`](reports/)。

| 文档 | 主题 |
|---|---|
| [`reports/web-research-directions-0717.md`](reports/web-research-directions-0717.md) | **下一步方向评估**：25 份文档通读 + 网络调研（Mahjax/Mortal/PerfectDou/Tjong/Gumbel），方向 0/1/2 执行计划 |
| [`reports/godmode-ptie-0717.md`](reports/godmode-ptie-0717.md) | **方向 0+2 结果**：god-mode 上界（信息通道 ≤1.2pp）；PTIE critic 门 FAIL（corr 0.253≈0.231，SNR 根因=内在随机性）；方向 2 关闭 |
| [`reports/jax-rl-0717.md`](reports/jax-rl-0717.md) | **方向 1（已关闭）**：JAX 引擎（547k steps/s）+ KL 锚 PPO pilot 50M 步；筛查 +1.2% → 5000-pair 确认 +0.2%（winner's curse），不晋升；jaxenv 留存为基础设施 |
| [`reports/duplicate-reanalysis-0716.md`](reports/duplicate-reanalysis-0716.md) | **Duplicate 配对统计 bug 修正**：fable-5 的「best 不如 Baseline」结论反转，全部历史 pkl 重算 |
| [`reports/endgame-solver-ab-0716.md`](reports/endgame-solver-ab-0716.md) | **方向 A/B 判死**：终盘精确求解四重证据阴性（oracle 上界为零）；点炮 82% 属默听，前沿移交方向 D |
| [`reports/silent-tenpai-d-0716.md`](reports/silent-tenpai-d-0716.md) | **方向 D**：默听检测离线解决（seq AUC 0.92）但接入不复现胜率；winner's curse 被协议拦截的实例 |
| [`reports/recent-work.md`](reports/recent-work.md) | 近期工作汇总：V3-NN-BE1、网络训练与自对弈循环（详细版） |
| [`reports/eval_v3-report.md`](reports/eval_v3-report.md) | eval_v3（ukeire + wait + defense + Numba）实现与 benchmark |
| [`reports/mcts-eval2-report.md`](reports/mcts-eval2-report.md) | MCTS-Eval2、Eval2Ctx、对手建模 B+D、去 deepcopy 加速 |
| [`reports/performance-depth2-report.md`](reports/performance-depth2-report.md) | 性能优化（PyPy/Numba/Cython）与 depth=2 ExpectiMax |
| [`reports/route-a-report.md`](reports/route-a-report.md) | 路线 A：原项目 eval2 + 已见牌信息 |
| [`reports/shanten-ukeire-experiment.md`](reports/shanten-ukeire-experiment.md) | Shanten + Ukeire Agent 实验 |

## 设计与路线图

 located in [`designs/`](designs/)。

| 文档 | 主题 |
|---|---|
| [`designs/ai-roadmap.md`](designs/ai-roadmap.md) | 麻将 AI 各主流路线对比与晋北麻将适配度分析 |
| [`designs/algo-proposals.md`](designs/algo-proposals.md) | 更一致的弃牌算法设计方案（Shanten/Ukeire/Expectimax/Defense） |
| [`designs/eval-improvement-plan.md`](designs/eval-improvement-plan.md) | 评估函数改进计划（ukeire、wait quality、defense、报听、对手建模） |
| [`designs/mahjong-ai-research-designs.md`](designs/mahjong-ai-research-designs.md) | 麻将 AI 调研 + 四种从零设计方案的实现速报 |
| [`designs/silent-tenpai-seq-model.md`](designs/silent-tenpai-seq-model.md) | **方向 D（进行中）**：默听检测 + 弃牌序列特征模型；数据/模型/三级验收门 |

## 其他

| 文档 | 内容 |
|---|---|
| [`rules.md`](rules.md) | 晋北麻将核心规则 |
| [`expectimax-todos.md`](expectimax-todos.md) | ExpectiMax 框架下的改进 TODO 清单 |

## 归档说明

- 已合并/删除的文档：`recent-work.md`（内容并入 `handoff.md`）。
- 实验报告以**完成时的结论**为准，后续若代码/模型变化导致结论过时，请在对应 report 末尾追加 "更新" 段落，不要直接覆盖原始数据。
