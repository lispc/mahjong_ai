# 文档索引

> 本文档是 `docs/` 目录的总索引。新来的协作者建议按以下顺序阅读。

## 必读（从这里开始）

| 文档 | 内容 | 更新频率 |
|---|---|---|
| [`handoff.md`](handoff.md) | **当前项目状态、最强配置、下一步建议**。换机器或换人时先看这篇。 | 每次重大迭代 |
| [`eval-protocol.md`](eval-protocol.md) | **评测协议：标准考场、晋升/放弃门槛、目标函数结论**。任何 benchmark 决策前必读。 | 协议变更时 |
| [`plan-beliefjax-0720.md`](plan-beliefjax-0720.md) | 最近一批执行计划（BeliefExp→JAX + 防守蒸馏，**已结案**：P2 判决防守知识可蒸馏但不破均衡） | 已结案 |
| [`plan-0718.md`](plan-0718.md) | 简化路径结论 + 2-ply/对手池批次（**已结案**，零晋升） | 已结案 |
| [`plan-scratch-0718.md`](plan-scratch-0718.md) | from-scratch 优雅管线（M1 过 / M2 败，M2' 可选未做） | 已结案 |

## 实验报告（按时间倒序）

 located in [`reports/`](reports/)。

| 文档 | 主题 |
|---|---|
| [`reports/used-paircoef-ab-0720.md`](reports/used-paircoef-ab-0720.md) | **used 修复 + pair_coef A/B 双阴性**：used-aware eval2 −2.6% 显著负（更守但不更强第四次同构）；pair_coef 1.0/0.6 无差异维持现状 |
| [`reports/eval2-alternatives-0719.md`](reports/eval2-alternatives-0719.md) | **eval2 无法超越的考古结论**（精确 2 步随机 DP）+ C1 结案（深度饱和）；文献对照 |
| [`reports/gumbel-deploy-0719.md`](reports/gumbel-deploy-0719.md) | π' 部署（S1）+ NN 叶 A/B 双证伪；平台 accounting 规则 |
| [`reports/jax-rl-0717.md`](reports/jax-rl-0717.md) | **方向 1/1b**：JAX 引擎 + KL 锚 PPO；Gumbel AZ 闭环（断代前晋升，断代后反转）；jaxenv 基础设施 |
| [`reports/selfplay-bootstrap-0717.md`](reports/selfplay-bootstrap-0717.md) | **方向 E 判死**：15 候选零晋升；outcome 级/配对因果 RL 均无法改进 best |
| [`reports/godmode-ptie-0717.md`](reports/godmode-ptie-0717.md) | **方向 0+2**：god-mode 上界（信息通道 ≤1.2pp）；PTIE critic 门 FAIL（SNR 根因=内在随机性） |
| [`reports/web-research-directions-0717.md`](reports/web-research-directions-0717.md) | 25 份文档通读 + 网络调研（Mahjax/Mortal/PerfectDou/Tjong/Gumbel），方向 0/1/2 执行计划 |
| [`reports/duplicate-reanalysis-0716.md`](reports/duplicate-reanalysis-0716.md) | **Duplicate 配对统计 bug 修正**：fable-5 的「best 不如 Baseline」结论反转，全部历史 pkl 重算 |
| [`reports/endgame-solver-ab-0716.md`](reports/endgame-solver-ab-0716.md) | **方向 A/B 判死**：终盘精确求解四重证据阴性（oracle 上界为零）；点炮 82% 属默听 |
| [`reports/silent-tenpai-d-0716.md`](reports/silent-tenpai-d-0716.md) | **方向 D**：默听检测离线解决（seq AUC 0.92）但接入不复现胜率 |
| [`reports/project_history.md`](reports/project_history.md) | 按时间线的完整实验日志（至 2026-07-07，全部断代前） |
| [`reports/rl-ppo-report.md`](reports/rl-ppo-report.md) | PPO 端到端 RL 实验报告（方案 B，坍缩与引分惩罚教训） |
| [`reports/ablation_report.md`](reports/ablation_report.md) | Hybrid-FullAction 减法消融（搜索层贡献 ~32pp，断代前口径） |
| [`reports/search_distillation_report.md`](reports/search_distillation_report.md) | Path A/B：nnpolicy MC 标签与 exact depth-2 蒸馏双阴性 |
| [`reports/future_directions_analysis.md`](reports/future_directions_analysis.md) | 未来方向分析（2026-07-04，多数已执行/证伪） |
| [`reports/fable-5-execution-0706.md`](reports/fable-5-execution-0706.md) | duplicate 基建 + Cython eval0 + 终盘标签 + wait_dist 头 |
| [`reports/recent-work.md`](reports/recent-work.md) | 近期工作汇总：V3-NN-BE1、网络训练与自对弈循环（详细版） |
| [`reports/eval_v3-report.md`](reports/eval_v3-report.md) | eval_v3（ukeire + wait + defense + Numba）实现与 benchmark |
| [`reports/mcts-eval2-report.md`](reports/mcts-eval2-report.md) | MCTS-Eval2、Eval2Ctx、对手建模 B+D、去 deepcopy 加速 |
| [`reports/performance-depth2-report.md`](reports/performance-depth2-report.md) | 性能优化（PyPy/Numba/Cython）与 depth=2 ExpectiMax |
| [`reports/route-a-report.md`](reports/route-a-report.md) | 路线 A：原项目 eval2 + 已见牌信息（used 修复 A/B 的历史先例） |
| [`reports/shanten-ukeire-experiment.md`](reports/shanten-ukeire-experiment.md) | Shanten + Ukeire Agent 实验 |

## 设计与路线图

 located in [`designs/`](designs/)。

| 文档 | 主题 |
|---|---|
| [`designs/conv-bc-roadmap.md`](designs/conv-bc-roadmap.md) | 卷积 BC 路线图（方向 1–9 全执行；C3 分层策略唯一未试） |
| [`designs/td-lambda-plan.md`](designs/td-lambda-plan.md) | TD(λ) 值函数计划（已证伪：TD + V3-NN-PC = 死路） |
| [`designs/ai-roadmap.md`](designs/ai-roadmap.md) | 麻将 AI 各主流路线对比与晋北麻将适配度分析 |
| [`designs/algo-proposals.md`](designs/algo-proposals.md) | 更一致的弃牌算法设计方案（Shanten/Ukeire/Expectimax/Defense） |
| [`designs/eval-improvement-plan.md`](designs/eval-improvement-plan.md) | 评估函数改进计划（ukeire、wait quality、defense、报听、对手建模） |
| [`designs/mahjong-ai-research-designs.md`](designs/mahjong-ai-research-designs.md) | 麻将 AI 调研 + 四种从零设计方案的实现速报 |
| [`designs/silent-tenpai-seq-model.md`](designs/silent-tenpai-seq-model.md) | 方向 D：默听检测 + 弃牌序列特征模型；数据/模型/三级验收门 |

## 其他

| 文档 | 内容 |
|---|---|
| [`rules.md`](rules.md) | 晋北麻将核心规则 |
| [`expectimax-todos.md`](expectimax-todos.md) | ExpectiMax 框架下的改进 TODO 清单 |

## 归档说明

- 已合并/删除的文档：`recent-work.md`（内容并入 `handoff.md`）。
- 实验报告以**完成时的结论**为准，后续若代码/模型变化导致结论过时，请在对应 report 末尾追加 "更新" 段落，不要直接覆盖原始数据。
