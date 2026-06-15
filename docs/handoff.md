# Handoff：换机器继续工作的指南

> 本文档记录当前项目状态、最强配置、已 push 的数据/checkpoint，以及建议的后续路径。换到新机器后先读这篇。

---

## 1. 当前最强配置

默认 V3-NN agent 现在就是：

```python
BeliefExpectimaxV3Agent(
    'V3-NN',
    expectimax_depth=1,
    max_candidates=5,
    leaf_evaluator='nn',
    candidate_policy='baseline_eval1'   # 默认已改
)
```

对应模型：

- `output/nn_model.npz` + `output/nn_model_config.json`
  - Policy-Value Net，`hidden_dim=256`
- `output/nn_value_model_mc.npz` + `output/nn_value_model_mc_config.json`
  - Deep Value Net，`hidden_dims=[512,256,128]`

200 局 benchmark（4 workers）：

```
Baseline    : win 26.5%, deal-in 24.5%, Elo 1526, 117ms
BeliefExp   : win 25.0%, deal-in 10.0%, Elo 1456,  78ms
V3-NN (BE1) : win 22.0%, deal-in 19.5%, Elo 1568,  95ms
V3-NN-PC    : win 22.5%, deal-in 15.5%, Elo 1450,  92ms
```

V3-NN-BE1 在 Elo 上超过 baseline，点炮率明显更低。

---

## 2. 已 push 的数据与 Checkpoint

以下文件已加入 git（它们在 `.gitignore` 里默认被忽略，本次用 `-f` 强制跟踪）：

| 文件 | 说明 |
|---|---|
| `output/nn_model.npz` | 当前 policy-value 网络权重 |
| `output/nn_model_config.json` | policy net 配置（input_dim=175, hidden_dim=256） |
| `output/nn_value_model_mc.npz` | 当前 deep value 网络权重 |
| `output/nn_value_model_mc_config.json` | value net 配置（arch=deep, hidden_dims=[512,256,128]） |
| `output/nn_training_data_mc.npz` | 46k 条 MC 数据（BeliefExp 自对弈 + 8 rollout） |
| `output/nn_training_data_selfplay.npz` | 50k 条 V3-NN 自对弈数据（4 rollout） |
| `output/nn_training_data_merged.npz` | 96k 条合并训练数据 |

到新机器后，拉下来即可直接跑 benchmark 或继续训练。

---

## 3. 环境要求

- Python 3.10+
- `pip install -r requirements.txt`
- **MLX** 用于 NN 训练/推理：
  ```bash
  pip install mlx mlx-metal
  ```
  > 原机器是 Apple Silicon + macOS，新机器如果是 NVIDIA/Linux，需要换成对应 MLX 后端或改为 PyTorch/TensorFlow 版本（需额外移植）。

---

## 4. 到新机器后先验证

```bash
# 1. 跑测试
python run_tests.py

# 2. 跑 benchmark 确认模型能加载、结果接近上文
python tmp/benchmark_new_models.py 100 4
```

如果 `tmp/benchmark_new_models.py` 没随 repo 过来，可以现场写：

```python
import agent
from algo.agents.belief_expectimax import BeliefExpectimaxAgent
from algo.agents.belief_expectimax_v3 import BeliefExpectimaxV3Agent
from driver.tournament import run_tournament
from checker.report import compute_metrics, compute_elo

def make_baseline(): return agent.Agent('Baseline', verbose=False)
def make_beliefexp(): return BeliefExpectimaxAgent('BeliefExp', verbose=False)
def make_v3nn(): return BeliefExpectimaxV3Agent('V3-NN', expectimax_depth=1, max_candidates=5, leaf_evaluator='nn')
def make_v3nn_pc(): return BeliefExpectimaxV3Agent('V3-NN-PC', expectimax_depth=1, max_candidates=5, leaf_evaluator='nn', candidate_policy='nn')

configs = [make_baseline, make_beliefexp, make_v3nn, make_v3nn_pc]
names = ['Baseline', 'BeliefExp', 'V3-NN', 'V3-NN-PC']
results = run_tournament(configs, n_games=100, n_workers=4)
metrics = compute_metrics(results, names)
elo = compute_elo(results, names)
for n in names:
    print(n, metrics[n]['win_rate'], metrics[n]['deal_in_rate'], elo[n])
```

---

## 5. 建议的后续路径

按优先级排序：

### 5.1 验证 rollout policy 对 label 质量的影响（成本最低）

当前 MC value 的 rollout 用的是 greedy `eval0`，很弱。建议先改 `algo/nn/mc_value.py` 里的 `_greedy_discard`：

```python
def _greedy_discard(hand14):
    return algo.select(hand14, _EMPTY_CONTEXT)[0]
```

然后跑一个小规模对比实验（100 局），训练 value net 后 benchmark。如果提升明显，再跑 1000 局。

如果新机器有 GPU/Metal，也可以试 **NN policy rollout**，预计只比 greedy eval0 慢 3–4 倍（约 1 小时），上限更高。

### 5.2 跑新一轮 1000 局自对弈 + 重训练

用当前最强配置生成数据：

```bash
python scripts/self_play_loop.py 1000 6 4 1 100 20
```

这会：
1. 用 V3-NN-BE1 自对弈 1000 局；
2. 与历史 MC 数据合并；
3. 训练 candidate；
4. 跑 100 局 benchmark 与 current best 比较；
5. 只有 Elo 提升 ≥20 才替换。

### 5.3 继续优化网络/特征

- 如果 rollout 改进后 value net 仍不稳定，再考虑加特征（shanten、ukeire、dora 等）；
- 如果新机器算力更强，可以尝试把 `expectimax_depth` 提到 2，或给 DetMCTS 加 NN value 截断。

---

## 6. 代码改动摘要

最近修改的核心文件（都已 commit/push）：

- `algo/agents/belief_expectimax_v3.py`：默认 `candidate_policy='baseline_eval1'`，新增多种候选策略；
- `algo/nn/value_model.py`：`MahjongValueNetDeep` 支持可配置 `hidden_dims`；
- `algo/nn/nn_leaf.py`：按 config 加载对应尺寸的 value net；
- `algo/nn/nn_policy.py`：MLX 导入延迟到函数内部，避免父进程加载；
- `scripts/self_play_loop.py`：父进程零 MLX 导入，支持模型筛选门；
- `scripts/train_value_net_mc.py`：支持 `hidden_dims` 参数；
- `docs/recent-work.md`：算法与实验详细记录；
- `README.md`：加入 NN Agent 与自对弈章节。

---

## 7. 注意事项

- `output/` 目录仍在 `.gitignore` 中，本次 push 是**选择性强制 add**；之后新增的普通 output 文件不会自动进 git。
- 旧模型备份 `*_old128.*` 没 push，如果新机器上想对比，需要手动从原机器复制或重新训练。
- 如果新机器不是 macOS/Apple Silicon，MLX 可能无法直接运行，需要把 `algo/nn/model.py` / `value_model.py` 和训练脚本移植到 PyTorch。
