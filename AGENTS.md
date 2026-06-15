# AGENTS.md

> 本文件面向后续继续开发本项目的 AI agent / 协作者。阅读顺序：
> 1. 本文件（环境、当前状态、硬件利用）
> 2. `docs/handoff.md`（最强配置与下一步建议）
> 3. `docs/README.md`（文档索引）

---

## 1. 项目是什么

一个**晋北麻将 AI 研究与对战平台**，核心围绕：
- 规则：推倒胡、不能吃牌、报听后锁手牌。
- 算法：ExpectiMax / MCTS + 神经网络（Policy-Value Net + Deep Value Net）。
- 当前最强 agent：`BeliefExpectimaxV3Agent`（V3-NN-BE1），见 `docs/handoff.md`。

---

## 2. 环境与依赖

### 2.1 推荐环境

- **conda**: `/home/scroll/miniforge3`
- **环境名**: `mahjong`
- **Python**: 3.10
- **激活方式**:
  ```bash
  source /home/scroll/miniforge3/etc/profile.d/conda.sh
  conda activate mahjong
  ```

### 2.2 关键依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numba numpy cython
```

> **注意**：NN 后端已切换到 **PyTorch**。原 MLX 版本备份在 `algo/nn/model_mlx.py`、`algo/nn/value_model_mlx.py`、`scripts/train_nn_mlx.py`、`scripts/train_value_net_mc_mlx.py`。由于 PyTorch 与 MLX 的 CUDA 包版本冲突，**不要同时在一个环境里安装两者**。如需恢复 MLX 实验，请另建环境。

### 2.3 验证环境

```bash
source /home/scroll/miniforge3/etc/profile.d/conda.sh && conda activate mahjong
PYTHONPATH=. python run_tests.py
PYTHONPATH=. python tmp/benchmark_new_models.py 50 4
```

---

## 3. 如何充分利用高配服务器

服务器配置：
- **CPU**: AMD EPYC 7702，128 逻辑核
- **GPU**: 4 × NVIDIA RTX 3090（24 GB 显存）
- **内存**: ~1 TB

### 3.1 核心原则

- **GPU 用于 NN 推理/训练**；
- **CPU 用于对局模拟和 MC rollout**；
- **瓶颈通常在 CPU 上的 MC rollout**，而不是 GPU 训练。

### 3.2 自对弈数据生成（GPU 并行）

每个对局里的 V3-NN agent 需要 GPU 做 NN 推理。单 GPU 会被多个 worker 抢占，因此**按 GPU 拆成 4 个独立进程**是最有效的用法：

```bash
bash scripts/generate_selfplay_4gpu.sh <总局数> <每 GPU workers> <seed_base>
```

示例（1000 局，每 GPU 32 workers，共 128 逻辑核跑满）：
```bash
bash scripts/generate_selfplay_4gpu.sh 1000 32 800000
```

这会在后台启动 4 个进程，分别用 `CUDA_VISIBLE_DEVICES=0/1/2/3`，输出：
- `output/selfplay_raw_1000_gpu{0,1,2,3}.pkl`
- `output/selfplay_raw_1000_gpu{0,1,2,3}.log`

### 3.3 MC rollout value label 计算（CPU 并行）

MC rollout 中所有玩家用 Baseline（`algo.select` / eval2）决策，是纯 CPU 任务，应开满 128 核：

```bash
PYTHONPATH=. python scripts/compute_mc_values.py \
  output/selfplay_raw_1000.pkl \
  output/nn_training_data_selfplay_baseline_rollout_1000.npz \
  4 128
```

参数含义：`<raw.pkl> <out.npz> <n_rollouts> <n_workers>`。

> 若仍太慢，可进一步把每个样本的多次 rollout 拆成独立任务（当前是每个 worker 串行跑完 4 次 rollout），但 128 workers 时 CPU 通常已满载。

### 3.4 训练（GPU 0 即可）

模型很小（Policy-Value ~45k 参数，Deep Value ~100k 参数），单 GPU 训练 60 epochs 只需 ~1 分钟，不需要多卡 DDP。

```bash
PYTHONPATH=. python scripts/train_nn.py output/nn_training_data_merged.npz 60 256 0.001 256
PYTHONPATH=. python scripts/train_value_net_mc.py output/nn_training_data_merged.npz 60 256 0.001 512,256,128
```

### 3.5 Benchmark / Tournament

`driver/tournament.py` 使用 `ProcessPoolExecutor`，默认按 `n_workers` 并行。推荐：

```bash
PYTHONPATH=. python tmp/benchmark_new_models.py 200 4
```

> 注意：tournament 里每个 worker 也会加载 PyTorch 模型做 NN 推理，4 workers 共享 GPU 0。若想让 tournament 也利用 4 GPU，可手动拆成 4 个独立进程分别跑不同 benchmark，或改造 tournament 支持按 worker 分配 GPU。

### 3.6 监控资源利用率

```bash
# CPU / 负载
watch -n 1 'top -bn1 | grep -E "load|Cpu" | head -3'

# GPU
watch -n 1 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'

# 某个后台任务进度
tail -f output/compute_mc_values_1000.log
```

目标：
- 数据生成阶段：4 GPU 都接近 100%，CPU 也接近满载。
- MC rollout 阶段：CPU 接近 100%，GPU 接近 0%。
- 训练阶段：GPU 0 高占用，CPU 低占用。

---

## 4. 当前最强配置

```python
BeliefExpectimaxV3Agent(
    'V3-NN',
    expectimax_depth=1,
    max_candidates=5,
    leaf_evaluator='nn',
    candidate_policy='baseline_eval1',
    verbose=False,
)
```

对应模型（PyTorch `.pt`）：
- `output/nn_model.pt` + `output/nn_model_config.json`
- `output/nn_value_model_mc.pt` + `output/nn_value_model_mc_config.json`

详见 `docs/handoff.md`。

---

## 5. 重要代码约定

### 5.1 NN 代码位置

| 文件 | 职责 |
|---|---|
| `algo/nn/model.py` | Policy-Value Net（PyTorch） |
| `algo/nn/value_model.py` | Deep Value Net（PyTorch） |
| `algo/nn/nn_leaf.py` | ExpectiMax 叶子估值接口 |
| `algo/nn/nn_policy.py` | NN policy 候选生成接口 |
| `algo/nn/features.py` | 175 维特征编码 |
| `algo/nn/mc_value.py` | MC rollout 快速对局 + value label |

### 5.2 数据文件

| 文件 | 说明 |
|---|---|
| `output/nn_training_data_mc.npz` | 46k 历史 MC 数据（BeliefExp + eval0 rollout） |
| `output/nn_training_data_selfplay.npz` | 50k 历史 V3-NN 自对弈数据 |
| `output/nn_training_data_merged.npz` | 96k 合并数据 |
| `output/selfplay_raw_*.pkl` | 原始自对弈样本（context, hand14, action, features），等待计算 MC value |

### 5.3 模型文件格式

- 当前使用 **PyTorch `.pt`**。
- 配置文件是 `.json`，包含 `framework: "pytorch"`。
- 不要提交 `.pt`/`.npz` 模型权重到 git（`output/` 已在 `.gitignore`）。
- 配置文件 `output/nn_model_config.json` 和 `output/nn_value_model_mc_config.json` 被 git 跟踪，修改后应提交。

---

## 6. 常用命令速查

```bash
# 测试
PYTHONPATH=. python run_tests.py

# 训练
PYTHONPATH=. python scripts/train_nn.py output/nn_training_data_merged.npz 60 256 0.001 256
PYTHONPATH=. python scripts/train_value_net_mc.py output/nn_training_data_merged.npz 60 256 0.001 512,256,128

# 自对弈数据生成（4 GPU）
bash scripts/generate_selfplay_4gpu.sh 1000 32 800000

# 合并 4 GPU 的 pkl（示例）
PYTHONPATH=. python -c "
import pickle, os
all_samples = []
for gpu in range(4):
    with open(f'output/selfplay_raw_1000_gpu{gpu}.pkl', 'rb') as f:
        all_samples.extend(pickle.load(f))
with open('output/selfplay_raw_1000.pkl', 'wb') as f:
    pickle.dump(all_samples, f)
print(len(all_samples))
"

# 计算 MC value label（128 CPU）
PYTHONPATH=. python scripts/compute_mc_values.py \
  output/selfplay_raw_1000.pkl \
  output/nn_training_data_selfplay_baseline_rollout_1000.npz 4 128

# 合并历史 + 新数据
PYTHONPATH=. python -c "
from scripts.self_play_loop import merge_data_files
merge_data_files(
    'output/nn_training_data_merged_v2.npz',
    'output/nn_training_data_mc.npz',
    'output/nn_training_data_selfplay_baseline_rollout_1000.npz'
)
"

# Benchmark
PYTHONPATH=. python tmp/benchmark_new_models.py 200 4
```

---

## 7. 已知问题与注意事项

1. **MLX 与 PyTorch 不能共存**：当前 `mahjong` 环境只装 PyTorch。恢复 MLX 需另建环境。
2. **DataCollector 保存的是决策前状态**：`algo/agents/data_collectors.py` 中 `hand14` 和 `context` 快照必须在 `super().next()` 之前捕获，否则 MC rollout 会拿到不一致的 13 张手牌 / 弃牌后 context。
3. **`mc_value._greedy_discard` 返回 tile**：`algo.select(...)[0]` 返回的是 `(metric, tile)` 元组，取 tile 要用 `[0][1]`。
4. **tournament 默认只用一个 GPU**：大规模 benchmark 时若 GPU 0 成为瓶颈，可拆进程或改造 tournament。
5. **输出目录 `output/` 被 gitignore，但 config json 被跟踪**：修改模型配置后记得提交 `.json` 文件。
6. **不要提交 `.venv/`**：已加入 `.gitignore`。

---

## 8. 推荐工作流

1. 读 `docs/handoff.md` 确认当前最强配置和下一步。
2. 若下一步是自对弈迭代：
   - 用 `generate_selfplay_4gpu.sh` 生成原始样本；
   - 用 `compute_mc_values.py` 计算 MC value label；
   - 合并数据 → 训练 candidate → benchmark → 按 Elo 门限决定是否替换。
3. 若下一步是算法实验：先改 agent/eval，再跑 `tmp/benchmark_new_models.py` 或新建 benchmark 脚本验证。

---

## 9. Agent 工作守则（血泪教训）

以下规则来自实际操作中的重大失误，后续 agent 必须遵守。

1. **永不主动 kill 运行中的长任务**  
   除非用户明确说停止，或任务已确认 fatal error。已经跑了一大半的任务，先让它跑完拿到结果；改进方案放到下一批数据再做。

2. **长任务必须有 checkpoint/断点续跑**  
   任何预计运行 >10 分钟的任务，脚本必须支持 checkpoint。checkpoint 文件保留到最终输出文件确认生成成功，期间**严禁手动清理 `*.checkpoint*`**。

3. **清理任何文件前先列出并确认**  
   尤其不能清理 checkpoint、模型权重（`.pt`/`.npz`）、训练数据、原始对局样本。`rm -f` 是高危操作，使用前必须三思并向用户汇报。

4. **改脚本后必须先用小数据验证**  
   任何修改（尤其涉及多进程、文件格式、模型加载）必须先在 2 局 / 10 个样本级别跑通，再放大到全量。

5. **沉没成本优先**  
   imperfect 但完整的数据 >> 完美但没数据。不要为了小改进毁掉已经生成/计算了几个小时的数据。

6. **高风险操作前必须获得用户明确授权**  
   包括但不限于：kill 长任务、删除数据、重跑 >10 分钟的任务、切换底层 ML 框架、覆盖 current best 模型、push 到主分支。

7. **用户说“可以”不等于授权所有后续操作**  
   用户同意某个方向后，具体执行步骤中若涉及破坏已有成果，仍需单独确认。
