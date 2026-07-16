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
- 当前最强 agent：`HybridNNBeliefAgent`（Hybrid-FullAction-SoupDistilled），见 `docs/handoff.md` §1 与 §7。

---

## 2. 环境与依赖

### 2.1 推荐环境

> **⚠️ 2026-07 换机更新**：当前机器上旧 conda 环境 `mahjong`/`pypy39` **已不存在**。
> 但 **base 环境（Python 3.13）即自带 torch 2.12+cu126 + CUDA + 4×RTX3090**，
> 直接 `PYTHONPATH=. python3 ...` 即可跑测试 / 训练 / benchmark，无需重建 env。
> 下面 `mahjong`/`pypy39` 的说明为历史参考；如需 PyPy 加速 legacy MC 管线才需重建。

- **conda**: `/home/scroll/miniforge3`
- **主环境名**: `mahjong`（Python 3.10，PyTorch CUDA）
- **PyPy 环境名**: `pypy39`（Python 3.9 + PyPy 7.3.15，用于 MC value 计算）
- **激活方式**:
  ```bash
  source /home/scroll/miniforge3/etc/profile.d/conda.sh
  conda activate mahjong      # NN 训练/推理、自对弈生成、benchmark
  conda activate pypy39       # legacy eval2 MC value 计算
  ```

### 2.2 关键依赖

主环境 `mahjong`：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numba numpy cython
```

PyPy 环境 `pypy39`：
```bash
mamba create -n pypy39 python=3.9.18=1_73_pypy pip numpy -c conda-forge -y
conda activate pypy39
pip install numpy
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
- **legacy eval2 MC rollout 是 CPU 瓶颈**，用 PyPy 可加速 2-3 倍；
- **PyPy 不能 import Numba/torch**，因此 MC value 计算脚本必须保持纯 Python 依赖。

### 3.2 自对弈数据生成（GPU 并行）

每个对局里的 V3-NN agent 需要 GPU 做 NN 推理。单 GPU 会被多个 worker 抢占，因此**按 GPU 拆成 4 个独立进程**是最有效的用法：

```bash
bash scripts/generate_selfplay_4gpu.sh <总局数> <每 GPU workers> <seed_base>
```

示例（5000 局，每 GPU 32 workers，共 128 逻辑核跑满）：
```bash
bash scripts/generate_selfplay_4gpu.sh 5000 32 900000
```

这会在后台启动 4 个进程，分别用 `CUDA_VISIBLE_DEVICES=0/1/2/3`，输出：
- `output/selfplay_raw_5000_gpu{0,1,2,3}.pkl`
- `output/selfplay_raw_5000_gpu{0,1,2,3}.log`

### 3.3 MC rollout value label 计算（PyPy + CPU 并行）

MC rollout 中所有玩家用 legacy eval2（`algo.select`）决策，是纯 Python 任务。PyPy 对 eval2 有显著加速。

推荐做法：
1. 把 `output/selfplay_raw_N.pkl` 拆成 4 份；
2. 每份用 PyPy 32 workers 并行计算；
3. 最后合并 4 份 `.npz`。

```bash
source /home/scroll/miniforge3/etc/profile.d/conda.sh
conda activate pypy39
export PYTHONPATH=.
for p in 0 1 2 3; do
    pypy3 scripts/compute_mc_values.py \
        output/selfplay_raw_5000_part${p}.pkl \
        output/nn_training_data_selfplay_baseline_rollout_5000_part${p}.npz \
        4 32 240 200 250 > output/compute_mc_values_pypy_5000_part${p}.log 2>&1 &
done
wait
```

参数含义：`<raw.pkl> <out.npz> <n_rollouts> <n_workers> <timeout_per_task> <max_steps> <save_every>`。

> **关键优化**：`algo/eval/legacy.py` 的 `_eval0_cache` 已改为 `functools.lru_cache(maxsize=1_000_000)`。若无此限制，PyPy 长任务下每个 worker 的 cache 会无限增长，导致内存耗尽、BrokenProcessPool 或 OOM。

### 3.4 训练（GPU 0 即可）

模型很小（Policy-Value ~45k 参数，Deep Value ~100k 参数），单 GPU 训练 60 epochs 只需 ~1 分钟，不需要多卡 DDP。

```bash
PYTHONPATH=. python scripts/train_nn.py output/nn_training_data_selfplay_baseline_rollout_5000.npz 60 256 0.001 256
PYTHONPATH=. python scripts/train_value_net_mc.py output/nn_training_data_selfplay_baseline_rollout_5000.npz 60 256 0.001 512,256,128
```

### 3.5 Benchmark / Tournament

`driver/tournament.py` 使用 `ProcessPoolExecutor`，默认按 `n_workers` 并行。推荐：

```bash
bash scripts/benchmark_4gpu.sh 400 4
```

这会按 GPU 拆 4 个独立 benchmark 进程，每 GPU 跑 100 局、4 workers，充分利用 4 张卡。

### 3.6 监控资源利用率

```bash
# CPU / 负载 / 内存
watch -n 1 'top -bn1 | grep -E "load|Cpu|MiB Mem" | head -5'

# GPU
watch -n 1 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'

# 某个后台任务进度
tail -f output/compute_mc_values_pypy_5000_part0.log
```

目标：
- 数据生成阶段：4 GPU 都接近 100%，CPU 也接近满载。
- MC rollout 阶段：CPU 接近 100%，GPU 接近 0%，内存占用稳定在 ~200-400 GiB。
- 训练阶段：GPU 0 高占用，CPU 低占用。

---

## 4. 当前最强配置

```python
from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent

HybridNNBeliefAgent(
    'Hybrid-FullAction-SoupDistilled',
    nn_model_path='output/nn_full_action_best.pt',
    belief_kind='beliefexp',
    tenpai_threshold=28,
    device='cpu',
)
```

对应模型（PyTorch `.pt`）：
- `output/nn_full_action_best.pt` + `output/nn_full_action_best_config.json`
  - `TileConvNet`，128 channels / 6 residual blocks / 512 hidden
  - 带 dealin head、value head、tenpai head 与 **response head**（碰/杠/胡声明）
  - 来源：把 `nn_full_action_best.pt` 与 `nn_full_action_128000_epoch_07.pt` 做 model soup，再用该 soup 当教师蒸馏回单一模型

最近 benchmark（400 局同一 pool）：

| Agent | win | self | ron | deal-in | draw | Elo |
|---|---|---|---|---|---|---|
| Hybrid-newbest (SoupDist.) | 34.2% | 7.5% | 26.8% | 15.5% | 0.5% | 1629 |
| Hybrid-oldbest | 28.2% | 8.2% | 20.0% | 19.5% | 0.5% | 1519 |
| BeliefExp | 19.2% | 6.0% | 13.2% | 16.0% | 0.5% | 1502 |
| Baseline | 17.8% | 5.8% | 12.0% | 21.0% | 0.5% | 1350 |

对上一版旧 best `Hybrid-BE16k_t8`（200 局）：
- `Hybrid-FullAction-SoupDistilled`：胜率 40.5%，Elo 1601，点炮率 15.0%
- `Hybrid-BE16k_t8`：胜率 22.0%，Elo 1507，点炮率 20.5%

128k 行为克隆与 DPO/PPO/KTO 等后续实验均**未稳定超越**当前 SoupDistilled best；第二轮 soup/蒸馏 bootstrap 也已边际递减。详见 `docs/handoff.md §6.10–§6.14`。

> **2026-07-16 重要更正**：此前（07-06）记录的「duplicate 考场中 Baseline 显著强于当前 best（paired −20.2%）」是 **benchmark 脚本配对统计 bug**（候选与对手同名时前缀匹配误计），并非事实。用席位识别重算全部历史 pkl 后结论反转：**Hybrid-Best 在 duplicate 下显著强于 Baseline（+9.4%，5000 pairs）与 BeliefExp（+10.4%）**。但 soup→蒸馏这最后一环被证伪：NewBest vs OldBest 5000 pairs 仅 +0.2% [−0.5,+0.9]，score-proxy 亦为零——属 winner's curse，两模型视为同强。详见 `docs/reports/duplicate-reanalysis-0716.md`。**晋升/放弃决策一律按 `docs/eval-protocol.md`**：5000-pair duplicate arena、paired win diff CI 不含 0、独立种子复跑；Elo 不作依据。

---

## 5. 重要代码约定

### 5.1 NN 代码位置

| 文件 | 职责 |
|---|---|
| `algo/nn/model.py` | Policy-Value Net（PyTorch）；`TileConvNet` 现支持 `se_ratio`、`attn_heads`、`attn_layers`、`wait_dist_head` |
| `algo/nn/value_model.py` | Deep Value Net（PyTorch） |
| `algo/nn/nn_leaf.py` | ExpectiMax 叶子估值接口；可用 `MJ_NN_VALUE_MODEL` 指定 policy-value 网络当 value leaf |
| `algo/nn/nn_policy.py` | NN policy 候选生成接口；支持 `MJ_NN_POLICY_MODEL` 环境变量切换默认 policy 模型路径 |
| `algo/nn/features.py` | 175 维特征编码 |
| `algo/nn/mc_value.py` | MC rollout 快速对局 + value label |
| `algo/eval/endgame_solver.py` | 报听后终盘精确求解器（方向3） |
| `algo/agents/exact_endgame_agent.py` | 终盘使用精确求解器的 wrapper agent |

新增/扩展脚本：
- `scripts/rl/init_large_model.py`：按指定架构初始化 TileConvNet checkpoint
- `scripts/rl/train_large_model.py`：启动 large SE/attention 训练
- `scripts/rl/summarize_hpo.py`：汇总 HPO 训练日志
- `scripts/rl/make_model_soup.py`：同架构 checkpoint 权重平均
- `scripts/rl/gen_seq_opp_data.py`：对手序列数据生成（--mix 混合池，shard 断点续跑）
- `scripts/rl/train_seq_opp_model.py`：默听/待牌序列模型训练（seq/no-seq 消融，all/silent 拆分指标）
- `scripts/rl/oracle_endgame_gate.py`：完美待牌上界实验（方向 A/B 判死的关键证据）
- `scripts/rl/benchmark_duplicate.py`：duplicate 配对 benchmark（席位识别 + score-proxy 配对差）
- `scripts/rl/selfplay_bootstrap.py`：自对弈 bootstrap 管线（collect 含响应记录 / train_value / finetune / finetune_resp）
- `scripts/rl/peng_paired_eval.py`：god-state 快照 + 成对 rollout 因果评估（collect/evaluate/train，passfix_anchor 模式）

新增 agent（benchmark_pool token）：
- `hybridend:` = HybridNNBeliefEndgameAgent（Hybrid 接 exact-solver 搜索层，已证无增量）
- `besilent:` = BeliefSilentGuardAgent（BeliefExp + 序列模型默听防守）
- `hybridsilent:` = HybridNNBesilentAgent（Hybrid 接静默守卫搜索层）
- `hybridt:LABEL:PATH:THRESHOLD` = HybridNNBeliefAgent 可配置 tenpai_threshold
- `hybridfix:LABEL:PATH` = HybridNNBeliefTenpaiFixAgent（修复 _is_critical 的 tenpai_players 死代码；评测证实几乎无差，不晋升）

### 5.2 数据文件

| 文件 | 说明 |
|---|---|
| `output/nn_full_action_data_128000.npz` | 128k 局完整动作空间 BC 数据（~547万 discard / ~1681万 response 样本，16 GB） |
| `output/nn_full_action_data_32000.npz` | 32k 局完整动作空间 BC 数据（当前 best `nn_full_action_best.pt` 来源） |
| `output/nn_hybrid_soup_teacher_8000.npz` | 8000 局 Soup2 教师轨迹（334k discard / 2M response） |
| `output/nn_full_action_128000_epoch_{N}.pt` | 128k BC 每 epoch checkpoint（含 optimizer state，可 `--resume`） |
| `output/nn_training_data_selfplay_baseline_rollout_2000.npz` | 25569 条 2000 局 baseline rollout 数据（当前 best 来源） |
| `output/nn_training_data_selfplay_baseline_rollout_1000.npz` | 12835 条 1000 局 baseline rollout 数据 |
| `output/nn_training_data_mc.npz` | 46k 历史 MC 数据（BeliefExp + eval0 rollout） |
| `output/nn_training_data_selfplay.npz` | 50k 历史 V3-NN 自对弈数据 |
| `output/selfplay_raw_*.pkl` | 原始自对弈样本（context, hand14, action, features），等待计算 MC value |
| `output/duplicate_best_vs_baseline_400.pkl` | duplicate 评测原始结果（400 seeds，Hybrid vs Baseline） |
| `output/exact_endgame_labels_1000.npz` | 1000 局 BeliefExp 自对弈生成的 exact endgame 防守标签（13,843 样本） |
| `output/wait_dist_labels_*.npz` | 待牌分布监督样本（features + 34-dim wait one-hot） |
| `output/nn_wait_dist_tenpai_300.pt` | 300 局听牌样本上训练的 wait_dist head 初版 |

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

# 训练（legacy policy / value）
PYTHONPATH=. python scripts/train_nn.py output/nn_training_data_selfplay_baseline_rollout_2000.npz 60 256 0.001 256
PYTHONPATH=. python scripts/train_value_net_mc.py output/nn_training_data_selfplay_baseline_rollout_2000.npz 60 256 0.001 512,256,128

# 完整动作空间训练 / HPO
PYTHONPATH=. python scripts/rl/train_full_action.py \
    output/nn_hybrid_soup_teacher_8000.npz output/nn_full_action_best.pt \
    output/nn_full_action_hpo.pt --epochs 60 --batch 512 --lr 5e-5 \
    --optimizer adam --scheduler cosine --num_workers 4 --dp 0

# 初始化并训练 larger SE/attention 模型
PYTHONPATH=. python scripts/rl/init_large_model.py output/nn_full_action_large_se_init.pt \
    --channels 256 --n-blocks 8 --hidden-dim 1024 --se-ratio 16
PYTHONPATH=. python scripts/rl/train_large_model.py --gpu 0 \
    --init output/nn_full_action_large_se_init.pt --out output/nn_full_action_large_se.pt

# HPO 日志汇总
PYTHONPATH=. python scripts/rl/summarize_hpo.py

# 模型 soup
PYTHONPATH=. python scripts/rl/make_model_soup.py output/nn_full_action_soup.pt \
    output/nn_full_action_best.pt output/nn_full_action_128000_epoch_07.pt

# 自对弈数据生成（4 GPU）
bash scripts/generate_selfplay_4gpu.sh 5000 32 900000

# 合并 4 GPU 的 pkl（示例）
PYTHONPATH=. python -c "
import pickle, glob
all_samples = []
for pkl in sorted(glob.glob('output/selfplay_raw_5000_gpu*.pkl')):
    with open(pkl, 'rb') as f:
        all_samples.extend(pickle.load(f))
with open('output/selfplay_raw_5000.pkl', 'wb') as f:
    pickle.dump(all_samples, f)
print(len(all_samples))
"

# 计算 MC value label（PyPy，4 parts 并行）
source /home/scroll/miniforge3/etc/profile.d/conda.sh
conda activate pypy39
export PYTHONPATH=.
for p in 0 1 2 3; do
    pypy3 scripts/compute_mc_values.py \
        output/selfplay_raw_5000_part${p}.pkl \
        output/nn_training_data_selfplay_baseline_rollout_5000_part${p}.npz \
        4 32 240 200 250 > output/compute_mc_values_pypy_5000_part${p}.log 2>&1 &
done
wait

# Benchmark（4 GPU）
bash scripts/benchmark_4gpu.sh 400 4

# 任意 4 agent 同一 pool benchmark
SEATS="hybrid:newbest:output/nn_full_action_best.pt:beliefexp,beliefexp,baseline,v3nnpc" \
    PYTHONPATH=. python3 scripts/rl/benchmark_pool.py 400 16

# Duplicate（复式）benchmark：Hybrid vs Baseline，固定对手三件套
PYTHONPATH=. python3 scripts/rl/benchmark_duplicate.py \
    --a hybrid:Best:output/nn_full_action_best.pt \
    --b baseline \
    --opponents baseline,beliefexp,hybrid:Base:output/nn_full_action_best.pt \
    --n-seeds 400 --workers 32

# 生成 exact endgame 防守标签
PYTHONPATH=. python3 scripts/rl/generate_exact_endgame_labels.py \
    output/exact_endgame_labels_1000.npz 1000 32

# 生成 wait distribution 标签
PYTHONPATH=. python3 scripts/rl/generate_wait_dist_labels.py \
    output/wait_dist_labels_10000.npz 10000 32

# 训练 wait_dist head（在 current best backbone 上）
PYTHONPATH=. python3 scripts/rl/train_wait_dist.py \
    output/wait_dist_labels_10000.npz output/nn_full_action_best.pt \
    output/nn_wait_dist.pt --epochs 60 --batch 512 --device cuda:0
```

---

## 7. 已知问题与注意事项

1. **MLX 与 PyTorch 不能共存**：当前 `mahjong` 环境只装 PyTorch。恢复 MLX 需另建环境。
2. **PyPy 不能 import Numba**：`algo/nn/mc_value.py` 已做兼容，PyPy 下自动跳过 `fast_eval` import。
3. **legacy eval2 cache 必须限制大小**：`algo/eval/legacy.py` 使用 `lru_cache(maxsize=1_000_000)`，避免 PyPy 长任务内存爆炸。
4. **DataCollector 保存的是决策前状态**：`algo/agents/data_collectors.py` 中 `hand14` 和 `context` 快照必须在 `super().next()` 之前捕获，否则 MC rollout 会拿到不一致的 13 张手牌 / 弃牌后 context。
5. **`mc_value._greedy_discard` 返回 tile**：`algo.select(...)[0]` 返回的是 `(metric, tile)` 元组，取 tile 要用 `[0][1]`。
6. **`MJ_NN_POLICY_MODEL` 环境变量**：`algo/nn/nn_policy.py` 支持通过该变量切换默认 policy 模型（例如 MC rollout 的 `nnpolicy` 模式想使用 `output/nn_full_action_best.pt` 时设置）。
7. **tournament 默认只用一个 GPU**：大规模 benchmark 时若 GPU 0 成为瓶颈，用 `scripts/benchmark_4gpu.sh` 拆 4 进程。
8. **输出目录 `output/` 被 gitignore，但 config json 被跟踪**：修改模型配置后记得提交 `.json` 文件。
9. **不要提交 `.venv/`**：已加入 `.gitignore`。
10. **`tests/legacy_test.py` 已知失败（2026-07 起）**：`test_select` 断言 `select()[:2] == [2, 22]`，实际返回 `[22, 2]`——Cython eval2 集成（commit `05a303e`）后同分候选的 tie-break 顺序变化所致，**与强度无关**（同分值），clean tree 上同样失败。如需消除，把断言改为集合比较；勿为此改动 legacy eval 语义（Baseline 依赖 `algo.select`）。
11. **HybridNNBeliefAgent 无 `.context` 属性**：引擎 `getattr(agent, 'context', None)` 得 None，导致 Hybrid 在对战中**从不报听**（PPOAgent tenpai head 被跳过）。引擎无报听计分，当前评测目标下无碍；接入计分前需重新评估（见 `docs/eval-protocol.md` §1 备注）。
12. **HybridNNBeliefAgent._is_critical 的 tenpai 分支是死代码（2026-07-17 确认几乎无害）**：误用 `getattr(ctx,'tenpai',set())`（ContextV3 实为 `tenpai_players`），「对手报听→搜索」从未触发，只有弃牌数阈值生效。修复变体 `hybridfix` 1000-pair 仅 +0.1%（报听几乎都发生在 ≥28 弃牌后），**不改原类**；若未来报听提前（如引擎接入报听收益）需重估。
13. **Hybrid 的 melds 列表 quirk**：Hybrid 把同一 melds 列表共享给 nn_agent/belief_agent，`add_meld` 被三个组件各调一次 → 每个副露在列表中出现 3 次（`full_hand()` 巧合得到正确 3 张；gang 少 1 张）。写快照/状态注入代码时必须按此生产表示复刻（见 `scripts/rl/peng_paired_eval.py::_inject`）。
14. **RL/自对弈 bootstrap 判死（2026-07-17，15 候选零晋升）**：outcome 级 RL（AWBC/AWR）与配对因果标签 RL 均无法改进当前 best；根因=信用分配 SNR + 选择偏差混杂 + 误差状态在 175 维特征上不可分 + 在位者近最优。剩余方向：belief/danger 信号入特征重训。详见 `docs/reports/selfplay-bootstrap-0717.md`。复用资产：`output/peng_eval_v1.npz`（12k 配对 Δ）、`output/bootstrap_v{1,2}_merged.npz`、`scripts/rl/selfplay_bootstrap.py`、`scripts/rl/peng_paired_eval.py`。

---

## 8. 推荐工作流

1. 读 `docs/handoff.md` 确认当前最强配置和下一步。
2. 若下一步是自对弈迭代：
   - 用 `generate_selfplay_4gpu.sh` 生成原始样本；
   - 拆成 4 份，用 PyPy `compute_mc_values.py` 计算 MC value label；
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

8. **长任务必须定期输出进度日志**  
   任何预计运行 >10 分钟的任务，除了 checkpoint 外，还必须定期打印进度（如每 30 秒或每 1% 输出 `completed/total` 与 ETA）。进度日志写 `stdout`/`stderr` 并落盘，方便随时 tail 查看，避免盲等和误判任务卡住。多进程任务应使用共享计数器 + reporter thread 实现。

---

## 10. PPO 端到端 RL 管线（2026-07 新增，方案 B）

> 详见 `docs/reports/rl-ppo-report.md` 与 `docs/handoff.md §6`。

**代码位置**：

| 文件 | 职责 |
|---|---|
| `algo/rl/selfplay.py` | `PPOActorAgent`（NN policy 采样 + 轨迹记录，复用 ContextV3）+ 对局 runner + 对手池 + 多进程收集 |
| `algo/rl/reward.py` | 终局 result → 每座位标量奖励（win/deal_in/other_loss/draw 可配） |
| `algo/rl/ppo.py` | GAE(λ) + 轨迹展平（γ=1，终局稀疏奖励） |
| `scripts/rl/train_ppo.py` | PPO 训练主脚本（warm-start/`--init`、`--n-opponents/--opponents`、`--draw-reward`、熵退火、每 iter checkpoint） |
| `algo/agents/ppo_agent.py` | 加载 PPO 权重的对战 agent（合法 argmax，~1 ms/步，兼容 tournament） |
| `scripts/rl/benchmark_pool.py` | 任意 4 agent 同一 pool 比 Elo（避免跨 run 漂移）；`benchmark_rl.py`、`sanity_selfplay.py` |
| `algo/agents/opp_defensive_agent.py` | 用对手听牌概率放大 deal-in head 惩罚的纯前馈 agent（`oppdef` token） |
| `algo/agents/hybrid_nn_belief_opp_agent.py` | 当对手听牌概率高时提前切 BeliefExp 的 Hybrid agent（`hybridopp` token） |
| `algo/agents/danger_aware_agent.py` | 用 tile danger 预测模型防御重排的纯前馈 agent（`danger` token） |
| `scripts/rl/run_ablation_study.py` | 对当前最强 pipeline 做减法消融，输出 `docs/reports/ablation_report.md` |

**常用命令**（base 环境，`PYTHONPATH=. python3`）：
```bash
# 训练（自对弈 + 引分惩罚）
python3 scripts/rl/train_ppo.py --iters 80 --games-per-iter 512 --workers 32 \
    --device cuda:0 --draw-reward -0.4 --ent-coef 0.008 --ent-coef-final 0.001 --tag nn_rl_ppo_C
# 同一 pool 严谨 benchmark
SEATS="ppo:C:output/nn_rl_ppo_C.pt,beliefexp,baseline,v3nnpc" \
    python3 scripts/rl/benchmark_pool.py 400 40
```

**关键教训**：
1. **纯自对弈会坍缩到消极引分均衡**（draw=0 ≻ loss=−1）；引分惩罚 + 熵退火才能解锁学习。
2. **vs frozen 自对弈胜率 ≠ 真实强度**；只有固定强对手 benchmark 才算数（且要同一 pool，Elo 会跨 run 漂移）。
3. **让弱学习者直接打强敌（每局 2 家 Baseline/BeliefExp）适得其反**（几乎全负、梯度失效、退化）。
4. **纯前馈 PPO policy 仍弱于搜索型 agent**（胜率 ~10% vs BeliefExp 47%）；报酬整形的真实收益是「更守」（点炮 18%→15%）。
5. 产物一律写 `output/nn_rl_ppo_*`，**从不覆盖 `nn_model.pt` / `nn_value_model_mc.pt` / 任何 best**。
6. **RL+搜索融合（已试）**：把 PPO policy 当 V3 的候选生成器——纯 PPO 候选变差（会漏搜索最优弃牌）；PPO∪nn_model 并集只比「同候选数纯 nn」高 ~1.5%（噪声内），增益基本源于「候选更多」而非 RL。未对 V3-NN-PC 稳健提升。用 `scripts/rl/benchmark_pool.py` 的 `v3rlcand:`/`v3rlunion:`/`v3nnpck:` token 复现。
7. **大杠杆（卷积 + BC）= 当前最强 NN 策略**：`TileConvNet`（1D-Conv/ResNet over 34 牌轴 + GroupNorm，`algo/nn/model.py`，`build_model(config)` 按 `arch` 构造）+ `scripts/rl/pretrain_bc.py` 在 `nn_training_data_merged.npz` 上监督预训练（val acc 0.710）。产物 **`output/nn_conv_bc.pt`（纯前馈 ~1 ms）400 局公平池胜率 25.0%，与 Baseline(26.0%)/BeliefExp(25.8%) 打平**。部署：`PPOAgent(model_path='output/nn_conv_bc.pt')`。**PPO 自对弈在其上无加分；融合受弱 `nn_value_model_mc.pt` 拖累不划算。** 结论：真正起作用的是**卷积架构 + 全量 BC**，而非 RL 自对弈本身。
8. **benchmark 必须 `torch.set_num_threads(1)`**（`benchmark_pool.py` 已设）：多进程 fork 后 torch 线程过度订阅会让 benchmark 慢几十倍。Elo 在小样本/相近对手下不可信，**以胜率为准**，且要同一 pool（Elo 跨 run 漂移）。
9. **进一步压榨 conv-BC 均失败（纯前馈天花板）**：对手式 PPO 精调退化（convFT 18.8% < convBC 22.0%）；花色置换增广（`pretrain_bc.py::_suit_perms`，6×）修复过拟合但 val acc 仍 0.711、实战持平。BC ≈ 教师(eval2) ≈ Baseline/BeliefExp。
10. **突破天花板：NN + BeliefExp Hybrid + Search Trace Distillation**。用纯 `BeliefExpectimaxAgent` 当教师，对 `TileConvNet` 做 search trace distillation（soft target，T=4），8000 局数据 + 128/6/512 大网络得到 `output/nn_conv_bc_beliefexp_trace_8000_big_t4.pt`，组装成 `Hybrid-BE8k_t4` 后在 2000 局公平池胜率 25.7%、点炮 15.5%、Elo 1567，成为新的当前最强配置。详见 `docs/designs/conv-bc-roadmap.md`。

---

## 11. Offline RL / DPO 候选方案（128k 数据后的下一步）

PPO 在 128k checkpoint 上发散，说明 **online self-play RL 当前不稳定**。下一步可尝试 **offline RL**，直接在已有 128k BC 数据或 tournament 数据上做策略优化，避免在线采样的分布漂移。

### 11.1 方案 A：DPO（Direct Preference Optimization）

**思路**：把 128k 数据或自对弈 tournament 结果转成「偏好对」`(state, chosen_action, rejected_action)`，用 DPO loss 让模型更愿意选能带来更好结局的动作。

**偏好对构造方式（按复杂度）**：
1. **Outcome-level**：同一局里，最终赢家的动作 > 输家的动作；或按最终排名给每个动作打 reward，再 pair。
2. **Action-level**：对同一状态，用当前 BC 模型采样多个候选，用 fast value/搜索评估哪个更好，构造 preferred/rejected。
3. **Self-play pair**：让 BC 模型和另一个模型对打，收集轨迹后用比赛结果生成 pair。

**优点**：不需要训练 reward model，实现相对简单；天然与 BC 初始化兼容。
**风险**：偏好对质量决定一切；如果 pair 本身噪声大（麻将方差大），会训偏。

### 11.2 方案 B：Reward-Weighted BC / RWR

**思路**：给 128k 数据里的每个样本赋权，权重 = `exp(β * normalized_return)`，然后做加权 BC。

- 可用最终对局得分、排名、或手搓 heuristics（如是否点炮、是否和牌）作为 return。
- 超参 β 控制保守/激进：β=0 就是普通 BC；β 大则只保留高回报样本。

**优点**：几乎不改动训练代码，只是把 `CrossEntropyLoss` 改成 weighted。
**风险**：128k 数据来自 BC 模型自身，return 分布可能很窄，权重后容易过拟合到少数“赢的轨迹”。

### 11.3 方案 C：Filtered BC / BC-Best

**思路**：过滤出“高质量”子集再训练。

- 只保留最终和牌/获胜玩家的决策样本；
- 或过滤掉明显错误样本（如点炮前的弃牌、被和时的弃牌）。

**优点**：最简单；如果 128k 里有大量低质量样本，filtering 可能比加数据更有效。
**风险**：BC 数据来自模型自身，输赢不完全由当前动作决定，filtering 可能丢掉很多正常样本。

### 11.4 方案 D：Offline Actor-Critic（IQL / CQL 风格）

**思路**：用 128k 数据训练 Q 函数或 V 函数，再用 policy extraction 得到策略。

- 对每个 `(state, action, final_return)` 样本训练 Q(s,a)；
- 用 advantage-weighted regression（AWR/IQL）提取 policy。

**优点**：理论上能利用 return 信号直接优化长期收益。
**风险**：动作空间大（discard 34 + response 4），需要大量数据；实现复杂度明显高于 BC/DPO；容易过拟合到离线数据。

### 11.5 方案 E：Rejection Sampling / Best-of-N Distillation

**思路**：对 128k 数据里的每个状态，用当前 BC 模型采样 N 个候选动作，用快速 evaluator（如 `algo.select` 或 conv-BC value）选出最佳，再用这些 best 样本 fine-tune BC。

**优点**：不需要 reward model 或偏好对，直接 distillation；N 越大越接近“搜索增强版 BC”。
**风险**：速度较慢（每个状态要评估 N 次）；如果 evaluator 不准，蒸馏对象就有偏差。

### 11.6 2026-07 调研结论与执行计划

2026-07 中旬又做了一次广义棋牌 AI 调研（不限于麻将），主要结论：

- **稀疏 reward 是最大瓶颈**：Suphx、Tjong、Evo-Sparrow 等都强调 reward shaping / fan-backward / 全局 reward 预测，而不是直接用 ±1 终局信号做 DPO/PPO。
- **KTO 比 DPO 更适合二元反馈**：KTO 只需要每个样本标“好/坏”，不需要配对，对麻将这种高方差场景更友好。
- **对手建模仍是麻将核心**：SIMCAT、Suphx、OMIS（NeurIPS 2024）都显示，准确推断对手手牌/听牌状态能显著降低点炮率。
- **进化策略 / 搜索蒸馏是可行替代**：Evo-Sparrow 用 CMA-ES 优化 LSTM 权重避开梯度崩溃；Suphx 用运行时搜索+蒸馏把搜索能力注入 NN。

**已决定同时启动两条路线**：

1. **A. Reward shaping + KTO**：在现有 128k 数据上，把 win/loss/draw 作为二元反馈，用 KTO loss 微调完整动作 policy。预期快速验证“KTO 是否比 DPO 更稳”。后续若有效，再加入 shanten/ukeire 等 shaped reward。
2. **C. 对手建模**：生成新的自对弈数据，记录每个决策点所有对手的隐藏手牌和听牌状态，训练一个**对手听牌/手牌预测**的辅助网络；最终接入 belief 更新或 NN 的额外输入。

**不再优先**：跨状态 outcome-level DPO（已验证无效）、PPO 在线自对弈（已发散）。备选：Evo-Sparrow 式 CMA-ES、Suphx 式运行时搜索蒸馏。

**建议先快速验证 B 和 C**（各 1–2 小时），如果有效再尝试 DPO；D 作为长期备选。
