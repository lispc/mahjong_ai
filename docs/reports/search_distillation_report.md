# Search Distillation 实验报告（Path A / Path B）

> 时间：2026-07-06 至 2026-07-08
> 目标：通过更强的 value/policy label 突破 `Hybrid-FullAction-SoupDistilled` 天花板。
> 结论：**两条路均失败**。

---

## 1. 背景与动机

当前最强 `output/nn_full_action_best.pt` 来自「model soup + 蒸馏」，已接近 BC 天花板。Fable-5 评审后决定同时尝试：

- **Path A**：用更强的 rollout policy（训练好的 NN policy）生成 MC value labels，训练更大的 deep value net。
- **Path B**：用 exact depth-2 expectimax search 生成 search value / search action labels，蒸馏回 NN。

---

## 2. Path A：nnpolicy MC Rollout Value Labels

### 2.1 代码改动

- `algo/nn/nn_policy.py` 新增 `MJ_NN_POLICY_MODEL` 环境变量支持，允许 `nnpolicy` rollout 指定任意 policy 模型。

### 2.2 数据生成

| 阶段 | 局数 | 样本数 | 耗时 | 备注 |
|---|---|---|---|---|
| Pilot | 100 | 1,327 | 138s | 4 rollouts，64 workers，0 bad |
| 放大 | 3,750 | **50,815** | 372s | CPU-only nnpolicy rollout |

环境变量：
```bash
export MJ_ROLLOUT_POLICY=nnpolicy
export MJ_NN_POLICY_MODEL=output/nn_full_action_best.pt
```

### 2.3 训练

训练更大的 deep value net（1024/512/256）：
```bash
PYTHONPATH=. python3 scripts/train_value_net_mc.py \
    output/nn_training_data_selfplay_nnpolicy_rollout_3750.npz \
    80 256 0.001 1024,512,256 0.0 \
    --out output/nn_value_model_mc_nnpolicy_3750
```

结果：best val_loss 仅 **0.8040**（epoch 13），之后严重过拟合。

### 2.4 Benchmark

50 局，`EXACT_DEPTH2=1`，`MJ_NN_VALUE_MODEL=output/nn_value_model_mc_nnpolicy_3750.pt`：

| Agent | win | deal-in | Elo |
|---|---|---|---|
| V3d-2-nn-baseline_eval1 | 16.0% | 30.0% | 1437 |
| Hybrid-Best | 36.0% | 10.0% | 1533 |

### 2.5 结论

**阴性**。4-rollouts nnpolicy label 噪声太大，训出的 value net 当 leaf 明显弱于 baseline。

---

## 3. Path B：Exact Depth-2 Search Distillation

### 3.1 代码改动

- `scripts/rl/gen_search_value_data.py`：新增 `--games-per-task` 参数，支持细粒度断点续跑。
- `scripts/rl/distill_search.py`：从 backbone 初始化，联合训练 policy + value head。
- `scripts/rl/distill_search_dpo.py`：search-policy 的 DPO/ranking 蒸馏。
- `scripts/train_value_net_mc.py`：新增 `--out` 参数，避免覆盖默认 `nn_value_model_mc.pt`。

### 3.2 数据生成

#### B1. 250 局 pilot（leaf=nn，exact depth-2）

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    EXACT_DEPTH2=1 python3 scripts/rl/gen_search_value_data.py \
    output/nn_search_value_v3d2_exact_250.npz 250 4 \
    --depth 2 --leaf nn --cand-policy baseline_eval1 --seed-base 830000
```

- 输出：`output/nn_search_value_v3d2_exact_250.npz`
- 样本：11,932
- 耗时：4,233s（≈70 分钟）
- Value 分布：`mean=4.313, std=5.074, min=-1.064, max=61.000`

#### B2. 5,000 局放大（leaf=eval0）

为加速，把 leaf 从 `nn` 换成 `eval0`，workers 从 4 提到 32：

```bash
PYTHONPATH=. python3 scripts/rl/gen_search_value_data.py \
    output/nn_search_value_v3d2_eval0_5000.npz 5000 32 \
    --depth 2 --leaf eval0 --cand-policy baseline_eval1 \
    --seed-base 840000 --games-per-task 10
```

- 输出：`output/nn_search_value_v3d2_eval0_5000.npz`
- 样本：**238,882**
- 耗时：39,297s（≈10.9 小时）
- Value 分布：`mean=4.103, std=4.796, min=0.000, max=100.000`

#### B3. 5,000 局（leaf=nn）

意外发现 nn leaf 因 Cython + batch NN 评估反而更快：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 -u scripts/rl/gen_search_value_data.py \
    output/nn_search_value_v3d2_nn_5000.npz 5000 4 \
    --depth 2 --leaf nn --cand-policy baseline_eval1 \
    --seed-base 850000 --games-per-task 5
```

- 输出：`output/nn_search_value_v3d2_nn_5000.npz`
- 样本：**261,521**
- 耗时：9,966.8s（≈2.77 小时）
- Value 分布：`mean=4.612, std=5.624, min=-1.500, max=100.000`

注意：第一次尝试 4 GPU × 8 workers 导致 CUDA 死锁，后改用单 GPU 4 workers 稳定运行。

### 3.3 训练

#### BC Policy + Value Distillation

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 scripts/rl/distill_search.py \
    output/nn_search_value_v3d2_eval0_5000.npz \
    output/nn_full_action_best.pt \
    output/nn_search_distill_v3d2_eval0_5000.pt \
    --epochs 80 --batch 512 --lr 5e-5
```

| 数据 | val disc acc | value MSE |
|---|---|---|
| eval0 leaf 238k | **0.745** | 32.8 |
| nn leaf 261k | 0.624 | 38.4 |

#### Value Distillation（独立 deep value net）

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python3 scripts/train_value_net_mc.py \
    output/nn_search_value_v3d2_eval0_5000.npz \
    80 256 1e-3 512,256,128 0.0 \
    --out output/nn_search_value_v3d2_eval0_5000
```

结果：best val_loss ~28，train_loss 持续下降但 validation 震荡过拟合。

#### DPO Ranking Distillation

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 scripts/rl/distill_search_dpo.py \
    output/nn_search_value_v3d2_eval0_5000.npz \
    output/nn_full_action_best.pt \
    output/nn_search_dpo_v3d2_eval0_5000.pt \
    --epochs 40 --batch 512 --lr 5e-5 --beta 0.1 --value-weight
```

结果：best val acc **0.681**。

### 3.4 Benchmark 结果

#### Teacher 本身强度

| Teacher | 局数 | win | deal-in | Elo |
|---|---|---|---|---|
| V3d-2-eval0-baseline_eval1 | 400 | 17.5% | 19.5% | 1457 |
| V3d-2-nn-baseline_eval1 | 100 | 18.0% | 20.0% | 1393 |
| Hybrid-Best | 400 | 34.0% | 17.0% | 1625 |
| Baseline | 400 | 22.8% | 23.5% | 1446 |

#### Student 蒸馏结果

| Student | 数据来源 | 训练方法 | win | deal-in | Elo |
|---|---|---|---|---|---|
| PPO-SearchDistill | 12k eval0 leaf | BC policy | 12.0% | 34.0% | 1384 |
| PPO-SearchDistill5k | 238k eval0 leaf | BC policy | 9.0% | 26.0% | 1384 |
| V3-NN-PC | 238k eval0 leaf | Value distill | 15.0% | 21.0% | 1404 |
| PPO-SearchDPO5k | 238k eval0 leaf | DPO | 13.0% | 28.0% | 1412 |
| PPO-SearchDistillNN5k | 261k nn leaf | BC policy | 12.0% | 28.0% | 1436 |

### 3.5 结论

**Path B 阴性**。无论 leaf=eval0 还是 leaf=nn，exact depth-2 expectimax teacher 都无法超越 Hybrid-Best；蒸馏出的 student 全部弱于 Baseline/BeliefExp。

可能原因：
1. depth-2 search 的 leaf value 误差在期望和传播中累积。
2. `max_candidates=5` 的候选空间太窄，搜索找不到真正优势动作。
3. 当前 `nn_value_model_mc.pt` 质量不足以支撑深度搜索。
4. search value 是绝对值而非动作优势，直接回归难以恢复 ranking。

---

## 4. 产物清单

| 文件 | 说明 |
|---|---|
| `scripts/rl/gen_search_value_data.py` | exact search label 数据生成 |
| `scripts/rl/distill_search.py` | BC policy+value 蒸馏 |
| `scripts/rl/distill_search_dpo.py` | DPO ranking 蒸馏 |
| `output/nn_search_value_v3d2_exact_250.npz` | 250 局 nn leaf search labels |
| `output/nn_search_value_v3d2_eval0_5000.npz` | 5,000 局 eval0 leaf search labels |
| `output/nn_search_value_v3d2_nn_5000.npz` | 5,000 局 nn leaf search labels |
| `output/nn_search_distill_v3d2_eval0_5000.pt` | eval0 leaf BC distill 模型 |
| `output/nn_search_distill_v3d2_nn_5000.pt` | nn leaf BC distill 模型 |
| `output/nn_search_value_v3d2_eval0_5000.pt` | eval0 leaf value distill 模型 |
| `output/nn_search_dpo_v3d2_eval0_5000.pt` | eval0 leaf DPO 模型 |

---

## 5. 后续建议

Path A/B 失败后，下一个值得尝试的方向：

1. **Cython 化 eval2 / expectimax**：把搜索热路径编译加速，测试 depth-2/3 + 更多候选是否能产生更强 teacher。
2. **Exact endgame defensive head**：用 `output/exact_endgame_labels_1000.npz` 的 13,843 精确终局标签训练 defensive decision head。
3. **Wait distribution 头**：把待牌分布预测集成到 endgame solver 和 belief 更新中。

详见 `docs/handoff.md` §4。
