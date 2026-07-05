# Handoff：换机器继续工作的指南

> 本文档记录当前项目状态、最强配置、已 push 的数据/checkpoint，以及建议的后续路径。换到新机器后先读这篇。

---

## 1. 当前最强配置

经过 search trace distillation 迭代，当前最强实用配置为 **Hybrid-BE16k_t8**：

```python
# benchmark token: hybrid:BE16k_t8:output/nn_conv_bc_beliefexp_trace_16000_big_t8.pt:beliefexp
# 对应类：algo.agents.hybrid_nn_belief_agent.HybridNNBeliefAgent
HybridNNBeliefAgent(
    'Hybrid-BE16k_t8',
    nn_model_path='output/nn_conv_bc_beliefexp_trace_16000_big_t8.pt',
    belief_kind='beliefexp',
    tenpai_threshold=28,
    device='cpu',
)
```

2000 局 benchmark（4-agent pool）：

```
Agent          win      self     ron      deal-in    draw     Elo      avg_ms
Baseline       0.244    0.063    0.181    0.212      ...      1499     ~300
Hybrid-Base    0.214    0.055    0.160    0.176      ...      1425     ~100-300(critical)
Hybrid-BE8k_t8 0.257    0.065    0.192    0.172      ...      1495     ~100-300(critical)
Hybrid-BE16k_t8 0.258   0.068    0.191    0.163      ...      1581     ~100-300(critical)
```

**Hybrid-BE16k_t8** Elo **1581**，胜率 25.8%，点炮 16.3%，同时优于 Baseline、Hybrid-Base 和 Hybrid-BE8k_t8。

对应模型（PyTorch `.pt`）：

- `output/nn_conv_bc_beliefexp_trace_16000_big_t8.pt` + `output/nn_conv_bc_beliefexp_trace_16000_big_t8_config.json`
  - `TileConvNet`，128 channels / 6 residual blocks / 512 hidden，带 dealin head 与 value head
  - 训练数据：16000 局纯 `BeliefExpectimaxAgent` 搜索轨迹（`output/nn_teacher_beliefexp_trace_16000.npz`，734073 样本）
  - 蒸馏设置：α=0.5，T=8，β=0.3，λ_dealin=0.5

备份：

- `output/nn_conv_bc_beliefexp_trace_8000_big_t8.pt` / `..._config.json`（上一版本候选）
- `output/nn_conv_bc_beliefexp_trace_8000_big_t4.pt` / `..._config.json`（再上一版本候选）
- `output/nn_conv_bc_beliefexp_trace_4000_big.pt` / `..._config.json`（上一代候选 Hybrid-BE4k_big）
- `output/nn_conv_bc_hybrid_2000.pt` / `..._config.json`（上一代稳健候选 Hybrid-Base）
- `output/nn_conv_bc_dealin_2000_l07.pt` / `..._config.json`（纯前馈首选）
- `output/nn_model_best_1581.pt` / `output/nn_value_model_mc_best_1581.pt`（历史 V3-NN-PC best）

> **项目状态：PPO 自对弈微调和 8× 数据缩放均已证伪。** 当前最佳仍为 `Hybrid-BE16k_t8`；剩余未验证的大方向是「完整动作空间 NN 化（吃/碰/杠/报听）」和「换教师/推理时搜索」。

---

## 2. 已 push 的数据与 Checkpoint

以下文件已加入 git（模型权重在 `.gitignore` 里默认被忽略，用 `-f` 强制跟踪）：

| 文件 | 说明 |
|---|---|
| `output/nn_model.pt` | 当前 policy-value 网络权重（PyTorch） |
| `output/nn_model_config.json` | policy net 配置 |
| `output/nn_value_model_mc.pt` | 当前 deep value 网络权重（PyTorch） |
| `output/nn_value_model_mc_config.json` | value net 配置 |
| `output/nn_training_data_selfplay_baseline_rollout_5000.npz` | 68529 条 5000 局 baseline rollout MC value 数据 |
| `output/nn_training_data_selfplay_baseline_rollout_2000.npz` | 25569 条 2000 局 baseline rollout MC value 数据 |
| `output/nn_training_data_selfplay_baseline_rollout_1000.npz` | 12835 条 1000 局 baseline rollout MC value 数据 |
| `output/nn_teacher_beliefexp_trace_16000.npz` | 734073 条 16000 局纯 BeliefExp 教师搜索轨迹（当前 best 来源，本地存在） |
| `output/nn_teacher_beliefexp_trace_128k.npz` | 5885961 条 128000 局纯 BeliefExp 教师搜索轨迹（8× 缩放实验，未超越 16k，本地存在） |

---

## 3. 环境要求

- Python 3.10（主环境 `mahjong`）
- conda: `/home/scroll/miniforge3`
- PyTorch CUDA 用于 NN 训练/推理：
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  pip install numba numpy cython
  ```
- **PyPy 3.9** 用于加速 legacy eval2 MC value 计算：
  ```bash
  mamba create -n pypy39 python=3.9.18=1_73_pypy pip numpy -c conda-forge -y
  ```
- **MLX 版本已备份**：`algo/nn/model_mlx.py`、`algo/nn/value_model_mlx.py`、`scripts/train_nn_mlx.py`、`scripts/train_value_net_mc_mlx.py`。由于 MLX 与 PyTorch CUDA 包版本冲突，不要在一个环境里同时安装两者。

---

## 4. 到新机器后先验证

```bash
source /home/scroll/miniforge3/etc/profile.d/conda.sh
conda activate mahjong
PYTHONPATH=. python run_tests.py
PYTHONPATH=. python tmp/benchmark_new_models.py 100 4
```

---

## 5. 建议的后续路径

### 5.1 已验证：CPython 64 workers 是 baseline rollout 的甜点

- 在 128 CPU core 机器上，单 part 64 workers 跑 100 样本约 70s，96/128 workers 收益很小。
- 4 parts × 32 workers 并发会严重抢 CPU/cache，反而比 2 parts × 64 workers 慢很多。
- 推荐并发策略：**2 parts × 64 workers**，两批跑完 4 parts。

### 5.2 已验证：nnpolicy rollout 不可行

- 尝试用训练好的 Policy Net top-1 作为 MC rollout policy，生成 10000 局（134,358 样本）。
- 训练出的 V3-NN-PC 在 400 局 benchmark 中 Elo 仅 **1386**，远低于当前 best **1581**。
- 结论：NN policy 作为 rollout policy 太弱，value label 质量差，导致模型退化。应继续使用 **baseline (`algo.select`) rollout**。

### 5.3 已放弃：单纯放大 baseline rollout 数据规模

10000 局 baseline rollout（134,358 样本）已完成，但训练出的 V3-NN-PC Elo 仅 **1528**（默认网络）和 **1462**（扩大网络 + weight decay），均低于当前 best **1580**。

关键原因：
- value net 在 10000 局数据上 val_loss 最低只能到 ~0.78，而 5000 局数据可达 ~0.199
- 单纯增加样本量没有降低 label 噪声，反而让网络难以拟合
- 扩大网络（1024 hidden policy + 1024/512/256 value）+ weight decay 也无法解决

结论：**数据量不是当前瓶颈，数据/标签质量和算法结构才是**。已恢复 true best_1581 模型。

### 5.4 当前主攻方向：特征工程（已验证 2000 局，效果不佳）

DetMCTS + NN value 截断已快速验证：400 局 benchmark Elo 仅 **1315**，远低于当前 best 1580，暂时放弃该路线。

#### 特征工程第一次尝试（2000 局 baseline rollout + 291 维输入）

已实现 291 维输入（原 175 维）：

- 手牌质量：向听数(1) + 有效进张(1) + 待牌分布(34)
- 自己的弃牌历史(34)
- 壁牌/筋牌安全度(34)
- 对手花色偏好(12)
- 保留原特征：手牌(34)、剩余牌山(34)、对手弃牌(3×34)、报听 flag(4)、进度(1)

同步修复了 `features.py` 中 `_suit_of_tile` 的 tile value 编码错误，并在 `nn_leaf.py` 中补上了 leaf value 推理时缺失的 36 维手牌质量特征。

生成 2000 局（25,315 样本）baseline rollout 数据：

- 使用 `DataCollectorBaseline`（eval0 + baseline_eval1）决策
- `MJ_FAST_ROLLOUT=1` 加速 MC rollout
- 输出：`output/nn_training_data_selfplay_baseline_rollout_2000_v2.npz`

训练结果：

- Policy net val_acc 仅 **0.346**，value net best val_loss **0.743**（严重过拟合）
- 400 局 benchmark：V3-NN-PC Elo **1379**，远低于 best_1581 的 **1581**
- V3-NN-PC 平均决策时间 **3.36s**（原 best 约 155ms），新 quality 特征显著拖慢 leaf evaluation

可能原因：

1. 数据量不足（25k vs 旧 best 68k），且来自较弱策略（eval0+baseline_eval1），分布与 V3-NN-PC 不匹配。
2. 手牌质量特征计算昂贵，leaf evaluation 调用 `eval_v2.shanten/ukeire/winning_tiles`，使 expectimax 极慢。
3. 新增特征（尤其是 suji/safety、opp_suit_pref）可能噪声大于信号。

当前已恢复 `output/nn_model.pt` / `output/nn_value_model_mc.pt` 为 best_1581（175 维）。

#### 已验证失败的其他方向

- **DetMCTS / MCTS 替代 ExpectiMax**
  - Flat MC（10 worlds × 6 candidates，fast rollout）：Elo ~1391，决策时间 ~211ms
  - NN rollout（3 worlds × 4 candidates）：Elo ~1353，决策时间 ~394ms
  - 结论：当前 DetMCTS 实现无法替代 BeliefExpectimax/V3-NN-PC。

- **V3-NN-PC 配置调优**
  - 测试了 max_candidates / expectimax_depth / defense_margin 的多种组合。
  - depth=2 极慢（100 局 4 workers 跑 46 分钟以上），不实用。
  - depth=1 的初步结果也未超过 best_1581。

- **Expert Iteration / outcome 加权训练**
  - 用当前 best_1581 V3-NN-PC 自对弈 500 局（6,845 样本），value label 用最终 outcome。
  - Policy net val_acc 仅 0.36，value net best val_loss 0.737，严重过拟合。
  - 200 局 benchmark：V3-NN-PC Elo **1389**，远低于 best_1581。
  - 结论：500 局 + outcome label 噪声太大，无法提升；扩大数据量或改用 MC value label 可能再试，但当前证据不乐观。

#### 当前状态

- **当前最强配置仍为 V3-NN-PC（175-dim，Elo 1581）**。
- `output/nn_model.pt` + `output/nn_value_model_mc.pt` 已恢复为 best_1581。

#### 仍开放的长期方向

- 收集/生成更高质量的训练数据（如人类对局、多 agent 混合对局）；
- 改进 value label（如用更强 rollout policy 或结合 outcome 与 MC value）；
- 扩展动作空间（吃、碰、杠、报听决策）；
- 对手建模与防守推理；
- 若继续特征工程，需先解决推理速度问题，并找到与 175-dim 模型迭代的路径。

---

## 6. PPO 端到端 RL 探索（2026-07，换机后新增）

> 详见 [`reports/rl-ppo-report.md`](reports/rl-ppo-report.md)。

**换机现状**：旧 conda 环境 `mahjong`/`pypy39` 已不在，但 **base 环境即有 torch 2.12+cu126 + 4×RTX3090**，`PYTHONPATH=. python3` 可直接跑。`nn_model.pt` warm-start 健在；`best_1581` 备份与 5000 局数据已丢失（PPO 不需要）。

**新增管线**（`algo/rl/` + `scripts/rl/` + `algo/agents/ppo_agent.py`）：真正的端到端 RL 闭环——自对弈在线采样 → GAE(λ) → PPO clip 更新（masked policy + value + entropy）→ checkpoint → benchmark。全链路验证正确，1 ms/步。

**诚实结论**：
- 纯自对弈会坍缩到「消极引分均衡」（draw=0 ≻ loss=−1）；引分惩罚 + 熵退火可解锁学习。
- 直接对抗强敌（每局 2 家 Baseline/BeliefExp）**适得其反**（弱学习者常年负、梯度无法指向赢）。
- 400 局同一 pool 严谨 benchmark：PPO 变种彼此在 ~70 Elo 内（噪声级），报酬整形带来的真实收益是**点炮率 18%→15%（更守）**，而非更强。
- **纯前馈 PPO policy 对搜索型 Baseline/BeliefExp 仍弱**（胜率 ~10% vs ~47%），与「NN 路线只有带搜索的 V3-NN-PC 才强」一致。
- **RL+搜索融合（已做）**：把 PPO policy 当 `BeliefExpectimaxV3Agent` 的候选生成器。纯 PPO 候选变差；PPO∪nn_model 并集在 600 局对照里胜率 24.2%，只比「同候选数纯 nn」(nn10, 22.7%) 高 1.5%（噪声内），增益基本可用「候选更多」解释，**未对 V3-NN-PC 产生稳健提升**。根因是 PPO policy 本身不够强。
- 产物 `output/nn_rl_ppo_{A,C,...}.pt`（**未覆盖 nn_model.pt / 任何 best**）；代表模型 `nn_rl_ppo_C.pt`。

### 6.1 大杠杆：卷积网络 + 监督预训练(BC) —— **当前最强 NN 策略**

「纯前馈弱」被证伪：那是**小 MLP(82k) 的锅**。换成 `TileConvNet`（1D-Conv/ResNet over 34 牌轴 + GroupNorm，283k 参数，`algo/nn/model.py`）+ 用 `nn_training_data_merged.npz`(96721 条) 做**监督预训练(BC)**（`scripts/rl/pretrain_bc.py`，val acc **0.710**）：

- **`output/nn_conv_bc.pt`（conv-BC，纯前馈 ~1 ms/步）与项目最强搜索 agent 打平**：400 局公平池胜率 **25.0%** vs Baseline 26.0% / BeliefExp 25.8%（且快 200–300 倍）。用 `algo.agents.ppo_agent.PPOAgent(model_path='output/nn_conv_bc.pt')` 部署。
- **PPO 自对弈在 conv-BC 之上无加分**（convBC 35.1% ≈ convPPO 33.2%）；强初始化下 vanilla PPO 到顶。
- **进一步压榨也未超过 conv-BC**：对手式 PPO 精调**反而退化**（convFT 18.8% < convBC 22.0%）；花色置换增广（6×）修复了过拟合但 val acc 仍 0.711、实战持平。**BC 天花板 ≈ 教师(eval2) ≈ Baseline/BeliefExp**。
- **融合（conv-BC 当 V3 候选器）在公平池反而不如 conv-BC 单独**（20.8% vs 25.0%），因搜索复用的 `nn_value_model_mc.pt` 偏弱、re-rank 帮倒忙。
- **建议**：把 conv-BC 作为新的强基线/部署模型。若继续：训**卷积 value 网络**替换弱 leaf 再评估融合；或搜索在环的 AlphaZero 式迭代突破 BC 上限。

**下一步（按 ROI）**：① 训 conv value 网络替换弱 leaf，重测融合；② AlphaZero 式（policy 目标=搜索访问分布）迭代；③ 更多/更优 BC 数据；④ 扩展动作空间。

### 6.2 AlphaZero 式深搜索蒸馏（2026-07，进行中）

**动机**：conv-BC 到了「模仿 eval2 教师」的天花板。要超过，需要一个**比现有 agent 更强的教师**。思路：用 **depth≥2 的 expectimax 深搜索**（conv-BC 当候选生成器剪枝 + value leaf）当教师，把它的着法/价值蒸馏回 conv 网络，迭代。

**关键前置验证（先做，避免白跑几小时）**：depth=2 搜索必须先在小规模 benchmark 里被证明**明显强于 conv-BC**，才值得大规模产数据蒸馏。若 depth=2(leaf=eval0) ≈ eval2 ≈ Baseline（并不强于 conv-BC），则需转向「self-play 真实 outcome 训练 value（AlphaZero bootstrap）」这条更慢更不确定的路。

**工程要求（AGENTS.md §9）**：数据生成/训练都要 checkpoint + 断点续跑；depth=2 很慢，大规模产数据是小时级，分片并行 + 定期保存；`*.checkpoint*` 在最终产物确认前禁清理。

**验证结果（2026-07，已做）：教师并不强于 conv-BC → 蒸馏无意义。** 80 局：depth-1(conv-BC 候选+conv-BC value leaf)、depth-2(eval0 leaf) 胜率均 ~20-21% ≈ conv-BC，均低于 BeliefExp(30%, pool 依赖)。工具留存：`benchmark_pool.py` 的 `v3deep:` token、`nn_leaf` 的 `MJ_NN_VALUE_MODEL` env、`scripts/rl/gen_teacher_data.py`。根因：搜索强度 = leaf value 质量，而一切现有 value 都是 eval2 级，conv-BC 已编码之。

### 6.3 定向蒸馏 BeliefExp + 危险特征 + 天花板定论（2026-07）

用防守更好的 BeliefExp 当教师蒸馏（`gen_teacher_data.py teacher=beliefexp`，229k 真实-outcome 样本 → `output/nn_conv_bc_be.pt`）：**模仿 acc 0.816**（高！），但 **benchmark 未继承 BeliefExp 的低点炮**（19% vs 14.8%），不强于 conv-BC。**根因：BeliefExp 防守靠实时危险度信号（tile_danger/听牌/筋牌/per-player 信念），175 维特征没有这些 → 网络复现不了防守的那 18.4%。**

**追加危险度/防守特征（212 维）+ conv 重训**：实现 `extract_features_ext`（手牌/牌山/弃牌/危险度地图 + 对手危险等级），生成 229,481 条 BeliefExp 教师数据，训练 `output/nn_conv_bc_ext.pt`。val acc 进一步提升到 **0.844**，但 benchmark 点炮反而 **21.2%**（> conv-BC 15.2% / BeliefExp 15.5%）。网络把 danger 信号用来「更激进地搏牌」，而非「fold」。

**追加验证 A：危险样本加权 BC**。在 `pretrain_bc.py` 中对高危险状态样本加权（α=2.0/5.0），点炮仅从 21.2% 微降到 19.0%（α=2.0），仍远高于 conv-BC base；α=5.0 时 val acc 下降、点炮仍 20.0%。**BC loss 对稀疏高代价点炮错误不敏感，加权无法解决。**

**追加验证 B：conv-BC value head 升级搜索融合**。用 `MJ_NN_VALUE_MODEL` 把 conv-BC value 接入 `BeliefExpectimaxV3Agent` leaf：
- V3-NN-PC + conv-BC value leaf：residual 模式胜率 18.0%（< conv-BC 单独 23.0%），pure 模式 7.5%（scale 失配）；
- V3d-1-nn（conv-BC 候选 + conv-BC value leaf）：22.0% vs conv-BC 19.3%，互角。
**搜索融合无提升**：conv-BC value 没有提供超越 conv-BC policy 的新信息。

**最终定论**：175/212 维特征、监督/RL/搜索/增广框架下 **纯前馈 conv-BC（`output/nn_conv_bc.pt`）就是天花板**。但引入 **deal-in auxiliary loss** 后得到 `output/nn_conv_bc_dealin_2000_l07.pt`，首次降低点炮；再与 `BeliefExpectimaxAgent` 做 **Hybrid 分层结合** 后，得到实用上超越纯 conv-BC 的部署形态：`hybrid:dealin07:output/nn_conv_bc_dealin_2000_l07.pt:beliefexp`（400 局胜率 34.8%，点炮 18.8%，接近 BeliefExp 且更快）。若不计速度，胜率上限仍是 `BeliefExpectimaxAgent`。



### 6.4 三大后续方向验证 + 显式点炮代价改造（2026-07）

用户要求继续验证 **pMCPA / MCTS-PUCT with conv-BC / Oracle-Guided Distillation / 显式点炮代价 / true perfect-info rollout oracle**。完成情况：

- **pMCPA**（`algo/agents/adaptive_conv_agent.py`）：最佳配置 K=128/epochs=1/lr=5e-5 仅 +1.6% 绝对胜率，不稳定。
- **MCTS/PUCT**：`v3deep:1-nn` 给 conv-BC +2.7% 胜率，但仍低于 BeliefExp/Baseline。
- **Oracle-Guided Distillation**：BeliefExp/safety oracle 全部阴性。
- **显式点炮代价 auxiliary loss**：**阳性**。在 conv-BC 上增加 deal-in head，用 2000 局 perfect-info 即时点炮标签训练（λ=0.7），800 局公平池点炮从 **19.1% 降到 16.6%**，胜率 21.2%（vs base 23.5%），是首次稳健降低点炮。
- **PPO reward shaping（deal-in reward -5）**：点炮也降（16.0%），但胜率损失更大（19.0%），不如辅助 loss。
- **True perfect-info rollout oracle**：
  - conv-BC greedy rollout 单局 ~70s，50 局生成在 8 线程下超过 20 分钟，无法规模化；
  - shanten-minimizing rollout oracle 速度可接受，但 oracle 太弱，200 局数据训练出的 normal policy val acc 仅 43%，400 局胜率仅 1.5%；
  - **结论：当前资源下 true perfect-info rollout oracle 不可行**。

- **NN + BeliefExp 结合**：
  - 搜索内部替换 candidate/leaf（V3-RLunion）效果有限；
  - **Hybrid 分层策略成功**：实现 `HybridNNBeliefAgent`（平时 NN，对手报听/终盘切 BeliefExp）；
  - **Hybrid-dealin07** 在 400 局池中胜率 **34.8%**（BeliefExp 35.2%），点炮 **18.8%**（BeliefExp 21.0%），是速度-胜率-防守的最佳折中。
- **Bootstrap 两代**：
  - 一代：用 Hybrid-dealin07 当教师生成 2000 局数据；
  - 发现直接把 Hybrid 蒸馏进 deal-in NN 没有稳定提升；
  - **用纯 BC 在 Hybrid 数据上训练 `nn_conv_bc_hybrid_2000.pt`，再组装 Hybrid，得到 `Hybrid-hybridBase`**：400 局胜率 **25.2%**，点炮 **14.5%**，比 Hybrid-dealin07 更稳健；
  - 二代：用 Hybrid-hybridBase 当教师再生成 2000 局，训练 `nn_conv_bc_hybrid_v2.pt`，组装 `Hybrid-hybridV2`；
  - **二代没有继续提升**（胜率 22.8%，点炮 17.8%），当前框架下 bootstrap 基本收敛。

**最终结论**：
- **前馈 conv-BC 路线天花板已触及**，但 **NN + BeliefExp Hybrid 是实用层面的突破**；
- 最稳健部署形态：`hybrid:hybridBase:output/nn_conv_bc_hybrid_2000.pt:beliefexp`（400 局胜率 25.2%，点炮 14.5%）
- 胜率优先的 Hybrid：`hybrid:dealin07:output/nn_conv_bc_dealin_2000_l07.pt:beliefexp`
- 若不计速度追求胜率上限：仍用 `BeliefExpectimaxAgent`；
- 若必须纯前馈：用 `output/nn_conv_bc_dealin_2000_l07.pt`；
- 继续同方向 bootstrap 已收敛，再提升需更大网络、更强教师或动作空间改造。

**当前候选**（2026-07-05 及 128k 实验后）：
- **当前最佳（完整动作空间）**：`Hybrid-FullAction-32k`（`output/nn_full_action_best.pt` + `_config.json`， belief_kind='beliefexp'）
  - 200 局 vs old best `Hybrid-BE16k_t8`：胜率 40.5%，Elo 1601，点炮 15.0%
  - 400 局公平 pool：胜率 33.8%，Elo 1680，点炮 16.8%
- 上一版本稳健候选：`hybrid:BE16k_t8:output/nn_conv_bc_beliefexp_trace_16000_big_t8.pt:beliefexp`（Elo 1581）
- 128k 未确认候选：`output/nn_full_action_128000_epoch_07.pt`（Elo ~1621，需更大局数验证）
- 胜率上限：`BeliefExpectimaxAgent`
- 基线：`output/nn_conv_bc.pt`

---

## 6.5 完整动作空间 128k 缩放 + PPO 微调（2026-07）

在 `Hybrid-FullAction-32k` 基础上，把训练数据从 32k 局放大到 **128k 局**，并尝试用 PPO 在 128k checkpoint 上继续微调。

**128k 行为克隆**：
- 数据：`output/nn_full_action_data_128000.npz`（128k 局，~547万 discard / ~1681万 response 样本）。
- 模型：`TileConvNet` 128/6/512，带 dealin/value/tenpai/response head。
- 训练：从 `output/nn_full_action_best.pt` 热启，30 epoch，3 GPU DataParallel；脚本已支持每 epoch checkpoint + `--resume`。
- 结果：
  - Epoch 1 val disc_acc **0.9444**；
  - 后续 29 epoch **完全 plateau**，loss/acc 几乎不变；
  - Epoch 1 Elo **1601**，Epoch 7 Elo **1621**（128k 里最高），Epoch 30 Elo **1566**。
- 结论：**单纯放大 BC 数据到 128k 没有稳定收益**，最终模型反而不如中间 epoch。

**PPO 在 128k checkpoint 上微调**：
- 从 128k Epoch 2 启动，GPU0，10 iter × 100 局自对弈。
- 发散：entropy 从 0.06 涨到 **0.595**，KL 连续 early-stop；vs frozen 胜率仅 **4.3%**。
- 产物 `output/nn_full_action_ppo_128k.pt`；benchmark Elo **1424**，比初始化 checkpoint 弱约 180 分。
- 结论：**当前 PPO 超参不适合在强 BC 初始化上继续优化**。

**DPO（完整动作，outcome-level 偏好对）**：
- 实现 `scripts/rl/train_full_action_dpo.py`；从 32k best 热启，用 128k 数据里的 `v_discard/v_response` 构造赢-vs-输偏好对。
- 10 epochs，β=0.1，lr=5e-5；discard DPO acc 从 0.656 提到 0.717，response 几乎没动（0.055）。
- Benchmark（200 局）：DPO Elo **1333**，胜率 5.0%，点炮 26.0%；明显弱于 BC32k（1612）和 PPO（1596）。
- 结论：**跨状态 outcome-level 偏好对不适合当前数据**，DPO 学到区分样本但没学到更强策略。

**当前状态**：best 仍为 `Hybrid-FullAction-32k`（`output/nn_full_action_best.pt`）；128k Epoch 7 是未确认的候选。DPO 已验证无效。根据 2026-07 中旬广义棋牌 AI 调研，**同时启动两条路线**：
1. **A. Reward shaping + KTO**：用 KTO（二元反馈，无需配对）替代 DPO，在 128k 数据上微调完整动作 policy；
2. **C. 对手建模**：生成带对手隐藏状态的新自对弈数据，训练对手听牌/手牌预测网络，接入 belief/NN。

---

## 6.6 KTO / 对手建模验证结果（2026-07-04）

### A. KTO 实验：阴性

实现 `scripts/rl/train_full_action_kto.py`，用 128k 完整动作数据的 `v_discard/v_response` 做二元反馈（reward > 0 desirable，reward < 0 undesirable）：

- **主实验**（GPU0，β=0.1，λ_D=1，λ_U=2）：KL reference point `z0` 从 6.8 发散到 8.4，discard 没有 desirable 样本（d_acc=0），loss 进入饱和区，无有效学习。
- **消融**（GPU2 β=0.5；GPU3 bc_weight=0.1）：`z0` 稳定但 desirable 准确率仅 0.06–0.08，提升极慢。
- **根因**：`v_discard/v_response` 只是每局最终 seat reward（+1/-1），不是动作级价值；正样本仅占 ~25%，信号太弱、噪声太大。
- **结论**：outcome-level KTO 走不通；若再做离线 RL，必须先获得**动作级价值估计**（MC rollout、value net、或 shaped reward）。

### C. 对手建模：已接入，效果有限

实现 `scripts/rl/gen_opponent_data.py` 与 `scripts/rl/train_opponent_model.py`，生成 16k 局数据（68.5 万 snapshots），训练 MLP/Conv 对手听牌预测器：

- MLP 256/128：val_acc **0.840**；MLP 512/256/128：0.843；Conv：0.842。
- 基线（常数预测多数类）0.826，模型仅略高。

#### 接入方式 1：OppDefensiveAgent

`algo/agents/opp_defensive_agent.py`：在 deal-in head 惩罚上再乘以对手听牌概率（`oppdef` token）。

- 400 局 pool（vs PPO-base / Defensive / Baseline）：
  - PPO-base：win 10.8%，deal-in 18.8%，Elo 1566
  - OppDef：win 8.5%，deal-in 17.0%，Elo 1367
  - Defensive（deal-in only）：win 9.5%，deal-in 21.0%，Elo 1438
- **结论**：deal-in head 惩罚本身就会让纯 policy 变弱；再叠加对手信号无净收益。

#### 接入方式 2：HybridNNBeliefOppAgent

`algo/agents/hybrid_nn_belief_opp_agent.py`：当对手听牌概率超阈值时，提前把 Hybrid 从 NN policy 切换到 BeliefExp 搜索（`hybridopp` token）。

- 400 局 pool（vs Hybrid / Baseline / V3-NN-PC）：
  - Hybrid：win 34.2%，deal-in 16.8%，Elo 1506
  - HybridOpp：win 33.2%，deal-in 17.0%，Elo 1612
- 胜率基本打平，Elo 因 pairwise 计算波动；未观察到稳健提升。

#### 总体结论

- 当前对手模型准确率不足（只比常数预测高 ~1.5%），无法提供足够强的额外信号。
- 两种接入方式都**没有超越现有 Hybrid-FullAction-32k**。
- 若继续对手建模，需：① 大幅提升准确率（目标 >0.90）；② 预测对手具体待牌/花色偏好，而不只是“是否听牌”；③ 或直接把对手特征作为 NN policy 的额外输入并重新训练（数据需重新生成）。

### A2. 动作级价值 + Advantage-Weighted BC（AWBC）：Hybrid 内有微弱阳性

实现 `scripts/rl/train_full_action_awbc.py`：

1. 用 `output/nn_value_model_mc.pt` 估计每个样本决策前状态的价值 `V(s)`；
2. 用最终 seat reward `R` 计算优势 `A = R - V(s)`；
3. 只保留 `A >= min_adv` 的样本，并以 `exp(A / τ)` 加权做 BC 微调。

**训练配置 v1**：`--weight-temp 1.0 --min-adv -0.2 --bc-weight 0.1`，10 epoch，GPU0。  
**训练配置 v2**：`--weight-temp 0.5 --min-adv 0.0 --bc-weight 0.0`（更激进加权）。  
**训练配置 v3**：`--value-is-policy`，用 `output/nn_full_action_valueft.pt` 的 value head 做基线；`--weight-temp 0.5 --min-adv 0.0 --bc-weight 0.0`。

**结果**：

| 配置 | pool | win | deal-in | Elo |
|---|---|---|---|---|
| Hybrid-hyb | vs baseline/v3nnpc | 34.2% | 16.8% | 1506 |
| Hybrid-awbc v1 | 同上 | 33.2% | 17.0% | 1612 |
| Hybrid-hyb | 800 局 | 34.2% | 16.6% | 1542 |
| Hybrid-awbc v1 | 800 局 | 33.2% | 17.8% | 1597 |
| Hybrid-hyb | ablation pool | 35.0% | 18.0% | 1653 |
| Hybrid-awbc v2 | ablation pool | 29.5% | 19.8% | 1515 |
| Hybrid-hyb | AWBC v3 pool | 32.2% | 17.2% | 1454 |
| Hybrid-awbc v3 | AWBC v3 pool | 35.0% | 17.5% | 1608 |
| Hybrid-hyb | AWBC v3 pool 800 | 34.5% | 16.5% | 1542 |
| Hybrid-awbc v3 | AWBC v3 pool 800 | 33.2% | 17.6% | 1597 |

- v1/v3 在 400 局都曾略好，但 800 局均与 base 打平；Elo consistently 略高，胜率未形成统计显著超越。
- 纯 PPO 形态中 AWBC 显著降低点炮（24.0% → 19.5%），但胜率不变。
- **结论**：AWBC 思路可行，但当前 value net 质量仍是瓶颈；需要更强的 conv value net 或 search-value 标签才能越过 BC 天花板。

value head 微调产物：`output/nn_full_action_valueft.pt`（val_mse 0.6758）。

---

## 6.7 减法消融报告（2026-07-04）

为系统理解哪些改进真正有效，运行了针对 `Hybrid-FullAction-32k` 的减法消融实验：每个 pool 400 局、anchor 固定，量化各组件贡献。

详见 **`docs/reports/ablation_report.md`**。关键结论：

1. **BeliefExp 搜索是最大正收益**：去掉搜索后纯 NN policy 胜率下降 32.2%。
2. **完整动作空间 response head 贡献第二**：full-action policy 比纯 conv-BC 高约 22 个百分点。
3. **deal-in head 对 Hybrid 胜率无贡献**：在「NN + BeliefExp」中搜索已提供足够防守。
4. **数据缩放天花板**：32k 后 128k 无稳定收益。
5. **可删除的组件**：128k 继续训练、对手建模、DPO/PPO/KTO、AWBC（未确认）。

三个后续突破方向的详细分析与历史对照，见 **`docs/reports/future_directions_analysis.md`**。

---

## 6.8 AlphaZero MCTS 迭代管线（已完成三轮 bootstrap，均未超越 base）

为实现「search → stronger value/policy → stronger search」的迭代，建立 AlphaZero 风格管线：

1. **MCTS self-play 生成 trace**：`scripts/rl/gen_alphazero_data.py`  
   - 用 `AlphaZeroMCTSAgent` 对当前 best policy 做 determinized PUCT；
   - 每步记录 `(features, visit_distribution, value_target)`；
   - 已加入 checkpoint/resume：每 50 局保存 `.checkpoint.npz`。

2. **在 trace 上训练 policy + value**：`scripts/rl/train_alphazero.py`  
   - policy：用 visit distribution 做 soft target；
   - value：用 trace 中的 outcome 或 MCTS value 做 MSE；
   - 同时保留 response head 的 BC。

3. **benchmark 新模型**：`scripts/rl/benchmark_az_vs_base.py`。

### 三轮结果汇总

| 轮次 | trace | value target | val_policy | 400 局 win | deal-in | Elo |
|---|---|---|---|---|---|---|
| v1 | 200局，n_sims=16 | outcome | 1.24 | 47.2% | 26.2% | 1419 |
| v2 | 500局，n_sims=16 | MCTS value | 1.18 | 42.5% | 24.8% | 1423 |
| policy-only | 200局，n_sims=16 | 不用 value | 1.21 | 47.5% | 25.0% | 1462 |
| v3 | 500局，**n_sims=32** | MCTS value | 1.27 | 44.0% | **28.2%** | 1440 |

- 三轮 AZ 模型均**未超越 base**（`nn_full_action_best.pt`，Elo 1581）。
- 提升 `n_sims=16 → 32` 没有改善，反而点炮率升高、胜率下降。
- 说明当前 PUCT + eval2 rollout 的 search target 质量不足，单纯堆 sims 无法让 policy 超过教师。

### 结论与下一步

- **短期继续 brute-force AZ 的 ROI 低**：500 局 + n_sims=32 已花约 5 h，效果更差；再升 depth/n_sims 会进入“天量级”。
- 可能瓶颈：
  1. MCTS 只搜弃牌，不搜 response/tenpai 宣言；
  2. 对手用 `eval2` rollout 过于悲观/保守，导致 search 偏好安全牌而非争胜；
  3. 200–500 局对麻将 still 太少，AlphaZero 通常需要 10k+ 局。
- 推荐先**暂停 AZ 迭代**，回到 `docs/reports/future_directions_analysis.md` 的另外两个方向：
  - **方向一**：训一个 conv value net 做 search value labels（不是 outcome），再试 AZ；
  - **方向二**：把 BeliefExp 危险信号蒸馏进 policy 输入 + deal-in head。

### 常用命令备份

**生成命令（16 workers）**：
```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=. python3 scripts/rl/gen_alphazero_data.py \
    output/nn_full_action_valueft.pt output/alphazero_trace_500_mctsvalue.npz 500 16 \
    --n-worlds 4 --n-sims 32 --max-depth 2 --device cuda \
    --value-target mcts --resume
```

**训练命令**：
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/rl/train_alphazero.py \
    output/alphazero_trace_500_mctsvalue.npz output/nn_full_action_data_128000.npz \
    output/nn_full_action_best.pt output/nn_full_action_az_mctsvalue.pt
```

**快速 benchmark 命令**：
```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 scripts/rl/benchmark_az_vs_base.py \
    output/nn_full_action_az_mctsvalue.pt 400 16 --device cuda
```

---

## 7. 未来方向备份（2026-07-02）

当前项目已验证：
- pMCPA / MCTS-PUCT / Oracle Distillation：阴性；
- Deal-in auxiliary loss：阳性（纯前馈防守提升）；
- NN + BeliefExp Hybrid：阳性（实用最强框架）；
- Bootstrap 两代：一代阳性、二代收敛。

继续提升的候选方向（按推荐顺序，2026-07 更新）：

1. **Offline RL / DPO / 加权 BC**：PPO 在线自对弈在强 BC 初始化上发散，下一步优先尝试离线方法（DPO、reward-weighted BC、filtered BC、Best-of-N distillation），利用已有 128k 数据做策略优化。
2. **更大网络 / 更强教师 / 特征改造**：若 offline RL 无效，再考虑容量或特征空间。
3. **动作空间 / 规则层面改造**：已纳入完整动作空间（吃/碰/杠/胡），但可进一步 hierarchical policy 或对手建模。

**当前执行方向**：1（A. Reward shaping + KTO）与 2（C. 对手建模）并行启动。
