# 性能优化与 depth=2 ExpectiMax 实验报告

## 1. 性能优化尝试

### 1.1 多进程并行
- 已在前序工作中实现（`driver/tournament.py` 使用 `ProcessPoolExecutor`）。
- 本次实验重点比较单解释器加速手段。

### 1.2 PyPy（已验证有效）
- 环境：`pypy3` (PyPy 7.3.17, Python 3.10.14)。
- 单线程 10 局：CPython 59.7s vs PyPy 29.4s，**约 2.0× 加速**。
- 8 工作进程 10 局（depth=1 版本）：CPython 16.9s vs PyPy 10.2s，**约 1.65× 加速**。
- 8 工作进程 100 局（depth=2 版本）：CPython 149.9s vs PyPy 未单独重跑；D2 决策耗时 PyPy 628ms、CPython 846ms，**PyPy 仍快约 1.35×**。
- 结论：**PyPy 是目前最稳定、无侵入的加速方案**，推荐用于长时间 benchmark。

### 1.3 Numba（当前环境受阻）
- 项目虚拟环境 Python 3.13.7 下 `pip install numba` 失败，错误为 `SSLEOFError`（与 PyPI/conda 的 SSL 握手异常）。
- 系统其他 Python 中 Numba 已安装但与 NumPy 版本不兼容：
  - Anaconda Python 3.13 + NumPy 2.4：Numba 要求 NumPy ≤2.1。
  - Homebrew Python 3.10 + NumPy 2.2.6：Numba 编译时基于 NumPy 1.x，运行时崩溃。
- 结论：**当前网络/依赖环境无法可靠安装可用 Numba**。若后续环境修复，可针对 34 牌数组的向听数做 `@njit` 加速。

### 1.4 Cython（编译成功，性能待调优）
- 编写了 `algo/_shanten_fast.pyx`，将手牌映射到标准 34 数组并用 C 递归 + bytes memo 实现 `shanten_fast_cy`。
- 编译命令：`/opt/anaconda3/bin/python setup.py build_ext --inplace`（与项目 venv 同为 Python 3.13.7，ABI 兼容）。
- 当前问题：
  - 小样本（≤3 张）结果正确；13 张手牌出现极慢或超时。
  - 主要原因是每次调用使用 Python `dict` 做 memo，且 key 为 `bytes(35)`，开销高于 Python 原版的 C 级 `lru_cache`。
  - 未做 C 级哈希表、未用标准 34 数组 shanten 快速算法，导致 naive 移植反而不见得快。
- 结论：**Cython 有潜力，但需要专门重写核心算法（如迭代 DP、C 级缓存）才能战胜 Python + lru_cache**。已保留 `.pyx` / `setup.py` 作为实验代码，未接入主流程。

### 1.5 推荐结论
- **立即可用：PyPy + 多进程**。
- **下一步若需更快**：
  1. 修复 Numba 环境（或降级到 Python 3.10/NumPy 2.0 的独立环境）。
  2. 用 Cython 实现一个基于 34 数组 + 位运算 / 小型查找表的专用 shanten 核，并配 C 级缓存。

---

## 2. depth=2 ExpectiMax + 剪枝

### 2.1 实现位置
- `algo/expectimax_agent.py` 已重构：
  - `depth=1`：保留历史版精确期望 + `lru_cache`，速度最快。
  - `depth>=2`：新增 `_expectimax_pruned`，带以下剪枝：
    1. **候选弃牌裁剪**：用 depth=0 启发式打分，只保留 top-K（默认 ply1 K=6，ply2 K=3）。
    2. **摸牌分支裁剪**：按概率降序，丢弃低概率分支（`min_draw_prob=0.005`），并限制 top-P（ply1 P=12，ply2 P=8）。
    3. **Alpha 剪枝**：chance 节点在累计期望值 + 剩余概率 × WIN_VALUE 无法超过当前最佳值时提前返回。
    4. **次 ply 降级**：第二 ply 使用更小的候选集与摸牌集，降低分支因子。

### 2.2 性能基准（100 局，8 工作进程，PyPy）

| AI | 胜率 | 自摸率 | 点和率 | 点炮率 | Elo | 平均决策耗时(ms) |
|---|---|---|---|---|---|---|
| Baseline（原项目 eval2） | 32.0% | 9.0% | 23.0% | 17.0% | 1608 | 193 |
| ExpectiMax-D1 | 17.0% | 3.0% | 14.0% | 19.0% | 1436 | 59 |
| ExpectiMax-D2 | 18.0% | 6.0% | 12.0% | 17.0% | 1486 | 629 |
| MCTS-D1 (250 samples) | 25.0% | 5.0% | 20.0% | 16.0% | 1470 | 63 |

- D2 决策耗时约为 D1 的 **10.6×**（629ms vs 59ms）。
- 在混合对手环境中，D2 相对 D1 的 Elo 提升约 **50**，但均明显落后于原项目 Baseline。

### 2.3 D1 vs D2 专场对局（100 局，8 工作进程，PyPy）

| AI | 胜率 | 自摸率 | 点和率 | 点炮率 | Elo | 平均决策耗时(ms) |
|---|---|---|---|---|---|---|
| D1 | 26.5% | 7.0% | 19.5% | 19.0% | 1498 | 52 |
| D2 | 22.0% | 4.5% | 17.5% | 18.0% | 1502 | 573 |

- D2 与 D1 几乎打平（Elo 差 4），**说明在现有 eval_v2 评估函数上，加深搜索 depth=2 的收益被评估噪声抵消**。

### 2.4 关键结论
- **depth=2 + 剪枝已实现并稳定运行**，PyPy 8 进程下 100 局约 150–200s。
- **当前评估函数（shanten + taatsu + tenpai）不足以让 depth=2 体现优势**。 deeper search 只是在同一评估误差上做更精细的期望，反而可能过拟合到次优方向。
- 原项目 Baseline（`algo.py` 的 `eval2`）显著强于新 eval_v2，提示应回归/融合原项目评估思想。

---

## 3. 后续建议

1. **优先改进评估函数**：
   - 引入防守项（弃牌导致对手和牌的概率 / 剩余危险牌）。
   - 引入对手听牌建模（基于公开弃牌推断）。
   - 融合原项目 `algo.py` 的 meld/pair 评估与新的 shanten 评估。
2. **再调优 depth=2 剪枝参数**：当前 K/P 为经验值；在评估函数更强后，可用小 tournament 自动搜索最优剪枝强度。
3. **PyPy 作为默认运行环境**：在 `README.md` / 实验脚本中标注推荐命令。
4. **Numba / Cython 留作后续**：环境允许或核心算法重写后，再尝试将其接入 `shanten_fast`。
