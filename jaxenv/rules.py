# -*- coding: utf-8 -*-
"""晋北麻将（推倒胡）牌型判定：预计算查找表 + JAX 查表实现。

思路（与 Mahjax 同款）：
- 每种花色（万/条/饼，9 种牌）的字牌向量有 5^9 = 1,953,125 种；字牌（7 种）5^7 = 78,125 种。
- 对每个花色计数向量离线预计算：
  1. win_mask (uint16): bit (g*2+p) 表示该花色全部牌能否恰好划分为 g 个面子 + p 个对子
     （p 只记录 0/1；>=2 个对子的划分对「4 面子 + 1 对子」无贡献，安全丢弃）。
  2. smax (int8[5]): smax[g] = 该花色在恰好提取 g 个面子时，能达到的最大 (对子数+搭子数)，
     不可达为 -1。向听公式只依赖 (总面子数 G, 总对子+搭子数 S)，且关于 G、S 单调，
     因此每个 g 只保留最大 s 是无损的（被支配状态 (g'>=g, s'>=s) 可安全剪枝）。
- 运行时：手牌 34 维计数 → 4 个花色 base-5 编码查表 → 合并 → 得到胡牌/向听。

语义对齐 `algo/eval/v2.py`：
- is_win: 14 张 = 4 面子 + 1 对子，或七对子（仅无副露）；有 m 个副露时闭手需 (4-m) 面子 + 1 对子。
- shanten: 13 张向听 = min(一般型, 七对子向听)，公式与 v2._shanten_state 的叶子公式一致：
    shanten = 8 - 2*G - min(S, 4-G) - min(1, max(0, S-(4-G)))
"""

import functools
import os

import numpy as np

# ---------------------------------------------------------------------------
# 牌 id 映射：Python 引擎 id (1-9 万, 11-19 条, 21-29 饼, 31-37 字) <-> idx 0-33
# ---------------------------------------------------------------------------

TILE_IDS = list(range(1, 10)) + list(range(11, 20)) + list(range(21, 30)) + list(range(31, 38))
TILE_TO_IDX = {t: i for i, t in enumerate(TILE_IDS)}


def pyid_to_idx(t):
    """Python 引擎牌 id -> 0-33 内部索引。"""
    if t < 30:
        return (t // 10) * 9 + (t % 10) - 1
    return 27 + (t - 31)


def tiles_to_counts(tiles):
    """Python id 序列 -> np.int8[34] 计数向量。"""
    c = np.zeros(34, np.int8)
    for t in tiles:
        c[pyid_to_idx(t)] += 1
    return c


# ---------------------------------------------------------------------------
# 离线查找表生成（纯 numpy/python，仅在 gen_tables.py 中调用）
# ---------------------------------------------------------------------------

def _digits(idx, k):
    ds = []
    for _ in range(k):
        ds.append(idx % 5)
        idx //= 5
    return ds


def _make_solvers(kind_count, suited):
    """返回 (smax_states, win_states)：idx -> frozenset。

    smax_states(idx): 该花色计数向量可提取的 (面子数 g, 对子+搭子数 s) 状态集
        （允许留孤张；按 (g'>=g, s'>=s) 支配关系剪枝，单调性保证无损）。
    win_states(idx): 该花色全部牌恰好划分完的 (g, p) 集合，p 仅保留 0/1。
    两个递归都与 algo/eval/v2.py 的 _shanten_state 提取选项完全一致：
    孤张(仅 smax) / 对子 / 刻子 / 顺子 / 两面·边张搭子(t,t+1) / 坎张搭子(t,t+2)。
    """
    P5 = [5 ** i for i in range(kind_count)]

    @functools.lru_cache(maxsize=None)
    def smax_states(idx):
        ds = _digits(idx, kind_count)
        if not any(ds):
            return frozenset({(0, 0)})
        i = next(j for j, d in enumerate(ds) if d > 0)
        out = set()

        def add(states, dg, dsn):
            for (g, s) in states:
                if g + dg <= 4:
                    out.add((g + dg, s + dsn))

        add(smax_states(idx - P5[i]), 0, 0)                       # 孤张
        if ds[i] >= 2:
            add(smax_states(idx - 2 * P5[i]), 0, 1)               # 对子
        if ds[i] >= 3:
            add(smax_states(idx - 3 * P5[i]), 1, 0)               # 刻子
        if suited and i <= 6 and ds[i + 1] > 0 and ds[i + 2] > 0:
            add(smax_states(idx - P5[i] - P5[i + 1] - P5[i + 2]), 1, 0)   # 顺子
        if suited and i <= 7 and ds[i + 1] > 0:
            add(smax_states(idx - P5[i] - P5[i + 1]), 0, 1)       # 两面/边张
        if suited and i <= 6 and ds[i + 2] > 0:
            add(smax_states(idx - P5[i] - P5[i + 2]), 0, 1)       # 坎张

        # 每个 g 只保留最大 s，再剪掉被更高 g 支配的条目
        best = {}
        for (g, s) in out:
            if s > best.get(g, -1):
                best[g] = s
        ks = sorted(best)
        pruned = [(g, best[g]) for g in ks
                  if all(best[g2] < best[g] for g2 in ks if g2 > g)]
        return frozenset(pruned)

    @functools.lru_cache(maxsize=None)
    def win_states(idx):
        ds = _digits(idx, kind_count)
        if not any(ds):
            return frozenset({(0, 0)})
        i = next(j for j, d in enumerate(ds) if d > 0)
        out = set()
        if ds[i] >= 2:
            for (g, p) in win_states(idx - 2 * P5[i]):
                if p < 1:
                    out.add((g, p + 1))
        if ds[i] >= 3:
            for (g, p) in win_states(idx - 3 * P5[i]):
                if g < 4:
                    out.add((g + 1, p))
        if suited and i <= 6 and ds[i + 1] > 0 and ds[i + 2] > 0:
            for (g, p) in win_states(idx - P5[i] - P5[i + 1] - P5[i + 2]):
                if g < 4:
                    out.add((g + 1, p))
        return frozenset(out)

    return smax_states, win_states


def compute_table_chunk(kind_count, suited, lo, hi):
    """计算索引区间 [lo, hi) 的表项。供 multiprocessing worker 调用。"""
    smax_states, win_states = _make_solvers(kind_count, suited)
    n = hi - lo
    win = np.zeros(n, np.uint16)
    smax = np.full((n, 5), -1, np.int8)
    for idx in range(lo, hi):
        for (g, p) in win_states(idx):
            win[idx - lo] |= np.uint16(1 << (g * 2 + p))
        for (g, s) in smax_states(idx):
            if s > smax[idx - lo, g]:
                smax[idx - lo, g] = s
    return lo, hi, win, smax


# ---------------------------------------------------------------------------
# JAX 查表判定
# ---------------------------------------------------------------------------

_TABLES = None

TABLES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables.npz')


def load_tables(path=None):
    """加载 tables.npz（进程内缓存，返回 numpy 数组）。

    注意：刻意缓存 numpy 而非 jnp 数组 —— jnp 数组若在 jit trace 期间创建会被
    JAX 提升为该次 trace 的 constvar tracer，缓存到全局后再被其他 trace 使用会
    触发 UnexpectedTracerError。每次 trace 内现做 jnp.asarray 则安全。
    """
    global _TABLES
    if _TABLES is None:
        d = np.load(path or TABLES_PATH)
        _TABLES = {k: d[k] for k in d.files}
    return _TABLES


def _build_comps():
    """枚举 g=g1+g2+g3+g4 (g<=4) 的全部组合，用于 4 花色合并。"""
    comps, gs = [], []
    for g in range(5):
        for a in range(5):
            for b in range(5 - a):
                for c in range(5 - a - b):
                    d = g - a - b - c
                    if 0 <= d <= 4:
                        comps.append((a, b, c, d))
                        gs.append(g)
    return np.array(comps, np.int32), np.array(gs, np.int32)


_COMPS, _COMP_G = _build_comps()  # [70, 4], [70]


def is_win_counts(counts, n_melds):
    """JAX: 闭手计数 counts[34] + n_melds 个副露 -> 是否胡牌。

    判定：4 花色各自「全划分」(g_i, p_i) 组合满足 sum(g_i) == 4 - n_melds 且
    sum(p_i) == 1；或七对子（仅 n_melds==0 且 14 张，与 v2.is_win 一致）。
    """
    import jax.numpy as jnp
    T = load_tables()
    suit_win = jnp.asarray(T['suit_win'], dtype=jnp.int32)
    honor_win = jnp.asarray(T['honor_win'], dtype=jnp.int32)
    c = counts.astype(jnp.int32)
    p5_9 = jnp.array([5 ** i for i in range(9)], jnp.int32)
    p5_7 = jnp.array([5 ** i for i in range(7)], jnp.int32)
    idxs = [jnp.dot(c[0:9], p5_9), jnp.dot(c[9:18], p5_9), jnp.dot(c[18:27], p5_9)]
    idx_h = jnp.dot(c[27:34], p5_7)
    masks = [suit_win[idxs[0]], suit_win[idxs[1]], suit_win[idxs[2]], honor_win[idx_h]]

    def to_wp(mask):
        bits = (mask >> jnp.arange(10, dtype=jnp.int32)) & 1
        return bits.reshape(5, 2).astype(bool)  # W[g, p]

    def merge(W1, W2):
        M = jnp.zeros((5, 2), bool)
        for g1 in range(5):
            for p1 in range(2):
                for g2 in range(5 - g1):
                    for p2 in range(2 - p1):
                        M = M.at[g1 + g2, p1 + p2].set(
                            M[g1 + g2, p1 + p2] | (W1[g1, p1] & W2[g2, p2]))
        return M

    M = merge(merge(merge(to_wp(masks[0]), to_wp(masks[1])), to_wp(masks[2])),
              to_wp(masks[3]))
    m = n_melds.astype(jnp.int32)
    win_general = M[4 - m, 1]
    seven_pairs = ((m == 0) & (c.sum() == 14) & ((c > 0).sum() == 7)
                   & ((c <= 2).all()))
    return win_general | seven_pairs


def shanten_general_counts(counts, n_melds):
    """JAX: 一般型向听（v2._shanten_state 叶子公式的查表版）。"""
    import jax.numpy as jnp
    T = load_tables()
    suit_smax = jnp.asarray(T['suit_smax'], dtype=jnp.int8)
    honor_smax = jnp.asarray(T['honor_smax'], dtype=jnp.int8)
    c = counts.astype(jnp.int32)
    p5_9 = jnp.array([5 ** i for i in range(9)], jnp.int32)
    p5_7 = jnp.array([5 ** i for i in range(7)], jnp.int32)
    idxs = [jnp.dot(c[0:9], p5_9), jnp.dot(c[9:18], p5_9), jnp.dot(c[18:27], p5_9)]
    idx_h = jnp.dot(c[27:34], p5_7)
    S = jnp.stack([suit_smax[idxs[0]], suit_smax[idxs[1]],
                   suit_smax[idxs[2]], honor_smax[idx_h]])  # [4,5] int8, -1 不可达

    comps = jnp.asarray(_COMPS)       # [70,4]
    comp_g = jnp.asarray(_COMP_G)     # [70]
    gathered = S[jnp.arange(4)[None, :], comps].astype(jnp.int16)  # [70,4]
    valid = (gathered >= 0).all(axis=1)
    sums = jnp.where(valid, gathered.sum(axis=1, dtype=jnp.int16), jnp.int16(-1))
    total_s = jnp.full((5,), -1, jnp.int16).at[comp_g].max(sums)   # 每 g 的最大 S

    g = jnp.arange(5, dtype=jnp.int16)
    m = n_melds.astype(jnp.int16)
    G = g + m
    ok = (total_s >= 0) & (G <= 4)
    missing = 4 - G
    useful = jnp.minimum(total_s, missing)
    excess = jnp.minimum(1, jnp.maximum(0, total_s - missing))
    sh = 8 - 2 * G - useful - excess
    sh = jnp.where(ok, sh, jnp.int16(99))
    return jnp.maximum(jnp.int16(-1), sh.min())


def shanten_counts(counts, n_melds):
    """JAX: 13 张向听 = min(一般型, 七对子)，与 v2.shanten(len==13) 一致。

    七对子向听仅在无副露时有意义（n_melds>0 时置 99 忽略）。
    """
    import jax.numpy as jnp
    gen = shanten_general_counts(counts, n_melds)
    c = counts.astype(jnp.int32)
    pairs = (c >= 2).sum()
    kinds = (c > 0).sum()
    sp = jnp.where(kinds >= 7,
                   jnp.maximum(0, 6 - pairs),
                   jnp.maximum(0, 6 - pairs + (7 - kinds)))
    sp = jnp.where(n_melds.astype(jnp.int32) == 0, sp, jnp.int32(99))
    return jnp.minimum(gen, sp).astype(jnp.int16)
