# -*- coding: utf-8 -*-
"""Shanten + Ukeire Agent，支持可选的 Expectimax 搜索深度。

方案 A（depth=0）：
  对每张候选弃牌，直接评估打出后 13 张手牌的价值。

方案 A + Expectimax（depth>=1）：
  用统一的价值函数做递归期望最大化：

      V(hand13, depth) =
        若 depth == 0: leaf_value(hand13)
        否则:  E_draw [ max_discard V(hand13 + draw - discard, depth-1) ]

  其中 max_discard 只在 leaf 价值最高的 top_k 候选中搜索，
  draw 抽样 n_samples 次（若 n_samples <= 0 则精确枚举所有剩余牌）。

性能说明：
- depth=0 使用纯 Python 的 eval_v2，兼容 PyPy。
- depth>=1 强烈建议用 CPython + Numba（eval_v3）；PyPy 下会退回到较慢的 eval_v2。
"""

import functools
import random
import agent
import tile
import algo.eval.v2 as eval_v2
import algo.context.v3 as context_v3


# 向听惩罚系数：每多 1 向听扣多少分。
# 必须足够大，确保"少 1 向听"优先于"多几张待牌"。
DEFAULT_SHANTEN_PENALTY = 100.0

# 自摸/和牌 terminal value
WIN_VALUE = 1000.0

# ---------------------------------------------------------------------------
# 尝试加载 Numba 加速的 eval_v3（CPython 专用；PyPy 不可用）
# ---------------------------------------------------------------------------
try:
    import numpy as np
    import algo.eval.v3 as eval_v3
    from algo.eval.v3 import hand_to_counts as _v3_hand_to_counts
    from algo.eval.v3 import shanten_nb as _v3_shanten_nb
    from algo.eval.v3 import _is_win_14 as _v3_is_win_14
    from algo.eval.v3 import _TILE_TO_IDX as _v3_TILE_TO_IDX
    _HAS_V3 = True
except Exception:
    _HAS_V3 = False


@functools.lru_cache(maxsize=200000)
def _shanten_fast_cached(hand_tuple):
    """shanten_fast 的全局缓存版（向听数只依赖手牌）。"""
    return eval_v2.shanten_fast(list(hand_tuple))


@functools.lru_cache(maxsize=200000)
def _is_win_cached(hand_tuple):
    """is_win 的全局缓存版（胡牌判断只依赖手牌）。"""
    return eval_v2.is_win(list(hand_tuple))


def _unique_tiles(hand):
    seen = set()
    out = []
    for t in hand:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _remove_one(hand, tile_value):
    hand = list(hand)
    hand.remove(tile_value)
    return hand


# ---------------------------------------------------------------------------
# Leaf value：两套实现，按环境自动选择
# ---------------------------------------------------------------------------

def _leaf_value_v2(hand13, context, shanten_penalty):
    """纯 Python leaf（兼容 PyPy）。"""
    hand_tuple = tuple(sorted(hand13))
    s = _shanten_fast_cached(hand_tuple)
    rem = context.remaining_wall(hand13)

    if s <= 0:
        total = 0.0
        for t, cnt in rem.items():
            if cnt > 0 and _is_win_cached(hand_tuple + (t,)):
                total += cnt
        return total

    total = 0.0
    for t, cnt in rem.items():
        if cnt > 0 and _shanten_fast_cached(hand_tuple + (t,)) < s:
            total += cnt
    return -shanten_penalty * s + total


def _leaf_value_v3(hand13, context, shanten_penalty):
    """Numba 加速 leaf（CPython 专用）。"""
    counts = _v3_hand_to_counts(hand13)
    s = _v3_shanten_nb(counts)
    rem = context.remaining_wall(hand13)

    if s <= 0:
        total = 0.0
        for t, cnt in rem.items():
            if cnt <= 0:
                continue
            idx = int(_v3_TILE_TO_IDX[t])
            if counts[idx] >= 4:
                continue
            c2 = counts.copy()
            c2[idx] += 1
            if _v3_is_win_14(c2):
                total += cnt
        return total

    total = 0.0
    for t, cnt in rem.items():
        if cnt <= 0:
            continue
        idx = int(_v3_TILE_TO_IDX[t])
        if counts[idx] >= 4:
            continue
        c2 = counts.copy()
        c2[idx] += 1
        if _v3_shanten_nb(c2) < s:
            total += cnt
    return -shanten_penalty * s + total


def leaf_value(hand13, context, shanten_penalty=DEFAULT_SHANTEN_PENALTY):
    """
    评估 13 张手牌的进攻价值（depth=0 的 leaf）。

    - 若已听牌：价值 = 所有待牌的剩余张数之和。
    - 若未听牌：价值 = -shanten_penalty * 向听数 + 能降低向听的所有牌的剩余张数之和。
    """
    if _HAS_V3:
        return _leaf_value_v3(hand13, context, shanten_penalty)
    return _leaf_value_v2(hand13, context, shanten_penalty)


# ---------------------------------------------------------------------------
# Expectimax
# ---------------------------------------------------------------------------

def _top_k_discards(hand14, context, k, shanten_penalty):
    """按 leaf_value 选出 top_k 候选弃牌（k<=0 返回所有 unique 弃牌）。"""
    discs = _unique_tiles(hand14)
    if k <= 0 or len(discs) <= k:
        return discs
    scored = []
    for d in discs:
        v = leaf_value(_remove_one(hand14, d), context, shanten_penalty)
        scored.append((v, d))
    scored.sort(reverse=True)
    return [d for _, d in scored[:k]]


def _expectimax_value_exact(hand13, context, depth, top_k, shanten_penalty, cache):
    """精确枚举摸牌的 expectimax（无采样）。"""
    key = (tuple(sorted(hand13)), depth)
    if key in cache:
        return cache[key]

    if depth == 0:
        val = leaf_value(hand13, context, shanten_penalty)
        cache[key] = val
        return val

    prob = context.tile_prob(hand13)
    if not prob:
        val = leaf_value(hand13, context, shanten_penalty)
        cache[key] = val
        return val

    val = 0.0
    for drawn, w in prob.items():
        hand14 = hand13 + [drawn]
        if _is_win_cached(tuple(sorted(hand14))):
            val += w * WIN_VALUE
            continue
        inner_discs = _top_k_discards(hand14, context, top_k, shanten_penalty)
        if not inner_discs:
            continue
        best = max(
            _expectimax_value_exact(
                _remove_one(hand14, d), context, depth - 1, top_k,
                shanten_penalty, cache
            )
            for d in inner_discs
        )
        val += w * best

    cache[key] = val
    return val


def _expectimax_value_sampled(hand13, context, depth, n_samples, top_k,
                              shanten_penalty, cache):
    """采样摸牌的 expectimax。n_samples 每层抽几次。"""
    key = (tuple(sorted(hand13)), depth)
    if key in cache:
        return cache[key]

    if depth == 0:
        val = leaf_value(hand13, context, shanten_penalty)
        cache[key] = val
        return val

    prob = context.tile_prob(hand13)
    if not prob:
        val = leaf_value(hand13, context, shanten_penalty)
        cache[key] = val
        return val

    tiles = list(prob.keys())
    weights = list(prob.values())

    total = 0.0
    for _ in range(n_samples):
        drawn = random.choices(tiles, weights=weights, k=1)[0]
        hand14 = hand13 + [drawn]
        if _is_win_cached(tuple(sorted(hand14))):
            total += WIN_VALUE
            continue

        inner_discs = _top_k_discards(hand14, context, top_k, shanten_penalty)
        if not inner_discs:
            continue

        best = max(
            _expectimax_value_sampled(
                _remove_one(hand14, d), context, depth - 1, n_samples, top_k,
                shanten_penalty, cache
            )
            for d in inner_discs
        )
        total += best

    val = total / n_samples
    cache[key] = val
    return val


def select(hand14, context,
           shanten_penalty=DEFAULT_SHANTEN_PENALTY,
           expectimax_depth=0,
           n_samples=0,
           top_k=0):
    """
    对 14 张手牌选最优弃牌。

    expectimax_depth=0: 直接 leaf_value（原始方案 A）。
    expectimax_depth=1: 1-ply expectimax（精确枚举摸牌）。
    expectimax_depth=2: 2-ply expectimax（n_samples>0 时采样，否则精确枚举）。
    top_k: 内层弃牌预选数量；<=0 表示不预选（枚举所有 unique 弃牌）。
    """
    assert len(hand14) == 14
    cache = {}
    best_disc = None
    best_value = -float('inf')

    for disc in _unique_tiles(hand14):
        hand13 = _remove_one(hand14, disc)
        if expectimax_depth == 0:
            value = leaf_value(hand13, context, shanten_penalty)
        elif expectimax_depth == 1:
            value = _expectimax_value_exact(
                hand13, context, 1, top_k, shanten_penalty, cache
            )
        elif n_samples > 0:
            value = _expectimax_value_sampled(
                hand13, context, expectimax_depth, n_samples, top_k,
                shanten_penalty, cache
            )
        else:
            value = _expectimax_value_exact(
                hand13, context, expectimax_depth, top_k, shanten_penalty, cache
            )

        if value > best_value:
            best_value = value
            best_disc = disc

    return best_disc


class ShantenUkeireAgent(agent.Agent):
    """Shanten + Ukeire Agent，可配置搜索深度。"""

    def __init__(self, name, verbose=False,
                 shanten_penalty=DEFAULT_SHANTEN_PENALTY,
                 expectimax_depth=0, n_samples=0, top_k=0):
        super().__init__(name, verbose)
        self.shanten_penalty = shanten_penalty
        self.expectimax_depth = expectimax_depth
        self.n_samples = n_samples
        self.top_k = top_k
        self.context = context_v3.ContextV3()

    def init_tiles(self, l):
        super().init_tiles(l)
        self.context = context_v3.ContextV3()

    def handle_msg(self, msg):
        if msg.type == 'put':
            self.context.see_tile(msg.data, msg.sender)
        elif msg.type == 'tenpai':
            self.context.declare_tenpai(msg.sender)
        return super().handle_msg(msg)

    def next(self):
        assert len(self.cur) == 14
        result = select(
            self.cur, self.context,
            shanten_penalty=self.shanten_penalty,
            expectimax_depth=self.expectimax_depth,
            n_samples=self.n_samples,
            top_k=self.top_k
        )
        self.cur.remove(result)
        self.context.see_tile(result, self.name)
        if self.verbose:
            print('出牌:' + tile.tile_to_str(result))
        return result


# ---------------------------------------------------------------------------
# 方案 3：直接复用 eval_v3 的 CEM 权重 + 基础防守
# ---------------------------------------------------------------------------

if _HAS_V3:
    class ShantenUkeireV3Agent(agent.Agent):
        """
        直接复用 algo.eval.v3 的 CEM 调优权重和 discard_safety。

        对每个候选弃牌：
            score = eval_v3.evaluate(hand14 - d) + defense_weight * discard_safety(d)
        选 score 最大的弃牌。

        需要 CPython + Numba；PyPy 下无法使用。
        """

        def __init__(self, name, verbose=False, defense_weight=2.0):
            super().__init__(name, verbose)
            self.defense_weight = defense_weight
            self.context = context_v3.ContextV3()

        def init_tiles(self, l):
            super().init_tiles(l)
            self.context = context_v3.ContextV3()

        def handle_msg(self, msg):
            if msg.type == 'put':
                self.context.see_tile(msg.data, msg.sender)
            elif msg.type == 'tenpai':
                self.context.declare_tenpai(msg.sender)
            return super().handle_msg(msg)

        def next(self):
            assert len(self.cur) == 14
            best_disc = None
            best_score = -float('inf')
            for disc in _unique_tiles(self.cur):
                score = eval_v3.discard_score(
                    self.cur, disc, self.context,
                    defense_weight=self.defense_weight
                )
                if score > best_score:
                    best_score = score
                    best_disc = disc
            self.cur.remove(best_disc)
            self.context.see_tile(best_disc, self.name)
            if self.verbose:
                print('出牌:' + tile.tile_to_str(best_disc))
            return best_disc
