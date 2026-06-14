# -*- coding: utf-8 -*-
"""评估函数 v3：ukeire + wait quality + 基础防守。

使用 Numba 加速核心循环；shanten 采用快速近似算法，牺牲少量精度换取速度。
"""

import numpy as np
from numba import njit

import algo
import tile
import context as ctx

# ---------------------------------------------------------------------------
# 牌编码：tile id -> 34 数组下标
# ---------------------------------------------------------------------------
_TILE_TO_IDX = np.zeros(38, dtype=np.int64)
for i in range(1, 10):
    _TILE_TO_IDX[i] = i - 1
for i in range(11, 20):
    _TILE_TO_IDX[i] = i - 11 + 9
for i in range(21, 30):
    _TILE_TO_IDX[i] = i - 21 + 18
for i in range(31, 38):
    _TILE_TO_IDX[i] = i - 31 + 27

_IDX_TO_TILE = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9,           # 万
    11, 12, 13, 14, 15, 16, 17, 18, 19,   # 条
    21, 22, 23, 24, 25, 26, 27, 28, 29,   # 筒
    31, 32, 33, 34, 35, 36, 37            # 字
], dtype=np.int64)


# ---------------------------------------------------------------------------
# Numba 核心
# ---------------------------------------------------------------------------

@njit(cache=True)
def _is_suited_idx(i):
    return i < 27


@njit(cache=True)
def _rank_idx(i):
    return i % 9 + 1


@njit(cache=True)
def _greedy_melds(c):
    """贪心提取面子，返回面子数并修改 c。"""
    groups = 0
    for i in range(34):
        while c[i] >= 3:
            c[i] -= 3
            groups += 1
    for i in range(27):
        r = i % 9
        if r <= 6:
            while c[i] > 0 and c[i + 1] > 0 and c[i + 2] > 0:
                m = c[i]
                if c[i + 1] < m:
                    m = c[i + 1]
                if c[i + 2] < m:
                    m = c[i + 2]
                c[i] -= m
                c[i + 1] -= m
                c[i + 2] -= m
                groups += m
    return groups


@njit(cache=True)
def _greedy_incomplete(c):
    """贪心计算对子+搭子数。"""
    tmp = c.copy()
    total = 0
    for i in range(34):
        while tmp[i] >= 2:
            total += 1
            tmp[i] -= 2
    for i in range(34):
        if tmp[i] == 0:
            continue
        if i < 27 and (i % 9) + 1 <= 8 and tmp[i + 1] > 0:
            total += 1
            tmp[i] -= 1
            tmp[i + 1] -= 1
        elif i < 27 and (i % 9) + 1 <= 7 and tmp[i + 2] > 0:
            total += 1
            tmp[i] -= 1
            tmp[i + 2] -= 1
    return total


@njit(cache=True)
def _shanten_fast_nb(counts):
    """快速近似向听数（pair 枚举 + 贪心面子）。"""
    best = 99
    for pair_idx in range(-1, 34):
        c = counts.copy()
        pairs = 0
        if pair_idx >= 0 and c[pair_idx] >= 2:
            c[pair_idx] -= 2
            pairs = 1
        elif pair_idx >= 0:
            continue
        groups = _greedy_melds(c)
        incomplete = _greedy_incomplete(c)
        missing = 4 - groups
        useful = incomplete if incomplete < missing else missing
        excess = 1 if incomplete > missing else 0
        s = 8 - 2 * groups - pairs - useful - excess
        if s < 0:
            s = 0
        if s < best:
            best = s
    return best


@njit(cache=True)
def _shanten_greedy_nb(counts):
    """更快但更粗略的向听数：无 pair 枚举，直接贪心面子 + 公式。"""
    c = counts.copy()
    groups = _greedy_melds(c)
    incomplete = _greedy_incomplete(c)
    missing = 4 - groups
    useful = incomplete if incomplete < missing else missing
    excess = 1 if incomplete > missing else 0
    s = 8 - 2 * groups - useful - excess
    return 0 if s < 0 else s


@njit(cache=True)
def _shanten_seven_pairs_nb(counts):
    pairs = 0
    kinds = 0
    for i in range(34):
        if counts[i] > 0:
            kinds += 1
        if counts[i] >= 2:
            pairs += 1
    if kinds >= 7:
        s = 6 - pairs
        return 0 if s < 0 else s
    s = 6 - pairs + (7 - kinds)
    return 0 if s < 0 else s


@njit(cache=True)
def shanten_nb(counts):
    s = _shanten_fast_nb(counts)
    sp = _shanten_seven_pairs_nb(counts)
    return s if s < sp else sp


@njit(cache=True)
def _is_win_14(counts):
    """14 张牌是否胡牌（一般型或七对）。"""
    # 一般型：尝试每个对子
    for p in range(34):
        if counts[p] < 2:
            continue
        c = counts.copy()
        c[p] -= 2
        if _greedy_melds(c) == 4:
            return True
    # 七对
    pairs = 0
    for i in range(34):
        if counts[i] == 2:
            pairs += 1
    if pairs == 7:
        return True
    return False


@njit(cache=True)
def _wait_type_quality(idx):
    """待牌型质量系数。"""
    if idx >= 27:
        # 字牌：单骑待，但别人容易打出生张/役牌
        return 0.55
    r = _rank_idx(idx)
    # 区分两面/坎张/边张/单骑需要知道缺少哪张，这里只按待牌位置给粗略分
    if 3 <= r <= 7:
        return 0.9
    if r == 2 or r == 8:
        return 0.7
    # r == 1 or 9
    return 0.8


# ---------------------------------------------------------------------------
# Python 接口
# ---------------------------------------------------------------------------

def hand_to_counts(hand):
    c = np.zeros(34, dtype=np.int64)
    for t in hand:
        c[_TILE_TO_IDX[t]] += 1
    return c


@njit(cache=True)
def _ukeire_nb(counts, remaining):
    """Numba 批量计算有效进张数；remaining 为 34 长度的剩余张数数组。"""
    base = _shanten_greedy_nb(counts)
    total = 0.0
    for idx in range(34):
        if remaining[idx] <= 0:
            continue
        c2 = counts.copy()
        c2[idx] += 1
        if _shanten_greedy_nb(c2) < base:
            total += remaining[idx]
    return total


@njit(cache=True)
def _wait_quality_nb(counts, remaining):
    """Numba 批量计算待牌质量。"""
    if _shanten_greedy_nb(counts) != 0:
        return 0.0
    score = 0.0
    for idx in range(34):
        if remaining[idx] <= 0:
            continue
        c2 = counts.copy()
        c2[idx] += 1
        if _is_win_14(c2):
            score += _wait_type_quality(idx) * remaining[idx]
    return score


def _remaining_array(context, hand):
    """把 context.remaining_wall(hand) 转成 34 长度 numpy 数组。"""
    rem = context.remaining_wall(hand)
    arr = np.zeros(34, dtype=np.float64)
    for idx in range(34):
        t = int(_IDX_TO_TILE[idx])
        arr[idx] = rem.get(t, 0)
    return arr


def ukeire(hand, context, counts=None):
    """
    计算手牌的有效进张数（扣除已见牌后的实际张数）。
    hand 长度为 13（或任意）。
    """
    if counts is None:
        counts = hand_to_counts(hand)
    remaining = _remaining_array(context, hand)
    return _ukeire_nb(counts, remaining)


def wait_quality(hand, context, counts=None):
    """
    若手牌已听牌，评估待牌质量；否则返回 0。
    """
    if counts is None:
        counts = hand_to_counts(hand)
    remaining = _remaining_array(context, hand)
    return _wait_quality_nb(counts, remaining)


# ---------------------------------------------------------------------------
# 原项目 algo.py 评估作为特征
# ---------------------------------------------------------------------------

def _algo_context(context):
    """把 context_v3 转成 algo.py 需要的 context.Context。"""
    c = ctx.Context()
    c.used = context.used.copy()
    return c


def algo_eval0_score(hand, context):
    """algo.py eval0：即时手牌结构分。"""
    return algo.eval0(hand, _algo_context(context))


def algo_eval1_score(hand, context):
    """algo.py eval1：一 ply 期望。"""
    return algo.eval1(hand, _algo_context(context))


def algo_eval2_score(hand, context):
    """algo.py eval2：两 ply 期望（Baseline 使用）。"""
    return algo.eval2(hand, _algo_context(context))


# ---------------------------------------------------------------------------
# 防守：弃牌安全度
# ---------------------------------------------------------------------------

def _tile_neighbors(t):
    """返回 t 的相邻/隔一张的牌（同花色）。"""
    if t >= 31:
        return []
    base = t // 10 * 10
    r = t % 10
    neighbors = []
    for dr in [-2, -1, 1, 2]:
        nr = r + dr
        if 1 <= nr <= 9:
            neighbors.append(base + nr)
    return neighbors


def discard_safety(tile_value, context):
    """
    返回弃牌安全度分数，越高越安全。
    基于：现物、筋牌、幺九字牌安全、中张危险。
    """
    # 自己或对手已经打过的牌（现物）很安全
    if tile_value in context.all_seen and context.all_seen[tile_value] > 0:
        # 出现次数越多越安全
        return 1.0 + 0.1 * context.all_seen[tile_value]

    score = 0.0
    # 字牌/幺九相对安全
    if tile_value >= 31:
        score += 0.35
    else:
        r = tile_value % 10
        if r == 1 or r == 9:
            score += 0.25
        elif r == 2 or r == 8:
            score += 0.10
        # 中张危险
        else:
            score -= 0.10

    # 筋/壁效应：邻居已被大量打出则相对安全
    neighbors = _tile_neighbors(tile_value)
    for n in neighbors:
        if n in context.all_seen:
            score += 0.05 * context.all_seen[n]

    return score


# ---------------------------------------------------------------------------
# 综合评估
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    'shanten': 7.5,      # CEM 初步调优结果
    'ukeire': 0.001,
    'wait': 0.46,
    'algo_eval0': 13.5,  # 原项目 eval0，快；ExpectiMax 自身会做摸牌期望
}


def evaluate(hand, context=None, weights=None):
    """
    综合评估手牌好坏。
    hand 通常为 13 张（评估状态）或 14 张（此时取 evaluate 前应先弃一张）。
    """
    w = DEFAULT_WEIGHTS if weights is None else weights
    counts = hand_to_counts(hand)
    sh = shanten_nb(counts)
    score = -sh * w['shanten']
    if context and w.get('ukeire', 0) != 0:
        score += ukeire(hand, context, counts) * w['ukeire']
    if context and w.get('wait', 0) != 0:
        score += wait_quality(hand, context, counts) * w['wait']
    if context and w.get('algo_eval0', 0) != 0:
        score += algo_eval0_score(hand, context) * w['algo_eval0']
    return score


def discard_score(hand, discard_tile, context, weights=None, defense_weight=2.0):
    """
    评估“打出 discard_tile 后”的综合价值，包含手牌评估 + 弃牌安全度。
    hand 为 14 张牌。
    """
    new_hand = list(hand)
    new_hand.remove(discard_tile)
    hand_score = evaluate(new_hand, context, weights)
    safety = discard_safety(discard_tile, context)
    return hand_score + defense_weight * safety
