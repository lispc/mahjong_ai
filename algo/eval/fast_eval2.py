# -*- coding: utf-8 -*-
"""Fast eval2 v4：完整 Numba 化的两 ply select。

目标：用 @njit 把整个候选弃牌 + 两 ply 期望循环编译掉，
速度接近 fast eval1，质量接近 legacy eval2。
"""

import numpy as np
from numba import njit

import algo.eval.v3 as eval_v3


# 模块级 float 常量，Numba 可识别
_W_SHANTEN = -7.5
_W_UKEIRE = 0.001
_W_WAIT_QUALITY = 0.46
_W_PAIRS = 0.5
_W_TAATSU = 0.8
_W_KANCHAN = -0.3


@njit(cache=True)
def _eval0_counts(counts, rem):
    """给定 34 维 counts，返回 eval0 metric。"""
    shan = eval_v3.shanten_nb(counts)
    uk = eval_v3._ukeire_nb(counts, rem)
    wq = eval_v3._wait_quality_nb(counts, rem)
    score = 0.0
    pairs = 0
    for i in range(34):
        if counts[i] >= 2:
            pairs += 1
    score += pairs * _W_PAIRS
    for i in range(27):
        rank = i % 9
        if counts[i] == 0:
            continue
        if rank < 8 and counts[i + 1] > 0:
            score += _W_TAATSU
        if rank < 7 and counts[i + 2] > 0:
            score += _W_KANCHAN
    return (_W_SHANTEN * shan +
            _W_UKEIRE * uk +
            _W_WAIT_QUALITY * wq +
            score)


@njit(cache=True)
def _eval1_counts(counts13, rem, tile_prob, top_k=30):
    """一 ply 期望：对 top_k 种摸牌算 eval0 期望。counts13 是 34 维。"""
    # 取概率最高的 top_k 个 index
    idxs = np.argsort(tile_prob)[-top_k:]
    total = 0.0
    prob_sum = 0.0
    for i in range(top_k):
        k = idxs[i]
        p = tile_prob[k]
        if p <= 0:
            continue
        prob_sum += p
        counts14 = counts13.copy()
        counts14[k] += 1
        total += p * _eval0_counts(counts14, rem)
    return total / prob_sum if prob_sum > 0 else 0.0


@njit(cache=True)
def _eval2_counts(counts13, rem, tile_prob, top_k=30):
    """两 ply 期望：对 top_k 种摸牌算 eval1 期望。counts13 是 34 维。"""
    idxs = np.argsort(tile_prob)[-top_k:]
    total = 0.0
    prob_sum = 0.0
    for i in range(top_k):
        k = idxs[i]
        p = tile_prob[k]
        if p <= 0:
            continue
        prob_sum += p
        counts14 = counts13.copy()
        counts14[k] += 1
        total += p * _eval1_counts(counts14, rem, tile_prob, top_k)
    return total / prob_sum if prob_sum > 0 else 0.0


@njit(cache=True)
def _select_numba(hand14_array, rem, tile_prob, top_k=30):
    """完整 Numba select：返回所有候选弃牌按 metric 排序后的 tile value 数组。"""
    counts = np.zeros(34, dtype=np.int64)
    for i in range(14):
        idx = eval_v3._TILE_TO_IDX[int(hand14_array[i])]
        counts[idx] += 1

    # 最多 14 个唯一候选
    metrics = np.full(14, -1e18, dtype=np.float64)
    tiles = np.full(14, -1, dtype=np.int64)
    n = 0
    handled = np.zeros(34, dtype=np.int64)

    for i in range(14):
        tile_val = hand14_array[i]
        tile_idx = eval_v3._TILE_TO_IDX[int(tile_val)]
        if handled[tile_idx]:
            continue
        handled[tile_idx] = 1
        if counts[tile_idx] == 0:
            continue
        counts13 = counts.copy()
        counts13[tile_idx] -= 1
        metric = _eval2_counts(counts13, rem, tile_prob, top_k)
        metrics[n] = metric
        tiles[n] = tile_val
        n += 1

    # 按 metric 降序排序
    order = np.argsort(metrics[:n])[::-1]
    result = np.empty(n, dtype=np.int64)
    for i in range(n):
        result[i] = tiles[order[i]]
    return result


class FastEval2:
    """两 ply 快速评估器，完整 Numba 实现。"""

    def __init__(self, context):
        self.context = context
        self.rem = eval_v3._remaining_array(context, ())
        total = self.rem.sum()
        self.tile_prob = self.rem / total if total > 0 else np.zeros(34, dtype=np.float64)

    def select(self, hand14):
        """返回按 eval2 排序的弃牌 tile 列表。"""
        assert len(hand14) == 14
        hand14_arr = np.array(hand14, dtype=np.int64)
        result = _select_numba(hand14_arr, self.rem, self.tile_prob, top_k=30)
        return list(result)


def select(hand14, context):
    """顶层接口。"""
    return FastEval2(context).select(hand14)
