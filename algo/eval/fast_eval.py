# -*- coding: utf-8 -*-
"""快速评估函数：用 v3 的 Numba shanten/ukeire/wait 替代 legacy eval0，
并只展开一 ply 期望（eval1）以换取速度。

设计目标：作为 MC rollout policy，在保证不离谱的前提下尽可能快。
"""

from functools import lru_cache

import algo.eval.v3 as eval_v3


def _hand_to_counts(hand):
    """把 tile value 列表转成 34 维计数（无 numpy 分配，纯 Python 用于小 hand）。"""
    return tuple(sorted(hand))


def _fast_metric_for_counts(counts, rem):
    """给定 34 维计数和剩余牌数组，返回综合评分。"""
    shan = eval_v3.shanten_nb(counts)
    uk = eval_v3._ukeire_nb(counts, rem)
    wq = eval_v3._wait_quality_nb(counts, rem)
    # 权重与 v3.DEFAULT_WEIGHTS 一致
    return -7.5 * shan + 0.001 * uk + 0.46 * wq


class FastEval1:
    """一 ply 快速评估器，带缓存。每个 rollout 使用一个实例。"""

    def __init__(self, context):
        self.context = context
        self.rem = eval_v3._remaining_array(context, ())

    @lru_cache(maxsize=200000)
    def _metric(self, hand_tuple):
        counts = eval_v3.hand_to_counts(hand_tuple)
        return _fast_metric_for_counts(counts, self.rem)

    def eval0(self, hand):
        return self._metric(tuple(hand))

    def eval1(self, hand):
        prob = self.context.tile_prob(hand)
        total = 0.0
        for k, p in prob.items():
            total += p * self._metric(tuple(sorted(hand + [k])))
        return total

    def select(self, hand14):
        """返回按 metric 排序的弃牌 tile 列表。"""
        assert len(hand14) == 14
        best = []
        handled = set()
        for idx in range(len(hand14)):
            t = hand14[idx]
            if t in handled:
                continue
            handled.add(t)
            hand13 = list(hand14)
            del hand13[idx]
            metric = self.eval1(hand13)
            best.append((metric, t))
        best.sort(reverse=True)
        return [t for _, t in best]


def select(hand14, context):
    """顶层接口：给定 14 张手牌和 context，返回排序后的弃牌列表。"""
    return FastEval1(context).select(hand14)
