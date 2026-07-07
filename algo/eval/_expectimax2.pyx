# cython: language_level=3
"""Cython 化的精确 depth-2 expectimax tree builder。

输入一个 hand13（牌值 list）和 34 维有效剩余张数，返回：
- leaf_hands: depth-0 叶子手牌 list（13 张牌值 list）
- tree: 描述回溯结构的 Python list，元素为 draw1 分支。

Python 侧负责调用 NN leaf value net 评估 leaf_hands，再按 tree 回代求值。
"""

import numpy as np
import algo.eval.v3 as eval_v3


# Python 侧函数，供 Cython 调用做 14 张胡牌判断
cdef object _is_win_hand(list hand14):
    return eval_v3._is_win_14(eval_v3.hand_to_counts(hand14))


cdef list _unique_tiles(list hand):
    """返回 hand 中首次出现的牌值（保持原顺序）。"""
    cdef dict seen = {}
    cdef list out = []
    cdef int t
    for t in hand:
        if t not in seen:
            seen[t] = True
            out.append(t)
    return out


cdef tuple _canonical_key(list hand):
    """把 13 张手牌排序后转成 tuple，用于 leaf dedup。"""
    return tuple(sorted(hand))


cdef int _get_leaf_idx(list hand, dict leaf_map, list leaf_hands):
    cdef tuple key = _canonical_key(hand)
    cdef object idx_obj = leaf_map.get(key)
    cdef int idx
    if idx_obj is None:
        idx = len(leaf_hands)
        leaf_map[key] = idx
        leaf_hands.append(hand)
    else:
        idx = idx_obj
    return idx


def build_depth2_tree(list hand13_in, list rem34_in):
    """构建精确 depth-2 expectimax 搜索树。

    Parameters
    ----------
    hand13_in : list[int]
        当前 13 张手牌（牌值，如 11, 12, ...）。
    rem34_in : list[float]
        34 维有效剩余张数，按 eval_v3 的 0..33 索引。

    Returns
    -------
    leaf_hands : list[list[int]]
        所有 depth-0 叶子手牌，去重后列表。
    tree : list
        draw1 分支列表。每个元素为以下两种之一：
        ('win', prob1)
        ('node', prob1, children)  其中 children 是 dict：discard_tile -> branches
        branches 中每个元素为：
        ('win', prob2)
        ('leaf', prob2, [leaf_idx, ...])
    """
    cdef list hand13 = hand13_in
    cdef list rem = rem34_in
    cdef double total1 = 0.0
    cdef int i
    for i in range(34):
        total1 += rem[i]

    cdef list leaf_hands = []
    cdef dict leaf_map = {}
    cdef list tree1 = []

    cdef int t1_idx, t2_idx
    cdef double c1, c2, p1, p2, total2
    cdef int tile1, tile2, x1, x2
    cdef list hand14, hand13_1, hand14_2, hand13_2
    cdef list uniq1, uniq2
    cdef dict children
    cdef list branches
    cdef list indices
    cdef int idx
    cdef list rem1

    if total1 <= 0.0:
        idx = _get_leaf_idx(hand13, leaf_map, leaf_hands)
        tree1.append(('leaf', 1.0, [idx]))
        return leaf_hands, tree1

    cdef object idx_to_tile = eval_v3._IDX_TO_TILE

    for t1_idx in range(34):
        c1 = rem[t1_idx]
        if c1 <= 0.0:
            continue
        p1 = c1 / total1
        tile1 = idx_to_tile[t1_idx]
        hand14 = hand13 + [tile1]
        if _is_win_hand(hand14):
            tree1.append(('win', p1))
            continue

        children = {}
        uniq1 = _unique_tiles(hand14)
        rem1 = list(rem)
        rem1[t1_idx] -= 1.0
        total2 = total1 - c1

        for x1 in uniq1:
            hand13_1 = list(hand14)
            hand13_1.remove(x1)
            branches = []

            if total2 <= 0.0:
                idx = _get_leaf_idx(hand13_1, leaf_map, leaf_hands)
                branches.append(('leaf', 1.0, [idx]))
            else:
                for t2_idx in range(34):
                    c2 = rem1[t2_idx]
                    if c2 <= 0.0:
                        continue
                    p2 = c2 / total2
                    tile2 = idx_to_tile[t2_idx]
                    hand14_2 = hand13_1 + [tile2]
                    if _is_win_hand(hand14_2):
                        branches.append(('win', p2))
                        continue

                    uniq2 = _unique_tiles(hand14_2)
                    indices = []
                    for x2 in uniq2:
                        hand13_2 = list(hand14_2)
                        hand13_2.remove(x2)
                        idx = _get_leaf_idx(hand13_2, leaf_map, leaf_hands)
                        indices.append(idx)
                    branches.append(('leaf', p2, indices))

            children[x1] = branches
        tree1.append(('node', p1, children))

    return leaf_hands, tree1
