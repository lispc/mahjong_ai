# -*- coding: utf-8 -*-
"""报听后终局精确求解器（方向3）。

晋北麻将"报听锁手"意味着报听者后续行为完全确定：
- 摸到胡牌即自摸；
- 否则打出刚摸到的牌；
- 不会换听、不会改策略。

因此一旦某家报听且牌山剩余不多，终局博弈树急剧收缩，可以
对"防守方打出候选牌后的点炮概率 / 终局 EV"做精确（或高置信度
蒙特卡洛）计算。输出可直接作为 ground-truth 防守标签。

当前版本先做简化但严格的 1-vs-1 模型：
- 只考虑一家报听者 vs 当前防守方；
- 已知报听者待牌集合 W（后续可由对手建模/信念推断提供）；
- 牌山剩余 R 张，随机排列；
- 模拟报听者按固定策略摸牌打牌，直到自摸或流局。
"""

import random
from typing import List, Set, Dict, Tuple

import tile


def _tenpai_player_draw_indices(wall_len: int, tenpai_offset: int):
    """
    报听者在剩余 wall 中的摸牌位置序列。
    tenpai_offset: 报听者距离下一次摸牌还差多少家摸牌。
        0=下一张就是报听者摸，1=隔一家，2=隔两家，3=隔三家。
    返回报听者在 wall 中的 index 序列（0-based）。
    """
    indices = []
    i = tenpai_offset
    while i < wall_len:
        indices.append(i)
        i += 4
    return indices


def exact_tenpai_ron_prob(discard: int,
                          tenpai_waits: Set[int],
                          wall_remaining: List[int],
                          tenpai_offset: int = 0) -> float:
    """
    精确计算防守方打出 discard 后，报听者通过 ron 或 self-draw 和牌
    的概率（1-vs-1 简化模型，忽略其他两家截胡/副露）。

    若 discard in tenpai_waits：ron 概率 = 1.0。
    否则：枚举报听者在 wall 中的摸牌序列，若某次摸到 waits 即自摸。
    """
    if discard in tenpai_waits:
        return 1.0

    # 只有牌山中存在待牌时才可能自摸
    wait_in_wall = any(t in tenpai_waits for t in wall_remaining)
    if not wait_in_wall:
        return 0.0

    draw_indices = _tenpai_player_draw_indices(len(wall_remaining), tenpai_offset)
    if not draw_indices:
        return 0.0

    # 精确组合计算：报听者在 draw_indices 对应位置上至少有一次抽到 waits。
    # 等价于：在 wall 的随机排列中，draw_indices 位置上的牌至少有一张在 waits。
    # 用补集：1 - P(所有 draw_indices 位置都不是 waits)。
    n = len(wall_remaining)
    k = len(draw_indices)
    w = sum(1 for t in wall_remaining if t in tenpai_waits)
    non_wait = n - w
    if non_wait < k:
        return 1.0

    # 超几何分布：从 non_wait 中选 k 张放在 draw_indices 位置的概率
    # = C(non_wait, k) / C(n, k)
    import math
    prob_no_wait = (math.comb(non_wait, k) / math.comb(n, k))
    return 1.0 - prob_no_wait


def exact_tenpai_ev(discard: int,
                    tenpai_waits: Set[int],
                    wall_remaining: List[int],
                    tenpai_offset: int = 0,
                    self_win_reward: float = 1.0,
                    deal_in_reward: float = -1.0,
                    draw_reward: float = 0.0) -> float:
    """
    简化 EV：防守方弃牌 discard 后的终局期望收益。
    - discard 点炮：deal_in_reward（默认 -1）
    - 报听者自摸：deal_in_reward（默认 -1，因为被自摸lose）
    - 流局：draw_reward（默认 0）
    - 忽略防守方自己胡牌（1-vs-1 简化）
    """
    ron_prob = 1.0 if discard in tenpai_waits else 0.0
    if ron_prob > 0:
        return deal_in_reward

    self_prob = exact_tenpai_ron_prob(discard, tenpai_waits, wall_remaining, tenpai_offset)
    draw_prob = 1.0 - self_prob
    return self_prob * deal_in_reward + draw_prob * draw_reward


def best_defensive_discard(hand14: List[int],
                           tenpai_waits: Set[int],
                           wall_remaining: List[int],
                           tenpai_offset: int = 0,
                           deal_in_reward: float = -1.0) -> Tuple[int, Dict[int, float]]:
    """
    在 hand14 中选择 EV 最高的弃牌（最不负 EV）。
    返回 (best_tile, {tile: ev})。
    """
    best_tile = None
    best_ev = -float('inf')
    evs = {}
    seen = set()
    for t in hand14:
        if t in seen:
            continue
        seen.add(t)
        ev = exact_tenpai_ev(t, tenpai_waits, wall_remaining, tenpai_offset,
                             deal_in_reward=deal_in_reward)
        evs[t] = ev
        if ev > best_ev:
            best_ev = ev
            best_tile = t
    return best_tile, evs


def monte_carlo_tenpai_ev(discard: int,
                          tenpai_waits: Set[int],
                          wall_remaining: List[int],
                          tenpai_offset: int = 0,
                          n_samples: int = 1000,
                          self_win_reward: float = 1.0,
                          deal_in_reward: float = -1.0,
                          draw_reward: float = 0.0,
                          rng=None) -> float:
    """
    蒙特卡洛版本（用于验证精确公式，或后续加入截胡/副露等无法解析
    处理的规则时 fallback）。
    """
    if rng is None:
        rng = random.Random(0)
    total = 0.0
    for _ in range(n_samples):
        shuffled = list(wall_remaining)
        rng.shuffle(shuffled)
        if discard in tenpai_waits:
            total += deal_in_reward
            continue
        draw_indices = _tenpai_player_draw_indices(len(shuffled), tenpai_offset)
        won = False
        for idx in draw_indices:
            if shuffled[idx] in tenpai_waits:
                total += deal_in_reward
                won = True
                break
        if not won:
            total += draw_reward
    return total / n_samples
