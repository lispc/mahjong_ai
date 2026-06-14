# -*- coding: utf-8 -*-
"""对手建模：花色偏好 + 弃牌安全度 + 听牌危险信号。

纯 Python 实现，避免依赖 numba，以便在 PyPy 下使用。
"""

import tile


_SUIT_NAMES = ['wan', 'suo', 'tong', 'honor']


def _suit_of(tile_value):
    """0=万 1=条 2=筒 3=字"""
    return tile_value // 10 if tile_value < 30 else 3


def _tile_base_safety(tile_value):
    """不考虑现物/筋牌，只看牌本身的位置安全度。"""
    if tile_value >= 31:
        return 0.35
    r = tile_value % 10
    if r == 1 or r == 9:
        return 0.25
    if r == 2 or r == 8:
        return 0.10
    # 中张
    return -0.10


def _tile_neighbors(tile_value):
    """同花色 ±1、±2 的牌。"""
    if tile_value >= 31:
        return []
    base = tile_value // 10 * 10
    r = tile_value % 10
    return [base + r + d for d in (-2, -1, 1, 2) if 1 <= r + d <= 9]


def discard_safety(tile_value, context):
    """
    与 algo/eval_v3.py 语义一致但纯 Python 实现的安全度函数。
    返回值越高越安全。
    """
    seen = context.all_seen.get(tile_value, 0)
    if seen > 0:
        # 现物：出现次数越多越安全
        return 1.0 + 0.1 * seen

    score = _tile_base_safety(tile_value)
    for n in _tile_neighbors(tile_value):
        if n in context.all_seen:
            score += 0.05 * context.all_seen[n]
    return score


def player_suit_weights(discards):
    """
    根据某玩家弃牌序列推断其可能保留/在做的花色权重。
    弃得越少的花色权重越高。
    """
    n = len(discards)
    if n < 3:
        return [0.25, 0.25, 0.25, 0.25]

    counts = [0, 0, 0, 0]
    for t in discards:
        counts[_suit_of(t)] += 1

    eps = 0.5
    raw = [n - counts[s] + eps for s in range(4)]
    total = sum(raw)
    return [r / total for r in raw]


def player_danger_level(discards, window=6):
    """
    基于弃牌序列推断该玩家是否接近/已经听牌。
    返回 0/1/2。
    """
    n = len(discards)
    if n < window + 2:
        return 0

    early = discards[:n - window]
    recent = discards[n - window:]

    early_safety = sum(_tile_base_safety(t) for t in early) / len(early)
    recent_safety = sum(_tile_base_safety(t) for t in recent) / len(recent)

    level = 0
    # 近期弃牌明显变安全：说明在收缩手牌
    if recent_safety > early_safety + 0.15:
        level += 1

    # 近期切出中张（4-6）较多：可能是听牌后调整/引诱
    mid_tiles = [t for t in recent if t < 31 and 4 <= t % 10 <= 6]
    if len(mid_tiles) >= 2:
        level += 1

    return level


def tile_danger_for_player(tile_value, player, context):
    """
    对某个具体对手，评估打出 tile_value 的危险程度。
    返回值越高越危险。
    """
    discards = context.discards.get(player, [])
    if not discards:
        return 0.0

    # 基础危险分：1 - 安全度
    base_danger = 1.0 - discard_safety(tile_value, context)

    # 花色偏好加成
    weights = player_suit_weights(discards)
    suit_weight = weights[_suit_of(tile_value)]

    # 听牌危险信号
    danger_level = player_danger_level(discards)

    # 花色偏好：相对均匀分布的偏离。
    # 某花色弃得少 -> 更危险；弃得多 -> 更安全。
    suit_deviation = suit_weight - 0.25

    # 听牌危险信号放大基础危险。
    signal = 1.0 + 0.3 * danger_level

    # 组合：基础危险 * 听牌信号 + 花色偏离调整（系数较小，避免过度反应）
    danger = base_danger * signal + 0.5 * suit_deviation
    return max(0.0, danger)


def tile_danger(tile_value, context, self_name):
    """
    对所有对手取最大危险分。
    """
    max_danger = 0.0
    for player in context.discards:
        if player == self_name:
            continue
        d = tile_danger_for_player(tile_value, player, context)
        if d > max_danger:
            max_danger = d
    return max_danger
