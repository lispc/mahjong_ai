from collections import defaultdict
import functools

import tile


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _count(hand):
    c = defaultdict(int)
    for t in hand:
        c[t] += 1
    return c


def _is_suited(t):
    return t < 30


def _suit_base(t):
    return t // 10 * 10


def _rank(t):
    return t % 10


def _valid_tiles():
    return sorted(set(tile.all_tiles()))


VALID_TILES = _valid_tiles()


# ---------------------------------------------------------------------------
# 胡牌判断（支持一般型和七对子）
# ---------------------------------------------------------------------------

def is_win(hand):
    """hand 长度为 14，判断是否胡牌。"""
    if len(hand) != 14:
        return False
    return _can_split(hand) or _is_seven_pairs(hand)


def _is_seven_pairs(hand):
    counts = _count(hand)
    if len(counts) != 7:
        return False
    return all(c == 2 for c in counts.values())


def _can_split(hand):
    """是否能拆成 4 个面子 + 1 个对子。"""
    counts = _count(hand)
    for t, c in counts.items():
        if c >= 2:
            remaining = hand[:]
            remaining.remove(t)
            remaining.remove(t)
            if _split_melds(remaining):
                return True
    return False


def _split_melds(tiles):
    if not tiles:
        return True
    counts = _count(tiles)
    t = min(counts.keys())
    c = counts[t]

    # 刻子
    if c >= 3:
        remaining = tiles[:]
        remaining.remove(t)
        remaining.remove(t)
        remaining.remove(t)
        if _split_melds(remaining):
            return True

    # 顺子（仅数牌）
    if _is_suited(t) and _rank(t) <= 7:
        if (t + 1) in counts and (t + 2) in counts:
            remaining = tiles[:]
            remaining.remove(t)
            remaining.remove(t + 1)
            remaining.remove(t + 2)
            if _split_melds(remaining):
                return True

    return False


# ---------------------------------------------------------------------------
# 精确向听数（一般型 + 七对子）
# ---------------------------------------------------------------------------

def shanten(hand):
    """
    计算手牌向听数。
    返回 -1 表示已胡牌（仅当 len(hand)==14 时可能）。
    返回 0 表示听牌。
    """
    if len(hand) == 14 and is_win(hand):
        return -1
    if len(hand) == 13:
        return min(_shanten_general(hand), _shanten_seven_pairs(hand))
    # 其他长度返回一个启发式估计
    return _shanten_general(hand)


def _shanten_seven_pairs(hand):
    counts = _count(hand)
    pairs = sum(1 for c in counts.values() if c >= 2)
    kinds = len(counts)
    if kinds >= 7:
        return max(0, 6 - pairs)
    # 不同种类不足 7 张，需要补齐种类
    return max(0, 6 - pairs + (7 - kinds))


# 缓存：状态用 (tuple(sorted(hand)), groups, pairs, taatsu) 表示
@functools.lru_cache(maxsize=50000)
def _shanten_state(hand_tuple, groups, pairs, taatsu):
    """
    hand 已排序。已提取 groups 个面子、pairs 个对子、taatsu 个搭子。
    返回从该状态到达听牌还需的最少“有效进张数”（即向听数）。
    """
    hand = list(hand_tuple)
    if not hand:
        # 标准向听数公式：
        # shanten = 8 - 2*groups - min(pairs+taatsu, 4-groups)
        #           - min(1, max(0, pairs+taatsu - (4-groups)))
        incomplete = pairs + taatsu
        missing = 4 - groups
        useful = min(incomplete, missing)
        excess_bonus = min(1, max(0, incomplete - missing))
        return max(-1, 8 - 2 * groups - useful - excess_bonus)

    # 取最小牌
    t = hand[0]
    counts = _count(hand)
    c = counts[t]

    best = 99

    # 1. 弃掉这张，作为单张
    idx = hand.index(t)
    rest = hand[:idx] + hand[idx + 1:]
    best = min(best, _shanten_state(tuple(rest), groups, pairs, taatsu))

    # 2. 作为对子
    if c >= 2:
        rest = hand[:]
        rest.remove(t)
        rest.remove(t)
        best = min(best, _shanten_state(tuple(rest), groups, pairs + 1, taatsu))

    # 3. 作为刻子
    if c >= 3:
        rest = hand[:]
        rest.remove(t)
        rest.remove(t)
        rest.remove(t)
        best = min(best, _shanten_state(tuple(rest), groups + 1, pairs, taatsu))

    # 4. 作为顺子起点（仅数牌）
    if _is_suited(t) and _rank(t) <= 7:
        if (t + 1) in counts and (t + 2) in counts:
            rest = hand[:]
            rest.remove(t)
            rest.remove(t + 1)
            rest.remove(t + 2)
            best = min(best, _shanten_state(tuple(rest), groups + 1, pairs, taatsu))

    # 5. 作为两面/边张搭子 (t, t+1)
    if _is_suited(t) and _rank(t) <= 8:
        if (t + 1) in counts:
            rest = hand[:]
            rest.remove(t)
            rest.remove(t + 1)
            best = min(best, _shanten_state(tuple(rest), groups, pairs, taatsu + 1))

    # 6. 作为坎张搭子 (t, t+2)
    if _is_suited(t) and _rank(t) <= 7:
        if (t + 2) in counts:
            rest = hand[:]
            rest.remove(t)
            rest.remove(t + 2)
            best = min(best, _shanten_state(tuple(rest), groups, pairs, taatsu + 1))

    return best


def _shanten_general(hand):
    return _shanten_state(tuple(sorted(hand)), 0, 0, 0)


# ---------------------------------------------------------------------------
# 快速向听数（用于 rollout/大量调用）
# ---------------------------------------------------------------------------

def shanten_fast(hand):
    """牺牲一点精确性换取速度；对一般型使用递归面子提取 + 贪心搭子计数。"""
    if len(hand) == 14 and is_win(hand):
        return -1
    return min(_shanten_fast_general(hand), _shanten_seven_pairs(hand))


def _greedy_incomplete(counts):
    """贪心计算剩余牌中最多能有多少个对子+搭子（不完整组合）。"""
    c = dict(counts)
    total = 0
    # 先尽可能多地取对子
    for t in list(c.keys()):
        while c.get(t, 0) >= 2:
            total += 1
            c[t] -= 2
            if c[t] == 0:
                del c[t]
    # 再取搭子：优先两面，再坎张
    for t in sorted(c.keys()):
        if c.get(t, 0) == 0:
            continue
        if _is_suited(t) and _rank(t) <= 8 and c.get(t + 1, 0) > 0:
            total += 1
            c[t] -= 1
            c[t + 1] -= 1
            if c[t] == 0:
                del c[t]
            if c[t + 1] == 0:
                del c[t + 1]
        elif _is_suited(t) and _rank(t) <= 7 and c.get(t + 2, 0) > 0:
            total += 1
            c[t] -= 1
            c[t + 2] -= 1
            if c[t] == 0:
                del c[t]
            if c[t + 2] == 0:
                del c[t + 2]
    return total


@functools.lru_cache(maxsize=50000)
def _shanten_fast_state(counts_tuple, groups):
    counts = dict(counts_tuple)
    if not counts:
        missing = 4 - groups
        return 2 * missing  # 无搭子时，每缺一个面子需 2 张（粗略）

    incomplete = _greedy_incomplete(counts)
    missing = 4 - groups
    useful = min(incomplete, missing)
    excess = min(1, max(0, incomplete - missing))
    best = max(0, 8 - 2 * groups - useful - excess)

    # 尝试继续提取面子
    for t in sorted(counts.keys()):
        c = counts[t]
        # 刻子
        if c >= 3:
            new_counts = dict(counts)
            new_counts[t] -= 3
            if new_counts[t] == 0:
                del new_counts[t]
            best = min(best, _shanten_fast_state(tuple(sorted(new_counts.items())), groups + 1))
        # 顺子
        if _is_suited(t) and _rank(t) <= 7 and counts.get(t + 1, 0) > 0 and counts.get(t + 2, 0) > 0:
            new_counts = dict(counts)
            for x in [t, t + 1, t + 2]:
                new_counts[x] -= 1
                if new_counts[x] == 0:
                    del new_counts[x]
            best = min(best, _shanten_fast_state(tuple(sorted(new_counts.items())), groups + 1))
    return best


def _shanten_fast_general(hand):
    counts = _count(hand)
    return _shanten_fast_state(tuple(sorted(counts.items())), 0)


# ---------------------------------------------------------------------------
# 搭子质量
# ---------------------------------------------------------------------------

def taatsu_quality(hand):
    """
    评估手牌中搭子、对子、浮牌的质量。
    分数越高越好。
    """
    counts = _count(hand)
    score = 0.0
    used = set()

    # 1. 面子：完整顺子/刻子已经完成，不额外加分（向听数已奖励）
    # 但可记录已用牌，避免重复

    # 2. 优先识别刻子中的“对子潜力”和顺子中的搭子
    # 简单做法：直接扫描所有可能的 2 张组合
    tiles = sorted(hand)
    n = len(tiles)

    for i in range(n):
        if i in used:
            continue
        t = tiles[i]

        # 对子
        if i + 1 < n and tiles[i + 1] == t and i + 1 not in used:
            used.add(i)
            used.add(i + 1)
            if _is_suited(t):
                # 中张对子更好
                if 3 <= _rank(t) <= 7:
                    score += 2.5
                else:
                    score += 2.0
            else:
                score += 1.5
            continue

        # 两面搭子
        if _is_suited(t) and i + 1 < n and tiles[i + 1] == t + 1 and i + 1 not in used:
            r = _rank(t)
            if 2 <= r <= 7:  # 两面
                used.add(i)
                used.add(i + 1)
                score += 3.0
                continue

        # 坎张搭子
        if _is_suited(t) and i + 1 < n and tiles[i + 1] == t + 2 and i + 1 not in used:
            used.add(i)
            used.add(i + 1)
            score += 1.5
            continue

    # 3. 剩余孤张
    for i in range(n):
        if i in used:
            continue
        t = tiles[i]
        if _is_suited(t):
            r = _rank(t)
            if 3 <= r <= 7:
                score += 0.4
            else:
                score += 0.15
        else:
            score += 0.1

    return score


# ---------------------------------------------------------------------------
# 听牌张数
# ---------------------------------------------------------------------------

def tenpai_tiles(hand):
    """
    若手牌已听牌，返回能胡的不同牌数量；否则返回 0。
    """
    if shanten_fast(hand) != 0:
        return 0
    cnt = 0
    counts = _count(hand)
    for t in VALID_TILES:
        if counts[t] >= 4:
            continue
        test_hand = hand + [t]
        if is_win(test_hand):
            cnt += 1
    return cnt


# ---------------------------------------------------------------------------
# 综合评估
# ---------------------------------------------------------------------------

# 默认权重
DEFAULT_WEIGHTS = {
    'shanten': 10.0,
    'taatsu': 0.5,
    'tenpai': 0.3,
}


def evaluate(hand, context=None, weights=None):
    """
    综合评估手牌好坏。使用快速向听数，适合大量调用。
    weights 可选：{'shanten': ..., 'taatsu': ..., 'tenpai': ...}
    """
    w = DEFAULT_WEIGHTS if weights is None else weights
    sh = shanten_fast(hand)
    tq = taatsu_quality(hand)
    tt = tenpai_tiles(hand) if sh == 0 else 0
    score = -sh * w['shanten'] + tq * w['taatsu'] + tt * w['tenpai']
    return score
