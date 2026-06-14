# -*- coding: utf-8 -*-
"""ExpectiMax agent with pruning for depth >= 2."""

from functools import lru_cache
import agent
import tile
import algo.eval.v2 as eval_v2
import algo.context.v2 as context_v2

WIN_VALUE = 1000.0

# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------

def _unique_tiles(hand):
    """Return unique tile values preserving first occurrence order."""
    seen = set()
    out = []
    for t in hand:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _remove_one(hand, tile_value):
    """Return a new hand with one occurrence of tile_value removed."""
    hand = list(hand)
    hand.remove(tile_value)
    return hand


def _top_discard_candidates(hand, context, weights, limit):
    """
    用快速启发式（depth=0 评估）对手牌中每个候选弃牌打分，返回得分最高的 limit 个不同弃牌。
    """
    if limit <= 0 or len(hand) <= limit:
        return _unique_tiles(hand)
    candidates = []
    seen = set()
    for i, disc in enumerate(hand):
        if disc in seen:
            continue
        seen.add(disc)
        hand13 = hand[:i] + hand[i + 1:]
        score = eval_v2.evaluate(hand13, context, weights=weights)
        candidates.append((score, disc))
    candidates.sort(reverse=True)
    return [disc for _, disc in candidates[:limit]]


# ---------------------------------------------------------------------------
# ExpectiMax
# ---------------------------------------------------------------------------

def select(hand, context, depth, weights=None,
           max_discard_candidates=8,
           max_draw_tiles=20,
           max_draw_tiles2=12,
           max_discard_candidates2=5,
           min_draw_prob=0.005):
    """Choose the discard that maximises expectimax value."""
    assert len(hand) == 14
    candidates = _top_discard_candidates(hand, context, weights, max_discard_candidates)

    best_tile = None
    best_value = -float('inf')
    for disc in candidates:
        hand13 = _remove_one(hand, disc)
        value = expectimax(hand13, context, depth, weights=weights,
                           max_discard_candidates=max_discard_candidates,
                           max_draw_tiles=max_draw_tiles,
                           max_draw_tiles2=max_draw_tiles2,
                           max_discard_candidates2=max_discard_candidates2,
                           min_draw_prob=min_draw_prob,
                           alpha=best_value)
        if value > best_value:
            best_value = value
            best_tile = disc
    return best_tile


def expectimax(hand, context, depth, weights=None,
               max_discard_candidates=8,
               max_draw_tiles=20,
               max_draw_tiles2=12,
               max_discard_candidates2=5,
               min_draw_prob=0.005,
               alpha=-float('inf')):
    """
    Expected value of `hand` (13 tiles) under greedy discards.

    当 depth==1 时使用带 lru_cache 的精确一 ply 期望（与历史版本一致，最快）。
    当 depth>=2 时使用 alpha 剪枝 + 候选裁剪的递归搜索。
    """
    if depth <= 1:
        return _expectimax_depth1(hand, context, weights)

    return _expectimax_pruned(
        tuple(sorted(hand)), depth, context, weights,
        max_discard_candidates, max_draw_tiles, max_draw_tiles2,
        max_discard_candidates2, min_draw_prob, alpha)


def _expectimax_depth1(hand, context, weights):
    """深度 1 精确期望；保留 lru_cache 以获得历史版本性能。"""
    @lru_cache(maxsize=20000)
    def rec(hand_tuple):
        h = list(hand_tuple)
        prob = context.tile_prob(h)
        if not prob:
            return eval_v2.evaluate(h, context, weights=weights)

        total = 0.0
        for t, p in prob.items():
            new_hand = h + [t]
            if eval_v2.is_win(new_hand):
                total += p * WIN_VALUE
                continue

            best = -float('inf')
            seen_discard = set()
            for i, disc in enumerate(new_hand):
                if disc in seen_discard:
                    continue
                seen_discard.add(disc)
                hand12 = new_hand[:i] + new_hand[i + 1:]
                value = eval_v2.evaluate(hand12, context, weights=weights)
                if value > best:
                    best = value
            if best == -float('inf'):
                best = eval_v2.evaluate(h, context, weights=weights)
            total += p * best
        return total

    return rec(tuple(sorted(hand)))


def _expectimax_pruned(hand_tuple, depth, context, weights,
                       max_discard_candidates, max_draw_tiles, max_draw_tiles2,
                       max_discard_candidates2, min_draw_prob, alpha):
    """
    带 alpha 剪枝的 ExpectiMax。

    节点类型交替：chance（摸牌）-> max（弃牌）-> chance -> ... -> evaluate。
    alpha 表示当前 max 层已知的最佳值；chance 层若即使所有未处理分支都取 WIN_VALUE
    仍无法超过 alpha，则提前返回。
    """
    # 用 Python dict 做轻量级 memo（不计入 alpha，alpha 仅用于提前退出）
    memo = {}

    def rec(state, d, node_alpha):
        # node_alpha: 当前 max 层调用者已知的最佳值；chance 层用它做上界剪枝。
        if d == 0:
            return eval_v2.evaluate(list(state), context, weights=weights)

        key = (state, d)
        if key in memo:
            return memo[key]

        h = list(state)
        prob = context.tile_prob(h)
        if not prob:
            val = eval_v2.evaluate(h, context, weights=weights)
            memo[key] = val
            return val

        # 按概率排序并裁剪低概率/过多的摸牌分支
        draws = sorted(prob.items(), key=lambda x: -x[1])
        draws = [(t, p) for t, p in draws if p >= min_draw_prob]
        if d == depth:
            draws = draws[:max_draw_tiles]
        else:
            draws = draws[:max_draw_tiles2]

        considered = 0.0
        total = 0.0
        for t, p in draws:
            considered += p
            new_hand = h + [t]
            if eval_v2.is_win(new_hand):
                total += p * WIN_VALUE
            else:
                # max 层：选择最佳弃牌，node_alpha 用于剪枝后续弃牌
                disc_limit = max_discard_candidates if d == depth else max_discard_candidates2
                candidates = _top_discard_candidates(new_hand, context, weights, disc_limit)
                best = -float('inf')
                for disc in candidates:
                    hand12 = _remove_one(new_hand, disc)
                    child_state = tuple(sorted(hand12))
                    val = rec(child_state, d - 1, best)
                    if val > best:
                        best = val
                        # 若已达成最高收益，无需再看该 max 层的其他弃牌
                        if best >= WIN_VALUE:
                            break
                if best == -float('inf'):
                    best = eval_v2.evaluate(h, context, weights=weights)
                total += p * best

            # alpha 剪枝：即使剩余所有摸牌都取 WIN_VALUE 也无法超过 node_alpha
            if node_alpha > -float('inf'):
                upper_bound = total + (1.0 - considered) * WIN_VALUE
                if upper_bound <= node_alpha:
                    # 返回一个不超过 node_alpha 的值即可让调用者放弃该分支
                    memo[key] = upper_bound
                    return upper_bound

        # 归一化：如果只考察了部分概率质量，按考察质量缩放平均
        if considered > 0:
            result = total / considered
        else:
            result = eval_v2.evaluate(h, context, weights=weights)
        memo[key] = result
        return result

    return rec(hand_tuple, depth, alpha)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ExpectiMaxAgent(agent.Agent):
    def __init__(self, name, depth=2, verbose=False, weights=None,
                 max_discard_candidates=8,
                 max_draw_tiles=20,
                 max_draw_tiles2=12,
                 max_discard_candidates2=5,
                 min_draw_prob=0.005):
        super().__init__(name, verbose)
        self.depth = depth
        self.weights = weights
        self.max_discard_candidates = max_discard_candidates
        self.max_draw_tiles = max_draw_tiles
        self.max_draw_tiles2 = max_draw_tiles2
        self.max_discard_candidates2 = max_discard_candidates2
        self.min_draw_prob = min_draw_prob
        self.context = context_v2.ContextV2()

    def init_tiles(self, l):
        super().init_tiles(l)
        self.context = context_v2.ContextV2()

    def add(self, t):
        return super().add(t)

    def handle_msg(self, msg):
        if msg.type == 'put':
            self.context.see_tile(msg.data)
        return super().handle_msg(msg)

    def next(self):
        assert len(self.cur) == 14
        result = select(self.cur, self.context, self.depth,
                        weights=self.weights,
                        max_discard_candidates=self.max_discard_candidates,
                        max_draw_tiles=self.max_draw_tiles,
                        max_draw_tiles2=self.max_draw_tiles2,
                        max_discard_candidates2=self.max_discard_candidates2,
                        min_draw_prob=self.min_draw_prob)
        self.cur.remove(result)
        self.context.see_tile(result)
        if self.verbose:
            print('出牌:' + tile.tile_to_str(result))
        return result
