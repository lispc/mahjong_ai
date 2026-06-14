# -*- coding: utf-8 -*-
"""ExpectiMax agent using eval_v3 (ukeire + wait quality + defense)."""

from functools import lru_cache
import agent
import tile
import algo.eval.v3 as eval_v3
import algo.context.v3 as context_v3

WIN_VALUE = 1000.0


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


def _top_discard_candidates(hand, context, weights, limit, defense_weight):
    """用 discard_score 对候选弃牌排序，返回 top limit 个。"""
    if limit <= 0 or len(hand) <= limit:
        return _unique_tiles(hand)
    scored = []
    seen = set()
    for i, disc in enumerate(hand):
        if disc in seen:
            continue
        seen.add(disc)
        score = eval_v3.discard_score(hand, disc, context, weights, defense_weight)
        scored.append((score, disc))
    scored.sort(reverse=True)
    return [disc for _, disc in scored[:limit]]


def select(hand, context, depth=1, weights=None, defense_weight=2.0,
           max_discard_candidates=8):
    """选择最佳弃牌。"""
    assert len(hand) == 14
    candidates = _top_discard_candidates(hand, context, weights,
                                         max_discard_candidates, defense_weight)
    best_tile = None
    best_value = -float('inf')
    for disc in candidates:
        hand13 = _remove_one(hand, disc)
        ev = expectimax(hand13, context, depth, weights, defense_weight)
        safety = eval_v3.discard_safety(disc, context)
        value = ev + defense_weight * safety
        if value > best_value:
            best_value = value
            best_tile = disc
    return best_tile


def expectimax(hand, context, depth, weights=None, defense_weight=2.2):
    """深度 1 精确期望；depth>1 退化为 depth=1（v3 先验证 depth=1）。"""
    # 内层递归使用轻量评估：保留 shanten + algo_eval0，去掉 ukeire/wait
    fast_weights = {
        'shanten': weights.get('shanten', 10.0),
        'algo_eval0': weights.get('algo_eval0', 13.5),
    } if weights else {'shanten': 10.0, 'algo_eval0': 13.5}

    @lru_cache(maxsize=20000)
    def rec(hand_tuple):
        h = list(hand_tuple)
        prob = context.tile_prob(h)
        if not prob:
            return eval_v3.evaluate(h, context, weights)

        total = 0.0
        for t, p in prob.items():
            new_hand = h + [t]
            # 14 张：检查是否自摸
            c14 = eval_v3.hand_to_counts(new_hand)
            if eval_v3._is_win_14(c14):
                total += p * WIN_VALUE
                continue

            # 选最优弃牌：手牌价值 + 弃牌安全度（内层用轻量评估）
            best = -float('inf')
            seen_discard = set()
            for i, disc in enumerate(new_hand):
                if disc in seen_discard:
                    continue
                seen_discard.add(disc)
                hand12 = new_hand[:i] + new_hand[i + 1:]
                value = (eval_v3.evaluate(hand12, context, fast_weights) +
                         defense_weight * eval_v3.discard_safety(disc, context))
                if value > best:
                    best = value
            if best == -float('inf'):
                best = eval_v3.evaluate(h, context, weights)
            total += p * best
        return total

    return rec(tuple(sorted(hand)))


class ExpectiMaxV3Agent(agent.Agent):
    def __init__(self, name, depth=1, verbose=False, weights=None,
                 defense_weight=2.2, max_discard_candidates=8):
        super().__init__(name, verbose)
        self.depth = depth
        self.weights = weights
        self.defense_weight = defense_weight
        self.max_discard_candidates = max_discard_candidates
        self.context = context_v3.ContextV3()

    def init_tiles(self, l):
        super().init_tiles(l)
        self.context = context_v3.ContextV3()

    def add(self, t):
        return super().add(t)

    def handle_msg(self, msg):
        if msg.type == 'put':
            self.context.see_tile(msg.data, msg.sender)
        return super().handle_msg(msg)

    def next(self):
        assert len(self.cur) == 14
        result = select(self.cur, self.context, self.depth,
                        weights=self.weights,
                        defense_weight=self.defense_weight,
                        max_discard_candidates=self.max_discard_candidates)
        self.cur.remove(result)
        self.context.see_tile(result, self.name)
        if self.verbose:
            print('出牌:' + tile.tile_to_str(result))
        return result
