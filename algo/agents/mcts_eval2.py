# -*- coding: utf-8 -*-
"""MCTS-style agent using eval2-with-context for leaf evaluation."""

import random
import agent
import tile
import algo
import context as ctx
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


def _type_context(context):
    c = ctx.Context()
    c.used = context.used.copy()
    return c


def _is_win(hand):
    """14 张牌是否胡牌。"""
    if len(hand) != 14:
        return False
    return algo.is_succ(hand)


def _best_inner_discard(hand14, context):
    """用 eval0 快速选最佳弃牌。"""
    type_ctx = _type_context(context)
    best_disc = None
    best_score = -float('inf')
    for disc in _unique_tiles(hand14):
        hand13 = _remove_one(hand14, disc)
        score = algo.eval0(hand13, type_ctx)
        if score > best_score:
            best_score = score
            best_disc = disc
    return best_disc


def _leaf_eval2(hand13, context):
    """用 eval2（带入已见牌）评估 13 张手牌。"""
    return algo.eval2(hand13, _type_context(context))


def _simulate_draw(hand13, drawn_tile, context):
    """摸一张牌后，选最优内层弃牌，再用 eval2 评估。"""
    hand14 = hand13 + [drawn_tile]
    if _is_win(hand14):
        return WIN_VALUE
    disc = _best_inner_discard(hand14, context)
    if disc is None:
        return _leaf_eval2(hand13, context)
    new_hand13 = _remove_one(hand14, disc)
    return _leaf_eval2(new_hand13, context)


def select(hand, context, samples=15, max_candidates=6):
    """
    对每个候选弃牌，采样若干次摸牌，估计期望价值。
    候选弃牌先用 eval0 预选 top-K。
    """
    assert len(hand) == 14
    type_ctx = _type_context(context)

    # 预选候选弃牌：按 eval0 打分取 top-K
    scored = []
    for disc in _unique_tiles(hand):
        hand13 = _remove_one(hand, disc)
        score = algo.eval0(hand13, type_ctx)
        scored.append((score, disc))
    scored.sort(reverse=True)
    candidates = [disc for _, disc in scored[:max_candidates]]

    prob = context.tile_prob(hand)
    tiles = list(prob.keys())
    weights = list(prob.values())

    best_disc = None
    best_value = -float('inf')
    for disc in candidates:
        hand13 = _remove_one(hand, disc)
        draw_prob = context.tile_prob(hand13)
        if not draw_prob:
            value = _leaf_eval2(hand13, context)
        else:
            total = 0.0
            # 按真实剩余概率采样
            d_tiles = list(draw_prob.keys())
            d_weights = list(draw_prob.values())
            for _ in range(samples):
                drawn = random.choices(d_tiles, weights=d_weights, k=1)[0]
                total += _simulate_draw(hand13, drawn, context)
            value = total / samples

        if value > best_value:
            best_value = value
            best_disc = disc

    return best_disc


class MCTSEval2Agent(agent.Agent):
    def __init__(self, name, samples=15, max_candidates=6, verbose=False):
        super().__init__(name, verbose)
        self.samples = samples
        self.max_candidates = max_candidates
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
        result = select(self.cur, self.context,
                        samples=self.samples,
                        max_candidates=self.max_candidates)
        self.cur.remove(result)
        self.context.see_tile(result, self.name)
        if self.verbose:
            print('出牌:' + tile.tile_to_str(result))
        return result
