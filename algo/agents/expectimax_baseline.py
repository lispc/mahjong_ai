# -*- coding: utf-8 -*-
"""基于原项目 algo.py eval2 + 基础防守的 Agent。"""

import agent
import tile
import algo
import context as ctx
import algo.eval.v3 as eval_v3
import algo.context.v3 as context_v3


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
    """把 context_v3 转成 algo.py 需要的 context.Context。"""
    c = ctx.Context()
    c.used = context.used.copy()
    return c


def select(hand, context, defense_weight=2.0, max_candidates=14):
    """
    对每个候选弃牌：
      score = algo.eval2(hand_after_discard) + defense_weight * discard_safety(discarded_tile)
    返回最佳弃牌。
    """
    assert len(hand) == 14
    type_ctx = _type_context(context)
    candidates = _unique_tiles(hand)

    best_tile = None
    best_value = -float('inf')
    for disc in candidates:
        hand13 = _remove_one(hand, disc)
        base_score = algo.eval2(hand13, type_ctx)
        safety = eval_v3.discard_safety(disc, context)
        value = base_score + defense_weight * safety
        if value > best_value:
            best_value = value
            best_tile = disc
    return best_tile


class ExpectiMaxBaselineAgent(agent.Agent):
    """
    核心评估使用原项目 algo.eval2（本身就是 2-ply ExpectiMax），
    并加入基础防守 penalty。
    """
    def __init__(self, name, verbose=False, defense_weight=2.0, max_candidates=14):
        super().__init__(name, verbose)
        self.defense_weight = defense_weight
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
                        defense_weight=self.defense_weight,
                        max_candidates=self.max_candidates)
        self.cur.remove(result)
        self.context.see_tile(result, self.name)
        if self.verbose:
            print('出牌:' + tile.tile_to_str(result))
        return result
