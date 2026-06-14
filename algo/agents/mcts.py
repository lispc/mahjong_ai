# -*- coding: utf-8 -*-
"""Monte-Carlo ExpectiMax agent: same depth as ExpectiMax but samples draws."""

import random
import agent
import tile
import algo.eval.v2 as eval_v2
import algo.context.v2 as context_v2


WIN_VALUE = 1000.0


def mc_expectimax_select(hand, context, depth=1, samples=60, eval_weights=None):
    """
    对每个候选弃牌，随机采样若干次摸牌并估计期望价值。
    与 ExpectiMax 使用相同的 evaluate，但用采样代替精确期望。
    """
    assert len(hand) == 14
    prob = context.tile_prob(hand)
    tiles = list(prob.keys())
    weights = list(prob.values())

    best_tile = None
    best_value = -float('inf')
    seen = set()

    for i, discard in enumerate(hand):
        if discard in seen:
            continue
        seen.add(discard)
        hand13 = hand[:i] + hand[i + 1:]

        total = 0.0
        n = 0
        for _ in range(samples):
            # 按概率采样一张摸牌
            drawn = random.choices(tiles, weights=weights, k=1)[0]
            new_hand = hand13 + [drawn]
            if eval_v2.is_win(new_hand):
                total += WIN_VALUE
                n += 1
                continue

            if depth >= 2:
                # depth-2：再选一次最优弃牌（与 ExpectiMax 一致）
                best_inner = -float('inf')
                seen_inner = set()
                for j, disc2 in enumerate(new_hand):
                    if disc2 in seen_inner:
                        continue
                    seen_inner.add(disc2)
                    hand12 = new_hand[:j] + new_hand[j + 1:]
                    v = eval_v2.evaluate(hand12, context, weights=eval_weights)
                    if v > best_inner:
                        best_inner = v
                total += best_inner
            else:
                # depth-1：直接评估 14 张牌
                total += eval_v2.evaluate(new_hand, context, weights=eval_weights)
            n += 1

        avg = total / n if n > 0 else -float('inf')
        if avg > best_value:
            best_value = avg
            best_tile = discard

    return best_tile


class MCTSAgent(agent.Agent):
    def __init__(self, name, depth=1, samples=60, verbose=False, weights=None):
        super().__init__(name, verbose)
        self.depth = depth
        self.samples = samples
        self.weights = weights
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
        result = mc_expectimax_select(self.cur, self.context,
                                      depth=self.depth, samples=self.samples,
                                      eval_weights=self.weights)
        self.cur.remove(result)
        self.context.see_tile(result)
        if self.verbose:
            print('出牌:' + tile.tile_to_str(result))
        return result
