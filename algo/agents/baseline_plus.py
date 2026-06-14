# -*- coding: utf-8 -*-
"""Baseline-plus：在原有 Baseline 基础上做三项最小侵入式增强。

1. 报听机制：手牌听牌且待牌足够多时主动报听，锁定手牌求自摸。
2. 尾盘精确求解：牌山剩余很少时，用精确概率 + 1-ply 期望代替 eval2  heuristic。
3. 速度优化：eval2 结果按手牌缓存，避免重复计算。

注意：不再尝试外部 defense overlay（已验证会削弱 Baseline）。
"""

import functools
import agent
import tile
import algo
import context as ctx_module
import algo.eval.v2 as eval_v2
import algo.context.v3 as context_v3


WIN_VALUE = 1000.0

DEFAULT_TENPAI_MIN_WAIT = 3
DEFAULT_ENDGAME_THRESHOLD = 16
DEFAULT_ENDGAME_MIN_WAIT = 2


@functools.lru_cache(maxsize=200000)
def _eval2_cached(hand_tuple):
    """eval2 缓存版：使用默认空 Context（与原 Baseline 一致）。"""
    c = ctx_module.Context()
    return algo.eval2(list(hand_tuple), c)


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


def _winning_tiles(hand13, context):
    """返回 hand13 的待牌列表（只包含剩余张数 > 0 的）。"""
    rem = context.remaining_wall(hand13)
    return [t for t in eval_v2.VALID_TILES
            if rem.get(t, 0) > 0 and eval_v2.is_win(hand13 + [t])]


class BaselinePlusAgent(agent.Agent):
    """增强版 Baseline：报听 + 尾盘精确 + eval2 缓存。"""

    def __init__(self, name, verbose=False,
                 tenpai_min_wait=DEFAULT_TENPAI_MIN_WAIT,
                 endgame_threshold=DEFAULT_ENDGAME_THRESHOLD,
                 endgame_min_wait=DEFAULT_ENDGAME_MIN_WAIT):
        super().__init__(name, verbose)
        self.tenpai_min_wait = tenpai_min_wait
        self.endgame_threshold = endgame_threshold
        self.endgame_min_wait = endgame_min_wait
        self.context = context_v3.ContextV3()

    def init_tiles(self, l):
        super().init_tiles(l)
        self.context = context_v3.ContextV3()

    def handle_msg(self, msg):
        if msg.type == 'put':
            self.context.see_tile(msg.data, msg.sender)
        elif msg.type == 'tenpai':
            self.context.declare_tenpai(msg.sender)
        return super().handle_msg(msg)

    def declare_tenpai(self, hand, context):
        """报听决策：听牌且待牌总剩余张数 >= threshold 时报听。"""
        if eval_v2.shanten(hand) != 0:
            return False
        waits = _winning_tiles(hand, context)
        total = sum(context.remaining_wall(hand).get(t, 0) for t in waits)
        return total >= self.tenpai_min_wait

    def _wall_remaining(self):
        """返回牌山中还能被摸到的牌数（不含他家手牌）。

        初始 wall=84，每次有玩家弃牌说明消耗了一张牌山牌。
        当前玩家已经摸完牌但尚未弃牌，所以再减 1。
        """
        total_seen = sum(self.context.all_seen.values())
        return 84 - total_seen - 1

    def _baseline_select(self):
        """原 Baseline 选牌，使用 eval2 缓存加速。"""
        scored = algo.select(
            self.cur,
            metric_f=lambda h, c: _eval2_cached(tuple(sorted(h)))
        )
        return scored[0][1]

    def _endgame_select(self):
        """
        尾盘 1-ply 期望：对每个候选弃牌，用 Context 的真实剩余概率
        计算 eval2(hand13, context)。eval2 内部已经是 1-ply 期望（枚举下张
        摸牌并按概率加权），所以这里只需在候选弃牌中取最大值。
        """
        best_disc = None
        best_value = -float('inf')

        for disc in _unique_tiles(self.cur):
            hand13 = _remove_one(self.cur, disc)
            value = algo.eval2(hand13, self.context)
            if value > best_value:
                best_value = value
                best_disc = disc

        return best_disc if best_disc is not None else self._baseline_select()

    def next(self):
        assert len(self.cur) == 14

        wall_remaining = self._wall_remaining()
        if wall_remaining <= self.endgame_threshold:
            result = self._endgame_select()
        else:
            result = self._baseline_select()

        self.cur.remove(result)
        self.context.see_tile(result, self.name)
        if self.verbose:
            print('出牌:' + tile.tile_to_str(result))
        return result
