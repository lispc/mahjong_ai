# -*- coding: utf-8 -*-
"""per-player tile-level 信念模型。

不维护任何玩家的精确手牌分布，只维护每个玩家对 34 种牌的“持有期望数量”。
核心假设：
- 玩家早期弃掉某花色越多，手里该花色越少；
- 玩家弃过的牌不可能再持有；
- 所有未知牌按花色偏好加权分配给各玩家。
"""

import tile
import algo.context.v3 as context_v3


_SUIT_OF = lambda t: t // 10 if t < 30 else 3


def _initial_suit_weights():
    return [1.0, 1.0, 1.0, 1.0]


def _update_suit_weights(weights, discarded_tile, alpha=0.3):
    """根据一次弃牌轻微降低该玩家对应花色的权重。"""
    s = _SUIT_OF(discarded_tile)
    weights[s] = max(0.1, weights[s] - alpha)
    # 重新归一化
    total = sum(weights)
    return [w / total for w in weights]


class PlayerBelief:
    """
    从 ContextV3 的公开信息构建每个玩家对每种牌的持有期望。

    使用方式：
        belief = PlayerBelief(context)
        p_opp_has_5wan = belief.expected_copies('p2', 5)
    """

    def __init__(self, context):
        self.context = context
        self._recompute()

    def _recompute(self):
        """根据当前 context 重新计算每个玩家的花色权重和 tile 期望。"""
        # 每个玩家的花色权重
        self.suit_weights = {}
        for player in self.context.discards:
            weights = _initial_suit_weights()
            for t in self.context.discards[player]:
                weights = _update_suit_weights(weights, t)
            self.suit_weights[player] = weights

        # 对每个 tile，计算每个玩家的期望持有张数
        self._expected = {}
        for player in self.context.discards:
            self._expected[player] = {}

        all_tiles = tile.all_tiles_as_dict()
        for t in all_tiles:
            # 全局未知张数（牌山 + 所有对手手牌）
            unknown = all_tiles[t] - self.context.used.get(t, 0)
            if unknown <= 0:
                for player in self.context.discards:
                    self._expected[player][t] = 0.0
                continue

            suit = _SUIT_OF(t)
            # 分母：所有可能持有该牌的玩家的花色权重之和
            denom = 0.0
            weights_for_tile = {}
            for player in self.context.discards:
                # 若该玩家已经弃过这张牌，则他不可能再持有一张完全相同的牌
                if t in self.context.discards[player]:
                    weights_for_tile[player] = 0.0
                else:
                    w = self.suit_weights[player][suit]
                    weights_for_tile[player] = w
                    denom += w

            if denom <= 0:
                for player in self.context.discards:
                    self._expected[player][t] = 0.0
                continue

            for player in self.context.discards:
                self._expected[player][t] = unknown * weights_for_tile[player] / denom

    def expected_copies(self, player, tile_value):
        """返回玩家 player 手里期望持有 tile_value 的张数。"""
        self._recompute()
        return self._expected.get(player, {}).get(tile_value, 0.0)

    def opponent_holds_probability(self, tile_value, self_name):
        """返回除自己外，至少有一名对手持有 tile_value 的概率近似（期望张数上限 1）。"""
        self._recompute()
        total = 0.0
        for player in self.context.discards:
            if player == self_name:
                continue
            total += self._expected.get(player, {}).get(tile_value, 0.0)
        return min(1.0, total)

    def effective_remaining(self, tile_value, remaining, self_name):
        """把 per-player 信念折算成“这张牌实际还在牌山的期望张数”。"""
        self._recompute()
        opp_expected = 0.0
        for player in self.context.discards:
            if player == self_name:
                continue
            opp_expected += self._expected.get(player, {}).get(tile_value, 0.0)
        return max(0.0, remaining.get(tile_value, 0) - opp_expected)
