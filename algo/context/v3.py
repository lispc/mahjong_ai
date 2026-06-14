# -*- coding: utf-8 -*-
"""ContextV3：在 ContextV2 基础上增加每家弃牌记录与全局已见牌。"""

from utils import dict_sub, count
import tile


class ContextV3:
    """
    维护已见牌信息，并记录每个玩家的弃牌序列，用于防守计算。
    """

    def __init__(self):
        self.used = {}          # tile -> 已见张数（全局）
        self.discards = {}      # player -> [tile, ...]
        self.all_seen = {}      # tile -> 已见张数（快速查询）
        self.tenpai_players = set()  # 已报听的玩家集合

    def see_tile(self, t, player=None):
        self.used[t] = self.used.get(t, 0) + 1
        self.all_seen[t] = self.all_seen.get(t, 0) + 1
        if player is not None:
            if player not in self.discards:
                self.discards[player] = []
            self.discards[player].append(t)

    def see_tiles(self, tiles, player=None):
        for t in tiles:
            self.see_tile(t, player)

    def declare_tenpai(self, player):
        """记录某玩家已报听。"""
        self.tenpai_players.add(player)

    def tile_prob(self, hand):
        """返回剩余牌山中每种牌被摸到的概率（均匀分布）。"""
        wall = dict_sub(dict_sub(tile.all_tiles_as_dict(), self.used), count(hand))
        s = sum(wall.values())
        if s == 0:
            return {}
        return {k: v / s for k, v in wall.items()}

    def remaining_wall(self, hand):
        """返回剩余牌山张数字典。"""
        return dict_sub(dict_sub(tile.all_tiles_as_dict(), self.used), count(hand))

    def copy(self):
        c = ContextV3()
        c.used = self.used.copy()
        c.all_seen = self.all_seen.copy()
        c.discards = {p: list(seq) for p, seq in self.discards.items()}
        c.tenpai_players = self.tenpai_players.copy()
        return c
