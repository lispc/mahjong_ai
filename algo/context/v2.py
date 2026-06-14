from utils import dict_sub, count
import tile


class ContextV2:
    """
    维护已见牌信息，用于修正摸牌概率。
    已见牌包括：自己手牌、所有玩家打出的牌、明刻/明杠等公开信息。
    """

    def __init__(self):
        self.used = {}  # tile -> 已见张数

    def see_tile(self, t):
        self.used[t] = self.used.get(t, 0) + 1

    def see_tiles(self, tiles):
        for t in tiles:
            self.see_tile(t)

    def tile_prob(self, hand):
        """
        返回剩余牌山中每种牌被摸到的概率（均匀分布）。
        hand 是当前手牌，会从牌山中扣除。
        """
        wall = dict_sub(dict_sub(tile.all_tiles_as_dict(), self.used), count(hand))
        s = sum(wall.values())
        if s == 0:
            return {}
        return {k: v / s for k, v in wall.items()}

    def remaining_wall(self, hand):
        """返回剩余牌山张数字典。"""
        return dict_sub(dict_sub(tile.all_tiles_as_dict(), self.used), count(hand))

    def copy(self):
        c = ContextV2()
        c.used = self.used.copy()
        return c
