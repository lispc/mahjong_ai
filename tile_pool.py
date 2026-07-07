import random
import tile


class Pool:
    def __init__(self, seed=None):
        self.tiles = tile.all_tiles()
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(self.tiles)
        else:
            random.shuffle(self.tiles)
        self.idx = 0

    def next(self):
        if self.idx >= len(self.tiles):
            return None
        item = self.tiles[self.idx]
        self.idx += 1
        return item

    def next_n(self, n=14):
        result = self.tiles[self.idx:(self.idx+n)]
        assert len(result) == n
        self.idx += n
        return result

    def draw_replacement(self):
        """从牌山尾部摸一张（杠后摸牌）。"""
        if self.idx >= len(self.tiles):
            return None
        return self.tiles.pop()
