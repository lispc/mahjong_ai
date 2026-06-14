from utils import *
import tile


class Context:
    def __init__(self):
        self.used = {}

    def tile_prob(self, tiles):
        delta = dict_sub(dict_sub(tile.all_tiles_as_dict(), self.used), count(tiles))
        s = sum(delta.values())
        # print('sum is', s, 'tiles', tiles, 'delta', delta)
        for k in delta:
            delta[k] /= s
        return delta

