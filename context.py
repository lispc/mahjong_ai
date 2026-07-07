from utils import *
import tile


_ALL_TILES_DICT = tile.all_tiles_as_dict()

# tile id -> index 0..33 for count vectors
_TILE_IDS = list(range(1, 10)) + list(range(11, 20)) + list(range(21, 30)) + list(range(31, 38))
_TILE_TO_IDX = {t: i for i, t in enumerate(_TILE_IDS)}


class Context:
    def __init__(self):
        self.used = {}

    def tile_prob(self, tiles):
        # fast inline count (tile ids go up to 37)
        c = [0] * 38
        for t in tiles:
            c[t] += 1
        delta = {}
        for k, v in _ALL_TILES_DICT.items():
            rem = v - self.used.get(k, 0) - c[k]
            if rem:
                delta[k] = rem
        s = sum(delta.values())
        for k in delta:
            delta[k] /= s
        return delta

    def tile_prob_counts(self, counts_tuple):
        """Same as tile_prob but accepts a 34-dim count tuple."""
        delta = {}
        for k, v in _ALL_TILES_DICT.items():
            rem = v - self.used.get(k, 0) - counts_tuple[_TILE_TO_IDX[k]]
            if rem:
                delta[k] = rem
        s = sum(delta.values())
        for k in delta:
            delta[k] /= s
        return delta

