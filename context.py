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


class UsedAwareContext(Context):
    """带 `all_tiles_as_dict()` 的 Context：让 algo.eval2 的 Cython 快路径
    真正使用 `used`（已见牌）条件化剩余牌分布。

    背景（2026-07-20，AGENTS.md §7.18）：`algo.eval2` 只在 context 有
    `all_tiles_as_dict` 方法时才把 used 传给 Cython；legacy `Context` 没有
    该方法，used 被静默忽略。默认 `Context` 行为保持不变，需要 used 生效的
    agent 显式使用本子类。
    """

    def all_tiles_as_dict(self):
        """返回已见牌字典（tile -> 张数），语义与 algo.eval2 快路径的约定一致。"""
        return self.used

