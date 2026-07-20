# -*- coding: utf-8 -*-
"""Baseline 的 pair_coef 变体（A/B 专用，2026-07-20）。

背景：arena 实际运行的 eval2 走 Cython 快路径，pair_coef 恒为 1.0；
而 `config.pair_coef = 0.6`（Python 路径语义，历史 PyPy MC 数据也用它）。
本 agent 与 Baseline 唯一区别是 eval2 使用显式 pair_coef，用于
「0.6（config 语义）vs 1.0（de-facto arena 语义）」的配对 A/B。

默认行为不受影响：Baseline 本身（agent.Agent）仍是 Cython 默认 1.0。
"""

import agent
import algo


class BaselinePairCoefAgent(agent.Agent):
    """与 Baseline 相同的 algo.select 贪心，仅 eval2 的 pair_coef 不同。"""

    def __init__(self, name, pair_coef=0.6, verbose=False):
        super().__init__(name, verbose)
        self.pair_coef = pair_coef

    def _metric(self, tiles, c=None):
        # select 会传入 context；本变体固定空 context（与 Baseline 一致），
        # 仅 pair_coef 不同
        return algo.eval2(tiles, pair_coef=self.pair_coef)

    def next(self):
        assert len(self.cur) >= 1
        result = algo.select(self.cur, False, metric_f=self._metric)[0]
        self.cur.remove(result)
        if self.verbose:
            import tile
            print('出牌:' + tile.tile_to_str(result))
        return result
