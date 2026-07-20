# -*- coding: utf-8 -*-
"""HybridNNBeliefNoMeldAgent：禁止碰/杠的 Hybrid 变体（2026-07-19，规则修复后）。

背景：arena 修复副露和牌判负后，Hybrid 的 NN response 头（jaxenv 训练，
副露可胡）积极碰牌，但 BeliefExp 搜索层用 full_hand()（副露只记 1 张）评估，
对副露手牌严重误判 → 副露局 0 胜且点炮率飙升。新 meta 下的强者
（Baseline/BeliefExp）全闭手。本变体直接禁止碰/杠，其余不变。
"""

from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent


class HybridNNBeliefNoMeldAgent(HybridNNBeliefAgent):
    def respond_peng(self, tile_val, context=None):
        return False

    def respond_gang(self, tile_val, context=None):
        return False
