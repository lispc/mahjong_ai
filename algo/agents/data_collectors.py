# -*- coding: utf-8 -*-
"""用于生成训练数据的 Agent 包装器。"""

from algo.agents.belief_expectimax import BeliefExpectimaxAgent
from algo.agents.belief_expectimax_v3 import BeliefExpectimaxV3Agent
from algo.nn.features import extract_features, tile_to_index


class DataCollectorBeliefExp(BeliefExpectimaxAgent):
    """
    BeliefExp 的数据采集版本。
    每次决策前记录特征向量，决策后记录所选动作。
    """

    def __init__(self, name, verbose=False, buffer=None, **kwargs):
        super().__init__(name, verbose=verbose, **kwargs)
        self.buffer = buffer if buffer is not None else []

    def next(self):
        assert len(self.cur) == 14
        features = extract_features(self.context, self.cur, self.name)
        hand14 = list(self.cur)
        ctx_snapshot = self.context.copy()
        disc = super().next()
        # 保存完整 snapshot，方便后续计算 MC rollout value label
        # 注意：必须保存弃牌前的 14 张手牌和决策前的 context
        self.buffer.append({
            'features': features,
            'action': tile_to_index(disc),
            'context': ctx_snapshot,
            'hand': hand14,
            'name': self.name,
        })
        return disc


class DataCollectorV3NN(BeliefExpectimaxV3Agent):
    """
    BeliefExpV3 + NN leaf + NN policy 候选的数据采集版本。
    """

    def __init__(self, name, verbose=False, buffer=None, **kwargs):
        super().__init__(name, verbose=verbose,
                         leaf_evaluator='nn', candidate_policy='nn',
                         **kwargs)
        self.buffer = buffer if buffer is not None else []

    def next(self):
        assert len(self.cur) == 14
        features = extract_features(self.context, self.cur, self.name)
        hand14 = list(self.cur)
        ctx_snapshot = self.context.copy()
        disc = super().next()
        # 注意：必须保存弃牌前的 14 张手牌和决策前的 context
        self.buffer.append({
            'features': features,
            'action': tile_to_index(disc),
            'context': ctx_snapshot,
            'hand': hand14,
            'name': self.name,
        })
        return disc
