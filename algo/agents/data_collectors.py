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

    支持两种 buffer 格式：
    - 旧（无 outcome）：用于 MC value label 管线，保持向后兼容
    - 新（带 outcome/step_idx/game_id）：用于 TD(λ) 管线，由 set_outcome 回填
    """

    def __init__(self, name, verbose=False, buffer=None, game_id=None, **kwargs):
        super().__init__(name, verbose=verbose,
                         leaf_evaluator='nn', candidate_policy='nn',
                         **kwargs)
        self.buffer = buffer if buffer is not None else []
        self.game_id = game_id

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
            'step_idx': len(self.buffer),
            'game_id': self.game_id,
        })
        return disc

    def set_outcome(self, outcome, terminal_reason='unknown'):
        """游戏结束后调用，把终局 outcome 回填到本局所有 buffer 项。

        outcome: +1 赢 / -1 输 / 0 流局（target_seat 视角）
        terminal_reason: 'tsumo_win' / 'ron_win' / 'ron_mine' /
                         'lose_tsumo' / 'lose_ron_others' / 'draw' / 'unknown'
        """
        for item in self.buffer:
            item['outcome'] = float(outcome)
            item['terminal_reason'] = terminal_reason
