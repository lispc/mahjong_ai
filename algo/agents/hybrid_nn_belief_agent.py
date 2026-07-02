# -*- coding: utf-8 -*-
"""Hybrid NN + BeliefExp Agent。

大多数决策使用快速 NN policy；当局面进入高风险/终盘时，切换到 BeliefExpectimax 做精确搜索。

切换条件（可配置）：
- 任一对手已报听；
- 或总弃牌数超过阈值（默认 28，约进入终盘）。

用法（benchmark_pool token）：
    hybrid:<label>:<nn_model_path>:<belief_kind>
其中 belief_kind = 'beliefexp' | 'v3nnpc' | 'v3deep:1-nn'（暂只支持 beliefexp）。
"""

import agent
from algo.agents.ppo_agent import PPOAgent
from algo.agents.belief_expectimax import BeliefExpectimaxAgent


class HybridNNBeliefAgent(agent.Agent):
    def __init__(self, name, nn_model_path='output/nn_conv_bc.pt',
                 belief_kind='beliefexp', tenpai_threshold=28, device='cpu',
                 temperature=None, verbose=False, nn_agent_class=None):
        super().__init__(name, verbose=verbose)
        if nn_agent_class is None:
            nn_agent_class = PPOAgent
        self.nn_agent = nn_agent_class(name, model_path=nn_model_path,
                                 device=device, temperature=temperature, verbose=False)
        if belief_kind == 'beliefexp':
            self.belief_agent = BeliefExpectimaxAgent(name, verbose=False)
        else:
            raise ValueError(f'unsupported belief_kind: {belief_kind}')
        self.tenpai_threshold = tenpai_threshold
        self._nn_model_path = nn_model_path
        self._nn_agent_class = nn_agent_class

    def init_tiles(self, l):
        super().init_tiles(l)
        self.nn_agent.init_tiles(l)
        self.belief_agent.init_tiles(l)

    def handle_msg(self, msg):
        self.nn_agent.handle_msg(msg)
        self.belief_agent.handle_msg(msg)
        return super().handle_msg(msg)

    def declare_tenpai(self, hand, context):
        # 先问 NN；若 NN 不报听再问 belief（保持行为一致）
        return self.nn_agent.declare_tenpai(hand, context)

    def _is_critical(self):
        ctx = self.nn_agent.context
        if ctx is None:
            return False
        # 任一对手报听
        tenpai_players = getattr(ctx, 'tenpai', set())
        if self.name in tenpai_players:
            tenpai_players = set(tenpai_players)
            tenpai_players.discard(self.name)
        if tenpai_players:
            return True
        # 终盘启发：总弃牌数
        total_discarded = sum(len(v) for v in getattr(ctx, 'discards', {}).values())
        if total_discarded >= self.tenpai_threshold:
            return True
        return False

    def next_with_trace(self):
        """返回 (tile, trace)。trace 仅在 critical 状态（使用 BeliefExp）时非空。"""
        if self._is_critical():
            self.belief_agent.cur = list(self.cur)
            tile_val, trace = self.belief_agent.next_with_trace()
            self.cur = list(self.belief_agent.cur)
            return tile_val, trace
        self.nn_agent.cur = list(self.cur)
        tile_val = self.nn_agent.next()
        self.cur = list(self.nn_agent.cur)
        return tile_val, None

    def next(self):
        return self.next_with_trace()[0]
