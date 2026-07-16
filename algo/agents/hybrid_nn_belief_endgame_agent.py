# -*- coding: utf-8 -*-
"""Hybrid NN + BeliefEndgame Agent。

与 HybridNNBeliefAgent 相同的分层结构：平时走快速 NN policy，critical 状态
（对手报听或终盘）切搜索层。区别在于搜索层从 BeliefExpectimaxAgent 换成
BeliefEndgameAgent——后者在终盘威胁下用 wait_dist3 预测三家待牌分布，
再用 exact endgame solver 计算每张候选弃牌的精确点炮期望。

用法（benchmark_pool token）：
    hybridend:<label>:<nn_model_path>[:<wait_model_path>]

默认 wait_model_path = output/nn_wait_dist3_10k.pt（BeliefEndgameAgent 默认值）。
"""

from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent
from algo.agents.belief_endgame_agent import BeliefEndgameAgent


class HybridNNBeliefEndgameAgent(HybridNNBeliefAgent):
    def __init__(self, name, nn_model_path='output/nn_conv_bc.pt',
                 wait_model_path=None, wall_threshold=20,
                 wait_prob_threshold=0.5, **kwargs):
        super().__init__(name, nn_model_path=nn_model_path, **kwargs)
        # 在 init_tiles 之前替换搜索层；init_tiles 会重新共享 hand/meld 列表
        self.belief_agent = BeliefEndgameAgent(
            name, verbose=False,
            wait_model_path=wait_model_path,
            wall_threshold=wall_threshold,
            wait_prob_threshold=wait_prob_threshold)
        self._wait_model_path = wait_model_path
