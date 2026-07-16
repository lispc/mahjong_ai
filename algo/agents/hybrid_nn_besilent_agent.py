# -*- coding: utf-8 -*-
"""Hybrid NN + BeliefSilentGuard Agent（方向 D3 候选）。

与 HybridNNBeliefAgent 相同的分层结构（平时 NN，critical 切搜索层），
搜索层换成 BeliefSilentGuardAgent——用 SeqOppModel 的默听概率扩展
`_danger_signal`、用三家 wait 分布增强 danger。

用法（benchmark_pool token）：
    hybridsilent:<label>:<nn_model_path>[:<seq_model_path>]
"""

from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent
from algo.agents.belief_silent_guard_agent import BeliefSilentGuardAgent


class HybridNNBesilentAgent(HybridNNBeliefAgent):
    def __init__(self, name, nn_model_path='output/nn_conv_bc.pt',
                 seq_model_path=None, tenpai_prob_threshold=0.5, **kwargs):
        super().__init__(name, nn_model_path=nn_model_path, **kwargs)
        # 在 init_tiles 之前替换搜索层；init_tiles 会重新共享 hand/meld 列表
        self.belief_agent = BeliefSilentGuardAgent(
            name, verbose=False,
            seq_model_path=seq_model_path,
            tenpai_prob_threshold=tenpai_prob_threshold)
        self._seq_model_path = seq_model_path
