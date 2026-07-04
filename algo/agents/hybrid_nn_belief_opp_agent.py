# -*- coding: utf-8 -*-
"""带对手听牌感知的 Hybrid NN + BeliefExp Agent。

在 HybridNNBeliefAgent 基础上，额外用对手模型估计三个对手的听牌概率。
当任一对手听牌概率超过阈值（默认 0.5）时，提前切换到 BeliefExpectimax 做精确搜索，
以加强防守；否则仍按原阈值（任一对手报听 / 终盘）切换。

用法（benchmark_pool token）：
    hybridopp:<label>:<nn_model_path>:<opp_model_path>

阈值可通过环境变量 OPP_TENPAI_THRESHOLD 调整（默认 0.5）。
"""
import os
import numpy as np
import torch

from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent
from algo.agents.opp_defensive_agent import _load_opp_net


class HybridNNBeliefOppAgent(HybridNNBeliefAgent):
    def __init__(self, name, nn_model_path='output/nn_full_action_best.pt',
                 opp_model_path=None, belief_kind='beliefexp', tenpai_threshold=28,
                 device='cpu', temperature=None, verbose=False, nn_agent_class=None):
        super().__init__(name, nn_model_path=nn_model_path, belief_kind=belief_kind,
                         tenpai_threshold=tenpai_threshold, device=device,
                         temperature=temperature, verbose=verbose,
                         nn_agent_class=nn_agent_class)
        if opp_model_path is None:
            opp_model_path = os.environ.get('OPP_MODEL_PATH', 'output/opponent_model.pt')
        self.opp_model_path = opp_model_path
        self.opp_tenpai_threshold = float(os.environ.get('OPP_TENPAI_THRESHOLD', '0.5'))
        self._opp_net = None

    def _opp_net_obj(self):
        if self._opp_net is None:
            self._opp_net, _ = _load_opp_net(self.opp_model_path, self.device)
        return self._opp_net

    def _max_opp_tenpai(self):
        """基于当前玩家视角，估计三个对手中最大听牌概率。"""
        from algo.nn.features import extract_features
        ctx = self.nn_agent.context
        if ctx is None:
            return 0.0
        feat = extract_features(ctx, self.full_hand(), self.name)
        x = torch.from_numpy(np.asarray(feat, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._opp_net_obj()(x).squeeze(0)
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
        return float(probs.max())

    def _is_critical(self):
        # 原有条件优先
        if super()._is_critical():
            return True
        # 对手模型预测有人即将/已经听牌，也切搜索
        try:
            if self._max_opp_tenpai() >= self.opp_tenpai_threshold:
                return True
        except Exception:
            pass
        return False
