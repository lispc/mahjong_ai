# -*- coding: utf-8 -*-
"""用待牌分布预测触发 BeliefExp 的 Hybrid Agent。

在 HybridNNBeliefAgent 基础上，用 wait_dist_head 估计下家待牌分布。
当下家最大待牌概率超过阈值（默认 0.5）时，认为下家可能已听牌，
提前切换到 BeliefExpectimax 做精确搜索以加强防守。

用法（benchmark_pool token）：
    hybridwait:<label>:<model_path>[:wait_threshold]

阈值可通过环境变量 WAIT_TENPAI_THRESHOLD 调整（默认 0.5）。
"""
import os
import numpy as np
import torch

from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent
from algo.nn.features import extract_features


_NUM_ACTIONS = 34


def _seat(name):
    return int(name.split('@')[-1]) if '@' in name else 0


def _next_seat(name):
    return (_seat(name) + 1) % 4


class HybridNNBeliefWaitDistAgent(HybridNNBeliefAgent):
    def __init__(self, name, nn_model_path='output/nn_wait_dist_10k_tenpai_conv.pt',
                 belief_kind='beliefexp', tenpai_threshold=28,
                 wait_threshold=None, device='cpu', temperature=None, verbose=False,
                 nn_agent_class=None):
        super().__init__(name, nn_model_path=nn_model_path, belief_kind=belief_kind,
                         tenpai_threshold=tenpai_threshold, device=device,
                         temperature=temperature, verbose=verbose,
                         nn_agent_class=nn_agent_class)
        if wait_threshold is None:
            wait_threshold = float(os.environ.get('WAIT_TENPAI_THRESHOLD', '0.5'))
        self.wait_threshold = wait_threshold
        self._cached_has_wait = None

    def _has_wait_dist(self):
        if self._cached_has_wait is not None:
            return self._cached_has_wait
        try:
            feats = extract_features(None, [11]*14, self.name)
            x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.nn_agent._net_obj()(x)
            self._cached_has_wait = (out[-1].shape[-1] == _NUM_ACTIONS)
        except Exception:
            self._cached_has_wait = False
        return self._cached_has_wait

    def _max_wait_prob(self):
        """基于当前玩家视角，估计下家最大待牌概率。"""
        if not self._has_wait_dist():
            return 0.0
        ctx = self.nn_agent.context
        if ctx is None:
            return 0.0
        feat = extract_features(ctx, self.full_hand(), self.name)
        x = torch.from_numpy(np.asarray(feat, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.nn_agent._net_obj()(x)
            wait_logits = out[-1].squeeze(0)
            probs = torch.sigmoid(wait_logits).cpu().numpy().astype(np.float64)
        return float(probs.max())

    def _is_critical(self):
        # 原有条件优先
        if super()._is_critical():
            return True
        # wait_dist 预测下家可能听牌，也切搜索
        try:
            if self._max_wait_prob() >= self.wait_threshold:
                return True
        except Exception:
            pass
        return False
