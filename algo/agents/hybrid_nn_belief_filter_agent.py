# -*- coding: utf-8 -*-
"""Hybrid + wait_dist / defensive head safety filter。

在 HybridNNBeliefAgent 基础上，当 non-critical 状态使用 NN policy 时，
额外用 wait_dist_head 或 defensive_head 的输出对高危险 tile 降分。
critical 状态仍切换到 BeliefExpectimax。

用法（benchmark_pool token）：
    hybridfilter:<label>:<model_path>[:<filter_kind>]

filter_kind = 'wait' | 'def' | 'both'（默认 both）
"""
import os
import numpy as np
import torch

from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent
from algo.nn.features import _TILE_TO_IDX, _IDX_TO_TILE


_NUM_ACTIONS = 34


class HybridNNBeliefFilterAgent(HybridNNBeliefAgent):
    def __init__(self, name, nn_model_path='output/nn_full_action_best.pt',
                 belief_kind='beliefexp', tenpai_threshold=28,
                 filter_kind=None, device='cpu', temperature=None, verbose=False,
                 nn_agent_class=None):
        super().__init__(name, nn_model_path=nn_model_path, belief_kind=belief_kind,
                         tenpai_threshold=tenpai_threshold, device=device,
                         temperature=temperature, verbose=verbose,
                         nn_agent_class=nn_agent_class)
        if filter_kind is None:
            filter_kind = os.environ.get('HYBRID_FILTER_KIND', 'both')
        self.filter_kind = filter_kind
        self.dealin_beta = float(os.environ.get('DEALIN_BETA', '2.0'))
        self.wait_beta = float(os.environ.get('WAIT_BETA', '0.5'))
        self.def_beta = float(os.environ.get('DEF_BETA', '1.0'))
        self._cfg = None

    def _model_cfg(self):
        if self._cfg is None:
            self._cfg = self.nn_agent._cfg
        return self._cfg

    def _has_head(self, key):
        return self._model_cfg().get(key, False)

    def _wait_probs(self):
        if not self._has_head('wait_dist_head'):
            return np.zeros(_NUM_ACTIONS, dtype=np.float64)
        feats = self.nn_agent._extract(self.nn_agent.context, self.full_hand(), self.name)
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.nn_agent._net_obj()(x)
            wait_logits = out[-1]
            probs = torch.sigmoid(wait_logits).cpu().numpy().astype(np.float64)
        return probs.squeeze()

    def _defensive_ev(self):
        if not self._has_head('defensive_head'):
            return np.zeros(_NUM_ACTIONS, dtype=np.float64)
        feats = self.nn_agent._extract(self.nn_agent.context, self.full_hand(), self.name)
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.nn_agent._net_obj()(x)
            ev = out[-1]
        return ev.cpu().numpy().astype(np.float64).squeeze()

    def _any_opponent_tenpai(self):
        ctx = self.nn_agent.context
        if ctx is None:
            return False
        tenpai = getattr(ctx, 'tenpai', set())
        if self.name in tenpai:
            tenpai = set(tenpai)
            tenpai.discard(self.name)
        return bool(tenpai)

    def next_with_trace(self):
        if self._is_critical():
            tile_val, trace = self.belief_agent.next_with_trace()
            return tile_val, trace

        # non-critical：用 NN policy，加 safety filter
        nn = self.nn_agent
        net = nn._net_obj()
        feats = nn._extract(nn.context, self.full_hand(), nn.name)
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(nn.device)
        with torch.no_grad():
            out = net(x)
            logits = out[0].squeeze(0)
            dealin = torch.zeros(_NUM_ACTIONS, device=nn.device)
            if len(out) > 2 and out[2] is not None and out[2].shape[-1] == _NUM_ACTIONS:
                dealin = torch.sigmoid(out[2].squeeze(0))

        logits = logits.detach().cpu().numpy().astype(np.float64)
        dealin = dealin.detach().cpu().numpy().astype(np.float64)

        legal = np.zeros(_NUM_ACTIONS, dtype=np.float32)
        for t in self.cur:
            legal[int(_TILE_TO_IDX[t])] = 1.0

        adjusted = logits - self.dealin_beta * dealin
        if self._any_opponent_tenpai():
            if self.filter_kind in ('wait', 'both') and self._has_head('wait_dist_head'):
                adjusted -= self.dealin_beta * self.wait_beta * self._wait_probs()
            if self.filter_kind in ('def', 'both') and self._has_head('defensive_head'):
                adjusted += self.def_beta * self._defensive_ev()

        masked = adjusted + (legal - 1.0) * 1e9
        if nn.temperature and nn.temperature > 1e-6:
            m = masked / nn.temperature
            m = m - m.max()
            probs = np.exp(m)
            probs = probs / probs.sum()
            a = int(np.random.choice(_NUM_ACTIONS, p=probs))
        else:
            a = int(np.argmax(masked))

        tile_val = int(_IDX_TO_TILE[a])
        self.cur.remove(tile_val)
        self.nn_agent.context.see_tile(tile_val, self.name)
        self.belief_agent.context.see_tile(tile_val, self.name)
        self.nn_agent._belief = None
        self.belief_agent._belief = None
        if self.verbose:
            import tile
            print('出牌:' + tile.tile_to_str(tile_val))
        return tile_val, None

    def next(self):
        return self.next_with_trace()[0]
