# -*- coding: utf-8 -*-
"""把对手待牌分布预测接入防御重排的对战 agent。

在 PPOAgent 基础上，用包含 wait_dist_head 的模型估计下家待牌分布。
当某张牌被下家等待的概率较高时，额外加重 deal-in 惩罚，以降低点炮风险。

超参（环境变量）：
    DEALIN_BETA    基础 deal-in 惩罚系数（默认 2.0）
    WAIT_BETA      待牌概率对惩罚的放大系数（默认 2.0）

用法（benchmark_pool token）：
    waitdef:<label>:<model_path>

model_path 需包含 wait_dist_head（如 output/nn_wait_dist_10k_tenpai_conv.pt）。
"""
import os
import numpy as np
import torch

from algo.agents.ppo_agent import PPOAgent
from algo.nn.features import _TILE_TO_IDX, _IDX_TO_TILE


_NUM_ACTIONS = 34


class WaitDistDefensiveAgent(PPOAgent):
    """PPOAgent + deal-in head + 下家待牌分布惩罚。"""

    def __init__(self, name, model_path='output/nn_wait_dist_10k_tenpai_conv.pt',
                 device='cpu', temperature=None, verbose=False):
        super().__init__(name, model_path=model_path, device=device,
                         temperature=temperature, verbose=verbose)
        self.dealin_beta = float(os.environ.get('DEALIN_BETA', '2.0'))
        self.wait_beta = float(os.environ.get('WAIT_BETA', '2.0'))

    def _wait_probs(self):
        """返回下家（seat+1）的 34 维待牌概率。"""
        if not self._has_wait_dist():
            return np.zeros(_NUM_ACTIONS, dtype=np.float64)
        feats = self._extract(self.context, self.full_hand(), self.name)
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self._net_obj()(x)
            wait_logits = out[-1].squeeze(0)
            probs = torch.sigmoid(wait_logits).cpu().numpy().astype(np.float64)
        return probs

    def _has_wait_dist(self):
        # forward 返回的最后一个 tensor 形状为 (34,) 时认为有 wait_dist_head
        if not hasattr(self, '_cached_has_wait'):
            self._cached_has_wait = False
            try:
                import torch
                from algo.nn.features import extract_features
                feats = extract_features(None, [11]*14, self.name)
                x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out = self._net_obj()(x)
                if out[-1].shape[-1] == _NUM_ACTIONS:
                    self._cached_has_wait = True
            except Exception:
                pass
        return self._cached_has_wait

    def next(self):
        assert len(self.cur) >= 1
        net = self._net_obj()

        feats = self._extract(self.context, self.full_hand(), self.name)
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = net(x)
            logits = out[0].squeeze(0)
            # deal-in head 存在时才使用；否则退化到 PPOAgent
            if len(out) > 2 and out[2] is not None:
                dealin_logits = out[2].squeeze(0)
                p_dealin = torch.sigmoid(dealin_logits)
            else:
                p_dealin = torch.zeros(_NUM_ACTIONS, device=self.device)

        logits = logits.detach().cpu().numpy().astype(np.float64)
        p_dealin = p_dealin.detach().cpu().numpy().astype(np.float64)

        legal = np.zeros(_NUM_ACTIONS, dtype=np.float32)
        for t in self.cur:
            legal[int(_TILE_TO_IDX[t])] = 1.0

        # 只在有人报听时启用待牌分布惩罚
        ctx = self.context
        tenpai_opps = getattr(ctx, 'tenpai', set()) if ctx else set()
        if self.name in tenpai_opps:
            tenpai_opps = set(tenpai_opps)
            tenpai_opps.discard(self.name)
        use_wait = bool(tenpai_opps) and self._has_wait_dist()

        if use_wait:
            p_wait = self._wait_probs()
            adjusted = logits - self.dealin_beta * (p_dealin + self.wait_beta * p_wait)
        else:
            adjusted = logits - self.dealin_beta * p_dealin

        masked = adjusted + (legal - 1.0) * 1e9

        if self.temperature and self.temperature > 1e-6:
            m = masked / self.temperature
            m = m - m.max()
            probs = np.exp(m)
            probs = probs / probs.sum()
            a = int(np.random.choice(_NUM_ACTIONS, p=probs))
        else:
            a = int(np.argmax(masked))

        tile_val = int(_IDX_TO_TILE[a])
        self.cur.remove(tile_val)
        self.context.see_tile(tile_val, self.name)
        self._belief = None
        if self.verbose:
            import tile
            print('出牌:' + tile.tile_to_str(tile_val))
        return tile_val
