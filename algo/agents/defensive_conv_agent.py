# -*- coding: utf-8 -*-
"""带 deal-in head 防御重排的 conv-BC agent。

在 PPOAgent 基础上，用训练好的 deal-in auxiliary head 估计每张弃牌的即时点炮概率，
从 policy logits 中减去 penalty，降低选到高危牌的概率。

Beta 通过环境变量 `DEALIN_BETA` 控制（默认 2.0）；越大越保守。

用法（benchmark_pool token）：
    ppo:defensive:<path>      # beta=DEALIN_BETA 或 2.0
"""

import os
import numpy as np
import torch

from algo.agents.ppo_agent import PPOAgent
from algo.nn.features import _TILE_TO_IDX, _IDX_TO_TILE


class DefensiveConvAgent(PPOAgent):
    def __init__(self, name, model_path='output/nn_conv_bc_dealin_500.pt',
                 device='cpu', temperature=None, verbose=False):
        super().__init__(name, model_path=model_path, device=device,
                         temperature=temperature, verbose=verbose)
        self.beta = float(os.environ.get('DEALIN_BETA', '2.0'))

    def next(self):
        assert len(self.cur) == 14
        net = self._net_obj()
        feats = self._extract(self.context, self.cur, self.name)
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = net(x)
            logits = out[0].squeeze(0)
            dealin_logits = out[2].squeeze(0)
            p_dealin = torch.sigmoid(dealin_logits)
        logits = logits.detach().cpu().numpy().astype(np.float64)
        p_dealin = p_dealin.detach().cpu().numpy().astype(np.float64)

        legal = np.zeros(34, dtype=np.float32)
        for t in self.cur:
            legal[int(_TILE_TO_IDX[t])] = 1.0

        # 防御重排：点炮概率越高，logit 惩罚越大
        adjusted = logits - self.beta * p_dealin
        masked = adjusted + (legal - 1.0) * 1e9

        if self.temperature and self.temperature > 1e-6:
            m = masked / self.temperature
            m = m - m.max()
            probs = np.exp(m)
            probs = probs / probs.sum()
            a = int(np.random.choice(34, p=probs))
        else:
            a = int(np.argmax(masked))

        tile_val = int(_IDX_TO_TILE[a])
        self.cur.remove(tile_val)
        return tile_val
