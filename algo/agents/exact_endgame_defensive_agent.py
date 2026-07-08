# -*- coding: utf-8 -*-
"""用 exact endgame defensive head 做防守重排的 agent。

在 PPOAgent 基础上，加载带 `defensive_head` 的模型，把 defensive head 输出的
34-dim EV 加到 policy logits 上（EV 越高越优先），只在对手报听后启用防御重排。

用法（benchmark_pool token）：
    exactend:<label>:<model_path>
"""

import os
import numpy as np
import torch

from algo.agents.ppo_agent import PPOAgent
from algo.nn.features import _TILE_TO_IDX, _IDX_TO_TILE


class ExactEndgameDefensiveAgent(PPOAgent):
    def __init__(self, name, model_path='output/nn_exact_endgame_defensive.pt',
                 device='cpu', temperature=None, verbose=False):
        super().__init__(name, model_path=model_path, device=device,
                         temperature=temperature, verbose=verbose)
        self.beta = float(os.environ.get('DEF_BETA', '1.0'))

    def next(self):
        assert len(self.cur) >= 1
        net = self._net_obj()
        feats = self._extract(self.context, self.full_hand(), self.name)
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = net(x)
            logits = out[0].squeeze(0)
            # defensive head 是最后一个 34-dim 输出
            defensive_ev = out[-1].squeeze(0)
        logits = logits.detach().cpu().numpy().astype(np.float64)
        defensive_ev = defensive_ev.detach().cpu().numpy().astype(np.float64)

        legal = np.zeros(34, dtype=np.float32)
        for t in self.cur:
            legal[int(_TILE_TO_IDX[t])] = 1.0

        # 只在有对手报听时启用防御重排
        danger = False
        if self.context is not None:
            tenpai = getattr(self.context, 'tenpai_players', set())
            if tenpai - {self.name}:
                danger = True

        if danger:
            adjusted = logits + self.beta * defensive_ev
        else:
            adjusted = logits
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
