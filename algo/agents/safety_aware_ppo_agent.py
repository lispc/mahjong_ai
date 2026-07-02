# -*- coding: utf-8 -*-
"""Safety-aware tenpai declaration on top of PPOAgent.

Uses the dealin head to estimate the expected deal-in risk of being locked
after declaring tenpai.  If the expected risk is too high, the agent declines
to declare tenpai even though the base policy/tenpai_head says yes.
"""

import os
import numpy as np
import torch

from algo.agents.ppo_agent import PPOAgent
from algo.nn.features import _TILE_TO_IDX
import algo.eval.v2 as eval_v2


class SafetyAwarePPOAgent(PPOAgent):
    def __init__(self, name, model_path='output/nn_conv_bc.pt', device='cpu',
                 temperature=0.0, verbose=False):
        super().__init__(name, model_path=model_path, device=device,
                         temperature=temperature, verbose=verbose)
        # Maximum acceptable expected deal-in probability per locked discard.
        self.risk_threshold = float(os.environ.get('SAFETY_TENPAI_RISK', '0.25'))

    def declare_tenpai(self, hand, context):
        # First ask the base tenpai policy / tenpai_head.
        base = super().declare_tenpai(hand, context)
        if not base:
            return False
        if context is None:
            return base

        net = self._net_obj()
        if not getattr(net, 'use_dealin', False):
            # No dealin head -> cannot estimate safety, fall back to base.
            return base

        try:
            remaining = context.remaining_wall(hand)
            waits = eval_v2.winning_tiles(hand, remaining)
            # Tiles we could draw after being locked and would be forced to discard.
            draw_tiles = [t for t, c in remaining.items() if c > 0 and t not in waits]
            if not draw_tiles:
                return base

            batch = []
            idxs = []
            weights = []
            for t in draw_tiles:
                hand14 = list(hand) + [t]
                feats = self._extract(context, hand14, self.name)
                batch.append(feats)
                idxs.append(int(_TILE_TO_IDX[t]))
                weights.append(float(remaining[t]))

            x = torch.from_numpy(np.asarray(batch, dtype=np.float32)).to(self.device)
            idx_t = torch.tensor(idxs, dtype=torch.long, device=self.device)
            w_t = torch.tensor(weights, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                dealin_logits = net(x)[2]
            # deal-in probability for the tile that would be discarded.
            p_dealin = torch.sigmoid(dealin_logits[torch.arange(len(draw_tiles)), idx_t])
            expected_risk = (p_dealin * w_t).sum() / w_t.sum()
            return bool(expected_risk.item() <= self.risk_threshold)
        except Exception:
            # Any unexpected failure falls back to the base decision.
            return base
