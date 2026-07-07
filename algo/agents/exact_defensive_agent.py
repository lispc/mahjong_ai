# -*- coding: utf-8 -*-
"""把终局精确防守 EV 预测接入防御重排的对战 agent。

在 PPOAgent 基础上，用包含 defensive_head 的模型估计每个候选弃牌的
exact endgame EV（负值，越高越安全）。在 discard 决策中对低 EV（高风险）
的 tile 额外降分。

超参（环境变量）：
    DEALIN_BETA    基础 deal-in 惩罚系数（默认 2.0）
    DEF_BETA       defensive EV 放大系数（默认 2.0）

用法（benchmark_pool token）：
    exactdef:<label>:<model_path>

model_path 需包含 defensive_head（如 output/nn_defensive_1000.pt）。
"""
import os
import numpy as np
import torch

from algo.agents.ppo_agent import PPOAgent
from algo.nn.features import _TILE_TO_IDX, _IDX_TO_TILE


_NUM_ACTIONS = 34


class ExactDefensiveAgent(PPOAgent):
    """PPOAgent + deal-in head + exact endgame EV 惩罚。"""

    def __init__(self, name, model_path='output/nn_defensive_1000.pt',
                 device='cpu', temperature=None, verbose=False):
        super().__init__(name, model_path=model_path, device=device,
                         temperature=temperature, verbose=verbose)
        self.dealin_beta = float(os.environ.get('DEALIN_BETA', '2.0'))
        self.def_beta = float(os.environ.get('DEF_BETA', '2.0'))
        self._cached_has_def = None

    def _has_defensive(self):
        if self._cached_has_def is not None:
            return self._cached_has_def
        self._cached_has_def = self._cfg.get('defensive_head', False)
        return self._cached_has_def

    def _defensive_ev(self):
        """返回 34 维 EV 预测（负值，越高越安全）。"""
        # 如果模型无 defensive_head 但有 wait_dist 等，也不应误用；这里依赖 config
        feats = self._extract(self.context, self.full_hand(), self.name)
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self._net_obj()(x)
            ev = out[-1].squeeze(0)
        return ev.cpu().numpy().astype(np.float64)

    def next(self):
        assert len(self.cur) >= 1
        net = self._net_obj()

        feats = self._extract(self.context, self.full_hand(), self.name)
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = net(x)
            logits = out[0].squeeze(0)
            if len(out) > 2 and out[2] is not None and out[2].shape[-1] == _NUM_ACTIONS:
                dealin_logits = out[2].squeeze(0)
                p_dealin = torch.sigmoid(dealin_logits)
            else:
                p_dealin = torch.zeros(_NUM_ACTIONS, device=self.device)

        logits = logits.detach().cpu().numpy().astype(np.float64)
        p_dealin = p_dealin.detach().cpu().numpy().astype(np.float64)

        legal = np.zeros(_NUM_ACTIONS, dtype=np.float32)
        for t in self.cur:
            legal[int(_TILE_TO_IDX[t])] = 1.0

        # 只在有人报听时启用 exact defensive EV；否则退化为 PPOAgent
        ctx = self.context
        tenpai_opps = getattr(ctx, 'tenpai', set()) if ctx else set()
        if self.name in tenpai_opps:
            tenpai_opps = set(tenpai_opps)
            tenpai_opps.discard(self.name)
        use_defensive = bool(tenpai_opps) and self._has_defensive()

        if use_defensive:
            ev_pred = self._defensive_ev()
            # EV 范围约 [-1, 0]；只惩罚明显危险（低 EV）的 tile
            adjusted = logits - self.dealin_beta * p_dealin + self.def_beta * ev_pred
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
