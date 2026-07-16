# -*- coding: utf-8 -*-
"""BeliefExpectimax + 序列模型默听防守（方向 D3）。

在 BeliefWaitDistAgent 基础上，用 SeqOppModel（GRU 弃牌序列编码，
`scripts/rl/train_seq_opp_model.py` 训练）替换 wait_dist3 模型：

- `tenpai` head 的默听概率并入 `_danger_signal`（对手未报听但模型认为
  其听牌概率超过阈值时进入安全模式）；
- `wait` head 的三家待牌分布并入 `_aggregate_danger`（沿用 wait_alpha 机制）。

报听步（decl_step）与对手副露通过 handle_msg 自行跟踪（ContextV3 不存）。

用法（benchmark_pool token）：
    besilent:<label>[:<model_path>]
默认 model_path = output/nn_seq_opp_v1.pt。
"""
import json
import os

import numpy as np
import torch

from algo.agents.belief_waitdist_agent import BeliefWaitDistAgent
from algo.agents.belief_endgame_agent import _seat
from algo.nn.features import extract_features, tile_to_index

_MAX_SEQ = 40


class BeliefSilentGuardAgent(BeliefWaitDistAgent):
    def __init__(self, name, verbose=False, seq_model_path=None,
                 tenpai_prob_threshold=0.5, **kwargs):
        super().__init__(name, verbose=verbose, **kwargs)
        if seq_model_path is None:
            seq_model_path = os.environ.get(
                'SEQ_MODEL_PATH', 'output/nn_seq_opp_v1.pt')
        self.seq_model_path = seq_model_path
        self.tenpai_prob_threshold = float(
            os.environ.get('TENPAI_PROB_THRESHOLD', tenpai_prob_threshold))
        self._seq_net = None
        self._decl_step = {}     # seat -> 报听步
        self._opp_meld_hot = {}  # seat -> 34-dim multi-hot
        self._out_cache = None
        self._cache_key = None

    def init_tiles(self, l):
        super().init_tiles(l)
        self._decl_step = {}
        self._opp_meld_hot = {}
        self._out_cache = None
        self._cache_key = None

    def handle_msg(self, msg):
        if msg.type == 'tenpai':
            s = _seat(msg.sender) if '@' in msg.sender else None
            if s is not None and s not in self._decl_step:
                self._decl_step[s] = len(self.context.discards.get(msg.sender, []))
        elif msg.type == 'meld':
            s = _seat(msg.sender) if '@' in msg.sender else None
            if s is not None:
                hot = self._opp_meld_hot.get(s)
                if hot is None:
                    hot = np.zeros(34, dtype=np.float32)
                    self._opp_meld_hot[s] = hot
                hot[tile_to_index(msg.data['tile'])] = 1.0
        return super().handle_msg(msg)

    def _load_seq_net(self):
        if self._seq_net is not None:
            return self._seq_net
        from scripts.rl.train_seq_opp_model import SeqOppModel
        cfg_path = self.seq_model_path.replace('.pt', '_config.json')
        with open(cfg_path) as f:
            cfg = json.load(f)
        net = SeqOppModel(use_seq=cfg.get('use_seq', True))
        sd = torch.load(self.seq_model_path, map_location='cpu')
        if isinstance(sd, dict) and 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        net.load_state_dict(sd)
        net.eval()
        net.to(self.device)
        self._seq_net = net
        return net

    def _model_inputs(self):
        """构造 (feats, opp_seq, opp_seq_len, opp_decl_step, opp_meld_hot)。"""
        feats = np.asarray(extract_features(self.context, list(self.cur), self.name),
                           dtype=np.float32)
        self_s = _seat(self.name)
        opp_seq = np.full((3, _MAX_SEQ), -1, dtype=np.int64)
        opp_seq_len = np.zeros(3, dtype=np.int64)
        opp_decl_step = np.full(3, -1, dtype=np.int64)
        opp_meld = np.zeros((3, 34), dtype=np.float32)
        players_by_seat = {}
        for p in self.context.discards:
            players_by_seat[_seat(p)] = p
        for rel in (1, 2, 3):
            s = (self_s + rel) % 4
            row = rel - 1
            p = players_by_seat.get(s)
            if p is not None:
                ids = [tile_to_index(t) for t in self.context.discards[p]][:_MAX_SEQ]
                opp_seq[row, :len(ids)] = ids
                opp_seq_len[row] = len(ids)
            opp_decl_step[row] = self._decl_step.get(s, -1)
            hot = self._opp_meld_hot.get(s)
            if hot is not None:
                opp_meld[row] = hot
        return feats, opp_seq, opp_seq_len, opp_decl_step, opp_meld

    def _seq_out(self):
        """返回 (tenpai_probs(3,), wait_probs(3,34))；按局面缓存。"""
        key = (sum(len(v) for v in self.context.discards.values()), len(self.cur))
        if self._cache_key == key and self._out_cache is not None:
            return self._out_cache
        net = self._load_seq_net()
        feats, seq, seq_len, decl_step, meld = self._model_inputs()
        with torch.no_grad():
            t = [torch.from_numpy(x).unsqueeze(0).to(self.device)
                 for x in (feats, seq, seq_len, decl_step, meld)]
            tl, wl = net(*t)
            out = (torch.sigmoid(tl).squeeze(0).cpu().numpy(),
                   torch.sigmoid(wl).squeeze(0).cpu().numpy())
        self._cache_key = key
        self._out_cache = out
        return out

    def _wait_probs(self):
        return self._seq_out()[1]

    def _danger_signal(self):
        if super()._danger_signal():
            return True
        try:
            tenpai_p, _ = self._seq_out()
            return bool((tenpai_p > self.tenpai_prob_threshold).any())
        except Exception:
            return False
