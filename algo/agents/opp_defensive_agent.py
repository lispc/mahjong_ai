# -*- coding: utf-8 -*-
"""把对手听牌预测接入防御重排的对战 agent。

在 DefensiveConvAgent 基础上，额外用对手模型估计三个对手的听牌概率。
当对手更可能听牌时，放大 deal-in head 的惩罚；反之则更偏向进攻。

超参（环境变量）：
    DEALIN_BETA    基础 deal-in 惩罚系数（默认 2.0）
    OPP_BETA       对手听牌概率对惩罚的放大系数（默认 2.0）

用法（benchmark_pool token）：
    oppdef:<label>:<model_path>:<opp_model_path>

若省略 opp_model_path，默认使用 output/opponent_model.pt 或环境变量 OPP_MODEL_PATH。
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn

from algo.agents.ppo_agent import PPOAgent
from algo.nn.features import _TILE_TO_IDX, _IDX_TO_TILE


_NUM_ACTIONS = 34
_OPP_NET_CACHE = {}


def _build_mlp(input_dim, hidden_dims, output_dim, dropout=0.2):
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def _load_opp_net(opp_model_path, device='cpu'):
    path = os.path.abspath(opp_model_path)
    key = (path, device)
    if key in _OPP_NET_CACHE:
        return _OPP_NET_CACHE[key]
    cfg_path = path.replace('.pt', '_config.json')
    with open(cfg_path) as f:
        cfg = json.load(f)
    net = _build_mlp(
        cfg['input_dim'], cfg.get('hidden_dims', [256, 128]),
        cfg.get('output_dim', 3), cfg.get('dropout', 0.2))
    sd = torch.load(path, map_location='cpu', weights_only=False)
    state = sd.get('model_state', sd)
    net.load_state_dict(state, strict=False)
    net.eval()
    net.to(device)
    _OPP_NET_CACHE[key] = (net, cfg)
    return net, cfg


class OppDefensiveAgent(PPOAgent):
    """PPOAgent + deal-in head + 对手听牌概率放大防御。"""

    def __init__(self, name, model_path='output/nn_full_action_best.pt',
                 opp_model_path=None, device='cpu', temperature=None, verbose=False):
        super().__init__(name, model_path=model_path, device=device,
                         temperature=temperature, verbose=verbose)
        if opp_model_path is None:
            opp_model_path = os.environ.get('OPP_MODEL_PATH', 'output/opponent_model.pt')
        self.opp_model_path = opp_model_path
        self.dealin_beta = float(os.environ.get('DEALIN_BETA', '2.0'))
        self.opp_beta = float(os.environ.get('OPP_BETA', '2.0'))
        self._opp_net = None

    def _opp_net_obj(self):
        if self._opp_net is None:
            self._opp_net, _ = _load_opp_net(self.opp_model_path, self.device)
        return self._opp_net

    def _seat_index(self):
        # name 格式如 P0@2，取 @ 前数字
        base = self.name.split('@')[0]
        return int(base[1:])

    def next(self):
        assert len(self.cur) >= 1
        net = self._net_obj()
        opp_net = self._opp_net_obj()

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

            opp_logits = opp_net(x).squeeze(0)
            opp_probs = torch.sigmoid(opp_logits).cpu().numpy().astype(np.float64)

        logits = logits.detach().cpu().numpy().astype(np.float64)
        p_dealin = p_dealin.detach().cpu().numpy().astype(np.float64)

        legal = np.zeros(_NUM_ACTIONS, dtype=np.float32)
        for t in self.cur:
            legal[int(_TILE_TO_IDX[t])] = 1.0

        # 对手听牌概率：下家/对家/上家
        # 当前出牌者点炮时，通常下家/对家/上家中有人荣和这张牌。
        # 用三个对手中最大听牌概率作为风险放大系数。
        max_opp_tenpai = float(opp_probs.max())
        # 也可以加权平均：avg = float(opp_probs.mean())

        # 防御重排：听牌概率越高，点炮惩罚越重
        adjusted = logits - self.dealin_beta * (1.0 + self.opp_beta * max_opp_tenpai) * p_dealin
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
