# -*- coding: utf-8 -*-
"""用 tile danger 预测模型做防御重排的对战 agent。

输入 175-dim 公开特征，danger 模型输出 34 个 tile 各自的被荣和概率。
在 PPOAgent 基础上，从 policy logits 减去 danger 惩罚。

超参（环境变量）：
    DANGER_BETA    danger 惩罚系数（默认 2.0）

用法（benchmark_pool token）：
    danger:<label>:<model_path>:<danger_model_path>

若省略 danger_model_path，默认使用 output/opponent_danger_model.pt 或环境变量 DANGER_MODEL_PATH。
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn

from algo.agents.ppo_agent import PPOAgent
from algo.nn.features import _TILE_TO_IDX, _IDX_TO_TILE


_NUM_ACTIONS = 34
_DANGER_NET_CACHE = {}


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


def _load_danger_net(danger_model_path, device='cpu'):
    path = os.path.abspath(danger_model_path)
    key = (path, device)
    if key in _DANGER_NET_CACHE:
        return _DANGER_NET_CACHE[key]
    cfg_path = path.replace('.pt', '_config.json')
    with open(cfg_path) as f:
        cfg = json.load(f)
    arch = cfg.get('arch', 'danger_conv')
    if arch == 'danger_conv':
        from scripts.rl.train_opponent_danger_model import DangerConvNet
        net = DangerConvNet(
            n_tile_ch=5, channels=cfg.get('channels', 64),
            n_blocks=cfg.get('n_blocks', 4), hidden=cfg.get('hidden', 128),
            output_dim=cfg['output_dim'], dropout=cfg.get('dropout', 0.2))
    else:
        net = _build_mlp(cfg['input_dim'], cfg.get('hidden_dims', [256, 128]),
                         cfg['output_dim'], cfg.get('dropout', 0.2))
    sd = torch.load(path, map_location='cpu', weights_only=False)
    state = sd.get('model_state', sd)
    net.load_state_dict(state, strict=False)
    net.eval()
    net.to(device)
    _DANGER_NET_CACHE[key] = (net, cfg)
    return net, cfg


class DangerAwareAgent(PPOAgent):
    def __init__(self, name, model_path='output/nn_full_action_best.pt',
                 danger_model_path=None, device='cpu', temperature=None, verbose=False):
        super().__init__(name, model_path=model_path, device=device,
                         temperature=temperature, verbose=verbose)
        if danger_model_path is None:
            danger_model_path = os.environ.get('DANGER_MODEL_PATH', 'output/opponent_danger_model.pt')
        self.danger_model_path = danger_model_path
        self.danger_beta = float(os.environ.get('DANGER_BETA', '2.0'))
        self._danger_net = None

    def _danger_net_obj(self):
        if self._danger_net is None:
            self._danger_net, _ = _load_danger_net(self.danger_model_path, self.device)
        return self._danger_net

    def next(self):
        assert len(self.cur) >= 1
        net = self._net_obj()
        danger_net = self._danger_net_obj()

        feats = self._extract(self.context, self.full_hand(), self.name)
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = net(x)[0].squeeze(0)
            danger_logits = danger_net(x).squeeze(0)
            p_danger = torch.sigmoid(danger_logits).cpu().numpy().astype(np.float64)

        logits = logits.detach().cpu().numpy().astype(np.float64)

        legal = np.zeros(_NUM_ACTIONS, dtype=np.float32)
        for t in self.cur:
            legal[int(_TILE_TO_IDX[t])] = 1.0

        adjusted = logits - self.danger_beta * p_danger
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
