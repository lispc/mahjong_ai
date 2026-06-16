# -*- coding: utf-8 -*-
"""Rollout Policy Net：蒸馏 legacy eval2 的快速弃牌策略网络。

只输出 34 维 policy logits，没有 value head。用于 MC rollout 中替代 eval2。
"""

import os
import json

import torch
import torch.nn as nn


class RolloutPolicyNet(nn.Module):
    def __init__(self, input_dim=175, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.policy_head = nn.Linear(dims[-1], 34)

    def forward(self, x):
        return self.policy_head(self.net(x))


_MODEL = None
_CONFIG = None


def _load_model():
    global _MODEL, _CONFIG
    if _MODEL is not None:
        return _MODEL, _CONFIG
    out_dir = 'output'
    config_path = os.path.join(out_dir, 'nn_rollout_policy_config.json')
    weights_path = os.path.join(out_dir, 'nn_rollout_policy.pt')
    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        raise FileNotFoundError(
            f'Rollout policy model not found: {config_path} or {weights_path}. '
            'Run scripts/train_rollout_policy.py first.')
    with open(config_path, 'r') as f:
        _CONFIG = json.load(f)
    _MODEL = RolloutPolicyNet(_CONFIG['input_dim'], _CONFIG['hidden_dims'])
    _MODEL.load_state_dict(torch.load(weights_path, map_location='cpu'))
    _MODEL.eval()
    if torch.cuda.is_available():
        _MODEL = _MODEL.cuda()
    return _MODEL, _CONFIG


def clear_model():
    global _MODEL, _CONFIG
    _MODEL = None
    _CONFIG = None


def policy_scores(hand14, context, player_name):
    """返回 34 维 logit 或概率。"""
    from algo.nn.features import extract_features
    import numpy as np

    model, _ = _load_model()
    features = extract_features(context, hand14, player_name)
    x = torch.tensor(features, dtype=torch.float32).reshape(1, -1)
    if torch.cuda.is_available():
        x = x.cuda()
    with torch.no_grad():
        logits = model(x).cpu().numpy().reshape(-1)
    return logits


def select(hand14, context, player_name='V3NN'):
    """返回按 rollout policy net 排序的弃牌 tile 列表。"""
    from algo.nn.features import _TILE_TO_IDX, _IDX_TO_TILE
    import numpy as np

    logits = policy_scores(hand14, context, player_name)
    legal_tiles = list(set(hand14))
    legal_indices = [_TILE_TO_IDX[t] for t in legal_tiles]

    # mask illegal
    mask = np.full(34, -1e9, dtype=np.float32)
    mask[legal_indices] = 0
    logits = logits + mask
    order = np.argsort(logits)[::-1]

    result = []
    seen = set()
    for idx in order:
        tile = _IDX_TO_TILE[idx]
        if tile not in seen:
            seen.add(tile)
            result.append(tile)
    return result
