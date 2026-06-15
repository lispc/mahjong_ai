# -*- coding: utf-8 -*-
"""用训练好的 Policy-Value 网络的 policy head 给候选弃牌打分（PyTorch）。"""

import os
import json
import numpy as np

from algo.nn.features import extract_features, _TILE_TO_IDX, _IDX_TO_TILE


_POLICY_MODEL = None
_POLICY_CONFIG = None


def _load_policy_model():
    global _POLICY_MODEL, _POLICY_CONFIG
    if _POLICY_MODEL is not None:
        return _POLICY_MODEL, _POLICY_CONFIG

    import torch
    from algo.nn.model import MahjongNet
    out_dir = 'output'
    config_path = os.path.join(out_dir, 'nn_model_config.json')
    weights_path = os.path.join(out_dir, 'nn_model.pt')
    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        raise FileNotFoundError(
            f'Policy model not found: {config_path} or {weights_path}. '
            'Run scripts/train_nn.py first.')
    with open(config_path, 'r') as f:
        _POLICY_CONFIG = json.load(f)
    _POLICY_MODEL = MahjongNet(_POLICY_CONFIG['input_dim'], _POLICY_CONFIG['hidden_dim'])
    _POLICY_MODEL.load_state_dict(torch.load(weights_path, map_location='cpu'))
    _POLICY_MODEL.eval()
    if torch.cuda.is_available():
        _POLICY_MODEL = _POLICY_MODEL.cuda()
    return _POLICY_MODEL, _POLICY_CONFIG


def policy_scores(hand14, context, player_name):
    """返回 34 维概率分布（按 tile index 0..33 排列，非法动作概率为 0）。"""
    import torch

    model, _ = _load_policy_model()
    features = extract_features(context, hand14, player_name)
    x = torch.tensor(features, dtype=torch.float32).reshape(1, -1)
    if torch.cuda.is_available():
        x = x.cuda()

    with torch.no_grad():
        logits, _ = model(x)
        logits = logits.cpu().numpy().reshape(-1)

    legal_indices = list({int(_TILE_TO_IDX[t]) for t in hand14})
    mask = np.zeros(34, dtype=np.float32)
    mask[legal_indices] = 1.0

    # 屏蔽非法动作
    logits = logits * mask + (mask - 1.0) * 1e9
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    probs = exp / np.sum(exp)
    return probs.astype(np.float32)


def top_discards(hand14, context, player_name, k=5):
    """返回 policy 概率最高的 k 个弃牌（tile value 列表）。"""
    probs = policy_scores(hand14, context, player_name)
    legal_tiles = list(set(hand14))
    legal_probs = [(t, float(probs[int(_TILE_TO_IDX[t])])) for t in legal_tiles]
    legal_probs.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in legal_probs[:k]]
