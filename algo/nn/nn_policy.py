# -*- coding: utf-8 -*-
"""用训练好的 Policy-Value 网络的 policy head 给候选弃牌打分。"""

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
    from algo.nn.model import MahjongNet
    out_dir = 'output'
    config_path = os.path.join(out_dir, 'nn_model_config.json')
    weights_path = os.path.join(out_dir, 'nn_model.npz')
    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        raise FileNotFoundError(
            f'Policy model not found: {config_path} or {weights_path}. '
            'Run scripts/train_nn.py first.')
    with open(config_path, 'r') as f:
        _POLICY_CONFIG = json.load(f)
    _POLICY_MODEL = MahjongNet(_POLICY_CONFIG['input_dim'], _POLICY_CONFIG['hidden_dim'])
    _POLICY_MODEL.load_weights(weights_path)
    return _POLICY_MODEL, _POLICY_CONFIG


def policy_scores(hand14, context, player_name):
    """返回 34 维概率分布（按 tile index 0..33 排列，非法动作概率为 0）。"""
    import mlx.core as mx
    model, _ = _load_policy_model()
    features = extract_features(context, hand14, player_name)
    x = mx.array(features).reshape(1, -1)
    logits, _ = model(x)

    legal_indices = list({int(_TILE_TO_IDX[t]) for t in hand14})
    mask = np.zeros(34, dtype=np.float32)
    mask[legal_indices] = 1.0
    mask_mx = mx.array(mask)
    # 屏蔽非法动作
    logits = logits * mask_mx + (mask_mx - 1.0) * 1e9
    logits = logits - mx.max(logits, axis=-1, keepdims=True)
    probs = mx.exp(logits) / mx.sum(mx.exp(logits), axis=-1, keepdims=True)
    return np.array(probs.reshape(-1).tolist(), dtype=np.float32)


def top_discards(hand14, context, player_name, k=5):
    """返回 policy 概率最高的 k 个弃牌（tile value 列表）。"""
    probs = policy_scores(hand14, context, player_name)
    legal_tiles = list(set(hand14))
    legal_probs = [(t, float(probs[int(_TILE_TO_IDX[t])])) for t in legal_tiles]
    legal_probs.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in legal_probs[:k]]
