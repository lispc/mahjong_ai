# -*- coding: utf-8 -*-
"""用训练好的 Policy-Value 网络的 policy head 给候选弃牌打分（PyTorch）。"""

import os
import json
import numpy as np

from algo.nn.features import extract_features, _TILE_TO_IDX, _IDX_TO_TILE


_POLICY_MODEL = None
_POLICY_CONFIG = None
_MODEL_CACHE = {}   # weights_path -> (model, config)，支持每实例用不同候选模型


def _load_policy_model():
    global _POLICY_MODEL, _POLICY_CONFIG
    if _POLICY_MODEL is not None:
        return _POLICY_MODEL, _POLICY_CONFIG
    out_dir = 'output'
    config_path = os.path.join(out_dir, 'nn_model_config.json')
    weights_path = os.path.join(out_dir, 'nn_model.pt')
    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        raise FileNotFoundError(
            f'Policy model not found: {config_path} or {weights_path}. '
            'Run scripts/train_nn.py first.')
    _POLICY_MODEL, _POLICY_CONFIG = _load_policy_model_from(weights_path, config_path)
    return _POLICY_MODEL, _POLICY_CONFIG


def _load_policy_model_from(weights_path, config_path=None):
    """按路径加载并缓存一个 policy 模型（用于 RL+搜索融合的候选生成）。

    config 缺省用 output/nn_model_config.json（PPO 模型架构与 nn_model 相同）。
    """
    key = os.path.abspath(weights_path)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    import torch
    from algo.nn.model import build_model
    if config_path is None:
        cand = weights_path.replace('.pt', '_config.json')
        config_path = cand if os.path.exists(cand) else os.path.join('output', 'nn_model_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    model = build_model(config)   # 支持 mlp / conv 架构
    sd = torch.load(weights_path, map_location='cpu')
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    model.load_state_dict(sd)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    _MODEL_CACHE[key] = (model, config)
    return model, config


def _policy_scores_core(model, hand14, context, player_name):
    import torch
    features = extract_features(context, hand14, player_name)
    x = torch.tensor(features, dtype=torch.float32).reshape(1, -1)
    if torch.cuda.is_available():
        x = x.cuda()
    with torch.no_grad():
        out = model(x)
        logits = out[0].cpu().numpy().reshape(-1)

    legal_indices = list({int(_TILE_TO_IDX[t]) for t in hand14})
    mask = np.zeros(34, dtype=np.float32)
    mask[legal_indices] = 1.0
    logits = logits * mask + (mask - 1.0) * 1e9
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    probs = exp / np.sum(exp)
    return probs.astype(np.float32)


def policy_scores(hand14, context, player_name):
    """返回 34 维概率分布（用默认全局 nn_model.pt）。"""
    model, _ = _load_policy_model()
    return _policy_scores_core(model, hand14, context, player_name)


def policy_scores_with_model(model, hand14, context, player_name):
    """返回 34 维概率分布（用指定 model，用于融合）。"""
    return _policy_scores_core(model, hand14, context, player_name)


def _topk_from_probs(probs, hand14, k):
    legal_tiles = list(set(hand14))
    legal_probs = [(t, float(probs[int(_TILE_TO_IDX[t])])) for t in legal_tiles]
    legal_probs.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in legal_probs[:k]]


def top_discards(hand14, context, player_name, k=5):
    """返回 policy 概率最高的 k 个弃牌（默认全局 nn_model.pt）。"""
    probs = policy_scores(hand14, context, player_name)
    return _topk_from_probs(probs, hand14, k)


def top_discards_with_model(model, hand14, context, player_name, k=5):
    """返回指定 model 的 policy 概率最高的 k 个弃牌（用于 RL+搜索融合）。"""
    probs = _policy_scores_core(model, hand14, context, player_name)
    return _topk_from_probs(probs, hand14, k)
