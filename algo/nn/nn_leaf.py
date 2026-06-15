# -*- coding: utf-8 -*-
"""把训练好的 NN 作为叶子估值器供 expectimax / MCTS 使用。

为了避免在递归过程中把不可哈希的 context 传入 lru_cache，这里采用“调用前设置当前
context”的方式：每次 agent 做决策前把当前局面写进线程无关的全局变量，递归叶子估值时
直接读取。引擎对单个 agent 的 next() 是串行调用，因此这种单例方式是安全的。
"""

import os
import json
import numpy as np

import mlx.core as mx
import algo
import context as ctx_module

from algo.nn.model import MahjongNet
from algo.nn.value_model import MahjongValueNet, MahjongValueNetDeep
from algo.nn.features import extract_features, _context_features, _hand_to_array


_EMPTY_CONTEXT = ctx_module.Context()


_MODEL = None
_CONFIG = None
_CURRENT_CONTEXT = None
_CURRENT_PLAYER = None
_CURRENT_HAND14 = None


def _load_model():
    global _MODEL, _CONFIG
    if _MODEL is not None:
        return _MODEL, _CONFIG
    out_dir = 'output'

    # 1) 优先使用 MC rollout 训练出的深度价值网络
    mc_config_path = os.path.join(out_dir, 'nn_value_model_mc_config.json')
    mc_weights_path = os.path.join(out_dir, 'nn_value_model_mc.npz')
    if os.path.exists(mc_config_path) and os.path.exists(mc_weights_path):
        with open(mc_config_path, 'r') as f:
            _CONFIG = json.load(f)
        if _CONFIG.get('arch') == 'deep':
            _MODEL = MahjongValueNetDeep(_CONFIG['input_dim'])
        else:
            _MODEL = MahjongValueNet(_CONFIG['input_dim'], _CONFIG.get('hidden_dim', 256))
        _MODEL.load_weights(mc_weights_path)
        return _MODEL, _CONFIG

    # 2) 其次使用独立训练的价值网络
    value_config_path = os.path.join(out_dir, 'nn_value_model_config.json')
    value_weights_path = os.path.join(out_dir, 'nn_value_model.npz')
    if os.path.exists(value_config_path) and os.path.exists(value_weights_path):
        with open(value_config_path, 'r') as f:
            _CONFIG = json.load(f)
        _MODEL = MahjongValueNet(_CONFIG['input_dim'], _CONFIG['hidden_dim'])
        _MODEL.load_weights(value_weights_path)
        return _MODEL, _CONFIG

    # 3) 回退到 policy-value 网络的 value head
    config_path = os.path.join(out_dir, 'nn_model_config.json')
    weights_path = os.path.join(out_dir, 'nn_model.npz')
    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        raise FileNotFoundError(
            f'NN model not found: {config_path} or {weights_path}. '
            'Run scripts/train_nn.py first.')
    with open(config_path, 'r') as f:
        _CONFIG = json.load(f)
    _MODEL = MahjongNet(_CONFIG['input_dim'], _CONFIG['hidden_dim'])
    _MODEL.load_weights(weights_path)
    return _MODEL, _CONFIG


def set_leaf_context(context, player_name, current_hand14=None):
    """在 expectimax 搜索前设置当前决策局面。"""
    global _CURRENT_CONTEXT, _CURRENT_PLAYER, _CURRENT_HAND14
    _CURRENT_CONTEXT = context
    _CURRENT_PLAYER = player_name
    _CURRENT_HAND14 = current_hand14


def clear_leaf_context():
    global _CURRENT_CONTEXT, _CURRENT_PLAYER, _CURRENT_HAND14
    _CURRENT_CONTEXT = None
    _CURRENT_PLAYER = None
    _CURRENT_HAND14 = None


def nn_leaf_value(hand):
    """返回当前手牌在当前已设 context 下的 NN 价值估计（标量 float）。

    独立价值网络输出的是未约束的标量；这里把它与原项目 eval0 相加作为残差，保留
    eval0 的强先验，同时让 NN 学习微调。
    """
    return nn_leaf_values_batch([hand])[0]


def nn_leaf_values_batch(hands):
    """批量评估多个手牌，返回 list[float]。

    把同一局面下的所有叶子手牌拼成一个大 batch 一次性前向，避免多次 MLX 小 batch
    调度的开销。
    """
    if not hands:
        return []
    model, _ = _load_model()
    ctx = _CURRENT_CONTEXT
    player = _CURRENT_PLAYER
    if ctx is None or player is None:
        raise RuntimeError('nn_leaf_values_batch called before set_leaf_context')

    # 上下文特征只算一次；叶子之间的差异只有手牌 34 维
    ctx_arr = _context_features(ctx, _CURRENT_HAND14, player)

    n = len(hands)
    hand_matrix = np.zeros((n, 34), dtype=np.float32)
    for i, hand in enumerate(hands):
        hand_matrix[i] = _hand_to_array(hand) / 4.0

    X = np.concatenate([hand_matrix, np.tile(ctx_arr, (n, 1))], axis=1)
    values = model(mx.array(X))
    values = np.array(values.tolist(), dtype=np.float32).reshape(-1)

    base_values = []
    for hand in hands:
        base_values.append(algo.eval0(hand, _EMPTY_CONTEXT) + float(values[len(base_values)]) * 2.0)
    return base_values
