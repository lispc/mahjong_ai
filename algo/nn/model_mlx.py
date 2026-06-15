# -*- coding: utf-8 -*-
"""轻量 Policy-Value 网络（MLX）。"""

import mlx.core as mx
import mlx.nn as nn


class MahjongNet(nn.Module):
    """
    输入 175 维局面特征，输出：
    - policy_logits: 34 维（对应 34 种牌）
    - value: 1 维（当前玩家最终获胜期望，[-1, 1]）
    """

    def __init__(self, input_dim=175, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.policy_head = nn.Linear(hidden_dim // 2, 34)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

    def __call__(self, x):
        h = mx.maximum(self.fc1(x), 0)  # ReLU
        h = mx.maximum(self.fc2(h), 0)
        policy_logits = self.policy_head(h)
        value = mx.tanh(self.value_head(h))
        return policy_logits, value


def loss_fn(model, X, y_policy, y_value, policy_weight=1.0, value_weight=0.5):
    """policy 交叉熵 + value MSE。"""
    logits, value = model(X)

    # policy loss: sparse cross entropy
    log_probs = mx.log_softmax(logits, axis=-1)
    policy_loss = -mx.mean(mx.take_along_axis(
        log_probs, mx.expand_dims(y_policy, -1), axis=-1))

    # value loss: MSE
    value_loss = mx.mean((value.squeeze(-1) - y_value) ** 2)

    return policy_weight * policy_loss + value_weight * value_loss, {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
    }
