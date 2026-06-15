# -*- coding: utf-8 -*-
"""独立的价值网络（只输出一个标量价值）。"""

import mlx.core as mx
import mlx.nn as nn


class MahjongValueNet(nn.Module):
    """默认两层 MLP 价值网络。"""

    def __init__(self, input_dim=175, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

    def __call__(self, x):
        h = mx.maximum(self.fc2(mx.maximum(self.fc1(x), 0)), 0)
        return self.value_head(h).squeeze(-1)


class MahjongValueNetDeep(nn.Module):
    """更深的价值网络：默认 fc 512 -> 256 -> 128 -> value。

    hidden_dims 可配置，例如 [1024, 512, 256] 以训练更大的网络。
    """

    def __init__(self, input_dim=175, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        dims = [input_dim] + list(hidden_dims)
        # 层名从 fc1 开始，保证默认 [512,256,128] 架构与旧权重兼容
        for i in range(1, len(dims)):
            setattr(self, f'fc{i}', nn.Linear(dims[i - 1], dims[i]))
        self.value_head = nn.Linear(dims[-1], 1)
        self._n_layers = len(dims) - 1

    def __call__(self, x):
        h = x
        for i in range(1, self._n_layers + 1):
            h = mx.maximum(getattr(self, f'fc{i}')(h), 0)
        return self.value_head(h).squeeze(-1)
