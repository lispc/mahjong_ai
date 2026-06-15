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
    """更深的价值网络：fc 512 -> 256 -> 128 -> value。"""

    def __init__(self, input_dim=175):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.value_head = nn.Linear(128, 1)

    def __call__(self, x):
        h = mx.maximum(self.fc1(x), 0)
        h = mx.maximum(self.fc2(h), 0)
        h = mx.maximum(self.fc3(h), 0)
        return self.value_head(h).squeeze(-1)
