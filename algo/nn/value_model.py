# -*- coding: utf-8 -*-
"""独立的价值网络（只输出一个标量价值）。"""

import mlx.core as mx
import mlx.nn as nn


class MahjongValueNet(nn.Module):
    def __init__(self, input_dim=175, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

    def __call__(self, x):
        h = mx.maximum(self.fc2(mx.maximum(self.fc1(x), 0)), 0)
        return self.value_head(h).squeeze(-1)
