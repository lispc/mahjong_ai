# -*- coding: utf-8 -*-
"""用训练好的 Policy-Value 网络做决策的 Agent。"""

import os
import json

import numpy as np
import agent
import tile
import algo.context.v3 as context_v3
import algo.eval.v2 as eval_v2
import mlx.core as mx

from algo.nn.model import MahjongNet
from algo.nn.features import extract_features, tile_to_index


class NNAgent(agent.Agent):
    """
    轻量 NN Policy Agent。

    参数：
        model_path: 训练好的 MLX 权重路径。
        sample: True 时按 policy 概率采样；False 时取最大概率。
    """

    def __init__(self, name, verbose=False,
                 model_path='output/nn_model.npz',
                 sample=False):
        super().__init__(name, verbose)
        self.model_path = model_path
        self.sample = sample
        self.context = context_v3.ContextV3()
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f'Model not found: {self.model_path}')
        config_path = self.model_path.replace('.npz', '_config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            hidden_dim = config.get('hidden_dim', 128)
        else:
            hidden_dim = 128
        self.model = MahjongNet(input_dim=175, hidden_dim=hidden_dim)
        self.model.load_weights(self.model_path)
        mx.eval(self.model.parameters())

    def init_tiles(self, l):
        super().init_tiles(l)
        self.context = context_v3.ContextV3()

    def handle_msg(self, msg):
        if msg.type == 'put':
            self.context.see_tile(msg.data, msg.sender)
        elif msg.type == 'tenpai':
            self.context.declare_tenpai(msg.sender)
        return super().handle_msg(msg)

    def declare_tenpai(self, hand, context):
        if context is None:
            return False
        if sum(len(v) for v in context.discards.values()) < 12:
            return False
        if eval_v2.shanten(hand) != 0:
            return False
        remaining = context.remaining_wall(hand)
        waits = eval_v2.winning_tiles(hand, remaining)
        if not waits:
            return False
        total_wait = sum(remaining.get(t, 0) for t in waits)
        if total_wait >= 4:
            return True
        for t in waits:
            if context.all_seen.get(t, 0) > 0 and remaining.get(t, 0) > 0:
                return True
        return False

    def next(self):
        assert len(self.cur) == 14

        features = extract_features(self.context, self.cur, self.name)
        X = mx.array(features.reshape(1, -1))
        logits, _ = self.model(X)
        logits = logits[0]

        # mask：只能打出手牌中有的牌
        legal = set(self.cur)
        legal_indices = {tile_to_index(t) for t in legal}
        mask = np.ones(34, dtype=np.float32)
        for idx in range(34):
            if idx not in legal_indices:
                mask[idx] = 0.0
        mask_mx = mx.array(mask)
        logits = logits * mask_mx + (mask_mx - 1.0) * 1e9

        if self.sample:
            probs = mx.softmax(logits, axis=-1)
            idx = int(mx.random.categorical(mx.log(probs)))
        else:
            idx = int(mx.argmax(logits, axis=-1))

        # 把 idx 映射回 tile value
        from algo.nn.features import _IDX_TO_TILE
        result = int(_IDX_TO_TILE[idx])

        self.cur.remove(result)
        self.context.see_tile(result, self.name)
        if self.verbose:
            print('出牌:' + tile.tile_to_str(result))
        return result
