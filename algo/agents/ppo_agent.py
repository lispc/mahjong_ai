# -*- coding: utf-8 -*-
"""用训练好的 PPO policy 网络直接决策的对战 agent（用于 benchmark）。

- 继承 BeliefExpectimaxV3Agent 以复用 ContextV3 上下文维护 + declare_tenpai 启发式；
- next() 覆盖为「NN policy 合法 argmax（或低温采样）」，不走 expectimax，纯前馈极快；
- 网络按 (path, hidden_dim) 在进程内缓存，兼容 ProcessPoolExecutor（fork）多进程 benchmark；
- 默认 CPU 推理（网络仅 ~82k 参数，CPU 亚毫秒，避免 fork+CUDA 多进程问题）。
"""

import os
import json
import numpy as np
import torch

from algo.agents.belief_expectimax_v3 import BeliefExpectimaxV3Agent
from algo.nn.features import extract_features, extract_features_ext, _TILE_TO_IDX, _IDX_TO_TILE

NUM_ACTIONS = 34
_NET_CACHE = {}


def _load_net(model_path, device='cpu'):
    key = (os.path.abspath(model_path), device)
    if key in _NET_CACHE:
        return _NET_CACHE[key]
    from algo.nn.model import build_model
    cfg_path = model_path.replace('.pt', '_config.json')
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(model_path), 'nn_model_config.json')
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
    else:
        cfg = {'arch': 'mlp', 'input_dim': 175, 'hidden_dim': 256}
    net = build_model(cfg)
    sd = torch.load(model_path, map_location='cpu')
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    # 允许加载含 tenpai_head 的模型到基础结构，或反之（仅加载匹配键）
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        # 新 head 缺失是预期行为；其它关键层缺失需警惕
        non_head = [k for k in missing if 'tenpai' not in k and 'dealin' not in k]
        if non_head:
            print(f'[PPOAgent] warning: missing non-optional keys: {non_head[:5]}')
    net.eval()
    net.to(device)
    _NET_CACHE[key] = (net, cfg)
    return net, cfg


class PPOAgent(BeliefExpectimaxV3Agent):
    def __init__(self, name, model_path='output/nn_rl_ppo.pt', device='cpu',
                 temperature=0.0, verbose=False):
        super().__init__(name, verbose=verbose)
        self.model_path = model_path
        self.device = device
        self.temperature = temperature   # 0 -> 贪婪 argmax
        self._net = None
        self._extract = extract_features

    def _net_obj(self):
        if self._net is None:
            self._net, cfg = _load_net(self.model_path, self.device)
            self._extract = extract_features_ext if cfg.get('features') == 'ext' else extract_features
        return self._net

    def declare_tenpai(self, hand, context):
        """优先使用 tenpai_head 做报听决策；否则回退到启发式。"""
        if not getattr(self, '_tenpai_use_head', None):
            # 第一次调用时根据模型配置决定
            net = self._net_obj()
            self._tenpai_use_head = self._cfg.get('tenpai_head', False)
        if self._tenpai_use_head and context is not None:
            try:
                feats = self._extract(context, hand, self.name)
                x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logit = self._net.tenpai_logit(x)
                return bool(logit.item() > 0.0)
            except Exception:
                # 推理失败时安全回退
                return super().declare_tenpai(hand, context)
        return super().declare_tenpai(hand, context)

    @property
    def _cfg(self):
        if self._net is None:
            self._net_obj()
        return _NET_CACHE[(os.path.abspath(self.model_path), self.device)][1]

    def next(self):
        assert len(self.cur) >= 1
        net = self._net_obj()
        # 特征用完整手牌（闭手+副露），但合法动作只在闭手上
        feats = self._extract(self.context, self.full_hand(), self.name)
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = net(x)[0]
        logits = logits.squeeze(0).detach().cpu().numpy().astype(np.float64)

        legal = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for t in self.cur:
            legal[int(_TILE_TO_IDX[t])] = 1.0
        masked = logits + (legal - 1.0) * 1e9

        if self.temperature and self.temperature > 1e-6:
            m = masked / self.temperature
            m = m - m.max()
            probs = np.exp(m)
            probs = probs / probs.sum()
            a = int(np.random.choice(NUM_ACTIONS, p=probs))
        else:
            a = int(np.argmax(masked))

        tile_val = int(_IDX_TO_TILE[a])
        self.cur.remove(tile_val)
        self.context.see_tile(tile_val, self.name)
        self._belief = None
        if self.verbose:
            import tile
            print('出牌:' + tile.tile_to_str(tile_val))
        return tile_val

    def _response_features(self, tile_val):
        """把 offered tile 并入完整手牌特征，让 response head 看见。"""
        hand = self.full_hand() + [tile_val]
        feats = self._extract(self.context, hand, self.name)
        return torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(self.device)

    def _response_action(self, tile_val, legal_mask):
        """返回 action index：0=pass, 1=peng, 2=gang, 3=hu。"""
        if not self._cfg.get('response_head', False):
            return None
        net = self._net_obj()
        x = self._response_features(tile_val)
        with torch.no_grad():
            out = net(x)
            response_logits = out[-1]  # 最后一个输出是 response_head
        logits = response_logits.squeeze(0).cpu().numpy().astype(np.float64)
        masked = logits + (legal_mask - 1.0) * 1e9
        if self.temperature and self.temperature > 1e-6:
            m = masked / self.temperature
            m = m - m.max()
            probs = np.exp(m)
            probs = probs / probs.sum()
            a = int(np.random.choice(4, p=probs))
        else:
            a = int(np.argmax(masked))
        return a

    def respond_hu(self, tile_val, context=None):
        legal = np.zeros(4, dtype=np.float32)
        legal[0] = 1.0
        if super().respond_hu(tile_val, context):
            legal[3] = 1.0
        a = self._response_action(tile_val, legal)
        return a == 3 if a is not None else bool(legal[3])

    def respond_peng(self, tile_val, context=None):
        legal = np.zeros(4, dtype=np.float32)
        legal[0] = 1.0
        if self._can_peng(tile_val):
            legal[1] = 1.0
        a = self._response_action(tile_val, legal)
        return a == 1 if a is not None else False

    def respond_gang(self, tile_val, context=None):
        legal = np.zeros(4, dtype=np.float32)
        legal[0] = 1.0
        if self._can_gang(tile_val):
            legal[2] = 1.0
        a = self._response_action(tile_val, legal)
        return a == 2 if a is not None else False


class HeuristicResponsePPOAgent(PPOAgent):
    """PPOAgent 的响应动作改为强启发式，用于生成 teacher 数据。

    - 胡：能胡必胡；
    - 杠：能杠必杠（杠可补牌，通常正期望）；
    - 碰：仅碰字牌或快听牌时的碰；
    - 弃牌仍走原 NN policy。
    """

    def respond_hu(self, tile_val, context=None):
        return super(PPOAgent, self).respond_hu(tile_val, context)

    def respond_gang(self, tile_val, context=None):
        return self._can_gang(tile_val)

    def respond_peng(self, tile_val, context=None):
        if not self._can_peng(tile_val):
            return False
        # 字牌/幺九优先碰
        if tile_val >= 31 or tile_val % 10 in (1, 9):
            return True
        # 向听数 ≤ 2 时倾向碰以加速听牌
        from algo.eval.v2 import shanten
        if shanten(self.full_hand()) <= 2:
            return True
        return False
