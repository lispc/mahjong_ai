# -*- coding: utf-8 -*-
"""方案 B：信念状态 + Expectimax Agent。

核心思想：
- 维护全局 tile-level 信念（context_v3 中的已见牌、每家弃牌序列、报听信息）。
- 使用项目已有的 2-ply expectimax 评估器 `algo.eval2` 作为叶子估值，
  但把概率分布替换为 ContextV3 维护的真实剩余分布。
- 弃牌时先用 `algo.eval0` 快速预选，再用 `algo.eval2` 精确评估；
  只在检测到对手听牌信号时才用危险度做安全 tie-breaking，避免过度防守。

这是一个"把不完全信息放进概率分布，再在前向搜索里取期望"的干净实现。
"""

import os
import agent
import tile
import algo
import context as ctx_module
import algo.eval.opponent as opponent
import algo.context.v3 as context_v3
import algo.eval.v2 as eval_v2


_NN_CACHE = {}


def _load_policy_net(model_path, device='cpu'):
    key = (os.path.abspath(model_path), device)
    if key in _NN_CACHE:
        return _NN_CACHE[key]
    import torch
    import json
    from algo.nn.model import build_model
    cfg_path = model_path.replace('.pt', '_config.json')
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(model_path), 'nn_model_config.json')
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
    else:
        cfg = {'arch': 'conv', 'input_dim': 175, 'channels': 96, 'n_blocks': 4,
               'hidden_dim': 256, 'n_tile_ch': 5, 'dealin_head': True}
    net = build_model(cfg)
    sd = torch.load(model_path, map_location=device)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    net.load_state_dict(sd, strict=False)
    net.eval()
    net.to(device)
    _NN_CACHE[key] = net
    return net


class BeliefExpectimaxAgent(agent.Agent):
    """
    信念 Expectimax Agent（方案 B）。

    参数：
        max_candidates: eval0 预选后进入 eval2 精确评估的候选数。
        defense_margin: safe_mode 下，允许为安全让步的进攻分数比例。
        tenpai_min_wait: 报听所需的最小待牌剩余张数。
        eval_backend: 'legacy' 使用 algo.eval2；'fast2' 使用 algo.eval.fast_eval2。
    """

    def __init__(self, name, verbose=False,
                 max_candidates=8,
                 defense_margin=0.03,
                 tenpai_min_wait=4,
                 nn_model_path=None,
                 nn_top_k=None,
                 device='cpu',
                 eval_backend='legacy'):
        super().__init__(name, verbose)
        self.max_candidates = max_candidates
        self.defense_margin = defense_margin
        self.tenpai_min_wait = tenpai_min_wait
        self.context = context_v3.ContextV3()
        self.nn_model_path = nn_model_path
        self.nn_top_k = nn_top_k
        self.device = device
        self._nn_model = None
        self._nn_extract = None
        self.eval_backend = eval_backend
        self._fast_eval = None

    def init_tiles(self, l):
        super().init_tiles(l)
        self.context = context_v3.ContextV3()
        self._fast_eval = None

    def handle_msg(self, msg):
        if msg.type == 'put':
            self.context.see_tile(msg.data, msg.sender)
        elif msg.type == 'tenpai':
            self.context.declare_tenpai(msg.sender)
        return super().handle_msg(msg)

    def _legacy_context(self):
        """把 ContextV3 的信念映射成 legacy context.Context（algo.eval2 所需）。"""
        c = ctx_module.Context()
        c.used = self.context.used.copy()
        return c

    def _get_fast_eval(self):
        if self._fast_eval is None:
            from algo.eval.fast_eval2 import FastEval2
            self._fast_eval = FastEval2(self.context)
        return self._fast_eval

    def _eval2(self, hand13):
        if self.eval_backend == 'fast2':
            fe = self._get_fast_eval()
            return fe.eval2(hand13)
        return algo.eval2(hand13, self._legacy_context())

    def declare_tenpai(self, hand, context):
        """听牌且待牌足够好时才报听。"""
        if context is None:
            return False
        # 避免过早锁死
        if sum(len(v) for v in context.discards.values()) < 12:
            return False
        if eval_v2.shanten(hand) != 0:
            return False
        remaining = context.remaining_wall(hand)
        waits = eval_v2.winning_tiles(hand, remaining)
        if not waits:
            return False
        total_wait = sum(remaining.get(t, 0) for t in waits)
        if total_wait >= self.tenpai_min_wait:
            return True
        # 有现物待牌也可报听
        for t in waits:
            if context.all_seen.get(t, 0) > 0 and remaining.get(t, 0) > 0:
                return True
        return False

    def _nn_top_candidates(self, candidates, k):
        """用 NN policy 从 unique tiles 里选 top-k 进入 eval2。"""
        if self._nn_model is None:
            self._nn_model = _load_policy_net(self.nn_model_path, self.device)
            from algo.nn.features import extract_features
            self._nn_extract = extract_features
        import torch
        import numpy as np
        from algo.nn.features import _TILE_TO_IDX
        x = self._nn_extract(self.context, self.cur, self.name)
        with torch.no_grad():
            xt = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            logits = self._nn_model(xt)[0].squeeze(0).cpu().numpy()
        tile_scores = [(float(logits[int(_TILE_TO_IDX[t])]), t) for t in candidates]
        tile_scores.sort(reverse=True, key=lambda x: x[0])
        return [t for _, t in tile_scores[:k]]

    def _unique_tiles(self, hand):
        seen = set()
        out = []
        for t in hand:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _remove_one(self, hand, tile_value):
        hand = list(hand)
        hand.remove(tile_value)
        return hand

    def _danger_signal(self):
        """是否需要进入安全模式：有对手报听或某对手危险等级 >= 1。"""
        if self.context.tenpai_players - {self.name}:
            return True
        for player, discards in self.context.discards.items():
            if player == self.name:
                continue
            if opponent.player_danger_level(discards) >= 1:
                return True
        return False

    def next_with_trace(self):
        """返回 (chosen_tile, trace)。

        trace = {
            'candidates': list of tile values in top_k,
            'scores': dict tile_value -> offense score (eval2 value),
            'dangers': dict tile_value -> danger,
            'selected_value': offense score of chosen tile,
        }
        """
        assert len(self.cur) == 14

        type_ctx = self._legacy_context()
        candidates = self._unique_tiles(self.cur)

        if self.nn_model_path is not None and self.nn_top_k is not None and self.nn_top_k > 0:
            top = self._nn_top_candidates(candidates, self.nn_top_k)
        else:
            scored = []
            for disc in candidates:
                hand13 = self._remove_one(self.cur, disc)
                score = algo.eval0(hand13, type_ctx)
                scored.append((score, disc))
            scored.sort(reverse=True)
            top = [disc for _, disc in scored[:self.max_candidates]]

        evaluated = []
        score_map = {}
        danger_map = {}
        for disc in top:
            hand13 = self._remove_one(self.cur, disc)
            offense = self._eval2(hand13)
            danger = opponent.tile_danger(disc, self.context, self.name)
            evaluated.append((offense, danger, disc))
            score_map[disc] = float(offense)
            danger_map[disc] = float(danger)

        best_offense = max(item[0] for item in evaluated)

        if self._danger_signal():
            margin = self.defense_margin + 0.02 * len(
                self.context.tenpai_players - {self.name})
            safe_candidates = [
                item for item in evaluated if item[0] >= best_offense - margin
            ]
            safe_candidates.sort(key=lambda x: x[1])
            result = safe_candidates[0][2]
        else:
            evaluated.sort(reverse=True, key=lambda x: x[0])
            result = evaluated[0][2]

        trace = {
            'candidates': list(top),
            'scores': score_map,
            'dangers': danger_map,
            'selected_value': float(score_map.get(result, best_offense)),
        }

        self.cur.remove(result)
        self.context.see_tile(result, self.name)
        if self.verbose:
            print('出牌:' + tile.tile_to_str(result))
        return result, trace

    def next(self):
        return self.next_with_trace()[0]
