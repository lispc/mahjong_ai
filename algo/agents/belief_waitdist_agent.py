# -*- coding: utf-8 -*-
"""BeliefExpectimax + wait_dist 增强 danger。

在 BeliefExpectimaxAgent 基础上，用 wait_dist_head / wait_dist3_head 预测对手的待牌分布，
把"某张牌被任一对手等待的概率"叠加到原有 danger 信号上，增强终盘防守。

- wait_dist_head：仅下家 34 维。
- wait_dist3_head：三家 102 维（下家/对家/上家）。

用法（benchmark_pool token）：
    bewait:<label>[:<wait_model_path>]
"""
import os
import numpy as np
import torch

from algo.agents.belief_expectimax import BeliefExpectimaxAgent
import algo.eval.opponent as opponent
from algo.nn.features import extract_features, _TILE_TO_IDX, _IDX_TO_TILE


_NUM_TILES = 34


def _seat(name):
    return int(name.split('@')[-1]) if '@' in name else 0


def _next_seat(name):
    return (_seat(name) + 1) % 4


class BeliefWaitDistAgent(BeliefExpectimaxAgent):
    """BeliefExp + wait_dist danger（支持下家 34-dim 或三家 102-dim head）。"""

    def __init__(self, name, verbose=False,
                 max_candidates=8, defense_margin=0.03,
                 tenpai_min_wait=4, nn_model_path=None,
                 nn_top_k=None, device='cpu', eval_backend='legacy',
                 wait_model_path=None, wait_alpha=2.0):
        super().__init__(name, verbose=verbose,
                         max_candidates=max_candidates,
                         defense_margin=defense_margin,
                         tenpai_min_wait=tenpai_min_wait,
                         nn_model_path=nn_model_path,
                         nn_top_k=nn_top_k,
                         device=device,
                         eval_backend=eval_backend)
        if wait_model_path is None:
            wait_model_path = os.environ.get(
                'WAIT_MODEL_PATH', 'output/nn_wait_dist3_10k.pt')
        self.wait_model_path = wait_model_path
        self.wait_alpha = float(os.environ.get('WAIT_ALPHA', wait_alpha))
        self._wait_net = None
        self._wait_cfg = None

    def _load_wait_net(self):
        if self._wait_net is not None:
            return self._wait_net, self._wait_cfg
        import json
        from algo.nn.model import build_model
        cfg_path = self.wait_model_path.replace('.pt', '_config.json')
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(os.path.dirname(self.wait_model_path),
                                    'nn_model_config.json')
        with open(cfg_path) as f:
            cfg = json.load(f)
        net = build_model(cfg)
        sd = torch.load(self.wait_model_path, map_location='cpu')
        if isinstance(sd, dict) and 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        net.load_state_dict(sd, strict=False)
        net.eval()
        net.to(self.device)
        self._wait_net = net
        self._wait_cfg = cfg
        return net, cfg

    def _wait_probs(self):
        """预测三名对手的 34 维待牌概率，返回 shape (3, 34)。

        第 0/1/2 行分别对应下家、对家、上家。
        """
        net, cfg = self._load_wait_net()
        feats = extract_features(self.context, list(self.cur), self.name)
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = net(x)
            wait_logits = out[-1]
            probs = torch.sigmoid(wait_logits).cpu().numpy().astype(np.float64)
        if cfg.get('wait_dist3_head', False):
            return probs.reshape(3, _NUM_TILES)
        # 仅下家 head：补零扩展到三对手
        arr = np.zeros((3, _NUM_TILES), dtype=np.float64)
        arr[0] = probs.squeeze()
        return arr

    def _aggregate_danger(self, disc):
        """原有 danger + 三名对手 wait_dist 待牌概率的最大值。"""
        base = opponent.tile_danger(disc, self.context, self.name)
        try:
            probs = self._wait_probs()
            idx = int(_TILE_TO_IDX[disc])
            wait_danger = float(probs[:, idx].max())
        except Exception:
            wait_danger = 0.0
        return base + self.wait_alpha * wait_danger

    def next_with_trace(self):
        assert len(self.cur) == 14

        type_ctx = self._legacy_context()
        candidates = self._unique_tiles(self.cur)

        if self.nn_model_path is not None and self.nn_top_k is not None and self.nn_top_k > 0:
            top = self._nn_top_candidates(candidates, self.nn_top_k)
        else:
            scored = []
            for disc in candidates:
                hand13 = self._remove_one(self.cur, disc)
                score = __import__('algo').eval0(hand13, type_ctx)
                scored.append((score, disc))
            scored.sort(reverse=True)
            top = [disc for _, disc in scored[:self.max_candidates]]

        evaluated = []
        score_map = {}
        danger_map = {}
        for disc in top:
            hand13 = self._remove_one(self.cur, disc)
            offense = self._eval2(hand13)
            danger = self._aggregate_danger(disc)
            evaluated.append((offense, danger, disc))
            score_map[disc] = float(offense)
            danger_map[disc] = float(danger)

        best_offense = max(item[0] for item in evaluated)

        if self._danger_signal():
            margin = self.defense_margin + 0.02 * len(
                self.context.tenpai_players - {self.name})
            safe = [item for item in evaluated if item[0] >= best_offense - margin]
            safe.sort(key=lambda x: x[1])
            result = safe[0][2]
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
            import tile
            print('出牌:' + tile.tile_to_str(result))
        return result, trace

    def next(self):
        return self.next_with_trace()[0]
