# -*- coding: utf-8 -*-
"""BeliefExpectimax + 终盘精确求解。

在 BeliefExpectimaxAgent 基础上，当检测到危险信号且牌山剩余较少时，
用 wait_dist3_head 预测三家对手的待牌分布，再用 exact endgame solver 计算
每张候选弃牌的精确点炮期望，选择最安全的弃牌。

用法（benchmark_pool token）：
    beend:<label>[:model_path]

默认 model_path = output/nn_wait_dist3_10k.pt。
"""
import os
import numpy as np
import torch

from algo.agents.belief_expectimax import BeliefExpectimaxAgent
from algo.eval.endgame_solver import exact_tenpai_ev
from algo.eval.v2 import winning_tiles
from algo.nn.features import extract_features, _TILE_TO_IDX, _IDX_TO_TILE


_NUM_TILES = 34


def _seat(name):
    # 兼容 duplicate 命名：name@0_a / name@1_b
    part = name.split('@')[-1]
    part = part.split('_')[0]
    return int(part) if part.isdigit() else 0


def _next_seat(name):
    return (_seat(name) + 1) % 4


class BeliefEndgameAgent(BeliefExpectimaxAgent):
    """BeliefExp + 终盘精确求解（三家待牌分布版）。"""

    def __init__(self, name, verbose=False,
                 max_candidates=8, defense_margin=0.03,
                 tenpai_min_wait=4, nn_model_path=None,
                 nn_top_k=None, device='cpu', eval_backend='legacy',
                 wait_model_path=None, wall_threshold=20,
                 wait_prob_threshold=0.5):
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
        self.wall_threshold = int(os.environ.get('WALL_THRESHOLD', wall_threshold))
        self.wait_prob_threshold = float(os.environ.get('WAIT_PROB_THRESHOLD', wait_prob_threshold))
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
        if not (cfg.get('wait_dist3_head', False) or cfg.get('wait_dist_head', False)):
            return np.zeros((3, _NUM_TILES), dtype=np.float64)
        feats = extract_features(self.context, list(self.cur), self.name)
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = net(x)
            wait_logits = out[-1]
            probs = torch.sigmoid(wait_logits).cpu().numpy().astype(np.float64)
        if cfg.get('wait_dist3_head', False):
            return probs.reshape(3, _NUM_TILES)
        # 仅下家 head：补零扩展到三家
        arr = np.zeros((3, _NUM_TILES), dtype=np.float64)
        arr[0] = probs.squeeze()
        return arr

    def _wall_remaining(self):
        """返回剩余牌山列表（基于 context.used 反推）。"""
        from tile import all_tiles_as_dict
        wall = all_tiles_as_dict()
        used = getattr(self.context, 'used', {})
        for t, c in used.items():
            wall[t] -= c
        for t in self.cur:
            wall[t] -= 1
        for _, t in self.melds:
            wall[t] -= 1
        return [t for t, c in wall.items() for _ in range(c)]

    def _predicted_waits_for_opponents(self):
        """为每名有待牌预测的对手返回 (seat_rel, waits_set)。

        优先使用已报听玩家；若某玩家未报听但其 wait_dist 概率最大值超过阈值，
        也纳入防守计算。
        seat_rel: 1=下家, 2=对家, 3=上家。
        """
        probs = self._wait_probs()
        self_seat = _seat(self.name)
        out = []
        declared = self.context.tenpai_players - {self.name}
        for player in self.context.discards:
            if player == self.name:
                continue
            opp_seat = _seat(player)
            rel = (opp_seat - self_seat) % 4
            if rel == 0:
                continue
            row = rel - 1
            waits = set()
            for idx in range(_NUM_TILES):
                if probs[row, idx] > self.wait_prob_threshold:
                    waits.add(int(_IDX_TO_TILE[idx]))
            if player in declared or (probs[row].max() > self.wait_prob_threshold):
                if waits:
                    out.append((rel, waits))
        return out

    def _any_endgame_threat(self):
        """是否存在需要 endgame 精确求解的威胁。"""
        return bool(self._predicted_waits_for_opponents())

    def _endgame_safety_score(self, disc):
        """返回弃牌 disc 的精确 endgame EV（三家叠加，越高越安全）。"""
        opp_waits = self._predicted_waits_for_opponents()
        if not opp_waits:
            return 0.0
        wall = self._wall_remaining()
        total_ev = 0.0
        for rel, waits in opp_waits:
            offset = rel - 1  # 下家 0, 对家 1, 上家 2
            total_ev += exact_tenpai_ev(disc, waits, wall, tenpai_offset=offset,
                                        deal_in_reward=-1.0, draw_reward=0.0)
        return total_ev

    def next_with_trace(self):
        # 在牌山剩余较少且存在待牌威胁时启用精确求解。
        # wall_remaining 包含对手暗牌，用 len - 39（3 对手 × 13 张）估计真实牌山。
        wall_remaining = self._wall_remaining()
        effective_wall = max(0, len(wall_remaining) - 39)
        if (effective_wall <= self.wall_threshold and self._any_endgame_threat()):
            try:
                candidates = self._unique_tiles(self.cur)
                # 计算 offense（eval2）和 defense（exact EV）
                evaluated = []
                for disc in candidates:
                    hand13 = self._remove_one(self.cur, disc)
                    offense = self._eval2(hand13)
                    safety = self._endgame_safety_score(disc)
                    evaluated.append((offense, safety, disc))
                # 在 offense 不太差的前提下选最安全
                best_offense = max(item[0] for item in evaluated)
                margin = self.defense_margin + 0.02 * len(
                    self.context.tenpai_players - {self.name})
                safe = [item for item in evaluated
                        if item[0] >= best_offense - margin]
                # 按 safety 降序，tie-break 按 offense
                safe.sort(key=lambda x: (x[1], x[0]), reverse=True)
                result = safe[0][2]
                self.cur.remove(result)
                self.context.see_tile(result, self.name)
                if self.verbose:
                    import tile
                    print('出牌:' + tile.tile_to_str(result))
                return result, {
                    'candidates': candidates,
                    'scores': {d: float(o) for o, s, d in evaluated},
                    'safeties': {d: float(s) for o, s, d in evaluated},
                    'selected_value': float(safe[0][0]),
                }
            except Exception:
                pass
        return super().next_with_trace()

    def next(self):
        return self.next_with_trace()[0]
