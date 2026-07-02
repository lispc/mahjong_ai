# -*- coding: utf-8 -*-
"""MCTS/PUCT with conv-BC prior + value。

第一版实现：Determinized Flat Monte Carlo with conv-BC。
- 候选生成：conv-BC policy top-k；
- 世界采样：根据当前信念采样对手手牌 + 牌山；
- 每个 (world, candidate) 跑一条快速 rollout（默认用 eval0，可切换为 conv-BC policy）；
- rollout 固定深度后，用 conv-BC value head 评估当前玩家手牌价值；
- 选平均价值最高的候选。

后续可升级为树搜索（PUCT）。
"""

import os
import random
import numpy as np
import torch
import copy

import agent
import tile
import algo
import context as ctx_module
import algo.context.v3 as context_v3
import algo.eval.v2 as eval_v2
from utils import dict_sub, count
from algo.nn.features import extract_features, extract_features_ext


_EMPTY_CONTEXT = ctx_module.Context()


def _fast_rollout_select(hand14):
    """快速 rollout policy：最大化 eval0(hand13, empty context)。"""
    best_disc = None
    best_score = -float('inf')
    seen = set()
    for disc in hand14:
        if disc in seen:
            continue
        seen.add(disc)
        hand13 = list(hand14)
        hand13.remove(disc)
        score = algo.eval0(hand13, _EMPTY_CONTEXT)
        if score > best_score:
            best_score = score
            best_disc = disc
    return best_disc


def _conv_bc_select(hand14, ctx, name, net, extract, device):
    """用 conv-BC policy 选一个弃牌。"""
    feats = extract(ctx, hand14, name)
    x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = net(x)
    logits = logits.squeeze(0).detach().cpu().numpy()
    from algo.nn.features import _TILE_TO_IDX
    legal = np.zeros(34, dtype=np.float32)
    for t in hand14:
        legal[int(_TILE_TO_IDX[t])] = 1.0
    masked = logits + (legal - 1.0) * 1e9
    a = int(np.argmax(masked))
    from algo.nn.features import _IDX_TO_TILE
    return int(_IDX_TO_TILE[a])


def _simulate_one_conv_value(args):
    """在一个采样世界里评估某个候选弃牌：fast rollout + conv-BC value 截断。"""
    (candidate, current_hand, opp_hands, wall,
     public_ctx_dict, locked_indices, cutoff_depth,
     net_state_dict, cfg, device, use_conv_rollout) = args

    from algo.nn.model import build_model
    net = build_model(cfg)
    net.load_state_dict(net_state_dict)
    net.eval()
    net.to(device)
    extract_fn = extract_features_ext if cfg.get('features') == 'ext' else extract_features

    ctx = context_v3.ContextV3()
    ctx.used = public_ctx_dict['used'].copy()
    ctx.all_seen = public_ctx_dict['all_seen'].copy()
    ctx.discards = {p: list(seq) for p, seq in public_ctx_dict['discards'].items()}
    ctx.tenpai_players = set(public_ctx_dict['tenpai_players'])

    cur_hand = list(current_hand)
    cur_hand.remove(candidate)
    hands = [cur_hand, list(opp_hands[0]), list(opp_hands[1]), list(opp_hands[2])]
    player_names = ['cur@0', 'opp1@1', 'opp2@2', 'opp3@3']
    wall = list(wall)
    turn = 1
    current_idx = 0
    locked = set(locked_indices)

    step = 0
    max_steps = cutoff_depth if cutoff_depth > 0 else 10000

    while wall and step < max_steps:
        drawn = wall.pop(0)
        hands[turn].append(drawn)

        if eval_v2.is_win(hands[turn]):
            return 1.0 if turn == current_idx else -0.3

        if turn in locked:
            discarded = drawn
            hands[turn].remove(discarded)
        else:
            if use_conv_rollout:
                discarded = _conv_bc_select(hands[turn], ctx, player_names[turn], net, extract_fn, device)
            else:
                discarded = _fast_rollout_select(hands[turn])
            hands[turn].remove(discarded)

        for j in range(4):
            if j == turn:
                continue
            if eval_v2.is_win(hands[j] + [discarded]):
                if j == current_idx:
                    return 1.0
                if turn == current_idx:
                    return -1.0
                return -0.3

        ctx.see_tile(discarded, player_names[turn])

        if (turn not in locked and len(hands[turn]) == 13 and
                eval_v2.shanten(hands[turn]) == 0):
            rem = ctx.remaining_wall(hands[turn])
            waits = eval_v2.winning_tiles(hands[turn], rem)
            if sum(rem.get(t, 0) for t in waits) >= 3:
                locked.add(turn)
                ctx.declare_tenpai(player_names[turn])

        turn = (turn + 1) % 4
        step += 1

    # 截断：用 conv-BC value head 评估当前玩家手牌
    # 当前玩家手牌应为 13 张
    hand_for_value = hands[current_idx]
    # 需要构造一个 feature 输入：用当前上下文 + 当前玩家手牌（14 张，补一张摸牌位？）
    # 但 value head 是对 14 张手牌设计的。这里用 13 张 + 一张虚拟 0（最近摸牌）
    # 更简单：直接返回 eval0 启发式 + value 不可用时的 fallback
    try:
        # 把手牌补到 14 张：复制最后一张（不影响太多）
        if len(hand_for_value) == 13:
            hand14v = hand_for_value + [hand_for_value[-1]]
        else:
            hand14v = hand_for_value
        feats = extract_fn(ctx, hand14v, player_names[current_idx])
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            _, value = net(x)
        v = float(value.detach().cpu().reshape(-1)[0])
        return v
    except Exception:
        return 0.0


class MCTSConvAgent(agent.Agent):
    """conv-BC prior + value 的 Determinized Flat MC agent。"""

    def __init__(self, name, model_path='output/nn_conv_bc.pt', device='cpu',
                 n_worlds=8, top_k=5, cutoff_depth=15,
                 use_conv_rollout=False, verbose=False):
        super().__init__(name, verbose)
        self.model_path = model_path
        self.device = device
        self.n_worlds = n_worlds
        self.top_k = top_k
        self.cutoff_depth = cutoff_depth
        self.use_conv_rollout = use_conv_rollout
        self.context = context_v3.ContextV3()
        self._net = None
        self._cfg = None
        self._extract = extract_features

    def _net_obj(self):
        if self._net is None:
            import json
            from algo.nn.model import build_model
            cfg_path = self.model_path.replace('.pt', '_config.json')
            if not os.path.exists(cfg_path):
                cfg_path = os.path.join(os.path.dirname(self.model_path), 'nn_model_config.json')
            if os.path.exists(cfg_path):
                self._cfg = json.load(open(cfg_path))
            else:
                self._cfg = {'arch': 'mlp', 'input_dim': 175, 'hidden_dim': 256}
            self._net = build_model(self._cfg)
            sd = torch.load(self.model_path, map_location='cpu')
            if isinstance(sd, dict) and 'model_state_dict' in sd:
                sd = sd['model_state_dict']
            self._net.load_state_dict(sd)
            self._net.eval()
            self._net.to(self.device)
            self._extract = extract_features_ext if self._cfg.get('features') == 'ext' else extract_features
        return self._net

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
        rem = context.remaining_wall(hand)
        waits = eval_v2.winning_tiles(hand, rem)
        if not waits:
            return False
        total_wait = sum(rem.get(t, 0) for t in waits)
        if total_wait >= 4:
            return True
        for t in waits:
            if context.all_seen.get(t, 0) > 0 and rem.get(t, 0) > 0:
                return True
        return False

    def _unique_tiles(self, hand):
        seen = set()
        out = []
        for t in hand:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _sample_world(self):
        all_tiles = tile.all_tiles_as_dict()
        unknown = dict_sub(dict_sub(all_tiles, self.context.used), count(self.cur))
        unknown_list = []
        for t, c in unknown.items():
            unknown_list.extend([t] * c)
        random.shuffle(unknown_list)
        opp1 = unknown_list[:13]
        opp2 = unknown_list[13:26]
        opp3 = unknown_list[26:39]
        wall = unknown_list[39:]
        return [opp1, opp2, opp3], wall

    def _public_context_dict(self):
        return {
            'used': self.context.used,
            'all_seen': self.context.all_seen,
            'discards': self.context.discards,
            'tenpai_players': list(self.context.tenpai_players),
        }

    def _top_k_candidates(self, net, extract):
        """用 conv-BC policy 选 top-k 合法候选。"""
        from algo.nn.features import _TILE_TO_IDX, _IDX_TO_TILE
        feats = extract(self.context, self.cur, self.name)
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = net(x)
        logits = logits.squeeze(0).detach().cpu().numpy()
        legal = np.zeros(34, dtype=np.float32)
        for t in self.cur:
            legal[int(_TILE_TO_IDX[t])] = 1.0
        masked = logits + (legal - 1.0) * 1e9
        topk_idx = np.argpartition(masked, -self.top_k)[-self.top_k:]
        topk_idx = topk_idx[np.argsort(-masked[topk_idx])]
        return [int(_IDX_TO_TILE[int(a)]) for a in topk_idx]

    def next(self):
        assert len(self.cur) == 14
        net = self._net_obj()
        extract = self._extract

        candidates = self._top_k_candidates(net, extract)
        public_ctx = self._public_context_dict()
        state_dict = net.state_dict()

        rewards = []
        for _ in range(self.n_worlds):
            opp_hands, wall = self._sample_world()
            for candidate in candidates:
                reward = _simulate_one_conv_value(
                    (candidate, list(self.cur), opp_hands, wall,
                     public_ctx, [], self.cutoff_depth,
                     state_dict, self._cfg, self.device, self.use_conv_rollout))
                rewards.append((candidate, reward))

        sums = {d: 0.0 for d in candidates}
        counts = {d: 0 for d in candidates}
        for d, r in rewards:
            sums[d] += r
            counts[d] += 1

        best_disc = candidates[0]
        best_value = -float('inf')
        for d in candidates:
            avg = sums[d] / max(counts[d], 1)
            if avg > best_value:
                best_value = avg
                best_disc = d

        self.cur.remove(best_disc)
        self.context.see_tile(best_disc, self.name)
        if self.verbose:
            print('出牌:' + tile.tile_to_str(best_disc))
        return best_disc
