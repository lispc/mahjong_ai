# -*- coding: utf-8 -*-
"""AlphaZero 风格：determinized PUCT，用于生成 search trace 数据。

简化假设：
- 只搜索当前玩家的弃牌决策；
- 对手使用 fast rollout policy（`algo.select` / eval2）；
- 忽略吃/碰/杠，只保留荣和与自摸；
- 每个候选弃牌后，模拟对手动作直到再次轮到当前玩家，形成一“步”；
- PUCT 树深度受 `max_depth` 限制，叶节点用 value net 评估；
- 运行 `n_worlds` 个 determinized 世界，每个世界跑 `n_sims` 次 simulation，
  最后聚合访问分布作为 policy target。

产物（每个决策）：
    (features, action_visit_distribution, mcts_value)

用法：
    agent = AlphaZeroMCTSAgent('P0', model_path='output/nn_full_action_best.pt',
                               n_worlds=4, n_sims=32, max_depth=3, device='cuda')
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
from algo.nn.features import extract_features, _TILE_TO_IDX, _IDX_TO_TILE


_EMPTY_CONTEXT = ctx_module.Context()
_PLAYER_ORDER = ['cur', 'opp1', 'opp2', 'opp3']
_NEXT = {'cur': 'opp1', 'opp1': 'opp2', 'opp2': 'opp3', 'opp3': 'cur'}


def _fast_select(hand14, ctx):
    """对手 fast policy：eval2 select 排名第一的牌。"""
    try:
        return algo.select(hand14, with_prob=False, metric_f=algo.eval2, c=ctx)[0]
    except Exception:
        return hand14[0]


def _extract(net, cfg, ctx, hand14, name, device):
    """提取 NN 输入，返回 (discard_logits, value)。"""
    feats = extract_features(ctx, hand14, name)
    x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = net(x)
    # 支持多输出 tuple
    if isinstance(out, tuple):
        discard_logits = out[0].squeeze(0)
        value = out[1]
        if value.dim() > 1:
            value = value.squeeze(-1)
    else:
        discard_logits = out.squeeze(0)
        value = torch.zeros(1, device=device)
    return discard_logits.detach().cpu().numpy(), float(value.detach().cpu().item())


def _legal_mask(hand):
    legal = np.zeros(34, dtype=np.float32)
    for t in hand:
        legal[int(_TILE_TO_IDX[t])] = 1.0
    return legal


def _sample_world(context, my_hand):
    """从当前信念采样对手手牌和剩余牌山。"""
    all_tiles = tile.all_tiles_as_dict()
    unknown = dict_sub(dict_sub(all_tiles, context.used), count(my_hand))
    unknown_list = []
    for t, c in unknown.items():
        unknown_list.extend([t] * c)
    random.shuffle(unknown_list)
    opp1 = unknown_list[:13]
    opp2 = unknown_list[13:26]
    opp3 = unknown_list[26:39]
    wall = unknown_list[39:]
    return {'opp1': opp1, 'opp2': opp2, 'opp3': opp3}, wall


def _make_context(ctx_dict):
    ctx = context_v3.ContextV3()
    ctx.used = ctx_dict['used'].copy()
    ctx.all_seen = ctx_dict['all_seen'].copy()
    ctx.discards = {p: list(seq) for p, seq in ctx_dict['discards'].items()}
    ctx.tenpai_players = set(ctx_dict['tenpai_players'])
    return ctx


def _context_to_dict(ctx):
    return {
        'used': ctx.used.copy(),
        'all_seen': ctx.all_seen.copy(),
        'discards': {p: list(seq) for p, seq in ctx.discards.items()},
        'tenpai_players': list(ctx.tenpai_players),
    }


def _check_ron(discarded, hands, current):
    """返回 (winning_player_or_None, current_deals_in)。"""
    for pid in _PLAYER_ORDER:
        if pid == current:
            continue
        if eval_v2.is_win(hands[pid] + [discarded]):
            return pid, (pid != 'cur')
    return None, False


def _transition(state, discard_tile, current='cur', max_steps=20):
    """当前玩家弃牌后，模拟对手直到再次轮到 current，或终局，或步数上限。

    若达到步数上限仍未到 current，返回 (None, 0.0) 视为流局价值。
    """
    hands = {k: list(v) for k, v in state['hands'].items()}
    wall = list(state['wall'])
    ctx = _make_context(state['ctx_dict'])
    locked = set(state['locked'])

    hands[current].remove(discard_tile)
    ctx.see_tile(discard_tile, 'cur' if current == 'cur' else current)

    winner, _ = _check_ron(discard_tile, hands, current)
    if winner is not None:
        return None, 1.0 if winner == 'cur' else -1.0

    turn = _NEXT[current]
    steps = 0
    while wall and steps < max_steps:
        drawn = wall.pop(0)
        hands[turn].append(drawn)

        if eval_v2.is_win(hands[turn]):
            return None, 1.0 if turn == 'cur' else -1.0

        if turn not in locked and len(hands[turn]) == 13 and eval_v2.shanten(hands[turn]) == 0:
            rem = ctx.remaining_wall(hands[turn])
            waits = eval_v2.winning_tiles(hands[turn], rem)
            if sum(rem.get(t, 0) for t in waits) >= 3:
                locked.add(turn)
                ctx.declare_tenpai(turn)

        if turn in locked:
            discarded = drawn
        else:
            discarded = _fast_select(hands[turn], ctx)
        hands[turn].remove(discarded)
        ctx.see_tile(discarded, turn)

        winner, _ = _check_ron(discarded, hands, turn)
        if winner is not None:
            return None, 1.0 if winner == 'cur' else -1.0

        turn = _NEXT[turn]
        steps += 1
        if turn == current:
            if not wall:
                return None, 0.0
            drawn = wall.pop(0)
            hands[current].append(drawn)
            if eval_v2.is_win(hands[current]):
                return None, 1.0
            return {
                'hands': hands,
                'wall': wall,
                'ctx_dict': _context_to_dict(ctx),
                'locked': locked,
            }, None

    return None, 0.0


class _Node:
    __slots__ = ('parent', 'action', 'children', 'n', 'q', 'p',
                 'is_terminal', 'reward', 'state', 'depth')

    def __init__(self, parent=None, action=None, prior=0.0,
                 state=None, depth=0):
        self.parent = parent
        self.action = action
        self.children = {}  # action -> _Node
        self.n = 0
        self.q = 0.0
        self.p = prior
        self.is_terminal = False
        self.reward = None  # terminal reward for 'cur'
        self.state = state
        self.depth = depth


def _puct_score(child, c_puct, parent_sqrt_n):
    if child.n == 0:
        q = 0.0
    else:
        q = child.q / child.n
    return q + c_puct * child.p * parent_sqrt_n / (1 + child.n)


class AlphaZeroMCTSAgent(agent.Agent):
    def __init__(self, name, model_path='output/nn_full_action_best.pt',
                 n_worlds=4, n_sims=32, max_depth=3, c_puct=2.0,
                 transition_max_steps=20,
                 device='cpu', temperature=1.0, verbose=False):
        super().__init__(name, verbose)
        self.model_path = model_path
        self.device = device
        self.n_worlds = n_worlds
        self.n_sims = n_sims
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.transition_max_steps = transition_max_steps
        self.temperature = temperature
        self.context = context_v3.ContextV3()
        self._net = None
        self._cfg = None

    def _net_obj(self):
        if self._net is None:
            import json
            from algo.nn.model import build_model
            cfg_path = self.model_path.replace('.pt', '_config.json')
            if not os.path.exists(cfg_path):
                cfg_path = os.path.join(os.path.dirname(self.model_path), 'nn_model_config.json')
            with open(cfg_path) as f:
                self._cfg = json.load(f)
            self._net = build_model(self._cfg)
            sd = torch.load(self.model_path, map_location='cpu', weights_only=False)
            state = sd.get('model_state', sd)
            self._net.load_state_dict(state, strict=False)
            self._net.eval()
            self._net.to(self.device)
        return self._net

    def init_tiles(self, l):
        super().init_tiles(l)
        self.context = context_v3.ContextV3()
        self._traces = []

    def handle_msg(self, msg):
        if msg.type == 'put':
            self.context.see_tile(msg.data, msg.sender)
        elif msg.type == 'tenpai':
            self.context.declare_tenpai(msg.sender)
        return super().handle_msg(msg)

    def declare_tenpai(self, hand, context):
        # 复用 Hybrid 启发式：剩余待牌 ≥4 或已现待牌
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
        return [t for t in hand if not (t in seen or seen.add(t))]

    def _root_state(self, opp_hands, wall):
        return {
            'hands': {'cur': list(self.cur), 'opp1': list(opp_hands[0]),
                      'opp2': list(opp_hands[1]), 'opp3': list(opp_hands[2])},
            'wall': list(wall),
            'ctx_dict': _context_to_dict(self.context),
            'locked': set(),
        }

    def _prior_over_candidates(self, hand):
        net = self._net_obj()
        logits, _ = _extract(net, self._cfg, self.context, hand, self.name, self.device)
        legal = _legal_mask(hand)
        masked = logits + (legal - 1.0) * 1e9
        probs = np.exp(masked - masked.max())
        probs /= probs.sum()
        return probs

    def _eval_leaf(self, state):
        net = self._net_obj()
        ctx = _make_context(state['ctx_dict'])
        _, value = _extract(net, self._cfg, ctx, state['hands']['cur'], self.name, self.device)
        return value

    def _run_puct_in_world(self, root_state, candidates, prior_probs):
        root = _Node(state=root_state, depth=0)
        # 预先展开 root children
        for a, p in zip(candidates, prior_probs):
            root.children[a] = _Node(parent=root, action=a, prior=p)

        for _ in range(self.n_sims):
            node = root
            path = [node]
            # selection
            while node.children and not node.is_terminal:
                parent_sqrt_n = np.sqrt(node.n + 1e-8)
                best_a = None
                best_score = -float('inf')
                for a, child in node.children.items():
                    s = _puct_score(child, self.c_puct, parent_sqrt_n)
                    if s > best_score:
                        best_score = s
                        best_a = a
                node = node.children[best_a]
                path.append(node)

            # 若 node 是刚被创建的 child，先 transition 拿到状态
            if node.state is None and node.parent is not None and not node.is_terminal:
                parent_state = node.parent.state
                next_state, reward = _transition(parent_state, node.action,
                                                  max_steps=self.transition_max_steps)
                if reward is not None:
                    node.is_terminal = True
                    node.reward = reward
                else:
                    node.state = next_state

            # expansion / evaluation
            if node.is_terminal:
                value = node.reward
            elif node.depth >= self.max_depth:
                value = self._eval_leaf(node.state)
            elif not node.children:
                # expand current-player node
                legal_mask = _legal_mask(node.state['hands']['cur'])
                logits, _ = _extract(self._net_obj(), self._cfg,
                                     _make_context(node.state['ctx_dict']),
                                     node.state['hands']['cur'], self.name, self.device)
                masked = logits + (legal_mask - 1.0) * 1e9
                probs = np.exp(masked - masked.max())
                probs /= probs.sum()
                legal_tiles = np.where(legal_mask > 0)[0]
                for idx in legal_tiles:
                    tile_val = int(_IDX_TO_TILE[int(idx)])
                    node.children[tile_val] = _Node(parent=node, action=tile_val,
                                                    prior=float(probs[idx]),
                                                    depth=node.depth + 1)
                # 选一个 child 继续评估（第一次访问时）
                if node.children:
                    node = random.choice(list(node.children.values()))
                    path.append(node)
                    # 新 child 需要 transition
                    parent_state = node.parent.state
                    next_state, reward = _transition(parent_state, node.action,
                                                      max_steps=self.transition_max_steps)
                    if reward is not None:
                        node.is_terminal = True
                        node.reward = reward
                        value = reward
                    else:
                        node.state = next_state
                        value = self._eval_leaf(node.state)
                else:
                    value = self._eval_leaf(node.state)
            else:
                value = self._eval_leaf(node.state)

            # backup
            for n in path:
                n.n += 1
                n.q += value

        # 收集 root 访问分布
        visits = np.zeros(34, dtype=np.float64)
        for a, child in root.children.items():
            visits[int(_TILE_TO_IDX[a])] = child.n
        total = visits.sum()
        if total > 0:
            visits /= total
        # mcts_value：按访问分布加权 Q
        mcts_value = 0.0
        for a, child in root.children.items():
            q = child.q / max(child.n, 1)
            mcts_value += (child.n / max(total, 1)) * q
        return visits, mcts_value

    def next(self):
        assert len(self.cur) == 14
        candidates = self._unique_tiles(self.cur)
        prior_probs_full = self._prior_over_candidates(self.cur)
        prior_probs = np.array([prior_probs_full[int(_TILE_TO_IDX[t])] for t in candidates])
        if prior_probs.sum() > 0:
            prior_probs /= prior_probs.sum()

        agg_visits = np.zeros(34, dtype=np.float64)
        agg_value = 0.0
        for _ in range(self.n_worlds):
            opp, wall = _sample_world(self.context, self.cur)
            root_state = self._root_state([opp['opp1'], opp['opp2'], opp['opp3']], wall)
            visits, value = self._run_puct_in_world(root_state, candidates, prior_probs)
            agg_visits += visits
            agg_value += value
        agg_visits /= self.n_worlds
        agg_value /= self.n_worlds

        # 记录 trace（由 data collector 在外部读取）
        self._last_trace = {
            'features': extract_features(self.context, self.cur, self.name),
            'visit_dist': agg_visits.astype(np.float32),
            'value': float(agg_value),
        }
        self._traces.append(self._last_trace)

        if self.temperature and self.temperature > 1e-6:
            logits = np.log(np.maximum(agg_visits, 1e-10)) / self.temperature
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            a = int(np.random.choice(34, p=probs))
        else:
            a = int(np.argmax(agg_visits))

        tile_val = int(_IDX_TO_TILE[a])
        self.cur.remove(tile_val)
        self.context.see_tile(tile_val, self.name)
        if self.verbose:
            print('出牌:' + tile.tile_to_str(tile_val))
        return tile_val

    def last_trace(self):
        return getattr(self, '_last_trace', None)

    def all_traces(self):
        return getattr(self, '_traces', [])
