# -*- coding: utf-8 -*-
"""用快速 rollout 估计当前局面的胜率，作为价值网络的监督信号。

每次 rollout 都会把未知牌随机分配给对手并补成牌山，然后让所有玩家按一个贪婪
rollout policy 打完该局，最后返回当前玩家的平均收益（+1/0/-1）。
"""

import os
import sys
import random
import copy

import algo
import tile
from utils import dict_sub, count
import algo.eval.v2 as eval_v2
import context as ctx_module


_EMPTY_CONTEXT = ctx_module.Context()

# 默认使用 legacy eval2 作为 rollout policy；可通过环境变量切换 fast rollout
_USE_FAST_ROLLOUT = os.environ.get('MJ_FAST_ROLLOUT', '0') == '1'

# rollout policy 选择：
#   baseline（默认，纯 CPU，PyPy 可用）
#   nnpolicy（只取 Policy-Value Net 的 policy head top-1，需 torch，CPython  only）
#   v3nnpc（完整 BeliefExpectimaxV3Agent，需 torch，最慢）
_ROLLOUT_POLICY = os.environ.get('MJ_ROLLOUT_POLICY', 'baseline')

# PyPy 不兼容 Numba，因此禁用 fast_eval import；CPython 正常可用
if _USE_FAST_ROLLOUT and not hasattr(sys, 'pypy_version_info'):
    import algo.eval.fast_eval as fast_eval
else:
    fast_eval = None


# V3-NN-PC rollout 相关对象，延迟初始化（避免 PyPy import torch）
_V3NNPC_AGENTS = None


def _get_v3nnpc_agents():
    """懒加载 4 个 V3-NN-PC agent 实例（共享全局 NN 模型）。"""
    global _V3NNPC_AGENTS
    if _V3NNPC_AGENTS is None:
        from algo.agents.belief_expectimax_v3 import BeliefExpectimaxV3Agent
        _V3NNPC_AGENTS = []
        for seat in range(4):
            _V3NNPC_AGENTS.append(
                BeliefExpectimaxV3Agent(
                    f'V3NN@{seat}',
                    expectimax_depth=1,
                    max_candidates=5,
                    leaf_evaluator='nn',
                    candidate_policy='nn',
                    verbose=False,
                )
            )
    return _V3NNPC_AGENTS


def _greedy_discard(hand14, evaluator=None, context=None, player_name=None):
    """使用 Baseline 策略做贪婪弃牌（默认 legacy eval2，可选 fast eval / nnpolicy）。"""
    if _ROLLOUT_POLICY == 'nnpolicy':
        from algo.nn import nn_policy
        # 没有上下文时退回到 baseline
        if context is None or player_name is None:
            return algo.select(hand14, _EMPTY_CONTEXT)[0][1]
        tiles = nn_policy.top_discards(hand14, context, player_name, k=1)
        return tiles[0]
    if evaluator is not None:
        return evaluator.select(hand14)[0]
    if _USE_FAST_ROLLOUT and fast_eval is not None:
        return fast_eval.select(hand14, _EMPTY_CONTEXT)[0]
    return algo.select(hand14, _EMPTY_CONTEXT)[0][1]


def _sample_deal(context, self_hand, self_name):
    """根据已见信息采样一个完整牌局：返回 hands dict、wall list、玩家顺序。"""
    all_tiles = tile.all_tiles_as_dict()
    unknown = dict_sub(dict_sub(all_tiles, context.used), count(self_hand))
    unknown_list = []
    for t, c in unknown.items():
        unknown_list.extend([t] * c)
    random.shuffle(unknown_list)

    def _seat(name):
        return int(name.split('@')[-1]) if '@' in name else 0

    known = set(context.discards.keys())
    known.add(self_name)
    # 补齐 4 个玩家，名字尽量沿用原始 context 中的格式
    dummy_id = 0
    while len(known) < 4:
        dummy = f'V3NN@{dummy_id}'
        if dummy not in known and dummy != self_name:
            known.add(dummy)
        dummy_id += 1
        if dummy_id > 100:  # 防止异常循环
            break

    players = sorted(known, key=_seat)
    # 保证 self_name 在 players 中
    if self_name not in players:
        players.append(self_name)

    hands = {}
    idx = 0
    for p in players:
        if p == self_name:
            hands[p] = list(self_hand)
        else:
            # 简化：每个对手补成 13 张
            hands[p] = unknown_list[idx:idx + 13]
            idx += 13
    wall = unknown_list[idx:]
    return hands, wall, players


def _outcome_for(self_name, winner, win_type, dealer):
    """把单局结果转换成当前玩家的收益。"""
    if win_type == 'draw':
        return 0.0
    if winner == self_name:
        return 1.0
    return -1.0


def _rollout_one(hands, wall, players, current_idx, locked, self_name, evaluator,
                 context=None, max_steps=200):
    """从当前玩家必须弃牌开始，跑一局快速 rollout。

    max_steps 用于防止极端中盘局面导致 rollout 无限长；超过则按流局返回 0。
    context 用于 nnpolicy rollout；baseline/fast rollout 不需要。
    """
    hands = {p: list(h) for p, h in hands.items()}
    wall = list(wall)
    wall_idx = 0
    turn = current_idx
    locked = set(locked)
    steps = 0

    # 当前玩家先弃牌
    if players[turn] in locked:
        discarded = hands[players[turn]][-1]
    else:
        discarded = _greedy_discard(hands[players[turn]], evaluator=evaluator,
                                     context=context, player_name=players[turn])
    hands[players[turn]].remove(discarded)

    # 点炮检查
    for j, p in enumerate(players):
        if j == turn:
            continue
        if eval_v2.is_win(hands[p] + [discarded]):
            return _outcome_for(self_name, p, 'ron', players[turn])

    turn = (turn + 1) % 4

    while wall_idx < len(wall):
        if steps >= max_steps:
            return 0.0
        steps += 1

        drawn = wall[wall_idx]
        wall_idx += 1
        player = players[turn]
        hands[player].append(drawn)

        if eval_v2.is_win(hands[player]):
            return _outcome_for(self_name, player, 'self', None)

        if player in locked:
            discarded = drawn
        else:
            discarded = _greedy_discard(hands[player], evaluator=evaluator,
                                         context=context, player_name=player)
        hands[player].remove(discarded)

        for j, p in enumerate(players):
            if j == turn:
                continue
            if eval_v2.is_win(hands[p] + [discarded]):
                return _outcome_for(self_name, p, 'ron', player)

        turn = (turn + 1) % 4

    return 0.0


def _sync_context_to_agents(context, agents, players):
    """把原始 context 的已见信息同步到每个 V3-NN-PC agent 的 context。"""
    for agent in agents:
        if agent.name not in players:
            continue
        agent.context = context.copy()
        # 重新初始化 belief，因为 context 变了
        agent._belief = None
        # agent.cur 会在 rollout 前被外部设置


def _rollout_one_v3nnpc(hands, wall, players, current_idx, locked_names, self_name,
                        max_steps=200):
    """用 V3-NN-PC agent 跑一局完整 rollout。

    mc_value 的采样状态是"当前玩家已摸牌、14 张、必须弃牌"，与 engine 的
    "下一个玩家摸牌"语义不同。因此先让当前玩家弃牌，再交给 engine 继续。
    """
    from driver.engine import play_game_from_state

    agents = _get_v3nnpc_agents()
    # 为每个玩家设置手牌和 context
    for i, p in enumerate(players):
        agents[i].name = p
        agents[i].init_tiles(list(hands[p]))

    # 当前玩家先弃牌（14 -> 13）。
    # BeliefExpectimaxV3Agent.next() 内部已经 self.cur.remove(result)。
    current = agents[current_idx]
    if current.name in locked_names:
        discarded = current.cur[-1]
        current.cur.remove(discarded)
    else:
        discarded = current.next()

    # 点炮检查
    for j, p in enumerate(players):
        if j == current_idx:
            continue
        if eval_v2.is_win(agents[j].cur + [discarded]):
            return _outcome_for(self_name, p, 'ron', current.name)

    # 通知所有 agent 这张弃牌（current.next() 已经更新过自己的 context）
    from agent import Message
    msg = Message(current.name, 'put', discarded)
    for i, other in enumerate(agents):
        if i == current_idx:
            continue
        other.handle_msg(msg)

    # 后续从下一个玩家开始由 engine 接管
    result = play_game_from_state(
        agents, wall, start_turn=(current_idx + 1) % len(players),
        locked_names=set(locked_names),
        verbose=False, record_time=False, record_log=False
    )
    return _outcome_for(self_name, result.get('winner'), result.get('win_type'), None)


def estimate_win_rate(context, hand14, self_name, n_rollouts=8, max_steps=200):
    """估计当前玩家在当前局面下的期望收益（-1~+1）。"""
    if n_rollouts <= 0:
        return 0.0

    hands, wall, players = _sample_deal(context, hand14, self_name)
    current_idx = players.index(self_name)
    locked = set()

    if _ROLLOUT_POLICY == 'v3nnpc':
        # 同步 context 到 agents（只同步一次，agents 内部会在 rollout 中更新）
        _sync_context_to_agents(context, _get_v3nnpc_agents(), players)
        total = 0.0
        for _ in range(n_rollouts):
            total += _rollout_one_v3nnpc(hands, wall, players, current_idx, locked,
                                         self_name, max_steps=max_steps)
        return total / n_rollouts

    evaluator = fast_eval.FastEval1(context) if _USE_FAST_ROLLOUT else None
    total = 0.0
    for _ in range(n_rollouts):
        total += _rollout_one(hands, wall, players, current_idx, locked, self_name,
                               evaluator=evaluator, context=context, max_steps=max_steps)
    return total / n_rollouts
