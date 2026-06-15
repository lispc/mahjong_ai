# -*- coding: utf-8 -*-
"""用快速 rollout 估计当前局面的胜率，作为价值网络的监督信号。

每次 rollout 都会把未知牌随机分配给对手并补成牌山，然后让所有玩家按一个贪婪
rollout policy 打完该局，最后返回当前玩家的平均收益（+1/0/-1）。
"""

import random
import copy

import algo
import tile
from utils import dict_sub, count
import algo.eval.v2 as eval_v2
import context as ctx_module


_EMPTY_CONTEXT = ctx_module.Context()


def _greedy_discard(hand14):
    """使用 Baseline 策略做贪婪弃牌（比 eval0 更强、更贴近真实对局）。"""
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
    # 补齐 4 个玩家
    dummy_id = 0
    while len(known) < 4:
        dummy = f'opp_{dummy_id}'
        if dummy not in known:
            known.add(dummy)
        dummy_id += 1

    players = sorted(known, key=_seat)

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


def _rollout_one(hands, wall, players, current_idx, locked, self_name):
    """从当前玩家必须弃牌开始，跑一局快速 rollout。"""
    hands = {p: list(h) for p, h in hands.items()}
    wall = list(wall)
    wall_idx = 0
    turn = current_idx
    locked = set(locked)

    # 当前玩家先弃牌
    if players[turn] in locked:
        discarded = hands[players[turn]][-1]
    else:
        discarded = _greedy_discard(hands[players[turn]])
    hands[players[turn]].remove(discarded)

    # 点炮检查
    for j, p in enumerate(players):
        if j == turn:
            continue
        if eval_v2.is_win(hands[p] + [discarded]):
            return _outcome_for(self_name, p, 'ron', players[turn])

    turn = (turn + 1) % 4

    while wall_idx < len(wall):
        drawn = wall[wall_idx]
        wall_idx += 1
        player = players[turn]
        hands[player].append(drawn)

        if eval_v2.is_win(hands[player]):
            return _outcome_for(self_name, player, 'self', None)

        if player in locked:
            discarded = drawn
        else:
            discarded = _greedy_discard(hands[player])
        hands[player].remove(discarded)

        for j, p in enumerate(players):
            if j == turn:
                continue
            if eval_v2.is_win(hands[p] + [discarded]):
                return _outcome_for(self_name, p, 'ron', player)

        turn = (turn + 1) % 4

    return 0.0


def estimate_win_rate(context, hand14, self_name, n_rollouts=8):
    """估计当前玩家在当前局面下的期望收益（-1~+1）。"""
    if n_rollouts <= 0:
        return 0.0

    hands, wall, players = _sample_deal(context, hand14, self_name)
    current_idx = players.index(self_name)
    locked = set()

    total = 0.0
    for _ in range(n_rollouts):
        total += _rollout_one(hands, wall, players, current_idx, locked, self_name)
    return total / n_rollouts
