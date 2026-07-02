# -*- coding: utf-8 -*-
"""生成 Oracle-Guided Distillation 所需数据。

用 BeliefExp 当教师打 N 局 self-play，记录完整 event_log，
然后重建每个决策点的完整状态（含对手手牌 + 牌山），提取：
- 普通特征（175 维，observable）
- Oracle 特征（311 维，perfect info）
- 动作、最终 outcome

输出 .npz 供后续训练 oracle policy 和蒸馏 normal policy。

用法：
    PYTHONPATH=. python3 scripts/rl/gen_oracle_data.py \
        output/nn_teacher_be_oracle.npz 2000 32 beliefexp 0
"""

import sys
import os
import json
import pickle
import random
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from driver.engine import play_game
from algo.agents.belief_expectimax import BeliefExpectimaxAgent
from algo.agents.belief_expectimax_v3 import BeliefExpectimaxV3Agent
from algo.nn.features import extract_features, extract_features_oracle
import agent as base_agent_mod
import context as ctx_module
import algo.context.v3 as context_v3


OUT = 'output'
NUM_ACTIONS = 34


def _teacher_agent(name, spec):
    if spec == 'beliefexp':
        return BeliefExpectimaxAgent(name, verbose=False)
    if spec == 'v3nnpc':
        return BeliefExpectimaxV3Agent(name, expectimax_depth=1, max_candidates=5,
                                       leaf_evaluator='nn', candidate_policy='nn')
    if spec == 'baseline':
        return base_agent_mod.Agent(name, verbose=False)
    raise ValueError(spec)


def _build_context_at_step(events_up_to_draw, players):
    """根据截止到某次摸牌的事件重建 ContextV3。"""
    ctx = context_v3.ContextV3()
    for ev in events_up_to_draw:
        if ev['type'] == 'discard':
            ctx.see_tile(ev['tile'], ev['player'])
        elif ev['type'] == 'tenpai':
            ctx.declare_tenpai(ev['player'])
    return ctx


def _extract_from_game(event_log, outcome_per_player):
    """从单局 event_log 提取 (X_normal, X_oracle, y, v, weights)。"""
    players = None
    hands = None
    samples = []

    # 事件索引 -> 截止到该次摸牌前的事件列表（用于重建 context）
    for idx, ev in enumerate(event_log):
        if ev['type'] == 'init':
            players = ev['players']
            hands = {p: list(h) for p, h in ev['hands'].items()}
        elif ev['type'] == 'draw':
            player = ev['player']
            tile_drawn = ev['tile']
            if player is None or hands is None:
                continue
            hands[player].append(tile_drawn)
            # 下一个事件应该是 discard（当前玩家的决策）
            next_ev = event_log[idx + 1] if idx + 1 < len(event_log) else None
            if next_ev is None or next_ev['type'] != 'discard':
                continue
            if next_ev.get('locked'):
                # 报听锁手，不是策略决策，跳过
                continue
            discarded = next_ev['tile']
            # 重建 context（截止到本次摸牌前，即决策前公开信息）
            ctx = _build_context_at_step(event_log[:idx], players)
            hand14 = list(hands[player])
            # 当前 wall = 所有剩余未摸牌
            wall_remaining = ev['wall_remaining']
            # 从 event_log 剩余部分重建 wall 列表
            wall = []
            for future in event_log[idx + 1:]:
                if future['type'] == 'draw':
                    wall.append(future['tile'])
            # 对手手牌
            all_hands = {p: list(h) for p, h in hands.items()}
            try:
                xn = extract_features(ctx, hand14, player)
                xo = extract_features_oracle(ctx, hand14, player, all_hands, wall)
                from algo.nn.features import _TILE_TO_IDX
                a = int(_TILE_TO_IDX[discarded])
                v = outcome_per_player.get(player, 0.0)
                samples.append((xn, xo, a, v))
            except Exception as e:
                # 特征提取失败时跳过
                print('feature extraction failed:', e)
            # 消费掉 discard 事件会在下一次循环处理；这里不重复修改 hands
        elif ev['type'] == 'discard':
            player = ev['player']
            tile_discarded = ev['tile']
            if hands and player in hands and tile_discarded in hands[player]:
                hands[player].remove(tile_discarded)
        elif ev['type'] == 'win':
            # 获胜者已经胡牌，不需要再处理
            pass

    return samples


def _play_one_game(teacher_spec, seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 31 - 1))
    agents = [_teacher_agent(f'T@{i}', teacher_spec) for i in range(4)]
    result = play_game(agents, record_log=True)
    event_log = result.get('event_log', [])
    # outcome per player
    outcome = {}
    winner = result.get('winner')
    win_type = result.get('win_type', 'draw')
    if win_type == 'draw':
        for p in result.get('players_order', []):
            outcome[p] = 0.0
    else:
        for p in result.get('players_order', []):
            if p == winner:
                outcome[p] = 1.0
            elif win_type == 'ron' and result.get('dealer') == p:
                outcome[p] = -1.0
            else:
                outcome[p] = -1.0
    return _extract_from_game(event_log, outcome)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else f'{OUT}/nn_teacher_be_oracle.npz'
    total_games = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    teacher = sys.argv[4] if len(sys.argv) > 4 else 'beliefexp'
    seed_base = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    print(f'Oracle data generation: {total_games} games, {workers} workers, teacher={teacher}')
    t0 = time.time()
    all_samples = []
    if workers <= 1:
        for i in range(total_games):
            all_samples.extend(_play_one_game(teacher, seed_base + i))
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_play_one_game, teacher, seed_base + i)
                       for i in range(total_games)]
            for fut in futures:
                all_samples.extend(fut.result())
    dt = time.time() - t0
    print(f'Generated {len(all_samples)} samples from {total_games} games in {dt:.1f}s')

    if not all_samples:
        print('No samples generated')
        return

    Xn = np.stack([s[0] for s in all_samples])
    Xo = np.stack([s[1] for s in all_samples])
    y = np.array([s[2] for s in all_samples], dtype=np.int64)
    v = np.array([s[3] for s in all_samples], dtype=np.float32)
    np.savez(out_path, Xn=Xn, Xo=Xo, y=y, v=v)
    print(f'Saved {out_path}: Xn={Xn.shape}, Xo={Xo.shape}, y={y.shape}, v={v.shape}')


if __name__ == '__main__':
    main()
