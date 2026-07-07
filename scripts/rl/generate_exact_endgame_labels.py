# -*- coding: utf-8 -*-
"""生成报听后终盘精确防守标签。

运行 N 局自对弈，捕获"对手报听后、当前玩家出牌前"的状态。
用对手真实手牌计算其待牌集合，再对每个合法弃牌计算 exact endgame EV。
输出 numpy 样本：(features, candidate_tiles, evs, chosen_tile, ...)。

用法：
    PYTHONPATH=. python3 scripts/rl/generate_exact_endgame_labels.py \
        output/exact_endgame_labels.npz 100 32
"""

import sys
import os
import argparse
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from driver import engine
from algo.agents.belief_expectimax import BeliefExpectimaxAgent
from algo.eval.endgame_solver import best_defensive_discard
from algo.eval.v2 import winning_tiles
from algo.nn.features import extract_features


def _make_agent():
    return BeliefExpectimaxAgent('BE', verbose=False)


def _wall_remaining(agents):
    """从 engine pool 推算剩余牌山（简化：基于初始 136 张和已见牌）。"""
    # 更简单的做法：从 context.used 反推
    ctx = getattr(agents[0], 'context', None)
    if ctx is None:
        return []
    used = getattr(ctx, 'used', {})
    from tile import all_tiles_as_dict
    wall = all_tiles_as_dict()
    for t, c in used.items():
        wall[t] -= c
    for ag in agents:
        for t in ag.cur:
            wall[t] -= 1
        for _, t in ag.melds:
            wall[t] -= 1
    return [t for t, c in wall.items() for _ in range(c)]


def _extract_state(agents, turn, info):
    """返回当前决策状态是否满足'有人报听且防守方需决策'，并收集信息。"""
    ctx = getattr(agents[turn], 'context', None)
    if ctx is None:
        return None
    tenpai = getattr(ctx, 'tenpai_players', set())
    opponents = [p for p in tenpai if p != agents[turn].name]
    if not opponents:
        return None
    if len(agents[turn].cur) != 14:
        return None
    wall = info.get('wall', [])
    # 暂时放宽终盘阈值以收集样本
    if len(wall) > 70:
        return None

    # 找第一个报听对手的真实手牌（自对弈中可见）
    opp_name = opponents[0]
    opp_agent = None
    for ag in agents:
        if ag.name == opp_name:
            opp_agent = ag
            break
    if opp_agent is None:
        return None
    opp_hand = list(opp_agent.cur)
    waits = set(winning_tiles(opp_hand, None))  # 不过滤剩余，只想要待牌种类
    if not waits:
        return None

    return {
        'self_name': agents[turn].name,
        'opponent_name': opp_name,
        'self_hand': list(agents[turn].cur),
        'opponent_hand': opp_hand,
        'waits': sorted(waits),
        'wall': wall,
        'tenpai_players': sorted(tenpai),
    }


def _play_one(seed):
    torch.set_num_threads(1)
    agents = [_make_agent() for _ in range(4)]
    for i, a in enumerate(agents):
        a.name = f'BE@{i}'
    samples = []

    def cb(ags, turn, event, info):
        if event == 'decision':
            s = _extract_state(ags, turn, info)
            if s is not None:
                # 计算每个候选弃牌的 exact EV
                best, evs = best_defensive_discard(
                    s['self_hand'], set(s['waits']), s['wall'],
                    tenpai_offset=0, deal_in_reward=-1.0)
                # 提取特征（用当前玩家视角）
                feats = extract_features(ags[turn].context, s['self_hand'],
                                         s['self_name'])
                s['features'] = np.asarray(feats, dtype=np.float32)
                s['evs'] = {int(k): float(v) for k, v in evs.items()}
                s['best_exact'] = int(best)
                samples.append(s)

    engine.play_game(agents, seed=seed, state_callback=cb)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='output .npz path')
    parser.add_argument('n_games', type=int, default=100)
    parser.add_argument('workers', type=int, default=8)
    args = parser.parse_args()

    all_samples = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_play_one, seed): seed
                   for seed in range(args.n_games)}
        for fut in as_completed(futures):
            all_samples.extend(fut.result())

    print(f'Collected {len(all_samples)} exact endgame samples from '
          f'{args.n_games} games')
    if not all_samples:
        return

    np.savez_compressed(args.output, samples=pickle.dumps(all_samples))
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
