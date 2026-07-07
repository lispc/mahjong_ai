# -*- coding: utf-8 -*-
"""生成 34 维待牌分布监督标签。

运行 N 局 BeliefExp 自对弈，在每个决策前捕获：
- 当前玩家视角的 175 维特征；
- 下家（seat+1）的真实 13/14 张手牌；
- 下家的待牌 one-hot（34 维）。

后续用于训练 wait_dist_head：从弃牌历史预测对手待牌分布。

用法：
    PYTHONPATH=. python3 scripts/rl/generate_wait_dist_labels.py \
        output/wait_dist_labels.npz 1000 32
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
from algo.eval.v2 import winning_tiles, shanten
from algo.nn.features import extract_features
from algo.eval.v3 import _IDX_TO_TILE


_TILE_TO_IDX_DICT = {int(t): int(i) for i, t in enumerate(_IDX_TO_TILE)}


def _make_agent():
    return BeliefExpectimaxAgent('BE', verbose=False)


def _seat(name):
    return int(name.split('@')[-1]) if '@' in name else 0


def _next_seat(name):
    return (_seat(name) + 1) % 4


def _find_agent(agents, seat):
    for ag in agents:
        if _seat(ag.name) == seat:
            return ag
    return None


def _wait_onehot(hand13_or_14):
    """返回 13/14 张手牌的待牌 one-hot（34 维）。"""
    arr = np.zeros(34, dtype=np.float32)
    waits = winning_tiles(list(hand13_or_14), None)
    for t in waits:
        arr[_TILE_TO_IDX_DICT[t]] = 1.0
    return arr


def _play_one(seed):
    torch.set_num_threads(1)
    agents = [_make_agent() for _ in range(4)]
    for i, a in enumerate(agents):
        a.name = f'BE@{i}'
    samples = []

    def cb(ags, turn, event, info):
        if event != 'decision':
            return
        if len(ags[turn].cur) != 14:
            return
        # 下家
        next_s = _next_seat(ags[turn].name)
        opp = _find_agent(ags, next_s)
        if opp is None or len(opp.cur) < 13:
            return
        # 只保留对手已听牌的状态
        if shanten(list(opp.cur)) != 0:
            return

        feats = extract_features(ags[turn].context, list(ags[turn].cur),
                                 ags[turn].name)
        wait_label = _wait_onehot(list(opp.cur))
        samples.append({
            'features': np.asarray(feats, dtype=np.float32),
            'wait_label': wait_label,
            'opponent_hand': list(opp.cur),
            'self_name': ags[turn].name,
            'opponent_name': opp.name,
        })

    engine.play_game(agents, seed=seed, state_callback=cb)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='output .npz path')
    parser.add_argument('n_games', type=int, default=1000)
    parser.add_argument('workers', type=int, default=32)
    args = parser.parse_args()

    all_samples = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_play_one, seed): seed
                   for seed in range(args.n_games)}
        for fut in as_completed(futures):
            all_samples.extend(fut.result())

    print(f'Collected {len(all_samples)} wait-distribution samples from '
          f'{args.n_games} games')
    if not all_samples:
        return

    np.savez_compressed(args.output, samples=pickle.dumps(all_samples))
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
