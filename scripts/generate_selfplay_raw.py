#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成自对弈原始样本（不含 MC rollout value），用于后续离线并行计算 label。"""
import sys
import os
import time
import random
import pickle
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.agents.data_collectors import DataCollectorV3NN
from driver import engine


def _outcome_for_agent(agent_name, result):
    win_type = result.get('win_type')
    if win_type == 'draw':
        return 0.0
    winner = result.get('winner')
    if agent_name == winner:
        return 1.0
    if win_type == 'ron' and result.get('dealer') == agent_name:
        return -1.0
    if win_type == 'self':
        return -1.0
    return 0.0


def _play_and_collect_raw(args):
    seed, target_seat = args
    random.seed(seed)
    agents = [DataCollectorV3NN('V3NN', verbose=False,
                                expectimax_depth=1, max_candidates=5)
              for _ in range(4)]
    random.shuffle(agents)
    for i, a in enumerate(agents):
        a.name = '{}@{}'.format(a.name, i)

    result = engine.play_game(agents, verbose=False, record_time=False)

    target_name = f'V3NN@{target_seat}'
    target_agent = None
    for a in agents:
        if a.name == target_name:
            target_agent = a
            break

    if target_agent is None:
        return []
    outcome = _outcome_for_agent(target_name, result)
    # 只保留计算 MC value 所需的最小信息，以及最终胜负用于 timeout fallback
    return [(item['context'], item['hand'], item['name'],
             item['features'], item['action'], outcome) for item in target_agent.buffer]


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    out_path = sys.argv[3] if len(sys.argv) > 3 else 'output/selfplay_raw.pkl'
    seed_offset = int(sys.argv[4]) if len(sys.argv) > 4 else 300000

    print(f'Generating {n_games} raw self-play games, workers={n_workers} ...')
    start = time.time()
    tasks = [(seed, 0) for seed in range(seed_offset, seed_offset + n_games)]
    all_samples = []
    completed = 0
    with Pool(n_workers) as pool:
        for samples in pool.imap_unordered(_play_and_collect_raw, tasks):
            all_samples.extend(samples)
            completed += 1
            if completed % 10 == 0 or completed == n_games:
                print(f'  ... {completed}/{n_games} games, {len(all_samples)} samples', flush=True)
    elapsed = time.time() - start
    print(f'Done in {elapsed:.1f}s: {len(all_samples)} samples')

    with open(out_path, 'wb') as f:
        pickle.dump(all_samples, f)
    print(f'Saved raw samples to {out_path}')


if __name__ == '__main__':
    main()
