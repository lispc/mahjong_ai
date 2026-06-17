#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速生成自对弈数据，用于验证 baseline rollout 的 MC value label 质量。

优化点：
- 每局只收集一个固定 seat 的玩家样本（减少单局 MC rollout 次数）；
- n_rollouts=1，先快速得到 directional 信号；
- 高并行 workers。
"""
import sys
import os
import time
import random
import numpy as np
from multiprocessing import Pool

# 使用 fast_eval 作为 MC rollout policy，否则 baseline select 太慢
os.environ['MJ_FAST_ROLLOUT'] = '1'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.agents.data_collectors import DataCollectorBaseline
from driver import engine
from algo.nn import mc_value


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


def _play_and_collect_single(seed, target_seat=0, n_rollouts=1):
    random.seed(seed)
    agents = [DataCollectorBaseline('Baseline', verbose=False)
              for _ in range(4)]
    random.shuffle(agents)
    for i, a in enumerate(agents):
        a.name = '{}@{}'.format(a.name, i)

    result = engine.play_game(agents, verbose=False, record_time=False)

    target_name = f'Baseline@{target_seat}'
    target_agent = None
    for a in agents:
        if a.name == target_name:
            target_agent = a
            break

    if target_agent is None or not target_agent.buffer:
        return []

    outcome = _outcome_for_agent(target_name, result)
    samples = []
    for item in target_agent.buffer:
        if n_rollouts > 0:
            mc_v = mc_value.estimate_win_rate(
                item['context'], item['hand'], item['name'],
                n_rollouts=n_rollouts)
        else:
            mc_v = outcome
        samples.append((item['features'], item['action'], mc_v))
    return samples


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    n_rollouts = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    out_path = sys.argv[4] if len(sys.argv) > 4 else 'output/nn_training_data_selfplay_baseline_rollout.npz'
    seed_offset = int(sys.argv[5]) if len(sys.argv) > 5 else 200000

    print(f'Generating {n_games} games, workers={n_workers}, rollouts={n_rollouts} ...')
    start = time.time()
    all_samples = []
    completed = 0
    tasks = [(seed, 0, n_rollouts) for seed in range(seed_offset, seed_offset + n_games)]
    with Pool(n_workers) as pool:
        for samples in pool.imap_unordered(_play_and_collect_single, tasks):
            all_samples.extend(samples)
            completed += 1
            if completed % 10 == 0 or completed == n_games:
                print(f'  ... {completed}/{n_games} games, {len(all_samples)} samples', flush=True)
    elapsed = time.time() - start
    print(f'Done in {elapsed:.1f}s: {len(all_samples)} samples')

    if not all_samples:
        return

    X = np.stack([s[0] for s in all_samples])
    y = np.array([s[1] for s in all_samples], dtype=np.int64)
    v = np.array([s[2] for s in all_samples], dtype=np.float32)
    np.savez_compressed(out_path, X=X, y=y, v=v)
    print(f'Saved to {out_path}: X{X.shape} y{y.shape} v{v.shape}')


if __name__ == '__main__':
    main()
