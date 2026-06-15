# -*- coding: utf-8 -*-
"""自对弈 + 重训练循环。

用当前最强的 V3-NN leaf / NN policy 候选 agent 自己打数据，生成 MC rollout value
标签，然后重新训练 policy-value 网络和深度价值网络。反复迭代可不断提升模型。
"""

import sys
import os
import time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.agents.data_collectors import DataCollectorV3NN
from algo.nn import mc_value
from driver import engine
import random


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


def _play_and_collect(seed, n_rollouts=4):
    """一局 V3-NN self-play，返回带 MC value 标签的样本。"""
    random.seed(seed)
    agents = [DataCollectorV3NN('V3NN', verbose=False,
                                expectimax_depth=1, max_candidates=5)
              for _ in range(4)]
    random.shuffle(agents)
    for i, a in enumerate(agents):
        a.name = '{}@{}'.format(a.name, i)

    result = engine.play_game(agents, verbose=False, record_time=False)

    outcomes = {a.name: _outcome_for_agent(a.name, result) for a in agents}
    samples = []
    for a in agents:
        outcome = outcomes[a.name]
        for item in a.buffer:
            if n_rollouts > 0:
                mc_v = mc_value.estimate_win_rate(
                    item['context'], item['hand'], item['name'],
                    n_rollouts=n_rollouts)
            else:
                mc_v = outcome
            samples.append((item['features'], item['action'], mc_v))
    return samples


def _play_and_collect_wrapper(args):
    return _play_and_collect(*args)


def generate_self_play_data(n_games, n_workers, n_rollouts, seed_offset=0):
    out_path = 'output/nn_training_data_selfplay.npz'
    print(f'Self-play: generating {n_games} games (workers={n_workers}, '
          f'rollouts={n_rollouts}) ...')
    start = time.time()
    all_samples = []
    completed = 0
    tasks = [(seed, n_rollouts) for seed in range(seed_offset, seed_offset + n_games)]
    with Pool(n_workers) as pool:
        for samples in pool.imap_unordered(_play_and_collect_wrapper, tasks):
            all_samples.extend(samples)
            completed += 1
            if completed % 50 == 0 or completed == n_games:
                print(f'  ... {completed}/{n_games} games done, '
                      f'{len(all_samples)} samples', flush=True)
    elapsed = time.time() - start
    print(f'Generated {len(all_samples)} samples in {elapsed:.1f}s')

    if not all_samples:
        return None

    X = np.stack([s[0] for s in all_samples])
    y = np.array([s[1] for s in all_samples], dtype=np.int64)
    v = np.array([s[2] for s in all_samples], dtype=np.float32)
    np.savez_compressed(out_path, X=X, y=y, v=v)
    print(f'Saved to {out_path}: X{X.shape} y{y.shape} v{v.shape}')
    return out_path


def train_models(data_path, policy_epochs=60, value_epochs=60):
    """在自对弈数据上训练 policy-value 网络和深度价值网络。"""
    print('Training policy-value net ...')
    os.system(f'python scripts/train_nn.py {data_path} {policy_epochs} 256 0.001 256')
    print('Training deep value net ...')
    os.system(f'python scripts/train_value_net_mc.py {data_path} {value_epochs} 256 0.001')


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    n_rollouts = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    loops = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    for loop in range(1, loops + 1):
        print(f'\n========== Self-play loop {loop}/{loops} ==========')
        seed_offset = (loop - 1) * n_games
        data_path = generate_self_play_data(n_games, n_workers, n_rollouts,
                                            seed_offset=seed_offset)
        if data_path is None:
            print('No data generated, abort.')
            break
        train_models(data_path)
        print(f'Loop {loop} complete.')


if __name__ == '__main__':
    main()
