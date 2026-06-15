# -*- coding: utf-8 -*-
"""用 BeliefExp 自对弈生成 NN 训练数据。

每个样本：(features, action_index, outcome)。
outcome 是这局最终该玩家获得的胜负标签：+1 获胜，-1 输牌，0 流局。
数据保存为 numpy 数组到 output/nn_training_data.npz。

为了避免每个决策都进行 IPC，改为每局完整结束后由工作进程一次性返回该局全部样本。
"""

import sys
import os
import time
import signal
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.agents.data_collectors import DataCollectorBeliefExp
from algo.nn import mc_value
from driver import engine
import random


_PER_GAME_TIMEOUT = 120  # 秒；单局超过此时长视为异常，返回空样本


class GameTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise GameTimeoutError('single game exceeded timeout')


def _outcome_for_agent(agent_name, result):
    """根据 engine.play_game 的返回结果计算某玩家的最终收益。"""
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


def _play_and_collect(seed, n_rollouts=0):
    """玩一局 BeliefExp 自对弈并返回该局全部样本。"""
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(_PER_GAME_TIMEOUT)
    try:
        random.seed(seed)
        agents = [DataCollectorBeliefExp('BeliefExp', verbose=False) for _ in range(4)]
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
    except GameTimeoutError:
        print(f'[timeout] seed={seed}', flush=True)
        return []
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _play_and_collect_wrapper(args):
    return _play_and_collect(*args)


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    out_path = sys.argv[3] if len(sys.argv) > 3 else 'output/nn_training_data.npz'
    seed_offset = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    n_rollouts = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    start = time.time()
    print(f'Generating {n_games} games with BeliefExp self-play '
          f'(workers={n_workers}, seed_offset={seed_offset}, '
          f'mc_rollouts={n_rollouts}) ...')

    all_samples = []
    completed = 0
    seeds = range(seed_offset, seed_offset + n_games)
    tasks = [(seed, n_rollouts) for seed in seeds]
    with Pool(n_workers) as pool:
        for samples in pool.imap_unordered(_play_and_collect_wrapper, tasks):
            all_samples.extend(samples)
            completed += 1
            if completed % 50 == 0 or completed == n_games:
                print(f'  ... {completed}/{n_games} games done, '
                      f'{len(all_samples)} samples so far', flush=True)

    elapsed = time.time() - start
    print(f'Generated {len(all_samples)} samples from {n_games} games in {elapsed:.1f}s '
          f'({len(all_samples)/elapsed:.1f} samples/s)')

    if not all_samples:
        print('No samples collected!')
        return

    X = np.stack([s[0] for s in all_samples])
    y = np.array([s[1] for s in all_samples], dtype=np.int64)
    v = np.array([s[2] for s in all_samples], dtype=np.float32)

    np.savez_compressed(out_path, X=X, y=y, v=v)
    print(f'Saved to {out_path}: X shape {X.shape}, y shape {y.shape}, v shape {v.shape}')


if __name__ == '__main__':
    main()
