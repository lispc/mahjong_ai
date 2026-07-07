# -*- coding: utf-8 -*-
"""生成 outcome value labels：用 Hybrid-Best 自对弈，
每步记录 (features, action)，游戏结束后给每步赋予当前玩家的最终收益 (+1/0/-1)。

用法：
  PYTHONPATH=. python3 scripts/rl/gen_outcome_value_data.py \
      output/nn_outcome_hybridbest_5k.npz 5000 32 \
      --seed-base 900000
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from algo.nn.features import extract_features, tile_to_index
from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent


def _outcome_for(name, winner, win_type):
    if win_type == 'draw':
        return 0.0
    if winner == name:
        return 1.0
    return -1.0


def _make_collector(model_path='output/nn_full_action_best.pt',
                    belief_kind='beliefexp'):

    class _Collector(HybridNNBeliefAgent):
        def __init__(self, nm):
            super().__init__(nm, nn_model_path=model_path,
                             belief_kind=belief_kind, device='cpu',
                             temperature=0.0)
            self.steps = []

        def next(self):
            feats = extract_features(self.nn_agent.context, self.cur, self.name)
            disc = super().next()
            self.steps.append((
                np.asarray(feats, dtype=np.float32),
                int(tile_to_index(disc)),
            ))
            return disc

    return _Collector


def _init_worker(model_path, belief_kind):
    global _COLLECTOR_CLS
    _COLLECTOR_CLS = _make_collector(model_path, belief_kind)


def _worker(args):
    n_games, seed_base = args
    torch.set_num_threads(1)
    import random
    from driver.engine import play_game
    cls = globals()['_COLLECTOR_CLS']
    Xs, ys, acts = [], [], []
    for g in range(n_games):
        random.seed(seed_base + g)
        agents = [cls(f'T@{s}') for s in range(4)]
        result = play_game(agents)
        winner = result.get('winner')
        win_type = result.get('win_type', 'draw')
        for a in agents:
            y = _outcome_for(a.name, winner, win_type)
            for feats, act in a.steps:
                Xs.append(feats)
                ys.append(y)
                acts.append(act)
    if not Xs:
        return None
    return {
        'X': np.stack(Xs, axis=0),
        'y': np.array(ys, dtype=np.float32),
        'a': np.array(acts, dtype=np.int64),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='output .npz path')
    parser.add_argument('n_games', type=int, help='total games')
    parser.add_argument('workers', type=int, help='parallel workers')
    parser.add_argument('--model-path', default='output/nn_full_action_best.pt')
    parser.add_argument('--belief-kind', default='beliefexp')
    parser.add_argument('--seed-base', type=int, default=900000)
    args = parser.parse_args()

    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_games = args.n_games
    workers = args.workers
    games_per_worker = max(1, n_games // workers)
    tasks = []
    total_assigned = 0
    for w in range(workers):
        ng = games_per_worker
        if w == workers - 1:
            ng = n_games - total_assigned
        if ng <= 0:
            break
        tasks.append((ng, args.seed_base + total_assigned))
        total_assigned += ng

    all_X, all_y, all_a = [], [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker,
                             initargs=(args.model_path, args.belief_kind)) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        for future in as_completed(futures):
            res = future.result()
            if res is None:
                continue
            all_X.append(res['X'])
            all_y.append(res['y'])
            all_a.append(res['a'])
            n = sum(len(x) for x in all_y)
            print(f'  collected {n} samples so far ...')

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    a = np.concatenate(all_a, axis=0)
    dt = time.time() - t0
    print(f'Done. {len(X)} samples from {n_games} games in {dt:.1f}s')
    print(f'Outcome mean={y.mean():.3f} std={y.std():.3f} win={(y>0).mean():.3f} lose={(y<0).mean():.3f}')
    np.savez(args.output, X=X, y=y, a=a)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
