# -*- coding: utf-8 -*-
"""生成 search-value labels：用 V3 deep search agent 自对弈，
每步记录 (features, selected_search_value, action)。

用法：
  PYTHONPATH=. python3 scripts/rl/gen_search_value_data.py \
      output/nn_search_value_v3d2_200.npz 200 12 \
      --depth 2 --leaf nn --cand output/nn_full_action_best.pt
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from algo.nn.features import extract_features, tile_to_index


def _make_collector(depth, leaf, cand, cand_policy, max_candidates=5):
    from algo.agents.belief_expectimax_v3 import BeliefExpectimaxV3Agent as Base

    class _Collector(Base):
        def __init__(self, nm):
            kwargs = dict(
                verbose=False,
                expectimax_depth=depth,
                max_candidates=max_candidates,
                leaf_evaluator=leaf,
                candidate_policy=cand_policy,
            )
            if cand_policy == 'nn':
                kwargs['candidate_model_path'] = cand
            super().__init__(nm, **kwargs)
            self.steps = []

        def next(self):
            feats = extract_features(self.context, self.cur, self.name)
            disc, trace = self.next_with_trace()
            self.steps.append((
                np.asarray(feats, dtype=np.float32),
                float(trace.get('selected_value', 0.0)),
                int(tile_to_index(disc)),
            ))
            return disc

    return _Collector


def _init_worker(depth, leaf, cand, cand_policy, max_candidates):
    global _COLLECTOR_CLS
    _COLLECTOR_CLS = _make_collector(depth, leaf, cand, cand_policy, max_candidates)


def _worker(args):
    n_games, seed_base = args
    torch.set_num_threads(1)
    import random
    from driver.engine import play_game
    Xs, vs, acts = [], [], []
    cls = globals()['_COLLECTOR_CLS']
    for g in range(n_games):
        random.seed(seed_base + g)
        agents = [cls(f'T@{s}') for s in range(4)]
        result = play_game(agents)
        for a in agents:
            for feats, val, act in a.steps:
                Xs.append(feats)
                vs.append(val)
                acts.append(act)
    if not Xs:
        return None
    return {
        'X': np.stack(Xs, axis=0),
        'v': np.array(vs, dtype=np.float32),
        'a': np.array(acts, dtype=np.int64),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='output .npz path')
    parser.add_argument('n_games', type=int, help='total games')
    parser.add_argument('workers', type=int, help='parallel workers')
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--leaf', default='nn')
    parser.add_argument('--cand', default='output/nn_full_action_best.pt',
                        help='candidate model path (used only when cand-policy=nn)')
    parser.add_argument('--cand-policy', default='nn',
                        help='candidate generation policy: nn | baseline_eval1 | ...')
    parser.add_argument('--max-candidates', type=int, default=5)
    parser.add_argument('--seed-base', type=int, default=700000)
    parser.add_argument('--save-every', type=int, default=0,
                        help='if >0, save checkpoint every N games')
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

    all_X, all_v, all_a = [], [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker,
                             initargs=(args.depth, args.leaf, args.cand, args.cand_policy, args.max_candidates)) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        for future in as_completed(futures):
            res = future.result()
            if res is None:
                continue
            all_X.append(res['X'])
            all_v.append(res['v'])
            all_a.append(res['a'])
            n = sum(len(x) for x in all_v)
            print(f'  collected {n} samples so far ...')

    X = np.concatenate(all_X, axis=0)
    v = np.concatenate(all_v, axis=0)
    a = np.concatenate(all_a, axis=0)
    dt = time.time() - t0
    print(f'Done. {len(X)} samples from {n_games} games in {dt:.1f}s')
    print(f'Value mean={v.mean():.3f} std={v.std():.3f} min={v.min():.3f} max={v.max():.3f}')
    np.savez(args.output, X=X, v=v, a=a)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
