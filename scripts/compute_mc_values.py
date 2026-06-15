#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线并行计算 MC rollout value label。"""
import sys
import os
import time
import pickle
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.nn import mc_value


def _compute_one(args):
    ctx, hand, name, features, action, n_rollouts = args
    v = mc_value.estimate_win_rate(ctx, hand, name, n_rollouts=n_rollouts)
    return features, action, v


def main():
    raw_path = sys.argv[1] if len(sys.argv) > 1 else 'output/selfplay_raw.pkl'
    out_npz = sys.argv[2] if len(sys.argv) > 2 else 'output/nn_training_data_selfplay_baseline_rollout.npz'
    n_rollouts = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    n_workers = int(sys.argv[4]) if len(sys.argv) > 4 else 64

    with open(raw_path, 'rb') as f:
        raw_samples = pickle.load(f)
    print(f'Loaded {len(raw_samples)} raw samples')

    tasks = [(ctx, hand, name, features, action, n_rollouts)
             for ctx, hand, name, features, action in raw_samples]

    print(f'Computing MC values: rollouts={n_rollouts}, workers={n_workers} ...')
    start = time.time()
    results = []
    completed = 0
    with Pool(n_workers) as pool:
        for res in pool.imap_unordered(_compute_one, tasks):
            results.append(res)
            completed += 1
            if completed % 100 == 0 or completed == len(tasks):
                print(f'  ... {completed}/{len(tasks)} samples', flush=True)
    elapsed = time.time() - start
    print(f'Done in {elapsed:.1f}s')

    X = np.stack([r[0] for r in results])
    y = np.array([r[1] for r in results], dtype=np.int64)
    v = np.array([r[2] for r in results], dtype=np.float32)
    np.savez_compressed(out_npz, X=X, y=y, v=v)
    print(f'Saved to {out_npz}: X{X.shape} y{y.shape} v{v.shape}')


if __name__ == '__main__':
    main()
