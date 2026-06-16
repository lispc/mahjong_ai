#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 rollout policy net 的训练数据：用 legacy eval2 给真实局面打标签。

输入：自对弈原始样本 (context, hand14, ...)
输出：nn_training_data_rollout_policy.npz，包含 X（175 维特征）和 y（legacy eval2 的 top1 action）
"""
import sys
import os
import time
import pickle
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.eval.legacy import select as legacy_select
from algo.nn.features import extract_features, tile_to_index


def _label_one(args):
    ctx, hand14, name = args
    try:
        ranked = legacy_select(hand14, with_prob=False, c=ctx)
        top_tile = ranked[0]
        features = extract_features(ctx, hand14, name)
        return features, tile_to_index(top_tile)
    except Exception:
        return None


def main():
    raw_path = sys.argv[1] if len(sys.argv) > 1 else 'output/selfplay_raw_20000.pkl'
    out_npz = sys.argv[2] if len(sys.argv) > 2 else 'output/nn_training_data_rollout_policy.npz'
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    max_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 50000

    with open(raw_path, 'rb') as f:
        raw_samples = pickle.load(f)
    print(f'Loaded {len(raw_samples)} raw samples')

    if len(raw_samples) > max_samples:
        import random
        random.seed(42)
        raw_samples = random.sample(raw_samples, max_samples)
        print(f'Sampled down to {len(raw_samples)}')

    tasks = [(ctx, hand, name) for ctx, hand, name, *_ in raw_samples]

    print(f'Labeling with legacy eval2: workers={n_workers} ...')
    start = time.time()
    results = []
    completed = 0
    with Pool(n_workers) as pool:
        for res in pool.imap_unordered(_label_one, tasks):
            completed += 1
            if res is not None:
                results.append(res)
            if completed % 1000 == 0 or completed == len(tasks):
                print(f'  ... {completed}/{len(tasks)} labeled, {len(results)} ok, {time.time()-start:.1f}s', flush=True)

    elapsed = time.time() - start
    print(f'Done in {elapsed:.1f}s: {len(results)} labeled samples')

    X = np.stack([r[0] for r in results])
    y = np.array([r[1] for r in results], dtype=np.int64)
    np.savez_compressed(out_npz, X=X, y=y)
    print(f'Saved to {out_npz}: X{X.shape} y{y.shape}')


if __name__ == '__main__':
    main()
