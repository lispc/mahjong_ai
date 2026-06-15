#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线并行计算 MC rollout value label（带超时、截断标记与 outcome fallback）。"""
import sys
import os
import time
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor, TimeoutError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.nn import mc_value


# quality flags
Q_OK = 0
Q_TIMEOUT = 1
Q_EXCEPTION = 2
Q_TRUNCATED = 3


def _compute_one(args):
    ctx, hand, name, features, action, outcome, n_rollouts, max_steps, timeout_per_task = args
    quality = Q_OK
    try:
        v = mc_value.estimate_win_rate(ctx, hand, name, n_rollouts=n_rollouts, max_steps=max_steps)
        # estimate_win_rate 内部无法直接告诉我们是否被截断，但这里保守地不标 truncated
    except TimeoutError:
        v = float(outcome)
        quality = Q_TIMEOUT
    except Exception:
        v = float(outcome)
        quality = Q_EXCEPTION
    return features, action, v, quality


def main():
    raw_path = sys.argv[1] if len(sys.argv) > 1 else 'output/selfplay_raw.pkl'
    out_npz = sys.argv[2] if len(sys.argv) > 2 else 'output/nn_training_data_selfplay_baseline_rollout.npz'
    n_rollouts = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    n_workers = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    timeout_per_task = int(sys.argv[5]) if len(sys.argv) > 5 else 180
    max_steps = int(sys.argv[6]) if len(sys.argv) > 6 else 200
    save_every = int(sys.argv[7]) if len(sys.argv) > 7 else 1000

    with open(raw_path, 'rb') as f:
        raw_samples = pickle.load(f)
    print(f'Loaded {len(raw_samples)} raw samples')

    # 兼容旧格式：没有 outcome 的样本默认 outcome=0.0
    samples = []
    for s in raw_samples:
        if len(s) == 5:
            ctx, hand, name, features, action = s
            outcome = 0.0
        else:
            ctx, hand, name, features, action, outcome = s
        samples.append((ctx, hand, name, features, action, outcome))

    # 断点续跑
    start_idx = 0
    results = []
    checkpoint_npz = out_npz + '.checkpoint.npz'
    if os.path.exists(checkpoint_npz):
        d = np.load(checkpoint_npz)
        completed = int(d['completed'])
        if completed > 0 and completed <= len(samples):
            results = list(zip(d['X'], d['y'], d['v'], d['q']))
            start_idx = completed
            print(f'Resuming from checkpoint: {completed}/{len(samples)} already done')

    tasks = [
        (ctx, hand, name, features, action, outcome, n_rollouts, max_steps, timeout_per_task)
        for ctx, hand, name, features, action, outcome in samples[start_idx:]
    ]

    print(f'Computing MC values: rollouts={n_rollouts}, workers={n_workers}, '
          f'timeout={timeout_per_task}s, max_steps={max_steps} ...')
    start = time.time()
    completed = start_idx

    def _save_checkpoint():
        if not results:
            return
        X = np.stack([r[0] for r in results])
        y = np.array([r[1] for r in results], dtype=np.int64)
        v = np.array([r[2] for r in results], dtype=np.float32)
        q = np.array([r[3] for r in results], dtype=np.int8)
        np.savez_compressed(checkpoint_npz, X=X, y=y, v=v, q=q, completed=np.array(completed))
        n_bad = int(np.sum(q != Q_OK))
        print(f'  -> checkpoint saved: {completed}/{len(samples)} ({n_bad} bad)', flush=True)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_compute_one, t): i for i, t in enumerate(tasks)}
        for future in futures:
            try:
                res = future.result(timeout=timeout_per_task)
                results.append(res)
                completed += 1
                if completed % save_every == 0 or completed == len(samples):
                    _save_checkpoint()
                    elapsed = time.time() - start
                    print(f'  ... {completed}/{len(samples)} samples, {elapsed:.1f}s', flush=True)
            except TimeoutError:
                # 这里不应该发生，因为 _compute_one 内部 catch 了 TimeoutError，
                # 但为防万一，用 outcome 填充
                idx = start_idx + futures[future]
                _, _, _, features, action, outcome, *_ = samples[idx]
                results.append((features, action, float(outcome), Q_TIMEOUT))
                completed += 1
                print(f'  ... {completed}/{len(samples)} outer TIMEOUT', flush=True)

    elapsed = time.time() - start
    print(f'Done in {elapsed:.1f}s')

    X = np.stack([r[0] for r in results])
    y = np.array([r[1] for r in results], dtype=np.int64)
    v = np.array([r[2] for r in results], dtype=np.float32)
    q = np.array([r[3] for r in results], dtype=np.int8)
    n_bad = int(np.sum(q != Q_OK))
    print(f'Total samples: {len(v)}, bad (timeout/exception/truncated): {n_bad} '
          f'({100*n_bad/len(v):.2f}%)')
    np.savez_compressed(out_npz, X=X, y=y, v=v, q=q)
    if os.path.exists(checkpoint_npz):
        os.remove(checkpoint_npz)
    print(f'Saved to {out_npz}: X{X.shape} y{y.shape} v{v.shape} q{q.shape}')


if __name__ == '__main__':
    main()
