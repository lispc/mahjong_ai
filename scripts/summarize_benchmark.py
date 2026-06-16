#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总多 GPU benchmark 的 results pkl 文件。"""
import sys
import os
import pickle
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checker.report import compute_metrics, compute_elo


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else 'output/benchmark_splits/results_seed*.pkl'
    names = sys.argv[2].split(',') if len(sys.argv) > 2 else None

    files = sorted(glob.glob(pattern))
    if not files:
        print(f'No files matched: {pattern}')
        sys.exit(1)

    all_results = []
    for f in files:
        print(f'Loading {f} ...')
        with open(f, 'rb') as fh:
            all_results.extend(pickle.load(fh))

    print(f'Total games: {len(all_results)}')
    if names is None:
        names = sorted({r[0] for r in all_results})

    metrics = compute_metrics(all_results, names)
    elo = compute_elo(all_results, names)

    print('\nCombined results:')
    print(f"{'Agent':<12} {'win':<8} {'self':<8} {'ron':<8} {'deal-in':<10} {'draw':<8} {'Elo':<8} {'avg_ms':<10}")
    for n in names:
        m = metrics[n]
        print(f"{n:<12} {m['win_rate']:<8.3f} {m['self_rate']:<8.3f} "
              f"{m['ron_rate']:<8.3f} {m['deal_in_rate']:<10.3f} "
              f"{m['draw_rate']:<8.3f} {elo[n]:<8.0f} {m['avg_decision_time'] * 1000:<10.1f}")


if __name__ == '__main__':
    main()
