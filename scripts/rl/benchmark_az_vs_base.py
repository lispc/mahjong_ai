# -*- coding: utf-8 -*-
"""Quick benchmark: AZ checkpoint vs Hybrid-FullAction-32k base.

用法：
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python3 scripts/rl/benchmark_az_vs_base.py \
        output/nn_full_action_az_epoch_24.pt 200 16
"""
import os
import sys
import time
import argparse
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent
from driver.tournament import run_tournament
from checker.report import compute_metrics, compute_elo


class HybridFactory:
    def __init__(self, name, model_path, device='cpu'):
        self.name = name
        self.model_path = model_path
        self.device = device

    def __call__(self):
        return HybridNNBeliefAgent(self.name, nn_model_path=self.model_path,
                                   belief_kind='beliefexp', device=self.device,
                                   temperature=0.0, verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('az_path', help='AZ checkpoint path')
    ap.add_argument('n_games', type=int, default=200)
    ap.add_argument('n_workers', type=int, default=16)
    ap.add_argument('--base-path', type=str, default='output/nn_full_action_best.pt')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    torch.set_num_threads(1)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    factories = [
        HybridFactory('Hybrid-AZ', args.az_path, args.device),
        HybridFactory('Hybrid-Base', args.base_path, args.device),
    ]
    names = ['Hybrid-AZ', 'Hybrid-Base']

    print(f'Benchmark {args.az_path} vs {args.base_path}')
    print(f'{args.n_games} games, {args.n_workers} workers, device={args.device}')
    t0 = time.time()
    results = run_tournament(factories, n_games=args.n_games,
                             verbose=False, n_workers=args.n_workers)
    dt = time.time() - t0
    metrics = compute_metrics(results, names)
    elo = compute_elo(results, names)
    print(f'\nTotal {dt:.1f}s ({dt/max(args.n_games,1):.2f}s per game)')
    for name in names:
        m = metrics[name]
        print('  {:12s}: win {:.1%}, self {:.1%}, ron {:.1%}, deal-in {:.1%}, '
              'draw {:.1%}, Elo {:.0f}'.format(
                  name, m['win_rate'], m['self_rate'], m['ron_rate'],
                  m['deal_in_rate'], m['draw_rate'], elo[name]))


if __name__ == '__main__':
    main()
