# -*- coding: utf-8 -*-
"""Benchmark PPO-128k vs 128k-BC-latest vs original PPO vs BC32k vs V3."""
import os
import glob
import argparse
from driver.tournament import run_tournament
from checker.report import compute_metrics, compute_elo
from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent

PPO128K_PATH = 'output/nn_full_action_ppo_128k.pt'
PPO_PATH = 'output/nn_full_action_ppo.pt'
BC32K_PATH = 'output/nn_full_action_best.pt'

NAMES = ['BC128k_latest', 'PPO128k', 'PPO', 'BC32k']


def _set_env():
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('NUMBA_NUM_THREADS', '1')
    import torch
    torch.set_num_threads(1)


def _latest_bc128k_path():
    files = glob.glob('output/nn_full_action_128000_epoch_*.pt')
    if not files:
        raise FileNotFoundError('no nn_full_action_128000_epoch_*.pt checkpoint')
    # 按文件名中的 epoch 数字排序
    def _epoch(p):
        base = os.path.basename(p)
        num = base.replace('nn_full_action_128000_epoch_', '').replace('.pt', '')
        return int(num)
    return max(files, key=_epoch)


def make_bc128k_latest():
    _set_env()
    path = _latest_bc128k_path()
    return HybridNNBeliefAgent('BC128k_latest', nn_model_path=path,
                               belief_kind='beliefexp', device='cpu',
                               temperature=None, verbose=False)


def make_ppo128k():
    _set_env()
    return HybridNNBeliefAgent('PPO128k', nn_model_path=PPO128K_PATH,
                               belief_kind='beliefexp', device='cpu',
                               temperature=None, verbose=False)


def make_ppo():
    _set_env()
    return HybridNNBeliefAgent('PPO', nn_model_path=PPO_PATH,
                               belief_kind='beliefexp', device='cpu',
                               temperature=None, verbose=False)


def make_best():
    _set_env()
    return HybridNNBeliefAgent('BC32k', nn_model_path=BC32K_PATH,
                               belief_kind='beliefexp', device='cpu',
                               temperature=None, verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-games', type=int, default=200)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=4100000)
    args = ap.parse_args()
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    _set_env()

    print(f'Using BC128k checkpoint: {_latest_bc128k_path()}')
    factories = [make_bc128k_latest, make_ppo128k, make_ppo, make_best]
    results = run_tournament(factories,
                             n_games=args.n_games, n_workers=args.workers,
                             seed_offset=args.seed, verbose=False)
    m = compute_metrics(results, NAMES)
    e = compute_elo(results, NAMES)
    print(f"{'Agent':<16} {'games':<8} {'win':<8} {'self':<8} {'ron':<8} {'deal-in':<10} {'draw':<8} {'Elo':<8} {'avg_ms':<10}")
    for n in NAMES:
        print(f"{n:<16} {m[n]['games']:<8} {m[n]['win_rate']:<8.3f} {m[n]['self_rate']:<8.3f} "
              f"{m[n]['ron_rate']:<8.3f} {m[n]['deal_in_rate']:<10.3f} {m[n]['draw_rate']:<8.3f} "
              f"{e[n]:<8.0f} {m[n]['avg_decision_time']*1000:<10.1f}")


if __name__ == '__main__':
    main()
