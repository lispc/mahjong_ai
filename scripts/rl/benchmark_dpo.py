# -*- coding: utf-8 -*-
"""Benchmark DPO model vs PPO, BC32k, V3."""
import os
import argparse
from driver.tournament import run_tournament
from checker.report import compute_metrics, compute_elo
from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent
from algo.agents.belief_expectimax_v3 import BeliefExpectimaxV3Agent

DPO_PATH = 'output/nn_full_action_dpo.pt'
PPO_PATH = 'output/nn_full_action_ppo.pt'
BC32K_PATH = 'output/nn_full_action_best.pt'
NAMES = ['DPO', 'PPO', 'BC32k', 'V3']


def _set_env():
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('NUMBA_NUM_THREADS', '1')
    import torch
    torch.set_num_threads(1)


def make_dpo():
    _set_env()
    return HybridNNBeliefAgent('DPO', nn_model_path=DPO_PATH,
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


def make_v3():
    _set_env()
    return BeliefExpectimaxV3Agent('V3', expectimax_depth=1, max_candidates=5,
                                   leaf_evaluator='nn', candidate_policy='nn',
                                   verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-games', type=int, default=200)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=4300000)
    args = ap.parse_args()
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    _set_env()

    factories = [make_dpo, make_ppo, make_best, make_v3]
    results = run_tournament(factories,
                             n_games=args.n_games, n_workers=args.workers,
                             seed_offset=args.seed, verbose=False)
    m = compute_metrics(results, NAMES)
    e = compute_elo(results, NAMES)
    print(f"{'Agent':<10} {'games':<8} {'win':<8} {'self':<8} {'ron':<8} {'deal-in':<10} {'draw':<8} {'Elo':<8} {'avg_ms':<10}")
    for n in NAMES:
        print(f"{n:<10} {m[n]['games']:<8} {m[n]['win_rate']:<8.3f} {m[n]['self_rate']:<8.3f} "
              f"{m[n]['ron_rate']:<8.3f} {m[n]['deal_in_rate']:<10.3f} {m[n]['draw_rate']:<8.3f} "
              f"{e[n]:<8.0f} {m[n]['avg_decision_time']*1000:<10.1f}")


if __name__ == '__main__':
    main()
