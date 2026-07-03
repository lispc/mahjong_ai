# -*- coding: utf-8 -*-
"""Benchmark：完整动作空间 NN vs 当前 best / BeliefExp V3。

用法：
    PYTHONPATH=. python3 scripts/rl/benchmark_full_action.py 200 8
"""

import os
import sys
import time
import argparse
import random
import numpy as np
import multiprocessing as mp

from driver.tournament import run_tournament
from checker.report import compute_metrics, compute_elo
from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent
from algo.agents.ppo_agent import PPOAgent
from algo.agents.belief_expectimax_v3 import BeliefExpectimaxV3Agent


NN_PATH = 'output/nn_full_action_4000.pt'
BEST_PATH = 'output/nn_conv_bc_beliefexp_trace_16000_big_t8.pt'
AGENT_NAMES = ['FullNN', 'Best', 'V3']


def _set_env():
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('NUMBA_NUM_THREADS', '1')
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def make_full():
    _set_env()
    return HybridNNBeliefAgent(
        'FullNN',
        nn_model_path=NN_PATH,
        belief_kind='beliefexp',
        device='cpu',
        temperature=None,
        verbose=False,
    )


def make_best():
    _set_env()
    return HybridNNBeliefAgent(
        'Best',
        nn_model_path=BEST_PATH,
        belief_kind='beliefexp',
        device='cpu',
        temperature=None,
        verbose=False,
    )


def make_v3():
    _set_env()
    return BeliefExpectimaxV3Agent(
        'V3',
        expectimax_depth=1,
        max_candidates=5,
        leaf_evaluator='nn',
        candidate_policy='nn',
        verbose=False,
    )


def summarize(results):
    metrics = compute_metrics(results, AGENT_NAMES)
    elo = compute_elo(results, AGENT_NAMES)
    print(f"{'Agent':<10} {'games':<8} {'win':<8} {'self':<8} {'ron':<8} {'deal-in':<10} {'draw':<8} {'Elo':<8} {'avg_ms':<10}")
    for n in AGENT_NAMES:
        m = metrics[n]
        print(f"{n:<10} {m['games']:<8} {m['win_rate']:<8.3f} {m['self_rate']:<8.3f} "
              f"{m['ron_rate']:<8.3f} {m['deal_in_rate']:<10.3f} {m['draw_rate']:<8.3f} "
              f"{elo[n]:<8.0f} {m['avg_decision_time']*1000:<10.1f}")
    print(f"Total games: {metrics['_meta']['total_games']}, draws: {metrics['_meta']['draws']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('n_games', type=int, default=200)
    ap.add_argument('workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=3000000)
    args = ap.parse_args()

    mp.set_start_method('spawn', force=True)
    _set_env()

    factories = [make_full, make_best, make_v3, make_best]
    t0 = time.time()
    results = run_tournament(factories, n_games=args.n_games, n_workers=args.workers,
                             seed_offset=args.seed, verbose=False)
    dt = time.time() - t0
    print(f'Finished {args.n_games} games in {dt:.1f}s')
    summarize(results)


if __name__ == '__main__':
    main()
