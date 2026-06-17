# -*- coding: utf-8 -*-
"""Benchmark multiple DeterminizedMCTS configurations.

注意：DetMCTS 内部也用 ProcessPoolExecutor，不要和 tournament 的 worker 数同时设太大，
否则会出现进程爆炸。推荐 tournament workers=4，DetMCTS internal workers<=4。
"""

import sys
import os
import time
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from algo.agents.belief_expectimax import BeliefExpectimaxAgent
from algo.agents.determinized_mcts import DeterminizedMCTSAgent
from driver.tournament import run_tournament
from checker.report import generate_report, compute_metrics, compute_elo


def make_baseline():
    return agent.Agent('Baseline', verbose=False)


def make_beliefexp():
    return BeliefExpectimaxAgent('BeliefExp', verbose=False)


def make_detmcts_fast_10x6():
    return DeterminizedMCTSAgent('DetMCTS-fast', n_worlds=10, top_k=6,
                                 max_workers=4)


def make_detmcts_nn_5x5():
    return DeterminizedMCTSAgent('DetMCTS-NN', n_worlds=5, top_k=5,
                                 max_workers=1, nn_rollout=True)


AGENTS_CONFIG = [
    make_baseline,
    make_beliefexp,
    make_detmcts_fast_10x6,
    make_detmcts_nn_5x5,
]
AGENT_NAMES = ['Baseline', 'BeliefExp', 'DetMCTS-fast', 'DetMCTS-NN']


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    print(f'Running {n_games} games with {workers} workers ...')
    print(f'Agents: {AGENT_NAMES}')
    start = time.time()
    results = run_tournament(AGENTS_CONFIG, n_games=n_games,
                             verbose=False, n_workers=workers)
    elapsed = time.time() - start

    path = f'output/results_detmcts_variants_{n_games}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    print('Raw results saved to:', path)

    report_path = generate_report(results, AGENT_NAMES,
                                  output_path='output/detmcts_variants_report.md')
    print('Report written to:', report_path)
    print(f'Total time: {elapsed:.1f}s ({elapsed/max(n_games,1):.2f}s per game)')

    metrics = compute_metrics(results, AGENT_NAMES)
    elo = compute_elo(results, AGENT_NAMES)
    print('\nQuick summary:')
    for name in AGENT_NAMES:
        m = metrics[name]
        print('  {}: win {:.1%}, self {:.1%}, ron {:.1%}, deal-in {:.1%}, '
              'draw {:.1%}, Elo {:.0f}, avg_time {:.1f}ms'.format(
                  name, m['win_rate'], m['self_rate'], m['ron_rate'],
                  m['deal_in_rate'], m['draw_rate'], elo[name],
                  m['avg_decision_time'] * 1000))


if __name__ == '__main__':
    main()
