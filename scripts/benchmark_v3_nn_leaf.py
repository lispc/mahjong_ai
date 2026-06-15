# -*- coding: utf-8 -*-
"""Benchmark BeliefExpectimaxV3 with NN leaf vs eval0 leaf vs BeliefExp."""

import sys
import os
import time
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from algo.agents.belief_expectimax import BeliefExpectimaxAgent
from algo.agents.belief_expectimax_v3 import BeliefExpectimaxV3Agent
from driver.tournament import run_tournament
from checker.report import generate_report, compute_metrics, compute_elo


def make_baseline():
    return agent.Agent('Baseline', verbose=False)


def make_beliefexp():
    return BeliefExpectimaxAgent('BeliefExp', verbose=False)


def make_v3_eval0():
    return BeliefExpectimaxV3Agent('V3-eval0', expectimax_depth=1,
                                   max_candidates=5, leaf_evaluator='eval0')


def make_v3_nn():
    return BeliefExpectimaxV3Agent('V3-NN', expectimax_depth=1,
                                   max_candidates=5, leaf_evaluator='nn')


AGENTS_CONFIG = [make_baseline, make_beliefexp, make_v3_eval0, make_v3_nn]
AGENT_NAMES = ['Baseline', 'BeliefExp', 'V3-eval0', 'V3-NN']


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    print(f'Running {n_games} games with {workers} workers ...')
    start = time.time()
    results = run_tournament(AGENTS_CONFIG, n_games=n_games,
                             verbose=False, n_workers=workers)
    elapsed = time.time() - start

    path = f'output/results_v3_nn_leaf_{n_games}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    print('Raw results saved to:', path)

    report_path = generate_report(results, AGENT_NAMES,
                                  output_path='output/v3_nn_leaf_report.md')
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
