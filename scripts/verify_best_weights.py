# -*- coding: utf-8 -*-
"""Verify CEM-tuned weights against default weights and baseline."""

import sys
import os
import time
import json
import pickle
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from algo.agents.expectimax import ExpectiMaxAgent
from driver.tournament import run_tournament
from checker.report import generate_report, compute_metrics, compute_elo


def make_baseline():
    return agent.Agent('Baseline', verbose=False)


def make_expectimax_default():
    return ExpectiMaxAgent('ExpectiMax-Default', depth=1, verbose=False, weights=None)


class WeightedExpectiMaxFactory:
    def __init__(self, weights, name):
        self.weights = weights
        self.name = name

    def __call__(self):
        return ExpectiMaxAgent(self.name, depth=1, verbose=False, weights=self.weights)


def main():
    parser = argparse.ArgumentParser(description='Verify best CEM weights')
    parser.add_argument('--weights', type=str, default='output/cem/best_weights.json',
                        help='path to best weights JSON')
    parser.add_argument('--games', type=int, default=100,
                        help='number of verification games')
    parser.add_argument('--workers', type=int, default=os.cpu_count(),
                        help='parallel workers')
    args = parser.parse_args()

    with open(args.weights, 'r', encoding='utf-8') as f:
        best_weights = json.load(f)

    print('Loaded best weights:', best_weights)

    agent_names = ['Baseline', 'ExpectiMax-Default', 'ExpectiMax-Tuned']
    agents_config = [
        make_baseline,
        make_expectimax_default,
        WeightedExpectiMaxFactory(best_weights, 'ExpectiMax-Tuned'),
        make_baseline,
    ]

    print('Running {} verification games with {} workers ...'.format(args.games, args.workers))
    start = time.time()
    results = run_tournament(agents_config, n_games=args.games,
                             verbose=False, n_workers=args.workers)
    elapsed = time.time() - start

    results_path = 'output/results_verify.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print('Raw results saved to:', results_path)

    report_path = generate_report(results, agent_names,
                                  output_path='output/ai_report_verify.md')
    print('Report written to:', report_path)
    print('Total time: {:.1f}s ({:.2f}s per game)'.format(
        elapsed, elapsed / max(args.games, 1)))

    metrics = compute_metrics(results, agent_names)
    elo = compute_elo(results, agent_names)
    print('\nQuick summary:')
    for name in agent_names:
        m = metrics[name]
        print('  {}: win {:.1%}, self {:.1%}, ron {:.1%}, deal-in {:.1%}, '
              'draw {:.1%}, Elo {:.0f}, avg_time {:.1f}ms'.format(
                  name, m['win_rate'], m['self_rate'], m['ron_rate'],
                  m['deal_in_rate'], m['draw_rate'], elo[name],
                  m['avg_decision_time'] * 1000))


if __name__ == '__main__':
    main()
