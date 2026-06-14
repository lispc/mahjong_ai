# -*- coding: utf-8 -*-
"""Run a tournament between ExpectiMax, MCTS and the baseline agent."""

import sys
import os
import time
import pickle
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from algo.agents.expectimax import ExpectiMaxAgent
from algo.agents.mcts import MCTSAgent
from driver.tournament import run_tournament
from checker.report import generate_report, compute_metrics, compute_elo


# Module-level factory functions so they are picklable for multiprocessing.
def make_baseline():
    return agent.Agent('Baseline', verbose=False)


def make_expectimax_d1():
    return ExpectiMaxAgent('ExpectiMax-D1', depth=1, verbose=False)


def make_expectimax_d2():
    return ExpectiMaxAgent('ExpectiMax-D2', depth=2, verbose=False,
                           max_discard_candidates=6,
                           max_draw_tiles=12,
                           max_draw_tiles2=8,
                           max_discard_candidates2=3,
                           min_draw_prob=0.005)


def make_mcts():
    return MCTSAgent('MCTS', depth=1, samples=250, verbose=False)


AGENTS_CONFIG = [
    make_baseline,
    make_expectimax_d1,
    make_expectimax_d2,
    make_mcts,
]

AGENT_NAMES = ['Baseline', 'ExpectiMax-D1', 'ExpectiMax-D2', 'MCTS']


def main():
    parser = argparse.ArgumentParser(description='Run mahjong AI tournament')
    parser.add_argument('n_games', type=int, nargs='?', default=200,
                        help='number of games to play')
    parser.add_argument('--workers', type=int, default=os.cpu_count(),
                        help='number of parallel workers (default: all cores)')
    args = parser.parse_args()

    print('Running {} games with {} workers ...'.format(args.n_games, args.workers))
    start = time.time()
    results = run_tournament(AGENTS_CONFIG, n_games=args.n_games,
                             verbose=False, n_workers=args.workers)
    elapsed = time.time() - start

    # 保存原始结果，方便后续重新生成报告或做深入分析
    results_path = 'output/results_{}.pkl'.format(args.n_games)
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print('Raw results saved to:', results_path)

    report_path = generate_report(results, AGENT_NAMES,
                                  output_path='output/ai_report.md')
    print('Report written to:', report_path)
    print('Total time: {:.1f}s ({:.2f}s per game)'.format(
        elapsed, elapsed / max(args.n_games, 1)))

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
