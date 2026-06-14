# -*- coding: utf-8 -*-
"""Head-to-head: Baseline vs Eval2Ctx."""

import sys
import os
import time
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from algo.agents.expectimax_eval2 import ExpectiMaxEval2Agent
from driver.tournament import run_tournament
from checker.report import generate_report, compute_metrics, compute_elo


def make_baseline():
    return agent.Agent('Baseline', verbose=False)


def make_eval2():
    return ExpectiMaxEval2Agent('Eval2Ctx', verbose=False)


AGENTS_CONFIG = [make_baseline, make_eval2, make_baseline, make_eval2]
AGENT_NAMES = ['Baseline', 'Eval2Ctx']


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    print('Running {} games with {} workers (Baseline vs Eval2Ctx) ...'.format(n_games, workers))
    start = time.time()
    results = run_tournament(AGENTS_CONFIG, n_games=n_games,
                             verbose=False, n_workers=workers)
    elapsed = time.time() - start

    path = 'output/results_eval2ctx_baseline_{}.pkl'.format(n_games)
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    print('Raw results saved to:', path)

    report_path = generate_report(results, AGENT_NAMES,
                                  output_path='output/eval2ctx_baseline_report.md')
    print('Report written to:', report_path)
    print('Total time: {:.1f}s ({:.2f}s per game)'.format(
        elapsed, elapsed / max(n_games, 1)))

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
