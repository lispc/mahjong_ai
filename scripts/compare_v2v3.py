# -*- coding: utf-8 -*-
"""Head-to-head: eval_v2 vs eval_v3."""

import sys
import os
import time
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.agents.expectimax import ExpectiMaxAgent
from algo.agents.expectimax_v3 import ExpectiMaxV3Agent
from driver.tournament import run_tournament
from checker.report import generate_report, compute_metrics, compute_elo


def make_v2():
    return ExpectiMaxAgent('V2', depth=1, verbose=False)


def make_v3():
    return ExpectiMaxV3Agent('V3', depth=1, verbose=False,
                             defense_weight=2.0,
                             max_discard_candidates=8)


AGENTS_CONFIG = [make_v2, make_v3, make_v2, make_v3]
AGENT_NAMES = ['V2', 'V3']


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    print('Running {} games with {} workers (V2 vs V3) ...'.format(n_games, workers))
    start = time.time()
    results = run_tournament(AGENTS_CONFIG, n_games=n_games,
                             verbose=False, n_workers=workers)
    elapsed = time.time() - start

    path = 'output/results_v2v3_{}.pkl'.format(n_games)
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    print('Raw results saved to:', path)

    report_path = generate_report(results, AGENT_NAMES,
                                  output_path='output/v2v3_report.md')
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
