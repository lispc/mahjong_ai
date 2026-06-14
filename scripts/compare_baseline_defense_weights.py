# -*- coding: utf-8 -*-
"""Test different BaseDef defense weights."""

import sys
import os
import time
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from algo.agents.expectimax_baseline import ExpectiMaxBaselineAgent
from driver.tournament import run_tournament
from checker.report import generate_report, compute_metrics, compute_elo


def make_baseline():
    return agent.Agent('Baseline', verbose=False)


def make_bd00():
    return ExpectiMaxBaselineAgent('BaseDef0.0', verbose=False, defense_weight=0.0)


def make_bd03():
    return ExpectiMaxBaselineAgent('BaseDef0.3', verbose=False, defense_weight=0.3)


def make_bd05():
    return ExpectiMaxBaselineAgent('BaseDef0.5', verbose=False, defense_weight=0.5)


AGENTS_CONFIG = [make_baseline, make_bd00, make_bd03, make_bd05]
AGENT_NAMES = ['Baseline', 'BaseDef0.0', 'BaseDef0.3', 'BaseDef0.5']


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    print('Running {} games with {} workers ...'.format(n_games, workers))
    start = time.time()
    results = run_tournament(AGENTS_CONFIG, n_games=n_games,
                             verbose=False, n_workers=workers)
    elapsed = time.time() - start

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
