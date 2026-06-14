# -*- coding: utf-8 -*-
"""Benchmark MCTS-Eval2 against others."""

import sys
import os
import time
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from algo.agents.expectimax_eval2 import ExpectiMaxEval2Agent, ExpectiMaxEval2DefenseAgent
from algo.agents.mcts_eval2 import MCTSEval2Agent
from algo.agents.mcts import MCTSAgent
from driver.tournament import run_tournament
from checker.report import generate_report, compute_metrics, compute_elo


def make_baseline():
    return agent.Agent('Baseline', verbose=False)


def make_eval2ctx():
    return ExpectiMaxEval2Agent('Eval2Ctx', verbose=False, max_candidates=6)


def make_eval2ctx_bd():
    return ExpectiMaxEval2DefenseAgent('Eval2Ctx+BD', verbose=False, defense_weight=0.3)


def make_mcts_eval2():
    return MCTSEval2Agent('MCTS-Eval2', samples=5, max_candidates=4, verbose=False)


def make_mcts():
    return MCTSAgent('MCTS', depth=1, samples=250, verbose=False)


AGENTS_CONFIG = [make_baseline, make_eval2ctx, make_eval2ctx_bd, make_mcts_eval2, make_mcts]
AGENT_NAMES = ['Baseline', 'Eval2Ctx', 'Eval2Ctx+BD', 'MCTS-Eval2', 'MCTS']


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    print('Running {} games with {} workers ...'.format(n_games, workers))
    start = time.time()
    results = run_tournament(AGENTS_CONFIG, n_games=n_games,
                             verbose=False, n_workers=workers)
    elapsed = time.time() - start

    path = 'output/results_mcts_eval2_{}.pkl'.format(n_games)
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    print('Raw results saved to:', path)

    report_path = generate_report(results, AGENT_NAMES,
                                  output_path='output/mcts_eval2_report.md')
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
