# -*- coding: utf-8 -*-
"""Grid search ExpectiMaxEval2Agent.max_candidates."""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from algo.agents.expectimax_eval2 import ExpectiMaxEval2Agent
from algo.agents.mcts_eval2 import MCTSEval2Agent
from algo.agents.mcts import MCTSAgent


class Eval2CtxNoTenpai(ExpectiMaxEval2Agent):
    """用于 max_candidates 调参的纯净版，不报听，避免 tenpai 机制干扰对比。"""
    def declare_tenpai(self, hand, context):
        return False
from driver.tournament import run_tournament
from checker.report import compute_metrics, compute_elo


def make_baseline():
    return agent.Agent('Baseline', verbose=False)


def make_eval2ctx_mc4():
    return Eval2CtxNoTenpai('Eval2Ctx-mc4', verbose=False, max_candidates=4)


def make_eval2ctx_mc6():
    return Eval2CtxNoTenpai('Eval2Ctx-mc6', verbose=False, max_candidates=6)


def make_eval2ctx_mc8():
    return Eval2CtxNoTenpai('Eval2Ctx-mc8', verbose=False, max_candidates=8)


def make_eval2ctx_mc10():
    return Eval2CtxNoTenpai('Eval2Ctx-mc10', verbose=False, max_candidates=10)


def make_mcts_eval2():
    return MCTSEval2Agent('MCTS-Eval2', samples=5, max_candidates=4, verbose=False)


def make_mcts():
    return MCTSAgent('MCTS', depth=1, samples=250, verbose=False)


FACTORIES = [
    make_baseline,
    make_eval2ctx_mc4,
    make_eval2ctx_mc6,
    make_eval2ctx_mc8,
    make_eval2ctx_mc10,
    make_mcts_eval2,
    make_mcts,
]
NAMES = ['Baseline', 'Eval2Ctx-mc4', 'Eval2Ctx-mc6', 'Eval2Ctx-mc8', 'Eval2Ctx-mc10', 'MCTS-Eval2', 'MCTS']


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()

    print('Running {} games with {} workers, max_candidates=[4,6,8,10] ...'.format(
        n_games, workers))
    start = time.time()
    results = run_tournament(FACTORIES, n_games=n_games,
                             verbose=False, n_workers=workers)
    elapsed = time.time() - start

    metrics = compute_metrics(results, NAMES)
    elo = compute_elo(results, NAMES)

    print('\nQuick summary:')
    print('  {:16s} {:>6s} {:>6s} {:>6s} {:>8s} {:>6s} {:>8s}'.format(
        'AI', 'win', 'self', 'ron', 'deal-in', 'Elo', 'time(ms)'))
    print('  ' + '-' * 60)
    for name in NAMES:
        m = metrics[name]
        print('  {:16s} {:>5.1%} {:>5.1%} {:>5.1%} {:>7.1%} {:>5.0f} {:>8.1f}'.format(
            name, m['win_rate'], m['self_rate'], m['ron_rate'],
            m['deal_in_rate'], elo[name], m['avg_decision_time'] * 1000))
    print('\nTotal time: {:.1f}s ({:.2f}s per game)'.format(
        elapsed, elapsed / max(n_games, 1)))


if __name__ == '__main__':
    main()
