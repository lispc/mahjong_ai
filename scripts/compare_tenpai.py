# -*- coding: utf-8 -*-
"""对比 Eval2Ctx 报听开启 vs 关闭。"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from algo.agents.expectimax_eval2 import ExpectiMaxEval2Agent
from algo.agents.mcts_eval2 import MCTSEval2Agent
from algo.agents.mcts import MCTSAgent
from driver.tournament import run_tournament
from checker.report import compute_metrics, compute_elo


class Eval2CtxNoTenpai(ExpectiMaxEval2Agent):
    def declare_tenpai(self, hand, context):
        return False


def make_baseline():
    return agent.Agent('Baseline', verbose=False)


def make_eval2ctx():
    return ExpectiMaxEval2Agent('Eval2Ctx+Tenpai', verbose=False, max_candidates=6)


def make_eval2ctx_no_tenpai():
    return Eval2CtxNoTenpai('Eval2Ctx-NoTenpai', verbose=False, max_candidates=6)


def make_mcts_eval2():
    return MCTSEval2Agent('MCTS-Eval2', samples=5, max_candidates=4, verbose=False)


def make_mcts():
    return MCTSAgent('MCTS', depth=1, samples=250, verbose=False)


AGENTS_CONFIG = [make_baseline, make_eval2ctx, make_eval2ctx_no_tenpai, make_mcts_eval2, make_mcts]
AGENT_NAMES = ['Baseline', 'Eval2Ctx+Tenpai', 'Eval2Ctx-NoTenpai', 'MCTS-Eval2', 'MCTS']


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
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
    print('Total time: {:.1f}s ({:.2f}s per game)'.format(
        elapsed, elapsed / max(n_games, 1)))


if __name__ == '__main__':
    main()
