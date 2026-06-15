# -*- coding: utf-8 -*-
"""Benchmark Shanten+Ukeire variants (and baselines) in a 4-player tournament.

Note: tournament expects exactly 4 agent factories. If you want to compare more
variants, run the script multiple times with different --agents selections.
"""

import sys
import os
import time
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from algo.agents.expectimax_eval2 import ExpectiMaxEval2Agent
from algo.agents.mcts import MCTSAgent
from algo.agents.baseline_plus import BaselinePlusAgent
from algo.agents.prob_efficiency import ProbEfficiencyAgent
from algo.agents.belief_expectimax import BeliefExpectimaxAgent
from algo.agents.belief_expectimax_v2 import BeliefExpectimaxV2Agent
from algo.agents.determinized_mcts import DeterminizedMCTSAgent
from algo.agents.shanten_ukeire import (
    ShantenUkeireAgent, ShantenUkeireV3Agent
)
from driver.tournament import run_tournament
from checker.report import generate_report, compute_metrics, compute_elo


def make_baseline():
    return agent.Agent('Baseline', verbose=False)


def make_eval2ctx():
    return ExpectiMaxEval2Agent('Eval2Ctx', verbose=False, max_candidates=6)


def make_mcts():
    return MCTSAgent('MCTS', depth=1, samples=250, verbose=False)


def make_su_d0():
    return ShantenUkeireAgent('SU-d0', verbose=False)


def make_su_d1k4():
    return ShantenUkeireAgent('SU-d1k4', verbose=False,
                              expectimax_depth=1, n_samples=0, top_k=4)


def make_suv3_d0():
    return ShantenUkeireV3Agent('SUv3-d0', verbose=False, defense_weight=0.0)


def make_suv3_d2():
    return ShantenUkeireV3Agent('SUv3-d2', verbose=False, defense_weight=2.0)


def make_suv3_d3():
    return ShantenUkeireV3Agent('SUv3-d3', verbose=False, defense_weight=3.0)


def make_baseline_plus():
    return BaselinePlusAgent('Baseline+', verbose=False)


def make_baseline_plus_tw8():
    return BaselinePlusAgent('Baseline+tw8', verbose=False,
                             tenpai_min_wait=8)


def make_baseline_plus_eg12():
    return BaselinePlusAgent('Baseline+eg12', verbose=False,
                             endgame_threshold=12)


def make_baseline_plus_noten():
    return BaselinePlusAgent('Baseline+noTen', verbose=False,
                             tenpai_min_wait=999)


def make_baseline_plus_noend():
    return BaselinePlusAgent('Baseline+noEnd', verbose=False,
                             endgame_threshold=0)


def make_baseline_plus_noten_noend():
    return BaselinePlusAgent('Baseline+noTnoE', verbose=False,
                             tenpai_min_wait=999,
                             endgame_threshold=0)


def make_prob_eff():
    return ProbEfficiencyAgent('ProbEff', verbose=False)


def make_prob_eff_exp():
    return ProbEfficiencyAgent('ProbEffExp', verbose=False,
                               use_expectation=True)


def make_prob_eff_aggr():
    return ProbEfficiencyAgent('ProbEffAggr', verbose=False,
                               lambda_def_base=0.2,
                               lambda_tenpai_opponent=0.8)


def make_prob_eff_safe():
    return ProbEfficiencyAgent('ProbEffSafe', verbose=False,
                               lambda_def_base=1.0,
                               lambda_tenpai_opponent=2.5)


def make_belief_exp():
    return BeliefExpectimaxAgent('BeliefExp', verbose=False)


def make_belief_exp_aggr():
    return BeliefExpectimaxAgent('BeliefExpAggr', verbose=False,
                                 defense_margin=0.06)


def make_belief_exp_cautious():
    return BeliefExpectimaxAgent('BeliefExpCautious', verbose=False,
                                 defense_margin=0.015)


def make_belief_exp_v2():
    return BeliefExpectimaxV2Agent('BeliefExpV2', verbose=False,
                                   defense_margin=0.06)


def make_det_mcts():
    return DeterminizedMCTSAgent('DetMCTS', verbose=False,
                                 n_worlds=8, top_k=6, max_workers=1)


def make_det_mcts_v2():
    return DeterminizedMCTSAgent('DetMCTSV2', verbose=False,
                                 n_worlds=3, top_k=4, max_workers=4,
                                 rollout_depth=16, belief_exp_rollout=True)


AGENTS = {
    'baseline': make_baseline,
    'baseline_plus': make_baseline_plus,
    'baseline_plus_tw8': make_baseline_plus_tw8,
    'baseline_plus_eg12': make_baseline_plus_eg12,
    'baseline_plus_noten': make_baseline_plus_noten,
    'baseline_plus_noend': make_baseline_plus_noend,
    'baseline_plus_noten_noend': make_baseline_plus_noten_noend,
    'prob_eff': make_prob_eff,
    'prob_eff_exp': make_prob_eff_exp,
    'prob_eff_aggr': make_prob_eff_aggr,
    'prob_eff_safe': make_prob_eff_safe,
    'belief_exp': make_belief_exp,
    'belief_exp_aggr': make_belief_exp_aggr,
    'belief_exp_cautious': make_belief_exp_cautious,
    'belief_exp_v2': make_belief_exp_v2,
    'det_mcts': make_det_mcts,
    'det_mcts_v2': make_det_mcts_v2,
    'eval2ctx': make_eval2ctx,
    'mcts': make_mcts,
    'su_d0': make_su_d0,
    'su_d1k4': make_su_d1k4,
    'suv3_d0': make_suv3_d0,
    'suv3_d2': make_suv3_d2,
    'suv3_d3': make_suv3_d3,
}

# tournament renames agents to name@seat; metrics strips @seat and matches this.
DISPLAY_NAMES = {
    'baseline': 'Baseline',
    'baseline_plus': 'Baseline+',
    'baseline_plus_tw8': 'Baseline+tw8',
    'baseline_plus_eg12': 'Baseline+eg12',
    'baseline_plus_noten': 'Baseline+noTen',
    'baseline_plus_noend': 'Baseline+noEnd',
    'baseline_plus_noten_noend': 'Baseline+noTnoE',
    'prob_eff': 'ProbEff',
    'prob_eff_exp': 'ProbEffExp',
    'prob_eff_aggr': 'ProbEffAggr',
    'prob_eff_safe': 'ProbEffSafe',
    'belief_exp': 'BeliefExp',
    'belief_exp_aggr': 'BeliefExpAggr',
    'belief_exp_cautious': 'BeliefExpCautious',
    'belief_exp_v2': 'BeliefExpV2',
    'det_mcts': 'DetMCTS',
    'det_mcts_v2': 'DetMCTSV2',
    'eval2ctx': 'Eval2Ctx',
    'mcts': 'MCTS',
    'su_d0': 'SU-d0',
    'su_d1k4': 'SU-d1k4',
    'suv3_d0': 'SUv3-d0',
    'suv3_d2': 'SUv3-d2',
    'suv3_d3': 'SUv3-d3',
}


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Shanten+Ukeire variants in 4-player games.')
    parser.add_argument('n_games', nargs='?', type=int, default=100,
                        help='Number of games to run.')
    parser.add_argument('n_workers', nargs='?', type=int, default=os.cpu_count(),
                        help='Number of parallel workers.')
    parser.add_argument('--agents', nargs=4,
                        default=['baseline', 'eval2ctx', 'mcts', 'suv3_d2'],
                        help='Exactly 4 agent names from: {}'.format(
                            ', '.join(AGENTS.keys())))
    args = parser.parse_args()

    configs = [AGENTS[name] for name in args.agents]
    names = [DISPLAY_NAMES[name] for name in args.agents]

    print('Running {} games with {} workers ...'.format(args.n_games, args.n_workers))
    print('Agents:', ', '.join(names))
    start = time.time()
    results = run_tournament(configs, n_games=args.n_games,
                             verbose=False, n_workers=args.n_workers)
    elapsed = time.time() - start

    safe_names = '_'.join(names)
    path = 'output/results_shanten_ukeire_{}_{}.pkl'.format(
        args.n_games, safe_names)
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    print('Raw results saved to:', path)

    report_path = generate_report(results, names,
                                  output_path='output/shanten_ukeire_report.md')
    print('Report written to:', report_path)
    print('Total time: {:.1f}s ({:.2f}s per game)'.format(
        elapsed, elapsed / max(args.n_games, 1)))

    metrics = compute_metrics(results, names)
    elo = compute_elo(results, names)
    print('\nQuick summary:')
    for name in names:
        m = metrics[name]
        print('  {}: win {:.1%}, self {:.1%}, ron {:.1%}, deal-in {:.1%}, '
              'draw {:.1%}, Elo {:.0f}, avg_time {:.1f}ms'.format(
                  name, m['win_rate'], m['self_rate'], m['ron_rate'],
                  m['deal_in_rate'], m['draw_rate'], elo[name],
                  m['avg_decision_time'] * 1000))


if __name__ == '__main__':
    import argparse
    main()
