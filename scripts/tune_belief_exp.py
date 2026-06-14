# -*- coding: utf-8 -*-
"""对 BeliefExpectimaxAgent 的核心超参做随机网格搜索。

每个参数组合会和三个固定对手打 n_games 局，记录胜率、点炮率和决策时间。
结果输出到 output/belief_exp_tuning.csv。
"""

import sys
import os
import time
import random
import itertools
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.agents.belief_expectimax import BeliefExpectimaxAgent
from algo.agents.baseline_plus import BaselinePlusAgent
from algo.agents.expectimax_eval2 import ExpectiMaxEval2Agent
from driver.tournament import run_tournament
from checker.report import compute_metrics


# ---------------------------------------------------------------------------
# Picklable factories
# ---------------------------------------------------------------------------

class BaselinePlusFactory:
    def __call__(self):
        return BaselinePlusAgent('Baseline+', verbose=False)


class Eval2CtxFactory:
    def __call__(self):
        return ExpectiMaxEval2Agent('Eval2Ctx', verbose=False, max_candidates=6)


class BeliefExpFactory:
    """Generate a picklable factory for a given parameter dict."""
    def __init__(self, params):
        self.params = params
        self.name = 'BeliefExp_m{}_c{}_t{}'.format(
            params['defense_margin'], params['max_candidates'], params['tenpai_min_wait'])

    def __call__(self):
        return BeliefExpectimaxAgent(self.name, verbose=False, **self.params)


FIXED_OPPONENTS = [BaselinePlusFactory(), Eval2CtxFactory(), BaselinePlusFactory()]

# 超参搜索空间
GRID = {
    'defense_margin': [0.0, 0.015, 0.03, 0.06, 0.10],
    'max_candidates': [4, 6, 8, 12],
    'tenpai_min_wait': [2, 3, 4, 6],
}


def evaluate_params(params, n_games=60, n_workers=8):
    factories = [BeliefExpFactory(params)] + FIXED_OPPONENTS
    start = time.time()
    results = run_tournament(factories, n_games=n_games, verbose=False, n_workers=n_workers)
    elapsed = time.time() - start

    agent_name = factories[0].name
    metrics = compute_metrics(results, [agent_name, 'Baseline+', 'Eval2Ctx', 'Baseline+'])
    m = metrics[agent_name]
    return {
        'params': params,
        'win_rate': m['win_rate'],
        'deal_in_rate': m['deal_in_rate'],
        'avg_time_ms': m['avg_decision_time'] * 1000,
        'games': m['games'],
        'elapsed': elapsed,
    }


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    n_games = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    all_combos = list(itertools.product(*GRID.values()))
    random.shuffle(all_combos)
    selected = all_combos[:n_trials]

    keys = list(GRID.keys())
    records = []
    for i, combo in enumerate(selected, 1):
        params = dict(zip(keys, combo))
        print('[{}/{}] Testing {}'.format(i, n_trials, params))
        record = evaluate_params(params, n_games=n_games, n_workers=n_workers)
        records.append(record)
        print('  win={:.1%} deal-in={:.1%} time={:.1f}ms games={} elapsed={:.1f}s'.format(
            record['win_rate'], record['deal_in_rate'], record['avg_time_ms'],
            record['games'], record['elapsed']))

    records.sort(key=lambda r: r['win_rate'], reverse=True)

    out_dir = 'output'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'belief_exp_tuning.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['defense_margin', 'max_candidates', 'tenpai_min_wait',
                        'win_rate', 'deal_in_rate', 'avg_time_ms', 'games', 'elapsed'])
        writer.writeheader()
        for r in records:
            row = dict(r['params'])
            row.update({
                'win_rate': r['win_rate'],
                'deal_in_rate': r['deal_in_rate'],
                'avg_time_ms': r['avg_time_ms'],
                'games': r['games'],
                'elapsed': r['elapsed'],
            })
            writer.writerow(row)

    print('\nTop 5 by win rate:')
    for r in records[:5]:
        print('  {} -> win={:.1%} deal-in={:.1%} time={:.1f}ms'.format(
            r['params'], r['win_rate'], r['deal_in_rate'], r['avg_time_ms']))
    print('Full results saved to', out_path)


if __name__ == '__main__':
    main()
