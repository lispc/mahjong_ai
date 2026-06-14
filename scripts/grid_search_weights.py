# -*- coding: utf-8 -*-
"""Grid search over evaluation weights."""

import sys
import os
import time
import json
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from algo.agents.expectimax import ExpectiMaxAgent
from driver.tournament import run_tournament
from checker.report import compute_elo


class WeightedExpectiMaxFactory:
    def __init__(self, weights):
        self.weights = weights

    def __call__(self):
        return ExpectiMaxAgent('ExpectiMax', depth=1, verbose=False, weights=self.weights)


def make_baseline():
    return agent.Agent('Baseline', verbose=False)


def evaluate_weights(weights, n_games=20, n_workers=None):
    if n_workers is None:
        n_workers = os.cpu_count()

    agents_config = [
        make_baseline,
        WeightedExpectiMaxFactory(weights),
        make_baseline,
        make_baseline,
    ]
    results = run_tournament(agents_config, n_games=n_games,
                             verbose=False, n_workers=n_workers)
    elo = compute_elo(results, ['Baseline', 'ExpectiMax'])
    metrics = {}
    for r in results:
        if r['win_type'] != 'draw' and r['winner'].startswith('ExpectiMax'):
            metrics['wins'] = metrics.get('wins', 0) + 1
    metrics['games'] = n_games
    metrics['win_rate'] = metrics.get('wins', 0) / n_games
    return elo['ExpectiMax'], metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Grid search weights')
    parser.add_argument('--games', type=int, default=20, help='games per config')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='workers')
    args = parser.parse_args()

    grid = {
        'shanten': [8.0, 10.0, 12.0],
        'taatsu': [0.3, 0.5, 0.7],
        'tenpai': [0.1, 0.3, 0.5],
    }

    keys = list(grid.keys())
    configs = list(itertools.product(*grid.values()))
    print('Grid search: {} configurations'.format(len(configs)))

    results = []
    best = None
    best_score = -float('inf')

    for idx, values in enumerate(configs):
        weights = {k: values[i] for i, k in enumerate(keys)}
        print('\n[{}/{}] Testing {}'.format(idx + 1, len(configs), weights))
        start = time.time()
        elo, m = evaluate_weights(weights, n_games=args.games, n_workers=args.workers)
        elapsed = time.time() - start
        print('  -> Elo {:.1f}, win_rate {:.1%}, time {:.1f}s'.format(
            elo, m['win_rate'], elapsed))

        results.append({'weights': weights, 'elo': elo, 'metrics': m})

        if elo > best_score:
            best_score = elo
            best = weights

    # Sort by Elo
    results.sort(key=lambda x: x['elo'], reverse=True)

    os.makedirs('output/grid_search', exist_ok=True)
    with open('output/grid_search/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print('\n=== Best weights ===')
    print(best)
    print('Best Elo:', best_score)

    with open('output/grid_search/best_weights.json', 'w', encoding='utf-8') as f:
        json.dump(best, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
