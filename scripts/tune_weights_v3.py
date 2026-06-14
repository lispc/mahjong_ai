# -*- coding: utf-8 -*-
"""Cross-Entropy Method for tuning eval_v3 weights."""

import sys
import os
import time
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from algo.agents.expectimax_v3 import ExpectiMaxV3Agent
from driver.tournament import run_tournament
from checker.report import compute_elo


class WeightedV3Factory:
    """Picklable factory that creates ExpectiMaxV3Agent with given weights."""
    def __init__(self, weights, defense_weight):
        self.weights = weights
        self.defense_weight = defense_weight

    def __call__(self):
        return ExpectiMaxV3Agent('V3', depth=1, verbose=False,
                                 weights=self.weights,
                                 defense_weight=self.defense_weight,
                                 max_discard_candidates=8)


def make_baseline():
    return agent.Agent('Baseline', verbose=False)


def evaluate_weights(weights, defense_weight, n_games=10, n_workers=None):
    """Run a small tournament and return the V3 Elo score."""
    if n_workers is None:
        n_workers = os.cpu_count()

    agents_config = [
        make_baseline,
        WeightedV3Factory(weights, defense_weight),
        make_baseline,
        make_baseline,
    ]

    results = run_tournament(agents_config, n_games=n_games,
                             verbose=False, n_workers=n_workers)
    elo = compute_elo(results, ['Baseline', 'V3'])
    return elo['V3']


def cem_search(initial_mu, initial_sigma, n_iter=5, n_samples=10,
               elite_frac=0.3, n_games=10, n_workers=None,
               output_dir='output/cem_v3'):
    """
    Cross-Entropy Method for eval_v3 weights.
    """
    os.makedirs(output_dir, exist_ok=True)
    keys = list(initial_mu.keys())
    mu = np.array([initial_mu[k] for k in keys], dtype=float)
    sigma = np.array([initial_sigma[k] for k in keys], dtype=float)
    elite_size = max(1, int(n_samples * elite_frac))

    history = []
    best_weights = None
    best_score = -float('inf')

    for it in range(n_iter):
        print('\n=== CEM iteration {}/{} ==='.format(it + 1, n_iter))
        start = time.time()

        samples = []
        scores = []
        for s in range(n_samples):
            w_vec = np.random.normal(mu, sigma)
            w_vec = np.maximum(w_vec, 0.001)
            weights = {k: float(w_vec[i]) for i, k in enumerate(keys) if k != 'defense_weight'}
            defense_weight = float(w_vec[keys.index('defense_weight')])
            print('  sample {}/{}: weights={}, defense={:.2f}'.format(
                s + 1, n_samples, weights, defense_weight))
            score = evaluate_weights(weights, defense_weight,
                                     n_games=n_games, n_workers=n_workers)
            samples.append((weights, defense_weight))
            scores.append(score)
            print('    -> Elo {:.1f}'.format(score))

            if score > best_score:
                best_score = score
                best_weights = (weights.copy(), defense_weight)

        elite_indices = np.argsort(scores)[-elite_size:]
        elite_vecs = [np.array([samples[i][0][k] for k in keys if k != 'defense_weight'] +
                               [samples[i][1]]) for i in elite_indices]

        mu = np.mean(elite_vecs, axis=0)
        sigma = np.std(elite_vecs, axis=0)
        sigma = np.maximum(sigma, 0.05 * np.abs(mu))

        mu_dict = {k: float(mu[i]) for i, k in enumerate(keys) if k != 'defense_weight'}
        mu_dict['defense_weight'] = float(mu[-1])
        sigma_dict = {k: float(sigma[i]) for i, k in enumerate(keys) if k != 'defense_weight'}
        sigma_dict['defense_weight'] = float(sigma[-1])

        history.append({
            'iteration': it,
            'mu': mu_dict,
            'sigma': sigma_dict,
            'samples': [{'weights': w, 'defense_weight': dw, 'score': sc}
                        for (w, dw), sc in zip(samples, scores)],
            'best_weights': best_weights[0],
            'best_defense_weight': best_weights[1],
            'best_score': best_score,
        })

        with open(os.path.join(output_dir, 'history.json'), 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - start
        print('Iteration {} done in {:.1f}s'.format(it + 1, elapsed))
        print('Current best: weights={}, defense={:.2f} -> Elo {:.1f}'.format(
            best_weights[0], best_weights[1], best_score))

    return best_weights, best_score, history


def main():
    parser = argparse.ArgumentParser(description='CEM weight tuning for eval_v3')
    parser.add_argument('--iter', type=int, default=5, help='CEM iterations')
    parser.add_argument('--samples', type=int, default=10, help='samples per iteration')
    parser.add_argument('--games', type=int, default=10, help='games per sample evaluation')
    parser.add_argument('--elite', type=float, default=0.3, help='elite fraction')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='parallel workers')
    parser.add_argument('--output', type=str, default='output/cem_v3', help='output dir')
    args = parser.parse_args()

    initial_mu = {
        'shanten': 10.0,
        'ukeire': 0.05,
        'wait': 0.5,
        'algo_eval0': 20.0,
        'defense_weight': 2.0,
    }
    initial_sigma = {
        'shanten': 3.0,
        'ukeire': 0.03,
        'wait': 0.3,
        'algo_eval0': 8.0,
        'defense_weight': 1.0,
    }

    print('Starting CEM for eval_v3')
    print('Initial mu:', initial_mu)
    print('Initial sigma:', initial_sigma)
    print('Iterations:', args.iter)
    print('Samples/iter:', args.samples)
    print('Games/sample:', args.games)
    print('Workers:', args.workers)

    best_weights, best_score, history = cem_search(
        initial_mu, initial_sigma,
        n_iter=args.iter,
        n_samples=args.samples,
        elite_frac=args.elite,
        n_games=args.games,
        n_workers=args.workers,
        output_dir=args.output,
    )

    print('\n=== CEM finished ===')
    print('Best weights:', best_weights[0])
    print('Best defense_weight:', best_weights[1])
    print('Best Elo:', best_score)

    with open(os.path.join(args.output, 'best_weights.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'weights': best_weights[0],
            'defense_weight': best_weights[1],
            'elo': best_score,
        }, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
