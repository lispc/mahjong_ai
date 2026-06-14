# -*- coding: utf-8 -*-
"""Cross-Entropy Method for tuning ExpectiMax evaluation weights."""

import sys
import os
import time
import json
import pickle
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from algo.agents.expectimax import ExpectiMaxAgent
from driver.tournament import run_tournament
from checker.report import compute_elo


class WeightedExpectiMaxFactory:
    """Picklable factory that creates ExpectiMaxAgent with given weights."""
    def __init__(self, weights):
        self.weights = weights

    def __call__(self):
        return ExpectiMaxAgent('ExpectiMax', depth=1, verbose=False, weights=self.weights)


def make_baseline():
    return agent.Agent('Baseline', verbose=False)


def evaluate_weights(weights, n_games=15, n_workers=None):
    """Run a small tournament and return the ExpectiMax Elo score."""
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
    return elo['ExpectiMax']


def cem_search(initial_mu, initial_sigma, n_iter=10, n_samples=20,
               elite_frac=0.2, n_games=15, n_workers=None,
               output_dir='output/cem'):
    """
    Cross-Entropy Method for weight tuning.

    Parameters
    ----------
    initial_mu : dict
        Initial weight mean, e.g. {'shanten': 10.0, 'taatsu': 0.5, 'tenpai': 0.3}
    initial_sigma : dict
        Initial standard deviation for each weight.
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
            # Ensure non-negative weights
            w_vec = np.maximum(w_vec, 0.01)
            weights = {k: float(w_vec[i]) for i, k in enumerate(keys)}
            print('  sample {}/{}: {}'.format(s + 1, n_samples, weights))
            score = evaluate_weights(weights, n_games=n_games, n_workers=n_workers)
            samples.append(weights)
            scores.append(score)
            print('    -> Elo {:.1f}'.format(score))

            if score > best_score:
                best_score = score
                best_weights = weights.copy()

        # Select elite
        elite_indices = np.argsort(scores)[-elite_size:]
        elite_vecs = [np.array([samples[i][k] for k in keys]) for i in elite_indices]

        # Update distribution
        mu = np.mean(elite_vecs, axis=0)
        sigma = np.std(elite_vecs, axis=0)
        # Add minimal noise to avoid premature collapse
        sigma = np.maximum(sigma, 0.05 * np.abs(mu))

        history.append({
            'iteration': it,
            'mu': {k: float(mu[i]) for i, k in enumerate(keys)},
            'sigma': {k: float(sigma[i]) for i, k in enumerate(keys)},
            'samples': samples,
            'scores': scores,
            'best': best_weights,
            'best_score': best_score,
        })

        # Save after each iteration
        with open(os.path.join(output_dir, 'history.json'), 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - start
        print('Iteration {} done in {:.1f}s'.format(it + 1, elapsed))
        print('Current best: {} -> Elo {:.1f}'.format(best_weights, best_score))

    return best_weights, best_score, history


def main():
    parser = argparse.ArgumentParser(description='CEM weight tuning')
    parser.add_argument('--iter', type=int, default=10, help='CEM iterations')
    parser.add_argument('--samples', type=int, default=20, help='samples per iteration')
    parser.add_argument('--games', type=int, default=15, help='games per sample evaluation')
    parser.add_argument('--elite', type=float, default=0.2, help='elite fraction')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='parallel workers')
    parser.add_argument('--output', type=str, default='output/cem', help='output dir')
    args = parser.parse_args()

    initial_mu = {'shanten': 10.0, 'taatsu': 0.5, 'tenpai': 0.3}
    initial_sigma = {'shanten': 3.0, 'taatsu': 0.3, 'tenpai': 0.2}

    print('Starting CEM weight tuning')
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
    print('Best weights:', best_weights)
    print('Best Elo:', best_score)

    with open(os.path.join(args.output, 'best_weights.json'), 'w', encoding='utf-8') as f:
        json.dump(best_weights, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
