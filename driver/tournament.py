# -*- coding: utf-8 -*-
"""Run repeated games with randomised seating."""

import random
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from driver import engine


def _play_one_game(agents_config, seed, verbose=False):
    """Worker function for one game; must be picklable."""
    random.seed(seed)
    agents = [factory() for factory in agents_config]
    random.shuffle(agents)
    for i, a in enumerate(agents):
        a.name = '{}@{}'.format(a.name, i)
    return engine.play_game(agents, verbose=verbose, record_time=True)


def run_tournament(agents_config, n_games=200, verbose=False, n_workers=1, seed_offset=0):
    """
    agents_config: a list of agent factory functions, one per seat.
    Each game the seat order is shuffled so that every agent type plays
    from different positions.

    Returns a list of result dicts from engine.play_game.
    """
    if n_workers <= 1:
        results = []
        for idx in range(n_games):
            results.append(_play_one_game(agents_config, seed=seed_offset + idx, verbose=verbose))
        return results

    results = [None] * n_games
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_play_one_game, agents_config, seed_offset + idx, verbose): idx
            for idx in range(n_games)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return results
