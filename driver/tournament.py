# -*- coding: utf-8 -*-
"""Run repeated games with randomised seating.

新增 duplicate（复式）赛制：同一副牌墙/种子下，让两个候选 agent 轮换坐同一
席位，其余三席固定，从而配对消除发牌运气。
"""

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


class _DuplicateTask:
    """Picklable config for one duplicate mirror game."""
    def __init__(self, seed, candidate_kind, position,
                 candidate_factory, opponent_factories, verbose=False):
        self.seed = seed
        self.candidate_kind = candidate_kind  # 'a' or 'b'
        self.position = position
        self.candidate_factory = candidate_factory
        self.opponent_factories = list(opponent_factories)
        self.verbose = verbose


def _play_one_duplicate_game(task):
    """Worker: play one game with a fixed wall seed and fixed seating."""
    # Candidate sits at `position`; opponents fill the remaining seats in order.
    factories = [None] * 4
    factories[task.position] = task.candidate_factory
    opp_idx = 0
    for i in range(4):
        if factories[i] is None:
            factories[i] = task.opponent_factories[opp_idx]
            opp_idx += 1

    random.seed(task.seed)
    agents = [f() for f in factories]
    for i, a in enumerate(agents):
        a.name = '{}@{}_{}'.format(a.name, i, task.candidate_kind)

    return engine.play_game(agents, seed=task.seed, verbose=task.verbose,
                            record_time=True)


def run_duplicate_tournament(candidate_a_factory, candidate_b_factory,
                             opponent_factories,
                             n_seeds=400, mirror_positions=False,
                             verbose=False, n_workers=1, seed_offset=0):
    """
    Duplicate (paired) tournament between two candidate agents.

    For each seed, the same wall is used for both candidates.  Opponents sit
    in fixed seats; the candidate occupies `position`.  If `mirror_positions`
    is False (default), only position 0 is mirrored: 2 games per seed.  If
    True, all four positions are mirrored: 8 games per seed, fully cancelling
    seating bias.

    Returns a list of result dicts (A games followed by B games, grouped by
    seed and position).
    """
    if len(opponent_factories) != 3:
        raise ValueError('opponent_factories must contain exactly 3 factories')

    positions = list(range(4)) if mirror_positions else [0]
    tasks = []
    for s in range(n_seeds):
        seed = seed_offset + s
        for pos in positions:
            tasks.append(_DuplicateTask(seed, 'a', pos,
                                        candidate_a_factory,
                                        opponent_factories, verbose))
            tasks.append(_DuplicateTask(seed, 'b', pos,
                                        candidate_b_factory,
                                        opponent_factories, verbose))

    if n_workers <= 1:
        results = [_play_one_duplicate_game(t) for t in tasks]
        return results

    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_play_one_duplicate_game, t): idx
            for idx, t in enumerate(tasks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return results
