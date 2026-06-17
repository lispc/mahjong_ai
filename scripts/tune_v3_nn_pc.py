# -*- coding: utf-8 -*-
"""Grid search V3-NN-PC hyper-parameters (depth=1 only).

Grid over max_candidates and defense_margin.
Each config plays N games vs Baseline + BeliefExp.
Output: output/v3_nn_pc_tuning.json
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from algo.agents.belief_expectimax import BeliefExpectimaxAgent
from algo.agents.belief_expectimax_v3 import BeliefExpectimaxV3Agent
from driver.tournament import run_tournament
from checker.report import compute_metrics, compute_elo


# Module-level current config for picklable factory
_CURRENT_CONFIG = None


def make_baseline():
    return agent.Agent('Baseline', verbose=False)


def make_beliefexp():
    return BeliefExpectimaxAgent('BeliefExp', verbose=False)


def make_v3_from_global():
    cfg = _CURRENT_CONFIG
    return BeliefExpectimaxV3Agent(
        cfg['name'], verbose=False,
        expectimax_depth=1,
        max_candidates=cfg['max_candidates'],
        defense_margin=cfg['defense_margin'],
        leaf_evaluator='nn',
        candidate_policy='nn',
    )


def main():
    global _CURRENT_CONFIG
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    tournament_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    configs = []
    for max_candidates in [3, 5, 8, 12]:
        for margin in [0.0, 0.03, 0.06, 0.1]:
            configs.append({
                'name': f'V3-NN-PC-c{max_candidates}-m{margin}',
                'max_candidates': max_candidates,
                'defense_margin': margin,
            })

    results = []
    for cfg in configs:
        print(f'\n=== {cfg["name"]} ===', flush=True)
        _CURRENT_CONFIG = cfg

        agents_config = [make_baseline, make_beliefexp,
                         make_v3_from_global, make_v3_from_global]
        agent_names = ['Baseline', 'BeliefExp', cfg['name'], cfg['name']]

        start = time.time()
        raw = run_tournament(agents_config, n_games=n_games,
                             verbose=False, n_workers=tournament_workers)
        elapsed = time.time() - start

        metrics = compute_metrics(raw, agent_names)
        elo = compute_elo(raw, agent_names)

        m = metrics[cfg['name']]
        e = elo[cfg['name']]
        record = {
            'config': cfg,
            'elo': e,
            'win_rate': m['win_rate'],
            'self_rate': m['self_rate'],
            'ron_rate': m['ron_rate'],
            'deal_in_rate': m['deal_in_rate'],
            'draw_rate': m['draw_rate'],
            'avg_decision_time_ms': m['avg_decision_time'] * 1000,
            'total_time_s': elapsed,
        }
        results.append(record)
        print(f'  Elo {e:.0f}, win {m["win_rate"]:.1%}, deal-in {m["deal_in_rate"]:.1%}, '
              f'time {m["avg_decision_time"]*1000:.1f}ms, total {elapsed:.1f}s',
              flush=True)

        out_path = 'output/v3_nn_pc_tuning.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'  saved {out_path}', flush=True)

    results.sort(key=lambda x: x['elo'], reverse=True)
    print('\n=== Summary (sorted by Elo) ===')
    for r in results:
        print(f"{r['config']['name']}: Elo {r['elo']:.0f}, "
              f"win {r['win_rate']:.1%}, deal-in {r['deal_in_rate']:.1%}, "
              f"time {r['avg_decision_time_ms']:.1f}ms")


if __name__ == '__main__':
    main()
