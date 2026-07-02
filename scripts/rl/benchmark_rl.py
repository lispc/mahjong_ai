# -*- coding: utf-8 -*-
"""Benchmark PPO agent vs 当前 best(V3-NN-PC) / Baseline / BeliefExp。

用法：
    PYTHONPATH=. python3 scripts/rl/benchmark_rl.py [n_games] [workers] [ppo_model_path]

每局 4 个座位：PPO / V3-NN-PC / Baseline / BeliefExp，座位随机洗牌。
输出 Elo / 胜率 / 点炮率 等，并写 output/rl_ppo_benchmark_report.md。
"""

import sys
import os
import time
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from algo.agents.belief_expectimax import BeliefExpectimaxAgent
from algo.agents.belief_expectimax_v3 import BeliefExpectimaxV3Agent
from algo.agents.ppo_agent import PPOAgent
from driver.tournament import run_tournament
from checker.report import generate_report, compute_metrics, compute_elo


PPO_MODEL = sys.argv[3] if len(sys.argv) > 3 else 'output/nn_rl_ppo.pt'


def make_ppo():
    return PPOAgent('PPO', model_path=PPO_MODEL, device='cpu', temperature=0.0)


def make_v3_nn_pc():
    return BeliefExpectimaxV3Agent('V3-NN-PC', expectimax_depth=1, max_candidates=5,
                                   leaf_evaluator='nn', candidate_policy='nn')


def make_baseline():
    return agent.Agent('Baseline', verbose=False)


def make_beliefexp():
    return BeliefExpectimaxAgent('BeliefExp', verbose=False)


AGENTS_CONFIG = [make_ppo, make_v3_nn_pc, make_baseline, make_beliefexp]
AGENT_NAMES = ['PPO', 'V3-NN-PC', 'Baseline', 'BeliefExp']


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    print(f'PPO model: {PPO_MODEL}')
    print(f'Running {n_games} games with {workers} workers ...')
    start = time.time()
    results = run_tournament(AGENTS_CONFIG, n_games=n_games, verbose=False,
                             n_workers=workers)
    elapsed = time.time() - start

    with open(f'output/results_rl_ppo_{n_games}.pkl', 'wb') as f:
        pickle.dump(results, f)
    generate_report(results, AGENT_NAMES, output_path='output/rl_ppo_benchmark_report.md')

    metrics = compute_metrics(results, AGENT_NAMES)
    elo = compute_elo(results, AGENT_NAMES)
    print(f'\nTotal time: {elapsed:.1f}s ({elapsed/max(n_games,1):.2f}s per game)')
    print('Quick summary:')
    for name in AGENT_NAMES:
        m = metrics[name]
        print('  {:10s}: win {:.1%}, self {:.1%}, ron {:.1%}, deal-in {:.1%}, '
              'draw {:.1%}, Elo {:.0f}, avg_time {:.1f}ms'.format(
                  name, m['win_rate'], m['self_rate'], m['ron_rate'],
                  m['deal_in_rate'], m['draw_rate'], elo[name],
                  m['avg_decision_time'] * 1000))


if __name__ == '__main__':
    main()
