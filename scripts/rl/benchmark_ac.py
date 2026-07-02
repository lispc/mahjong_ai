# -*- coding: utf-8 -*-
"""在同一 tournament 里直接比较多个 PPO 模型 + 强参照，避免跨 benchmark 的 Elo 漂移。

用法：PYTHONPATH=. python3 scripts/rl/benchmark_ac.py [n_games] [workers]
座位：PPO-A / PPO-C / Baseline / BeliefExp（各 1，座位随机洗牌）。
模型路径由环境变量 PPO_A / PPO_C 指定（默认 nn_rl_ppo_A.pt / nn_rl_ppo_C.pt）。
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent
from algo.agents.belief_expectimax import BeliefExpectimaxAgent
from algo.agents.ppo_agent import PPOAgent
from driver.tournament import run_tournament
from checker.report import compute_metrics, compute_elo

PPO_A = os.environ.get('PPO_A', 'output/nn_rl_ppo_A.pt')
PPO_C = os.environ.get('PPO_C', 'output/nn_rl_ppo_C.pt')


def make_a():
    return PPOAgent('PPO-A', model_path=PPO_A, device='cpu', temperature=0.0)


def make_c():
    return PPOAgent('PPO-C', model_path=PPO_C, device='cpu', temperature=0.0)


def make_baseline():
    return agent.Agent('Baseline', verbose=False)


def make_beliefexp():
    return BeliefExpectimaxAgent('BeliefExp', verbose=False)


AGENTS = [make_a, make_c, make_baseline, make_beliefexp]
NAMES = ['PPO-A', 'PPO-C', 'Baseline', 'BeliefExp']


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    print(f'A={PPO_A}  C={PPO_C}')
    print(f'Running {n_games} games with {workers} workers ...')
    t0 = time.time()
    results = run_tournament(AGENTS, n_games=n_games, verbose=False, n_workers=workers)
    dt = time.time() - t0
    metrics = compute_metrics(results, NAMES)
    elo = compute_elo(results, NAMES)
    print(f'\nTotal {dt:.1f}s')
    for name in NAMES:
        m = metrics[name]
        print('  {:10s}: win {:.1%}, self {:.1%}, ron {:.1%}, deal-in {:.1%}, '
              'draw {:.1%}, Elo {:.0f}'.format(
                  name, m['win_rate'], m['self_rate'], m['ron_rate'],
                  m['deal_in_rate'], m['draw_rate'], elo[name]))


if __name__ == '__main__':
    main()
