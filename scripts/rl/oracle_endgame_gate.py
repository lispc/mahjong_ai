# -*- coding: utf-8 -*-
"""Oracle gate 实验（方向 B 的止损门）。

问题：把 exact endgame solver 接入 BeliefExp 是否值得继续投入？
本实验给 BeliefEndgameAgent 换上**完美待牌**（通过同进程 agent 引用直接读
报听对手的真实手牌，仅限已报听者——这是部署时可得信息的边界），
与 BeliefExp 跑 paired duplicate。

判读：
- 若完美待牌相对 BeliefExp 都没有显著提升 → 瓶颈不在待牌预测质量，
  方向 B（训练更好的 wait_dist 模型）直接止损；
- 若完美待牌有显著提升 → 提升幅度就是 wait 预测的上界，值得投入
  训练部署匹配的 3 家待牌模型。

用法：
    PYTHONPATH=. python3 scripts/rl/oracle_endgame_gate.py \
        --n-seeds 2000 --workers 24 --output output/oracle_endgame_gate_2000.pkl
"""

import argparse
import math
import os
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from driver import engine
from algo.agents.belief_endgame_agent import BeliefEndgameAgent, _seat
from algo.eval.v2 import winning_tiles
from scripts.rl import benchmark_pool


class OracleBEEndAgent(BeliefEndgameAgent):
    """BeliefEndgameAgent，但用真实手牌给出已报听对手的精确待牌集合。

    `_table` 由 runner 在同进程内注入（list of agents）。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._table = None

    def _predicted_waits_for_opponents(self):
        if not self._table:
            return []
        self_seat = _seat(self.name)
        declared = self.context.tenpai_players - {self.name}
        out = []
        for opp in self._table:
            if opp.name == self.name or opp.name not in declared:
                continue
            if len(opp.cur) != 13:
                continue
            opp_seat = _seat(opp.name)
            rel = (opp_seat - self_seat) % 4
            if rel == 0:
                continue
            waits = set(winning_tiles(list(opp.cur), None))
            if waits:
                out.append((rel, waits))
        return out


class _Task:
    def __init__(self, seed, kind, position):
        self.seed = seed
        self.kind = kind          # 'a' = oracle, 'b' = beliefexp
        self.position = position


def _oracle_factory():
    return OracleBEEndAgent('OracleEnd', verbose=False)


def _play(task, opponent_tokens):
    opp_factories = [benchmark_pool._make_factory(t)[0] for t in opponent_tokens]
    factories = [None] * 4
    if task.kind == 'a':
        factories[task.position] = _oracle_factory
    else:
        factories[task.position] = benchmark_pool._make_factory('beliefexp')[0]
    oi = 0
    for i in range(4):
        if factories[i] is None:
            factories[i] = opp_factories[oi]
            oi += 1
    random.seed(task.seed)
    agents = [f() for f in factories]
    for a in agents:
        if isinstance(a, OracleBEEndAgent):
            a._table = agents
    for i, a in enumerate(agents):
        a.name = '{}@{}_{}'.format(a.name, i, task.kind)
    return engine.play_game(agents, seed=task.seed, record_time=True)


def _paired_ci(a_only, b_only, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    diff = (a_only - b_only) / n
    var = max((a_only + b_only) / n - diff ** 2, 0.0)
    se = math.sqrt(var / n)
    return diff, diff - z * se, diff + z * se


def main():
    import torch
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-seeds', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--opponents', default='baseline,v3nnpc,hybrid:Base:output/nn_full_action_best.pt')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    opp_tokens = [t.strip() for t in args.opponents.split(',')]
    assert len(opp_tokens) == 3

    tasks = []
    for s in range(args.n_seeds):
        tasks.append(_Task(args.seed_offset + s, 'a', 0))
        tasks.append(_Task(args.seed_offset + s, 'b', 0))

    print(f'Oracle endgame gate: OracleBEEnd vs BeliefExp, opponents={opp_tokens}')
    print(f'{args.n_seeds} pairs, {len(tasks)} games, workers={args.workers}')
    t0 = time.time()
    if args.workers <= 1:
        results = [_play(t, opp_tokens) for t in tasks]
    else:
        results = [None] * len(tasks)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_play, t, opp_tokens): i for i, t in enumerate(tasks)}
            for f in futs:
                results[futs[f]] = f.result()
    dt = time.time() - t0

    a_only = b_only = both = neither = 0
    a_m = b_m = 0
    n = 0
    for i in range(0, len(results) - 1, 2):
        ra, rb = results[i], results[i + 1]
        ca, cb = ra['players_order'][0], rb['players_order'][0]
        if not (ca.startswith('OracleEnd') and cb.startswith('BeliefExp')):
            continue
        n += 1
        aw = ra.get('winner') == ca
        bw = rb.get('winner') == cb
        a_m += aw
        b_m += bw
        if aw and bw:
            both += 1
        elif aw:
            a_only += 1
        elif bw:
            b_only += 1
        else:
            neither += 1

    diff, lo, hi = _paired_ci(a_only, b_only, n)
    print(f'\nTotal {dt:.1f}s, pairs={n}')
    print(f'marginal: Oracle {a_m/n:.1%}  BeliefExp {b_m/n:.1%}')
    print(f'A-only {a_only}  B-only {b_only}  both {both}  neither {neither}')
    print(f'Paired (Oracle - BeliefExp) = {diff:+.1%} [{lo:+.1%}, {hi:+.1%}]')
    if lo > 0:
        print('=> 完美待牌显著更强：wait 预测值得投入（上界即此幅度）')
    elif hi < 0:
        print('=> 完美待牌反而更弱（异常，需查 solver 逻辑）')
    else:
        print('=> 完美待牌也无显著提升：方向 B 止损，瓶颈不在待牌预测')

    if args.output:
        with open(args.output, 'wb') as f:
            pickle.dump({'args': vars(args), 'results': results,
                         'paired': {'n_pairs': n, 'a_only': a_only, 'b_only': b_only,
                                    'both': both, 'neither': neither,
                                    'diff': diff, 'ci_lo': lo, 'ci_hi': hi}}, f)
        print(f'Raw results saved to {args.output}')


if __name__ == '__main__':
    main()
