# -*- coding: utf-8 -*-
"""方向 0：全局 god-mode 上界测量（完美隐藏手牌信息 vs 当前 best）。

问题：当前 best（Hybrid-FullAction-SoupDistilled）距离「信息上限」还有多远？
此前只测过局部上界：终盘已报听防守 oracle（=0，`oracle_endgame_gate.py`）、
碰的配对因果效应（+0.117）。本实验测**全程完美隐藏手牌信息**的上界：

GodBeliefAgent = BeliefExpectimaxAgent + 两项完美信息升级：
1. **精确剩余分布**：把三家对手的闭手与副露计入 eval0/eval2 的 `used`，
   进攻分的摸牌概率从「信念均匀」变成「精确」（对手手牌不在牌山里）。
2. **精确点炮规避**：每张候选弃牌直接调用各对手自己的 `respond_hu`
   （与引擎裁决完全一致，含 Hybrid 对手的 NN 响应头），非点和候选中取
   进攻最大；全部点和（被迫）时取进攻最大。

同 seed 三局配对（duplicate 格式，pos 0 分别为 God / BeliefExp / Hybrid-Best），
对手 = 标准三件套 baseline,beliefexp,hybrid:Base。配对差给出：
- God − BeliefExp：同一搜索结构下**纯信息价值**；
- God − Hybrid：相对当前 best 的**剩余头部空间**（上界参考）；
- BeliefExp − Hybrid：sanity check（应 ≈ −7pp，见 eval-protocol §5）。

用法：
    PYTHONPATH=. python3 scripts/rl/god_mode_upper_bound.py \
        --n-seeds 2000 --workers 32 --output output/god_mode_ub_2000.pkl
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

import algo
import context as ctx_module
from driver import engine
from algo.agents.belief_expectimax import BeliefExpectimaxAgent
from scripts.rl import benchmark_pool


class GodBeliefAgent(BeliefExpectimaxAgent):
    """BeliefExp + 完美隐藏手牌信息（见模块 docstring）。

    `_table` 由 runner 在同进程内注入（list of agents）。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._table = None

    def _opponents(self):
        if not self._table:
            return []
        return [a for a in self._table if a is not self]

    def _legacy_context(self):
        """god 版：used 额外计入对手闭手与副露 → 剩余分布精确。"""
        c = ctx_module.Context()
        c.used = self.context.used.copy()
        for opp in self._opponents():
            for t in opp.cur:
                c.used[t] = c.used.get(t, 0) + 1
            for _, t in opp.melds:
                c.used[t] = c.used.get(t, 0) + 1
        return c

    def _god_ron(self, disc):
        """精确点和判断：任一对手实际会和这张牌（与引擎裁决一致）。"""
        for opp in self._opponents():
            if opp.respond_hu(disc, getattr(opp, 'context', None)):
                return True
        return False

    def next_with_trace(self):
        assert len(self.cur) == 14

        type_ctx = self._legacy_context()
        candidates = self._unique_tiles(self.cur)

        scored = []
        for disc in candidates:
            hand13 = self._remove_one(self.cur, disc)
            score = algo.eval0(hand13, type_ctx)
            scored.append((score, disc))
        scored.sort(reverse=True)
        top = [disc for _, disc in scored[:self.max_candidates]]

        evaluated = []
        score_map = {}
        ron_map = {}
        for disc in top:
            hand13 = self._remove_one(self.cur, disc)
            offense = self._eval2(hand13)
            ron = self._god_ron(disc)
            evaluated.append((offense, ron, disc))
            score_map[disc] = float(offense)
            ron_map[disc] = bool(ron)

        safe = [item for item in evaluated if not item[1]]
        pool = safe if safe else evaluated
        pool.sort(reverse=True, key=lambda x: x[0])
        result = pool[0][2]

        trace = {
            'candidates': list(top),
            'scores': score_map,
            'god_ron': ron_map,
            'n_safe': len(safe),
            'selected_value': float(score_map.get(result, 0.0)),
        }

        self.cur.remove(result)
        self.context.see_tile(result, self.name)
        return result, trace


class _Task:
    def __init__(self, seed, kind, position=0):
        self.seed = seed
        self.kind = kind          # 'g' = god, 'b' = beliefexp, 'h' = hybrid-best
        self.position = position


def _god_factory():
    return GodBeliefAgent('GodBelief', verbose=False)


def _play(task, opponent_tokens, hybrid_token):
    opp_factories = [benchmark_pool._make_factory(t)[0] for t in opponent_tokens]
    if task.kind == 'g':
        cand_factory = _god_factory
    elif task.kind == 'b':
        cand_factory = benchmark_pool._make_factory('beliefexp')[0]
    else:
        cand_factory = benchmark_pool._make_factory(hybrid_token)[0]
    factories = [None] * 4
    factories[task.position] = cand_factory
    oi = 0
    for i in range(4):
        if factories[i] is None:
            factories[i] = opp_factories[oi]
            oi += 1
    random.seed(task.seed)
    agents = [f() for f in factories]
    for a in agents:
        if isinstance(a, GodBeliefAgent):
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


def _score_proxy(r):
    """候选席位（pos 0）的推倒胡计分代理：自摸+3 / 点和+1 / 放炮−1 / 其他0。"""
    cand = r['players_order'][0]
    if r.get('winner') == cand:
        return 3.0 if r.get('win_type') == 'self' else 1.0
    if r.get('dealer') == cand:
        return -1.0
    return 0.0


def _paired_win_stats(wins_a, wins_b):
    """wins_a/wins_b: 按 pair 对齐的 0/1 列表。返回 (n, a_only, b_only, both, neither)。"""
    a_only = b_only = both = neither = 0
    n = 0
    for wa, wb in zip(wins_a, wins_b):
        n += 1
        if wa and wb:
            both += 1
        elif wa:
            a_only += 1
        elif wb:
            b_only += 1
        else:
            neither += 1
    return n, a_only, b_only, both, neither


def _paired_mean_ci(xs, ys, z=1.96):
    diffs = [x - y for x, y in zip(xs, ys)]
    n = len(diffs)
    if n == 0:
        return 0.0, 0.0, 0.0
    m = sum(diffs) / n
    var = sum((d - m) ** 2 for d in diffs) / max(n - 1, 1)
    se = math.sqrt(var / n)
    return m, m - z * se, m + z * se


def main():
    import torch
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-seeds', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--opponents',
                        default='baseline,beliefexp,hybrid:Base:output/nn_full_action_best.pt')
    parser.add_argument('--hybrid-token',
                        default='hybrid:Best:output/nn_full_action_best.pt')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    opp_tokens = [t.strip() for t in args.opponents.split(',')]
    assert len(opp_tokens) == 3

    tasks = []
    for s in range(args.n_seeds):
        seed = args.seed_offset + s
        tasks.append(_Task(seed, 'g'))
        tasks.append(_Task(seed, 'b'))
        tasks.append(_Task(seed, 'h'))

    print(f'God-mode upper bound: GodBelief vs BeliefExp vs {args.hybrid_token}')
    print(f'opponents={opp_tokens}')
    print(f'{args.n_seeds} seeds x3 = {len(tasks)} games, workers={args.workers}')
    t0 = time.time()
    if args.workers <= 1:
        results = [_play(t, opp_tokens, args.hybrid_token) for t in tasks]
    else:
        results = [None] * len(tasks)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_play, t, opp_tokens, args.hybrid_token): i
                    for i, t in enumerate(tasks)}
            for f in futs:
                results[futs[f]] = f.result()
    dt = time.time() - t0

    # 按 seed 对齐三路结果
    wins = {'g': [], 'b': [], 'h': []}
    scores = {'g': [], 'b': [], 'h': []}
    dealins = {'g': [], 'b': [], 'h': []}
    bad = 0
    for i in range(0, len(results) - 2, 3):
        rg, rb, rh = results[i], results[i + 1], results[i + 2]
        cg, cb, ch = rg['players_order'][0], rb['players_order'][0], rh['players_order'][0]
        if not (cg.startswith('GodBelief') and cb.startswith('BeliefExp')
                and ch.startswith('Hybrid')):
            bad += 1
            continue
        for kind, r, c in (('g', rg, cg), ('b', rb, cb), ('h', rh, ch)):
            wins[kind].append(r.get('winner') == c)
            scores[kind].append(_score_proxy(r))
            dealins[kind].append(r.get('dealer') == c and r.get('win_type') == 'ron')
    n = len(wins['g'])
    print(f'\nTotal {dt:.1f}s, pairs={n}' + (f'  (bad triples skipped: {bad})' if bad else ''))

    print('\n== marginal（候选席位胜率 / 点炮率 / 平均 score-proxy）==')
    for kind, label in (('g', 'GodBelief'), ('b', 'BeliefExp'), ('h', 'Hybrid-Best')):
        wr = sum(wins[kind]) / n
        dr = sum(dealins[kind]) / n
        sp = sum(scores[kind]) / n
        print(f'  {label:12s} win {wr:.1%}  deal-in {dr:.1%}  score-proxy {sp:+.3f}')

    print('\n== paired win diff（95% CI）==')
    paired = {}
    for a, b, label in (('g', 'b', 'God − BeliefExp'),
                        ('g', 'h', 'God − Hybrid'),
                        ('b', 'h', 'BeliefExp − Hybrid (sanity)')):
        nn, ao, bo, both, neither = _paired_win_stats(wins[a], wins[b])
        diff, lo, hi = _paired_ci(ao, bo, nn)
        paired[label] = {'n': nn, 'a_only': ao, 'b_only': bo,
                         'both': both, 'neither': neither,
                         'diff': diff, 'ci_lo': lo, 'ci_hi': hi}
        print(f'  {label:32s} {diff:+.1%} [{lo:+.1%}, {hi:+.1%}]'
              f'  (a_only {ao}, b_only {bo}, ties {both + neither})')

    print('\n== paired score-proxy diff（95% CI）==')
    for a, b, label in (('g', 'b', 'God − BeliefExp'),
                        ('g', 'h', 'God − Hybrid'),
                        ('b', 'h', 'BeliefExp − Hybrid (sanity)')):
        m, lo, hi = _paired_mean_ci(scores[a], scores[b])
        paired[label]['score_diff'] = m
        paired[label]['score_ci'] = (lo, hi)
        print(f'  {label:32s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]')

    print('\n== 决策耗时（秒/决策，均值）==')
    times = {}
    for r in results:
        if r and 'decision_times' in r:
            for name, ts in r['decision_times'].items():
                times.setdefault(name.split('@')[0], []).extend(ts)
    for name, ts in sorted(times.items()):
        print(f'  {name:20s} {sum(ts)/len(ts)*1000:.1f} ms  (n={len(ts)})')

    if args.output:
        with open(args.output, 'wb') as f:
            pickle.dump({'args': vars(args), 'results': results, 'paired': paired}, f)
        print(f'\nRaw results saved to {args.output}')


if __name__ == '__main__':
    main()
