# -*- coding: utf-8 -*-
"""jaxenv/duplicate_arena.py 验证（G1 门：deal identity + determinism + smoke）。

**独立运行，不加入 run_tests.py**：
    PYTHONPATH=. python3 jaxenv/test_duplicate_arena.py [--seed 7] [--det-seeds 64] [--smoke-seeds 128]

- test_deal_identity：若干 seed，同一 pair 的两条 lane（2k / 2k+1）init 后
  牌墙、4 家起手、摸牌指针逐 bit 相同；不同 seed 的牌墙不同（防 key 串线）。
- test_determinism：64-seed baseline vs beliefexp 完整跑两遍，终局
  winner/win_type/dealer 逐 bit 一致、paired 统计完全一致（全 argmax/纯函数
  agent，无采样 => 必须严格确定）。
- test_smoke：128 seeds baseline vs beliefexp（对手 baseline,beliefexp,beliefexp）
  跑通；打印统计；健全性检查（胜率 ∈ [0,1]、局数 = 2×seeds、
  配对计数 a_wins+b_wins+ties == n_pairs、胡牌局 winner 非空等）。
"""

import argparse
import time

import numpy as np

import jax
import jax.numpy as jnp

from jaxenv import env
from jaxenv import duplicate_arena as dup

SETUP = dict(type_a=dup.TYPE_EVAL2, type_b=dup.TYPE_BELIEF,
             opp_types=(dup.TYPE_EVAL2, dup.TYPE_BELIEF, dup.TYPE_BELIEF),
             a_name='Baseline', b_name='BeliefExp',
             opp_names=('Baseline', 'BeliefExp', 'BeliefExp'))


def test_deal_identity(n_seeds=8, seed_offset=100):
    keys = dup._pair_keys(n_seeds, seed_offset)
    assert keys.shape == (2 * n_seeds, 2)
    states = jax.vmap(env.init)(keys)
    wall = np.asarray(states.wall)
    hands = np.asarray(states.hands)
    head = np.asarray(states.wall_head)
    tail = np.asarray(states.wall_tail)
    drawn = np.asarray(states.drawn)
    for k in range(n_seeds):
        a, b = 2 * k, 2 * k + 1
        np.testing.assert_array_equal(wall[a], wall[b])
        np.testing.assert_array_equal(hands[a], hands[b])
        assert head[a] == head[b] and tail[a] == tail[b] and drawn[a] == drawn[b]
        # 每家起手 13 张、座位 0 多摸 1 张
        assert hands[a].sum() == 53 and hands[a, 0].sum() == 14
    for k in range(n_seeds - 1):
        assert not np.array_equal(wall[2 * k], wall[2 * (k + 1)]), \
            f'seed {k} 与 {k + 1} 牌墙相同（key 串线？）'
    print(f'[deal-identity] {n_seeds} pairs: lanes 2k/2k+1 wall+hands identical, '
          f'distinct across seeds: OK')


def _run(n_seeds, seed_offset):
    return dup.run_duplicate(SETUP['type_a'], SETUP['type_b'], SETUP['opp_types'],
                             n_seeds, seed_offset=seed_offset,
                             a_name=SETUP['a_name'], b_name=SETUP['b_name'],
                             opp_names=SETUP['opp_names'])


def test_determinism(n_seeds=64, seed_offset=7):
    t0 = time.time()
    r1 = _run(n_seeds, seed_offset)
    r2 = _run(n_seeds, seed_offset)
    for key in ('winner', 'win_type', 'dealer', 'done'):
        np.testing.assert_array_equal(r1[key], r2[key])
    assert r1['paired'] == r2['paired'], (r1['paired'], r2['paired'])
    print(f'[determinism] {n_seeds} seeds x2 runs: terminal arrays bitwise '
          f'identical, paired stats equal ({time.time() - t0:.1f}s): OK')
    return r1


def _check_sanity(out, n_seeds):
    res = out['results']
    assert len(res) == 2 * n_seeds
    p = out['paired']
    assert p['n_pairs'] == n_seeds
    assert p['a_wins'] + p['b_wins'] + p['ties'] == n_seeds
    decisive = 0
    for r in res:
        wt = r['win_type']
        if wt == 'draw':
            assert r['winner'] is None
        else:
            assert r['winner'] in r['players_order']
            decisive += 1
        if wt == 'ron':
            assert r['dealer'] in r['players_order'] and r['dealer'] != r['winner']
    assert 0.0 <= p['diff'] <= 1.0 or -1.0 <= p['diff'] <= 0.0
    for kind in ('a', 'b'):
        c = out['candidate'][kind]
        assert c['games'] == n_seeds
        for k in ('win_rate', 'self_rate', 'ron_rate', 'deal_in_rate', 'draw_rate'):
            assert 0.0 <= c[k] <= 1.0, (kind, k, c[k])
        assert abs(c['self_rate'] + c['ron_rate'] - c['win_rate']) < 1e-9
    return decisive


def test_smoke(n_seeds=128, seed_offset=7):
    t0 = time.time()
    out = _run(n_seeds, seed_offset)
    dt = time.time() - t0
    decisive = _check_sanity(out, n_seeds)
    p = out['paired']
    ca, cb = out['candidate']['a'], out['candidate']['b']
    print(f'[smoke] {n_seeds} seeds ({2 * n_seeds} games) in {dt:.1f}s '
          f'({n_seeds / dt:.1f} seeds/s): OK')
    print(f'  decisive {decisive}/{2 * n_seeds} '
          f'(draw rate {(2 * n_seeds - decisive) / (2 * n_seeds):.1%})')
    print(f'  Baseline seat: win {ca["win_rate"]:.1%} '
          f'(self {ca["self_rate"]:.1%} ron {ca["ron_rate"]:.1%} '
          f'dealin {ca["deal_in_rate"]:.1%} draw {ca["draw_rate"]:.1%})')
    print(f'  BeliefExp seat: win {cb["win_rate"]:.1%} '
          f'(self {cb["self_rate"]:.1%} ron {cb["ron_rate"]:.1%} '
          f'dealin {cb["deal_in_rate"]:.1%} draw {cb["draw_rate"]:.1%})')
    print(f'  paired: A {p["a_wins"]} B {p["b_wins"]} ties {p["ties"]}, '
          f'A-B = {p["diff"]:+.1%} [{p["ci_lo"]:+.1%}, {p["ci_hi"]:+.1%}], '
          f'score {p["score_diff"]:+.3f} '
          f'[{p["score_ci_lo"]:+.3f}, {p["score_ci_hi"]:+.3f}]')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--det-seeds', type=int, default=64)
    ap.add_argument('--smoke-seeds', type=int, default=128)
    args = ap.parse_args()

    dup.enable_compile_cache()
    test_deal_identity(seed_offset=args.seed)
    test_determinism(args.det_seeds, seed_offset=args.seed)
    test_smoke(args.smoke_seeds, seed_offset=args.seed)
    print('ALL DUPLICATE-ARENA TESTS PASSED')


if __name__ == '__main__':
    main()
