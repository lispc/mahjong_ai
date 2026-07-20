# -*- coding: utf-8 -*-
"""jaxenv/eval2jax.py 验证：eval0/eval2 弃牌 parity（vs arena 实际 Cython 路径）
+ 4×eval2 自对弈 env smoke。

- test_eval0_parity：随机 N 个 13/14/15 张手牌（136 牌山无放回采样），
  jax eval0_counts vs algo.eval0（Cython _fast_eval0，pair_coef=1.0 快路径）
  逐值相等。15 张组覆盖 eval2 内层可达的 5 面子路径（tables.npz g<=4 不够用的
  原因，见 gen_eval2_tables.py）。
- test_discard_parity：随机 N 个 14 张手牌，jax eval2_discard_idx vs
  algo.select(hand, False)[0] 弃牌选择。JAX 侧用整数分子精确比较 + idx 降序
  tie-break；Cython 用 float64 顺序求和，数学严格平局时 ±1ulp 可打破
  (metric, tile) 降序规则 —— 这类样本单独统计为 tie-split（选择不同但整数
  分子相等），不算 bug；无整数平局的真正不一致 assert 为 0。
- test_env_smoke：eager 数步（动作恒合法）+ 逐步 jit 跑 n 局 4×eval2 自对弈，
  对局正常结束；全程 n_melds==0、locked 全 False（eval2 不碰不杠不报听的不变式）。

用法：PYTHONPATH=. python3 jaxenv/test_eval2jax.py \
    [--eval0-n 2000] [--sel-n 1000] [--games 32] [--seed 7]
"""

import argparse
import time

import numpy as np

import jax
import jax.numpy as jnp

from jaxenv import env, rules
from jaxenv.eval2jax import (eval0_counts, eval2_discard_idx, eval2_action,
                             _discard_scores)

TILE_IDS = rules.TILE_IDS
TILE_TO_IDX = rules.TILE_TO_IDX


def _random_hands(rng, n, size):
    """从 136 牌山无放回采 n 个 size 张手牌 -> counts (n,34) int8 + tile id 列表。"""
    counts = np.zeros((n, 34), np.int8)
    hands = []
    for i in range(n):
        wall = rng.permutation(136) // 4
        draw = wall[:size]
        for t in draw:
            counts[i, t] += 1
        hands.append([TILE_IDS[t] for t in draw])
    return counts, hands


def test_eval0_parity(n, seed):
    import algo
    from algo.eval import _fast_eval0  # noqa: F401 -- 断言 Cython 快路径可用
    rng = np.random.default_rng(seed)
    total_bad = 0
    for size in (13, 14, 15):
        counts, hands = _random_hands(rng, n, size)
        jx = np.asarray(jax.jit(jax.vmap(eval0_counts))(jnp.asarray(counts)))
        py = np.array([algo.eval0(h) for h in hands])
        bad = int((jx != py).sum())
        total_bad += bad
        print(f'[eval0-parity] {size} 张: {n - bad}/{n} 完全一致 '
              f'(mismatch={bad})', flush=True)
        if bad:
            i = int(np.nonzero(jx != py)[0][0])
            print('  首个 mismatch: hand=', sorted(hands[i]),
                  'jax=', jx[i], 'py=', py[i], flush=True)
    assert total_bad == 0, f'eval0 parity 失败: {total_bad} 个不一致'
    print(f'[eval0-parity] passed ({3 * n} hands, exact match)', flush=True)


def test_discard_parity(n, seed):
    import algo
    rng = np.random.default_rng(seed + 1)
    counts, hands = _random_hands(rng, n, 14)
    counts_j = jnp.asarray(counts)
    jx_choice = np.asarray(jax.jit(jax.vmap(eval2_discard_idx))(counts_j))
    jx_scores = np.asarray(jax.jit(jax.vmap(_discard_scores))(counts_j))
    py_choice = np.array([TILE_TO_IDX[algo.select(h, False)[0]] for h in hands])

    same = int((jx_choice == py_choice).sum())
    tie_split = 0
    bad = 0
    for i in np.nonzero(jx_choice != py_choice)[0]:
        # Cython 选的牌与 JAX 选的牌整数分子相同 => 数学平局被 float 求和顺序打破
        if jx_scores[i, py_choice[i]] == jx_scores[i, jx_choice[i]]:
            tie_split += 1
        else:
            bad += 1
            if bad <= 3:
                print('  真 mismatch: hand=', sorted(hands[i]),
                      'jax=', TILE_IDS[jx_choice[i]], 'py=', TILE_IDS[py_choice[i]],
                      'N_jax=', int(jx_scores[i, jx_choice[i]]),
                      'N_py=', int(jx_scores[i, py_choice[i]]), flush=True)
    print(f'[discard-parity] {n} hands: 一致 {same} ({100 * same / n:.2f}%), '
          f'tie-split {tie_split}, 真不一致 {bad}', flush=True)
    assert bad == 0, f'弃牌 parity 失败: {bad} 个非平局不一致'
    print('[discard-parity] passed', flush=True)


def test_env_smoke(n=32, seed=99):
    # eager 数步：eval2_action 不依赖 jit、动作恒合法
    keys = jax.random.split(jax.random.PRNGKey(seed), 2)
    states = jax.vmap(env.init)(keys)
    for step in range(4):
        acts = jax.vmap(eval2_action)(states)               # eager（非 jit）
        masks = jax.vmap(env.legal_mask)(states)
        a, m, d = np.asarray(acts), np.asarray(masks), np.asarray(states.done)
        assert m[np.arange(2), a][~d].all(), f'eager: illegal action at step {step}'
        states, _, _ = jax.vmap(env.step)(states, acts)
    print('[env-smoke] eager 4 steps: all actions legal', flush=True)

    # 逐步 jit 跑完整局
    act_v = jax.jit(jax.vmap(eval2_action))
    step_v = jax.jit(jax.vmap(env.step))
    keys = jax.random.split(jax.random.PRNGKey(seed + 1), n)
    states = jax.vmap(env.init)(keys)
    steps = 0
    t0 = time.time()
    while not bool(jnp.all(states.done)):
        states, _, _ = step_v(states, act_v(states))
        steps += 1
        if steps > 1000:
            raise AssertionError('games did not finish in 1000 steps')
    dt = time.time() - t0
    max_melds = int(np.asarray(states.n_melds).max())
    locked_any = bool(np.asarray(states.locked).any())
    wt = np.asarray(states.win_type)
    print(f'[env-smoke] {n} games x {steps} steps jit: '
          f'win_type self/ron/draw = {(wt == 1).sum()}/{(wt == 2).sum()}/{(wt == 3).sum()}, '
          f'n_draws_mean={np.asarray(states.n_draws).mean():.1f}, '
          f'max_melds={max_melds}, locked_any={locked_any}, '
          f'{dt:.1f}s ({n * steps / max(dt, 1e-9):.0f} state-steps/s incl. compile)',
          flush=True)
    assert max_melds == 0 and not locked_any, 'eval2 不变式被破坏（碰杠/报听）'
    print('[env-smoke] passed', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval0-n', type=int, default=2000)
    ap.add_argument('--sel-n', type=int, default=1000)
    ap.add_argument('--games', type=int, default=32)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    t0 = time.time()
    test_eval0_parity(args.eval0_n, args.seed)
    print(f'  eval0 parity done ({time.time() - t0:.1f}s)', flush=True)
    t0 = time.time()
    test_discard_parity(args.sel_n, args.seed)
    print(f'  discard parity done ({time.time() - t0:.1f}s)', flush=True)
    test_env_smoke(args.games, args.seed + 2)
    print('[test_eval2jax] all passed', flush=True)


if __name__ == '__main__':
    main()
