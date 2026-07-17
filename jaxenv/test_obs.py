# -*- coding: utf-8 -*-
"""观测函数对齐验证（M5）。

非 jit 驱动 JAX 环境到随机中期状态（混合策略覆盖各 phase：碰/杠后、报听后、
CLAIM/TENPAI/DISCARD），对每个决策状态手工构建等价 Python ContextV3 + 手牌，
比较 ``algo/nn/features.py::extract_features`` 与 ``jaxenv/obs.py::observe``：
要求 max abs diff < 1e-6 且状态数 >= 1000。

Python 侧构造规则（与 observe 语义一一对应，含部署 quirk）：
- 手牌 = 闭手 + 每个副露牌种 ×3（full_hand() quirk），CLAIM 时 + offered tile；
- ctx.used / ctx.discards = 四家已入弃牌堆的牌（被声明的牌不在其中）；
- TENPAI 决策时 pending 已进自己 used / discards（PPOAgent.next 先 see_tile）；
- 报听集合 = locked 四座位。

用法：PYTHONPATH=. python3 jaxenv/test_obs.py [--states 1200] [--seed 7]
"""

import argparse
import time

import numpy as np

import jax
import jax.numpy as jnp

from jaxenv import env, obs as obs_mod
from algo.context.v3 import ContextV3
from algo.eval.v3 import _IDX_TO_TILE
from algo.nn.features import extract_features

PHASE_NAMES = {0: 'DISCARD', 1: 'CLAIM', 2: 'TENPAI'}
NAMES = [f'bot@{s}' for s in range(4)]

_step_jit = jax.jit(env.step)
_init_jit = jax.jit(env.init)
_mask_jit = jax.jit(env.legal_mask)
_observe_jit = jax.jit(obs_mod.observe)


@jax.jit
def _shanten_all(after, n_melds):
    from jaxenv import rules
    return jax.vmap(lambda c: rules.shanten_counts(c, n_melds))(after)


def _greedy_discard(hand_counts, n_melds, legal_mask_np):
    """34 个候选弃牌后的向听批量评估，取合法且向听最小者（tie 取下标小）。"""
    hand = jnp.asarray(hand_counts, dtype=jnp.int8)
    after = hand[None, :] - jnp.eye(34, dtype=jnp.int8)
    sh = np.array(_shanten_all(after, jnp.int8(n_melds)), copy=True)
    sh[legal_mask_np[:34] == 0] = 127
    return int(np.argmin(sh))


def choose_action(st, kind, rng):
    """按策略类型为当前决策状态选动作（host 侧）。"""
    phase = int(st.phase)
    mask = np.asarray(_mask_jit(st))
    legal = np.where(mask)[0]
    if phase == env.PHASE_DISCARD:
        if rng.random() < 0.10:
            return int(rng.choice(legal))
        turn = int(st.turn)
        return _greedy_discard(st.hands[turn], int(st.n_melds[turn]), mask)
    if phase == env.PHASE_CLAIM:
        stage = int(st.claim_stage)
        if stage == env.STAGE_HU:
            return 37 if mask[37] else 34
        if kind == 'aggr' or (kind == 'mixed' and rng.random() < 0.5):
            if stage == env.STAGE_GANG and mask[36]:
                return 36
            if stage == env.STAGE_PENG and mask[35]:
                return 35
        return 34
    # TENPAI
    if kind == 'passive_no':
        return 39
    return 38 if rng.random() < 0.9 else 39


def actor_of_np(st):
    if int(st.phase) == env.PHASE_CLAIM:
        return (int(st.turn) + int(st.claim_offset)) % 4
    return int(st.turn)


def build_python_feats(st):
    """从 host-numpy State 手工构建等价 ContextV3 + 手牌，返回 extract_features。"""
    p = actor_of_np(st)
    phase = int(st.phase)
    ctx = ContextV3()
    for s in range(4):
        for t in range(34):
            c = int(st.discards[s][t])
            if c:
                tv = int(_IDX_TO_TILE[t])
                ctx.discards.setdefault(NAMES[s], []).extend([tv] * c)
                ctx.used[tv] = ctx.used.get(tv, 0) + c
    for s in range(4):
        if bool(st.locked[s]):
            ctx.tenpai_players.add(NAMES[s])
    hand = []
    for t in range(34):
        tv = int(_IDX_TO_TILE[t])
        hand.extend([tv] * int(st.hands[p][t]))
        if int(st.meld_counts[p][t]) > 0:
            hand.extend([tv] * 3)
    if phase == env.PHASE_CLAIM:
        hand.append(int(_IDX_TO_TILE[int(st.pending_tile)]))
    if phase == env.PHASE_TENPAI:
        # 部署 quirk：PPOAgent.next 弃牌时已 see_tile 进自己的 ctx
        tv = int(_IDX_TO_TILE[int(st.pending_tile)])
        ctx.used[tv] = ctx.used.get(tv, 0) + 1
        ctx.discards.setdefault(NAMES[p], []).append(tv)
    return extract_features(ctx, hand, NAMES[p])


def collect_states(n_target, seed):
    """混合策略开多局，收集决策状态（host numpy State 列表）。"""
    kinds = ['aggr', 'passive_yes', 'passive_no', 'mixed']
    rng = np.random.default_rng(seed)
    collected = []
    game = 0
    while len(collected) < n_target and game < 200:
        kind = kinds[game % len(kinds)]
        state = _init_jit(jax.random.PRNGKey(seed * 100003 + game))
        game += 1
        if bool(state.done):  # 天胡：无决策点，重开
            continue
        steps = 0
        while not bool(state.done) and steps < 600:
            collected.append(jax.device_get(state))
            if len(collected) >= n_target:
                break
            a = choose_action(state, kind, rng)
            state, _, _ = _step_jit(state, jnp.int8(a))
            steps += 1
    return collected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--states', type=int, default=1200)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    t0 = time.time()
    states = collect_states(args.states, args.seed)
    print(f'collected {len(states)} states in {time.time() - t0:.1f}s')

    counts = {0: 0, 1: 0, 2: 0}
    n_meld_states = 0
    n_locked_states = 0
    max_diff = 0.0
    worst = None
    for st in states:
        phase = int(st.phase)
        counts[phase] += 1
        if int(st.meld_counts.sum()) > 0:
            n_meld_states += 1
        if int(st.locked.sum()) > 0:
            n_locked_states += 1
        f_jax = np.asarray(_observe_jit(st))
        f_py = build_python_feats(st)
        assert f_jax.shape == (175,) and f_py.shape == (175,)
        d = float(np.max(np.abs(f_jax - f_py)))
        if d > max_diff:
            max_diff = d
            worst = (phase, int(np.argmax(np.abs(f_jax - f_py))))

    print(f'phase counts: ' + ', '.join(f'{PHASE_NAMES[k]}={v}' for k, v in counts.items()))
    print(f'states with melds: {n_meld_states}, with locked: {n_locked_states}')
    print(f'max abs diff over {len(states)} states: {max_diff:.3e} (worst={worst})')

    # vmap / jit 兼容性
    import jaxenv.env as _e
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    batch = jax.vmap(_init_jit)(keys)
    ob = jax.vmap(obs_mod.observe)(batch)
    assert ob.shape == (4, 175), ob.shape
    ob2 = jax.jit(jax.vmap(obs_mod.observe))(batch)
    assert np.allclose(np.asarray(ob), np.asarray(ob2))
    print('vmap/jit compatibility OK')

    assert len(states) >= 1000, f'need >=1000 states, got {len(states)}'
    assert counts[0] >= 400, f'DISCARD coverage too low: {counts[0]}'
    assert counts[1] >= 100, f'CLAIM coverage too low: {counts[1]}'
    assert counts[2] >= 10, f'TENPAI coverage too low: {counts[2]}'
    assert n_meld_states >= 50, f'meld coverage too low: {n_meld_states}'
    assert n_locked_states >= 10, f'locked coverage too low: {n_locked_states}'
    assert max_diff < 1e-6, f'max diff {max_diff} >= 1e-6'
    print('ALL OBS ALIGNMENT TESTS PASSED')


if __name__ == '__main__':
    main()
