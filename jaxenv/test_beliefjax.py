# -*- coding: utf-8 -*-
"""jaxenv/beliefjax.py 验证：danger 逐值 parity + top-1 决策 parity + env smoke。

Ground truth = arena 实际代码路径（BeliefExpectimaxAgent 默认参数 + Cython
_fast_eval0 快路径）。采样自 BeliefExp 自对弈真实对局的决策点快照
（engine.play_game 的 state_callback，深拷贝 cur/context）。

- test_danger_parity：每个快照状态，jax tile_danger_vec(34 维) vs
  opponent.tile_danger（对每个合法牌）逐值比较，容差 1e-9（本测试开 x64，
  jax 侧为 float64；训练态 x64 关时为 float32，见 beliefjax.py 注记）。
- test_decision_parity：jax belief_action(DISCARD) vs next_with_trace()[0]。
  分类统计：一致 / offense tie-split（整数分子相等的数学平局，Cython float64
  求和序噪声打破 tile 降序，同 eval2jax 标准）/ danger tie-split（危险分支
  双方 danger 差 <1e-9 且均在 safe 集）/ 真不一致（assert 0）。
  另校验 top-8 候选内 jax 进攻分 N/15006 vs trace['scores'] 容差 1e-9。
- test_tenpai_parity：对满足引擎报听前置（弃后 shanten==0）的快照，
  jax TENPAI 分支 vs BeliefExpectimaxAgent.declare_tenpai（context 为
  next_with_trace see_tile 后的版本）逐一相等。
- test_env_smoke：eager 数步 + jit n 局 4×belief 自对弈，动作恒合法、对局
  正常结束、n_melds 恒 0（locked 允许，belief 会按启发式报听）。

用法：PYTHONPATH=. python3 jaxenv/test_beliefjax.py [--games 30] [--seed 7]
"""

import argparse
import time

import numpy as np

import jax
jax.config.update('jax_enable_x64', True)                   # danger 1e-9 需要 f64
import jax.numpy as jnp

from jaxenv import env, rules
from jaxenv import beliefjax
from jaxenv.beliefjax import belief_action, tile_danger_vec
from jaxenv.eval2jax import _eval2_num13, _eval0_int

TILE_IDS = rules.TILE_IDS
TILE_TO_IDX = rules.TILE_TO_IDX


# ---------------------------------------------------------------------------
# 快照采集与 Python 侧决策复算
# ---------------------------------------------------------------------------

def collect_snapshots(n_games, seed):
    from driver import engine
    from algo.agents.belief_expectimax import BeliefExpectimaxAgent
    snaps = []
    for g in range(n_games):
        agents = [BeliefExpectimaxAgent(f'p{k}', verbose=False) for k in range(4)]

        def cb(agents, turn, phase, kw):
            if phase != 'decision':
                return
            a = agents[turn]
            if a.name in kw['locked_names']:
                return                                        # 锁手无 next_with_trace
            snaps.append({
                'turn': turn,
                'hand': list(a.cur),
                'context': a.context.copy(),
                'hands': [list(x.cur) for x in agents],
                'locked': [x.name in kw['locked_names'] for x in agents],
                'drawn': kw['drawn'],
            })

        engine.play_game(agents, seed=seed * 100003 + g, state_callback=cb)
        if (g + 1) % 10 == 0:
            print(f'  collect {g + 1}/{n_games} games, {len(snaps)} snaps', flush=True)
    return snaps


def py_decide(snap):
    """用快照重建 agent 并重放决策 -> (tile, trace, hand13, tenpai_yes_or_None)。"""
    import algo.eval.opponent as opponent
    import algo.eval.v2 as eval_v2
    from algo.agents.belief_expectimax import BeliefExpectimaxAgent
    name = f"p{snap['turn']}"
    a = BeliefExpectimaxAgent(name, verbose=False)
    a.cur = list(snap['hand'])
    a.context = snap['context'].copy()
    tile, trace = a.next_with_trace()                         # see_tile 已发生
    hand13 = list(a.cur)
    tenpai = None
    if eval_v2.shanten(hand13) == 0:                          # 引擎报听前置
        tenpai = bool(a.declare_tenpai(hand13, a.context))
    # 决策时（pre-see_tile）的 danger 信号，用于分层统计
    ctx = snap['context']
    sig = bool(ctx.tenpai_players - {name}) or any(
        opponent.player_danger_level(d) >= 1
        for p, d in ctx.discards.items() if p != name)
    return tile, trace, tenpai, sig


def py_danger_vec(snap):
    import algo.eval.opponent as opponent
    name = f"p{snap['turn']}"
    ctx = snap['context']
    return np.array([opponent.tile_danger(t, ctx, name) for t in TILE_IDS])


# ---------------------------------------------------------------------------
# 快照 -> jaxenv State（wall 等无关字段填哑值）
# ---------------------------------------------------------------------------

def snaps_to_states(snaps, phase, post_hands=None):
    """phase: env.PHASE_DISCARD / PHASE_TENPAI。post_hands: 每快照 turn 座位的
    13 张弃后手牌（TENPAI 状态用；此时 context 须为 see_tile 后版本）。"""
    n = len(snaps)
    hands = np.zeros((n, 4, 34), np.int8)
    discards = np.zeros((n, 4, 34), np.int8)
    dseq = np.full((n, 4, 64), -1, np.int8)
    dlen = np.zeros((n, 4), np.int8)
    locked = np.zeros((n, 4), bool)
    turn = np.zeros(n, np.int8)
    drawn = np.full(n, -1, np.int8)
    for i, s in enumerate(snaps):
        for p in range(4):
            hp = s['hands'][p] if post_hands is None or p != s['turn'] \
                else post_hands[i]
            for t in hp:
                hands[i, p, TILE_TO_IDX[t]] += 1
            seq = s['context'].discards.get(f'p{p}', [])
            assert len(seq) <= 64
            for j, t in enumerate(seq):
                idx = TILE_TO_IDX[t]
                dseq[i, p, j] = idx
                discards[i, p, idx] += 1
            dlen[i, p] = len(seq)
            locked[i, p] = s['locked'][p]
        turn[i] = s['turn']
        if phase == env.PHASE_DISCARD and s['drawn'] is not None:
            drawn[i] = TILE_TO_IDX[s['drawn']]
    st = env.State(
        wall=jnp.zeros((n, 136), jnp.int8),
        wall_head=jnp.full(n, 53, jnp.int16),
        wall_tail=jnp.full(n, 135, jnp.int16),
        hands=jnp.asarray(hands),
        n_melds=jnp.zeros((n, 4), jnp.int8),
        meld_counts=jnp.zeros((n, 4, 34), jnp.int8),
        discards=jnp.asarray(discards),
        discard_seq=jnp.asarray(dseq),
        discard_len=jnp.asarray(dlen),
        turn=jnp.asarray(turn),
        pending_tile=jnp.full(n, -1, jnp.int8),
        drawn=jnp.asarray(drawn),
        claim_stage=jnp.zeros(n, jnp.int8),
        claim_offset=jnp.zeros(n, jnp.int8),
        claim_mask=jnp.zeros(n, jnp.int16),
        locked=jnp.asarray(locked),
        phase=jnp.full(n, phase, jnp.int8),
        done=jnp.zeros(n, bool),
        winner=jnp.full(n, -1, jnp.int8),
        win_type=jnp.zeros(n, jnp.int8),
        dealer=jnp.full(n, -1, jnp.int8),
        n_draws=jnp.zeros(n, jnp.int16),
    )
    return st


# ---------------------------------------------------------------------------
# parity gates
# ---------------------------------------------------------------------------

def test_danger_parity(snaps):
    states = snaps_to_states(snaps, env.PHASE_DISCARD)
    jx = np.asarray(jax.jit(jax.vmap(tile_danger_vec))(states))
    py = np.stack([py_danger_vec(s) for s in snaps])
    diff = np.abs(jx - py).max()
    print(f'[danger-parity] {len(snaps)} states x34 tiles: max|diff|={diff:.3e} '
          f'(tol 1e-9)', flush=True)
    assert diff < 1e-9, f'danger 逐值 parity 失败: {diff}'
    print('[danger-parity] passed', flush=True)


def test_decision_parity(snaps):
    states = snaps_to_states(snaps, env.PHASE_DISCARD)
    jx_act = np.asarray(jax.jit(jax.vmap(belief_action))(states)).astype(np.int32)
    same = tie_split = danger_split = bad = 0
    sig_count = 0
    max_score_diff = 0.0
    for i, s in enumerate(snaps):
        tile, trace, _, sig = py_decide(s)
        py_idx = TILE_TO_IDX[tile]
        jx_idx = jx_act[i]
        sig_count += int(sig)
        # 进攻分逐值校验（top-8 候选）：N/15006 vs trace['scores']
        hand = np.zeros(34, np.int32)
        for t in s['hand']:
            hand[TILE_TO_IDX[t]] += 1
        hands13 = hand[None, :] - np.eye(34, dtype=np.int32)
        n_all = np.asarray(jax.vmap(_eval2_num13)(jnp.asarray(hands13)))
        for t, sc in trace['scores'].items():
            d = abs(n_all[TILE_TO_IDX[t]] / 15006.0 - sc)
            max_score_diff = max(max_score_diff, d)
        if jx_idx == py_idx:
            same += 1
            continue
        if n_all[py_idx] == n_all[jx_idx]:
            tie_split += 1
            continue
        if sig:
            # 危险分支：双方 danger 差 <1e-9 且都 safe 则为 danger 平局噪声
            st = jax.tree.map(lambda x: x[i], states)
            dv = np.asarray(tile_danger_vec(st))
            if abs(dv[py_idx] - dv[jx_idx]) < 1e-9:
                danger_split += 1
                continue
        bad += 1
        if bad <= 3:
            print(f'  真 mismatch snap {i}: hand={sorted(s["hand"])} '
                  f'jax={TILE_IDS[jx_idx]} py={TILE_IDS[py_idx]} sig={sig} '
                  f'N_jax={int(n_all[jx_idx])} N_py={int(n_all[py_idx])}', flush=True)
    n = len(snaps)
    print(f'[decision-parity] {n} decisions (危险信号分层: True={sig_count} '
          f'False={n - sig_count}): 一致 {same} ({100 * same / n:.2f}%), '
          f'offense tie-split {tie_split}, danger tie-split {danger_split}, '
          f'真不一致 {bad}', flush=True)
    print(f'  进攻分逐值 max|N/15006 - trace_score|={max_score_diff:.3e} '
          f'(tol 1e-9)', flush=True)
    assert bad == 0, f'决策 parity 失败: {bad} 个非平局不一致'
    assert max_score_diff < 1e-9, '进攻分逐值超容差'
    print('[decision-parity] passed', flush=True)


def test_tenpai_parity(snaps):
    from algo.agents.belief_expectimax import BeliefExpectimaxAgent
    cases = []          # (snap_idx, py_tenpai)
    sub_snaps = []
    post_hands = []
    for i, s in enumerate(snaps):
        tile, _, tenpai, _ = py_decide(s)
        if tenpai is None:
            continue
        # TENPAI 状态的 context 需为 see_tile 后版本：重放决策拿 context
        a = BeliefExpectimaxAgent(f"p{s['turn']}", verbose=False)
        a.cur = list(s['hand'])
        a.context = s['context'].copy()
        a.next_with_trace()
        snap2 = dict(s)
        snap2['context'] = a.context
        cases.append((i, tenpai))
        sub_snaps.append(snap2)
        post_hands.append(list(a.cur))
    if not cases:
        raise AssertionError('快照中没有报听决策点，增加 --games')
    states = snaps_to_states(sub_snaps, env.PHASE_TENPAI, post_hands=post_hands)
    jx = np.asarray(jax.jit(jax.vmap(belief_action))(states)).astype(np.int32)
    mism = [k for k, (_, tenpai) in enumerate(cases)
            if jx[k] != (env.A_TENPAI_YES if tenpai else env.A_TENPAI_NO)]
    yes = sum(1 for _, t in cases if t)
    print(f'[tenpai-parity] {len(cases)} 个报听决策点 (yes={yes} no={len(cases) - yes}): '
          f'不一致 {len(mism)}', flush=True)
    for k in mism[:3]:
        print(f'  mismatch case {cases[k]}', flush=True)
    assert not mism, 'tenpai parity 失败'
    print('[tenpai-parity] passed', flush=True)


def test_env_smoke(n=32, seed=99):
    keys = jax.random.split(jax.random.PRNGKey(seed), 2)
    states = jax.vmap(env.init)(keys)
    for step in range(4):
        acts = jax.vmap(belief_action)(states)                # eager（非 jit）
        masks = jax.vmap(env.legal_mask)(states)
        a, m, d = np.asarray(acts), np.asarray(masks), np.asarray(states.done)
        assert m[np.arange(2), a][~d].all(), f'eager: illegal action at step {step}'
        states, _, _ = jax.vmap(env.step)(states, acts)
    print('[env-smoke] eager 4 steps: all actions legal', flush=True)

    act_v = jax.jit(jax.vmap(belief_action))
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
    assert max_melds == 0, 'belief 不变式被破坏（碰杠）'
    print('[env-smoke] passed', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=30)
    ap.add_argument('--min-snaps', type=int, default=2000,
                    help='快照数下限（parity 门要求 >=2000；小样本调试用可调低）')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    t0 = time.time()
    snaps = collect_snapshots(args.games, args.seed)
    print(f'collected {len(snaps)} snapshots from {args.games} games '
          f'({time.time() - t0:.1f}s)', flush=True)
    assert len(snaps) >= args.min_snaps, f'快照不足 {args.min_snaps}，增加 --games'

    test_danger_parity(snaps)
    test_decision_parity(snaps)
    test_tenpai_parity(snaps)
    test_env_smoke(32, args.seed + 2)
    print('[test_beliefjax] all passed', flush=True)


if __name__ == '__main__':
    main()
