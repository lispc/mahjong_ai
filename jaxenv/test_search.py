# -*- coding: utf-8 -*-
"""jaxenv/search.py 语义单元核对（方向 1b）。

用手写构造状态非 jit 调用 improved_policy / _q_discard，打印 π'/Q/声明位图/
响应头概率并人工核对：
1. 对手能胡（offset2 等 7 胡）→ 弃 7 的 Q ≈ -1/3（响应头 P(hu) 应 ≈1），π' 压低 7
2. 对手能碰（不能胡）→ 弃 7 走碰分支 V 截断，Q 远离 -1/3
3. 无人能声明 → mask9==0，Q = 全 pass 后根玩家摸牌 V，π' 归一
4. 报听锁手 → π' one-hot 在强制弃牌上
5. 两对手都能胡 7（一家还能杠 9）→ 弃 7 的 Q 比场景 1 更负；杠分支被激活
另：jit + vmap 形状/有限性 smoke。

用法：PYTHONPATH=. python3 jaxenv/test_search.py
"""

import json

import numpy as np

import jax
import jax.numpy as jnp

from jaxenv import env
from jaxenv.model_flax import build_model_flax
from jaxenv.obs import observe, obs_for_player
from jaxenv.search import improved_policy, _q_discard, HU_VALUE
from jaxenv.ppo import load_params

K, N_DRAWS, BETA = 8, 2, 8.0


# ---------------------------------------------------------------------------
# 构造工具（与 test_env.py 的 _mk_state/_wall_with 相同模式）
# ---------------------------------------------------------------------------

def _mk_state(wall, hands, **kw):
    hc = np.zeros((4, 34), np.int8)
    for p in range(4):
        for t in hands[p]:
            hc[p, t] += 1
    st = env.init(jax.random.PRNGKey(0))
    st = st.replace(
        wall=jnp.asarray(wall, jnp.int8),
        wall_head=jnp.int16(kw.get('wall_head', 53)),
        wall_tail=jnp.int16(kw.get('wall_tail', 135)),
        hands=jnp.asarray(hc),
        n_melds=jnp.asarray(kw.get('n_melds', [0, 0, 0, 0]), jnp.int8),
        meld_counts=jnp.asarray(kw.get('meld_counts', np.zeros((4, 34), np.int8)), jnp.int8),
        turn=jnp.int8(kw.get('turn', 0)),
        pending_tile=jnp.int8(kw.get('pending_tile', -1)),
        drawn=jnp.int8(kw.get('drawn', -1)),
        locked=jnp.asarray(kw.get('locked', [False] * 4)),
        phase=jnp.int8(kw.get('phase', env.PHASE_DISCARD)),
        done=bool(kw.get('done', False)),
        winner=jnp.int8(kw.get('winner', -1)),
        win_type=jnp.int8(kw.get('win_type', 0)),
    )
    return st


def _wall_with(head_tiles, tail_tiles=()):
    wall = np.zeros(136, np.int8)
    wall[53:53 + len(head_tiles)] = head_tiles
    if tail_tiles:
        wall[136 - len(tail_tiles):] = tail_tiles
    return wall


def after_discard(st, a):
    """根玩家（turn=0）弃 a 后的中间状态（pending=a）。"""
    return st.replace(hands=st.hands.at[0, a].add(-1),
                      pending_tile=jnp.int8(a), drawn=jnp.int8(-1))


def claim_bits(st, a):
    m = int(env._claim_ask_mask(after_discard(st, a)))
    return [(m >> i) & 1 for i in range(9)]


def resp_probs(params, model, st, a, p, stage):
    """玩家 p 对弃 a 的 response head P(claim)（{pass, stage动作} 二项 masked softmax）。"""
    o = obs_for_player(after_discard(st, a), jnp.int32(p), jnp.int8(a))
    r = model.apply({'params': params}, o[None])['response'][0]
    idx = {1: 3, 2: 2, 3: 1}[stage]     # hu/gang/peng
    return float(jax.nn.sigmoid(r[idx] - r[0]))


def q_of(params, model, st, a, n_rep=16):
    """_q_discard 对固定 key 多次重复取均值（降低 n_draws 采样噪声）。"""
    qs = [float(_q_discard(params, model, st, jnp.int8(a),
                           jax.random.PRNGKey(1000 + i), N_DRAWS))
          for i in range(n_rep)]
    return float(np.mean(qs)), float(np.std(qs))


def report(name, params, model, st, key):
    out = model.apply({'params': params}, observe(st)[None])
    logits = np.asarray(out['policy'][0])
    pi, q_top, v = improved_policy(params, model, st, key, K, N_DRAWS, BETA)
    legal = np.where(np.array(env.legal_mask(st)[:34]))[0]
    print(f'--- {name} ---')
    print(f'  V={float(v):+.4f}')
    print(f'  prior logits (legal): ' +
          ', '.join(f'{a}:{logits[a]:+.2f}' for a in legal))
    print(f'  pi\' (legal): ' +
          ', '.join(f'{a}:{float(pi[a]):.4f}' for a in legal))
    print(f'  q_top={np.array2string(np.asarray(q_top), precision=4)}')
    return pi, q_top, v


# ---------------------------------------------------------------------------
# 场景手牌
# ---------------------------------------------------------------------------

# 根玩家手牌（14 张）：含一张 7，其余 13 张各对手均无法声明（逐一核对过计数）
HAND0 = [7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
# 等 7 胡：000 111 222 345 + 7（来 7 成对即胡）
WAIT7 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 5, 7]
# 无法声明 HAND0 任何牌的路人手牌
NEUTRAL1 = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 23, 24]
NEUTRAL2 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 5, 8]
NEUTRAL3 = [28, 28, 29, 29, 30, 30, 31, 31, 24, 25, 26, 32, 33]


def main():
    with open('output/nn_full_action_best_config.json') as f:
        config = json.load(f)
    model = build_model_flax(config)
    params = load_params('output/nn_full_action_best_flax.msgpack')
    key = jax.random.PRNGKey(42)
    wall = _wall_with([5, 6, 8, 9])

    # --- 场景1：对手（offset2 玩家2）能胡 7 ---
    st = _mk_state(wall, [HAND0, NEUTRAL1, WAIT7, NEUTRAL3], turn=0)
    bits7 = claim_bits(st, 7)
    print('场景1 bits(弃7):', bits7, '(应只有 hu off2 = pos1)')
    assert bits7[1] == 1 and sum(bits7) == 1, bits7
    p_hu = resp_probs(params, model, st, 7, 2, 1)
    print(f'  响应头 P(hu|玩家2, 弃7) = {p_hu:.4f}（应 ≈1）')
    q7, sd7 = q_of(params, model, st, 7)
    q9, sd9 = q_of(params, model, st, 9)
    print(f'  Q(弃7)={q7:+.4f}±{sd7:.4f}（期望 ≈ {HU_VALUE:.4f}×{p_hu:.3f}）  '
          f'Q(弃9)={q9:+.4f}±{sd9:.4f}')
    assert q7 < -0.15, f'能胡的弃牌 Q 应显著为负: {q7}'
    assert q7 < q9, '能胡的 7 应比安全的 9 差'
    pi1, _, _ = report('场景1 对手能胡7', params, model, st, key)
    print(f'  >> pi\'[7]={float(pi1[7]):.4f} vs pi\'[9]={float(pi1[9]):.4f} '
          f'(若 7 在 top-k 内应被显著压低)')

    # --- 场景2：对手（offset1 玩家1）能碰 7、不能胡 ---
    hand_peng = [7, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20]
    st = _mk_state(wall, [HAND0, hand_peng, NEUTRAL1, NEUTRAL3], turn=0)
    bits7 = claim_bits(st, 7)
    print('场景2 bits(弃7):', bits7, '(应只有 peng off1 = pos6)')
    assert bits7[6] == 1 and sum(bits7) == 1, bits7
    p_peng = resp_probs(params, model, st, 7, 1, 3)
    q7p, _ = q_of(params, model, st, 7)
    print(f'  响应头 P(peng|玩家1, 弃7) = {p_peng:.4f}  Q(弃7)={q7p:+.4f}')
    assert q7p > -0.25, f'碰分支不应接近 -1/3: {q7p}'
    report('场景2 对手能碰7', params, model, st, key)

    # --- 场景3：无人能声明 ---
    st = _mk_state(wall, [HAND0, NEUTRAL1, NEUTRAL2, NEUTRAL3], turn=0)
    bits7, bits9 = claim_bits(st, 7), claim_bits(st, 9)
    print('场景3 bits(弃7):', bits7, ' bits(弃9):', bits9, '(应全 0)')
    assert sum(bits7) == 0 and sum(bits9) == 0
    q7n, sd7n = q_of(params, model, st, 7)
    print(f'  Q(弃7)={q7n:+.4f}±{sd7n:.4f}（= 全 pass 后根玩家摸牌 V）')
    pi3, _, _ = report('场景3 无人能声明', params, model, st, key)
    assert abs(float(jnp.sum(pi3)) - 1.0) < 1e-4

    # --- 场景4：报听锁手，强制打出摸到的 31 ---
    handL = [31, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    st = _mk_state(wall, [handL, NEUTRAL1, WAIT7, NEUTRAL3], turn=0,
                   locked=[True, False, False, False], drawn=31)
    pi4, _, _ = report('场景4 锁手强制弃牌', params, model, st, key)
    assert float(pi4[31]) > 0.999, f'锁手应 one-hot 在 31: {float(pi4[31])}'
    print(f'  >> pi\'[31]={float(pi4[31]):.6f}（one-hot ✓）')

    # --- 场景5：两家都能胡 7（玩家1 还能杠 9） ---
    wait7b = [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 13, 14, 7]
    st = _mk_state(wall, [HAND0, wait7b, WAIT7, NEUTRAL3], turn=0)
    bits7, bits9 = claim_bits(st, 7), claim_bits(st, 9)
    print('场景5 bits(弃7):', bits7, '(应 hu off1+off2)  '
          'bits(弃9):', bits9, '(应 gang off1 = pos3)')
    assert bits7[0] == 1 and bits7[1] == 1 and bits9[3] == 1
    p_hu1 = resp_probs(params, model, st, 7, 1, 1)
    p_hu2 = resp_probs(params, model, st, 7, 2, 1)
    q7m, _ = q_of(params, model, st, 7)
    print(f'  P(hu1)={p_hu1:.4f} P(hu2)={p_hu2:.4f}  Q(弃7)={q7m:+.4f}')
    assert q7m < q7 + 0.02, f'两家能胡应不比一家能胡好: {q7m} vs {q7}'
    report('场景5 两家能胡7/一家能杠9', params, model, st, key)

    # --- jit + vmap smoke ---
    keys = jax.random.split(jax.random.PRNGKey(7), 4)
    states = jax.vmap(env.init)(keys)
    f = jax.jit(jax.vmap(lambda s, kk: improved_policy(params, model, s, kk,
                                                       K, N_DRAWS, BETA)))
    pi_b, q_b, v_b = f(states, jax.random.split(jax.random.PRNGKey(8), 4))
    assert pi_b.shape == (4, 34) and q_b.shape == (4, K) and v_b.shape == (4,)
    assert bool(jnp.all(jnp.isfinite(pi_b))) and bool(jnp.all(jnp.isfinite(q_b)))
    ok = ~np.asarray(states.done)
    assert np.allclose(np.asarray(pi_b)[ok].sum(-1), 1.0, atol=1e-4), '未完成局 π\' 应归一'
    print(f'--- jit+vmap smoke ok: pi{pi_b.shape} q{q_b.shape} v{v_b.shape} ---')

    print('[test_search] all semantic checks passed')


if __name__ == '__main__':
    main()
