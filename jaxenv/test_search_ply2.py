# -*- coding: utf-8 -*-
"""jaxenv/search.py best-first 2-ply 语义核对（方向 1b 扩展）。

1. 场景1 复用（对手 WAIT7 等 7 胡）：2-ply Q(弃7) 仍 ≈ -1/3，π' 显著压低 7；
   并打印 top-5 候选的 1-ply vs 2-ply Q 对比。
2. 构造「1-ply 与 2-ply 判断应不同」的状态：根玩家已报听锁手，摸到安全牌 9
   （只能弃 9），wall 剩余全是 7（WAIT7 对手必胡）：
   - 1-ply jump 截断取 V(child)：价值头看不到对手手牌，报听手形 V 尚可；
   - 2-ply 叶值 = 下一状态（锁手，强制弃摸到的 7）的 search_value ≈ -1/3，
     即「下一手被迫点炮」。断言 Q2(弃9) 显著低于 Q1(弃9)。
3. jit + vmap 形状/有限性 smoke。

用法：PYTHONPATH=. python3 jaxenv/test_search_ply2.py
"""

import json

import numpy as np

import jax
import jax.numpy as jnp

from jaxenv import env
from jaxenv.model_flax import build_model_flax
from jaxenv.obs import observe
from jaxenv.search import (improved_policy, improved_policy_ply2,
                           _q_discard, _q_discard_ply2, HU_VALUE, NEG)
from jaxenv.ppo import load_params
from jaxenv.test_search import (_mk_state, _wall_with, HAND0, WAIT7,
                                NEUTRAL1, NEUTRAL3, claim_bits, resp_probs,
                                q_of)

K, N_DRAWS, BETA, K2, N_TOP2 = 8, 2, 32.0, 4, 2


def q2_of(params, model, st, a, n_rep=16):
    """_q_discard_ply2 对固定 key 多次重复取均值（降低 n_draws 采样噪声）。"""
    qs = [float(_q_discard_ply2(params, model, st, jnp.int8(a),
                                jax.random.PRNGKey(3000 + i), N_DRAWS, K2, BETA))
          for i in range(n_rep)]
    return float(np.mean(qs)), float(np.std(qs))


def topk_of(params, model, st, key, k=K):
    """复现 improved_policy_ply2 内部的含噪 top-k（同 key 拆分）。"""
    key_g = jax.random.split(key, 3)[0]
    out = model.apply({'params': params}, observe(st)[None])
    logits = out['policy'][0]
    legal = env.legal_mask(st)[:34]
    u = jnp.clip(jax.random.uniform(key_g, (34,)), 1e-9, 1.0 - 1e-9)
    g = -jnp.log(-jnp.log(u))
    pert = jnp.where(legal, logits + g, NEG)
    topv, topa = jax.lax.top_k(pert, k)
    return np.asarray(topa), np.asarray(topv > NEG / 2), np.asarray(logits)


def main():
    with open('output/nn_full_action_best_config.json') as f:
        config = json.load(f)
    model = build_model_flax(config)
    params = load_params('output/nn_full_action_best_flax.msgpack')
    wall = _wall_with([5, 6, 8, 9])

    # --- 场景1：对手（offset2 玩家2）能胡 7；2-ply Q(弃7) 仍 ≈ -1/3 ---
    st = _mk_state(wall, [HAND0, NEUTRAL1, WAIT7, NEUTRAL3], turn=0)
    q1_7, sd1 = q_of(params, model, st, 7)
    q2_7, sd2 = q2_of(params, model, st, 7)
    print(f'场景1  Q1(弃7)={q1_7:+.4f}±{sd1:.4f}  Q2(弃7)={q2_7:+.4f}±{sd2:.4f}'
          f'（胡分支截断一致，期望均 ≈ {HU_VALUE:.4f}）')
    assert q2_7 < -0.15, f'2-ply 下能胡的弃牌 Q 应显著为负: {q2_7}'
    assert abs(q2_7 - q1_7) < 0.1, f'1/2-ply 胡分支应接近: {q1_7} vs {q2_7}'

    # π' 压低 7：在 7 落入 top-k 的 key 上比较 π'[7] 与 prior softmax[7]
    out = model.apply({'params': params}, observe(st)[None])
    logits = np.asarray(out['policy'][0])
    legal = np.asarray(env.legal_mask(st)[:34])
    prior = np.asarray(jax.nn.softmax(jnp.where(legal, logits, NEG)))
    ratios, n_in = [], 0
    for i in range(32):
        key = jax.random.PRNGKey(100 + i)
        topa, valid, _ = topk_of(params, model, st, key)
        if 7 not in topa[valid]:
            continue
        n_in += 1
        pi2, _, _ = improved_policy_ply2(params, model, st, key,
                                         K, N_DRAWS, BETA, K2, N_TOP2)
        ratios.append(float(pi2[7]) / max(prior[7], 1e-9))
    ratios = np.array(ratios)
    print(f'场景1  7 落入 top-k 的 key {n_in}/32 个；'
          f'π\'[7]/prior[7] mean={ratios.mean():.4f} max={ratios.max():.4f}')
    assert n_in >= 4, f'7 应经常进入 top-k: {n_in}/32'
    assert ratios.mean() < 0.3, f'2-ply π\' 应显著压低 7: {ratios.mean()}'

    # top-5 候选的 1-ply vs 2-ply Q 对比（同一 key 的含噪排序）
    key = jax.random.PRNGKey(42)
    topa, valid, _ = topk_of(params, model, st, key)
    _, key_c, key_c2 = jax.random.split(key, 3)
    keys1 = jax.random.split(key_c, K)
    keys2 = jax.random.split(key_c2, N_TOP2)
    _, q_top2, v2 = improved_policy_ply2(params, model, st, key,
                                         K, N_DRAWS, BETA, K2, N_TOP2)
    print(f'场景1  top-5 候选 1-ply vs 2-ply Q（V={float(v2):+.4f}；'
          f'improved_policy_ply2 仅前 {N_TOP2} 个用 2-ply）：')
    for i in range(5):
        a = int(topa[i])
        q1 = float(_q_discard(params, model, st, jnp.int8(a), keys1[i], N_DRAWS))
        q2 = float(_q_discard_ply2(params, model, st, jnp.int8(a),
                                   keys2[i] if i < N_TOP2 else keys1[i],
                                   N_DRAWS, K2, BETA))
        used = '2-ply' if i < N_TOP2 else '1-ply'
        print(f'  #{i} tile={a:2d} logit={logits[a]:+.2f} '
              f'Q1={q1:+.4f} Q2={q2:+.4f} (π\' 用 {used}: {float(q_top2[i]):+.4f})')

    # --- 场景B：锁手被迫点炮——1-ply 看 V 尚可，2-ply 发现下一手必被胡 ---
    # 根玩家报听锁手，13 张报听手 + 摸到 9（只能弃 9）；wall 剩余全 7，
    # 玩家2 WAIT7 等 7 胡。1-ply：弃 9 全 pass，jump 摸 7 取 V（价值头看不到
    # 对手等 7）；2-ply：叶状态锁手强制弃 7 → 响应头 P(hu)≈1 → ≈ -1/3。
    wall7 = np.full(136, 7, np.int8)
    hand_tenpai = [9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    stB = _mk_state(wall7, [hand_tenpai, NEUTRAL1, WAIT7, NEUTRAL3], turn=0,
                    locked=[True, False, False, False], drawn=9)
    bits9 = claim_bits(stB, 9)
    print('场景B bits(弃9):', bits9, '(应全 0：9 安全)')
    assert sum(bits9) == 0
    q1_9, sd1 = q_of(params, model, stB, 9)
    q2_9, sd2 = q2_of(params, model, stB, 9)
    piB, _, vB = improved_policy_ply2(params, model, stB, jax.random.PRNGKey(7),
                                      K, N_DRAWS, BETA, K2, N_TOP2)
    print(f'场景B  Q1(弃9)={q1_9:+.4f}±{sd1:.4f}（1-ply jump 取 V：尚可）  '
          f'Q2(弃9)={q2_9:+.4f}±{sd2:.4f}（2-ply：下一手被迫弃 7 点炮 ≈ -1/3）')
    print(f'  锁手 π\'[9]={float(piB[9]):.6f}（应 one-hot）')
    assert float(piB[9]) > 0.999, f'锁手应 one-hot 在 9: {float(piB[9])}'
    assert q2_9 < q1_9 - 0.1, \
        f'2-ply 应显著差于 1-ply（被迫点炮 vs V 尚可）: Q2={q2_9} Q1={q1_9}'
    assert q2_9 < -0.2, f'2-ply Q 应接近 -1/3: {q2_9}'
    print('  >> 1-ply 与 2-ply 判断不同：Q1 依赖价值头（不知对手等 7），'
          'Q2 通过下一状态的 search_value 发现强制弃 7 被胡')

    # --- jit + vmap smoke ---
    keys = jax.random.split(jax.random.PRNGKey(7), 4)
    states = jax.vmap(env.init)(keys)
    f = jax.jit(jax.vmap(lambda s, kk: improved_policy_ply2(
        params, model, s, kk, K, N_DRAWS, BETA, K2, N_TOP2)))
    pi_b, q_b, v_b = f(states, jax.random.split(jax.random.PRNGKey(8), 4))
    assert pi_b.shape == (4, 34) and q_b.shape == (4, K) and v_b.shape == (4,)
    assert bool(jnp.all(jnp.isfinite(pi_b))) and bool(jnp.all(jnp.isfinite(q_b)))
    ok = ~np.asarray(states.done)
    assert np.allclose(np.asarray(pi_b)[ok].sum(-1), 1.0, atol=1e-4), \
        '未完成局 π\' 应归一'
    print(f'--- jit+vmap smoke ok: pi{pi_b.shape} q{q_b.shape} v{v_b.shape} ---')

    print('[test_search_ply2] all semantic checks passed')


if __name__ == '__main__':
    main()
