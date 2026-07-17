# -*- coding: utf-8 -*-
"""Gumbel-top-k 1-ply 搜索目标（方向 1b）：AlphaZero 式改进策略 π'。

对单个 DISCARD-phase State：
1. prior = 当前网络 policy logits(34) 与 value V（score/3 尺度，与 value head 一致）。
2. 合法弃牌上加大 Gumbel 噪声，取 top-k 候选。
3. 每个候选弃牌 a 的 Q（同一 score/3 尺度）：
   a. 模拟弃牌（hands 减一、pending=a），精确声明判定复用 env._claim_ask_mask
      （真实动力学，非 god-mode：响应概率来自 response head，而非知道对手底牌后的必然胡）。
   b. 9 个声明位（胡 off1..3 → 杠 off1..3 → 碰 off1..3）按 env 顺序走；每位若可声明，
      该玩家视角 CLAIM 语义 obs（obs_for_player）过 response head，取
      {pass, 当前阶段动作} 的二项 masked softmax 得 P(claim)。
      - 胡 → 分支 Q = -1/3（放炮者 score -1 的 /3），walk 结束。
      - 碰/杠 → 按 env 规则结算（杠后从牌山尾补牌，碰不摸牌；杠补牌若自摸/摸空则
        分支 Q=0——别家自摸根玩家得 0、流局得 0），随后「跳到根玩家下次摸牌后」取 V 截断
        （跳过中间对手的行动，v1 近似）。
      - pass → 顺序到下一个声明位。
   c. 全 pass / 无人可声明 → 弃牌入弃牌堆，同样跳到根玩家下次摸牌后取 V。
   d. 摸牌从 state.wall[head..tail] 真实余牌中无放回采 n_draws 张（gumbel-top-n），
      根玩家摸后即胡（自摸 +3 → /3 = +1）则该样本取 +1，否则 V(next_obs)；取均值。
4. π' = masked softmax(logits + beta*(Q_completed - V))；非 top-k 的合法动作用
   Q=V 补齐（completed-Q，等价于 logits 不变），非法动作 -1e9。

近似声明（v1）：
- 跳过报听决策点（弃牌后按不报听处理，locked flag 不变）。
- 响应结算后到根玩家下次决策之间的对手行动不模拟，直接跳到根玩家摸牌。

全部为纯函数（可 jit/vmap）。每个根决策共 3 次批量网络前向：
prior(1 行) + response(9k 行) + jump value(7·n_draws·k 行)。
"""

import jax
import jax.numpy as jnp

from . import env, rules
from .obs import observe, obs_for_player

NEG = -1e9
HU_VALUE = -1.0 / 3.0      # 放炮（score=-1）的 /3 尺度
SELFDRAW_VALUE = 1.0       # 自摸（score=+3）的 /3 尺度

_OFFS9 = jnp.array([1, 2, 3, 1, 2, 3, 1, 2, 3], jnp.int8)          # stage 内 offset
_POS9 = jnp.arange(9, dtype=jnp.int16)
# response head 索引：pass=0, peng=1, gang=2, hu=3（= action - 34）；stage 1/2/3 -> 3/2/1
_CLAIM_RESP_IDX = jnp.array([3, 3, 3, 2, 2, 2, 1, 1, 1], jnp.int32)


# ---------------------------------------------------------------------------
# 分支状态构造
# ---------------------------------------------------------------------------

def _branch_states(st, a):
    """根玩家 st.turn 弃 a 之后的 7 个分支后状态：[gang off1..3, peng off1..3, 全pass]。

    返回 (stacked State (7,), dead[7])；dead=True 的分支价值为 0
    （杠补牌自摸——别家自摸根玩家 score=0；或牌山摸空——流局 0）。
    pending 均已清除；turn 在碰/杠分支为声明者（jump 时会改回根玩家）。
    """
    d = st.turn.astype(jnp.int32)
    t = a.astype(jnp.int32)
    branches, deads = [], []
    for off in (1, 2, 3):   # 杠：3 张闭手成副露，从牌山尾补牌（env._step_claim.do_gang）
        c = (d + off) % 4
        s2 = st.replace(hands=st.hands.at[c, t].add(-3),
                        meld_counts=st.meld_counts.at[c, t].add(4),
                        n_melds=st.n_melds.at[c].add(1),
                        pending_tile=jnp.int8(-1), claim_stage=jnp.int8(0),
                        claim_offset=jnp.int8(0), claim_mask=jnp.int16(0))
        empty = s2.wall_head > s2.wall_tail
        tile = s2.wall[jnp.clip(s2.wall_tail, 0, env.WALL_SIZE - 1)].astype(jnp.int32)
        s3 = s2.replace(wall_tail=s2.wall_tail - jnp.int16(1),
                        hands=s2.hands.at[c, tile].add(1),
                        n_draws=s2.n_draws + jnp.int16(1),
                        turn=jnp.int8(c), drawn=jnp.int8(tile),
                        phase=jnp.int8(env.PHASE_DISCARD))
        win = rules.is_win_counts(s3.hands[c], s3.n_melds[c])
        branches.append(s3)
        deads.append(empty | win)
    for off in (1, 2, 3):   # 碰：2 张闭手成副露，不摸牌（env._step_claim.do_peng）
        c = (d + off) % 4
        s2 = st.replace(hands=st.hands.at[c, t].add(-2),
                        meld_counts=st.meld_counts.at[c, t].add(3),
                        n_melds=st.n_melds.at[c].add(1),
                        pending_tile=jnp.int8(-1), claim_stage=jnp.int8(0),
                        claim_offset=jnp.int8(0), claim_mask=jnp.int16(0),
                        turn=jnp.int8(c), drawn=jnp.int8(-1),
                        phase=jnp.int8(env.PHASE_DISCARD))
        branches.append(s2)
        deads.append(jnp.bool_(False))
    # 全 pass：弃牌入弃牌堆（env._resolve_pass_through 的记账部分）
    dl = st.discard_len[d].astype(jnp.int32)
    s_pass = st.replace(discards=st.discards.at[d, t].add(1),
                        discard_seq=st.discard_seq.at[d, dl].set(a.astype(jnp.int8)),
                        discard_len=st.discard_len.at[d].add(1),
                        pending_tile=jnp.int8(-1), claim_stage=jnp.int8(0),
                        claim_offset=jnp.int8(0), claim_mask=jnp.int16(0))
    branches.append(s_pass)
    deads.append(jnp.bool_(False))
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *branches)
    return stacked, jnp.stack(deads)


# ---------------------------------------------------------------------------
# 根玩家下次摸牌的 V 截断
# ---------------------------------------------------------------------------

def _root_draw_states(sb, key_b, root, n_draws):
    """单分支状态 sb：从 wall[head..tail] 无放回采 n_draws 张，构造根玩家摸牌后状态。

    返回 ((n_draws,) State, ok[n_draws])；ok=False 为余牌不足时的填充样本。
    采样用 gumbel-top-n（等价均匀无放回）。
    """
    head = sb.wall_head.astype(jnp.int32)
    tail = sb.wall_tail.astype(jnp.int32)
    g = jax.random.gumbel(key_b, (env.WALL_SIZE,))
    idx = jnp.arange(env.WALL_SIZE, dtype=jnp.int32)
    inpool = (idx >= head) & (idx <= tail)
    topv, topi = jax.lax.top_k(jnp.where(inpool, g, -jnp.inf), n_draws)
    ok = topv > -1e8
    tiles = sb.wall[topi].astype(jnp.int32)                       # (n_draws,)
    base = jax.tree.map(lambda x: jnp.broadcast_to(x, (n_draws,) + x.shape), sb)
    onehot = jax.nn.one_hot(tiles, 34, dtype=jnp.int8)            # (n_draws, 34)
    hands = base.hands.at[:, root, :].add(onehot)
    dr = base.replace(
        hands=hands,
        turn=jnp.broadcast_to(root.astype(jnp.int8), (n_draws,)),
        drawn=tiles.astype(jnp.int8),
        phase=jnp.full((n_draws,), env.PHASE_DISCARD, jnp.int8))
    return dr, ok


def _jump_values(params, model, stacked, root, keys, dead, n_draws):
    """7 个分支状态各自的「根玩家下次摸牌后」截断价值 [7]。"""
    B = stacked.hands.shape[0]
    dr, ok = jax.vmap(_root_draw_states, in_axes=(0, 0, None, None))(
        stacked, keys, root, n_draws)
    flat = jax.tree.map(lambda x: x.reshape((B * n_draws,) + x.shape[2:]), dr)
    obs = jax.vmap(observe)(flat)                                 # (B*n, 175)
    v = model.apply({'params': params}, obs)['value'][:, 0].reshape(B, n_draws)
    hands_root = dr.hands[:, :, root]                             # (B, n, 34)
    nmelds = jnp.broadcast_to(stacked.n_melds[:, root][:, None], (B, n_draws))
    win = jax.vmap(jax.vmap(rules.is_win_counts))(hands_root, nmelds)
    val = jnp.where(win, SELFDRAW_VALUE, v)
    val = jnp.where(ok, val, 0.0)
    count = ok.sum(-1).astype(jnp.float32)
    jumpv = jnp.where(count > 0, val.sum(-1) / jnp.maximum(count, 1.0), 0.0)
    return jnp.where(dead, 0.0, jumpv)


# ---------------------------------------------------------------------------
# 单候选弃牌的 Q
# ---------------------------------------------------------------------------

def _q_discard(params, model, state, a, key, n_draws):
    """根玩家（state.turn）在 DISCARD state 弃 a 的 Q（score/3 尺度）。"""
    d = state.turn.astype(jnp.int32)
    t = a.astype(jnp.int32)
    st = state.replace(hands=state.hands.at[d, t].add(-1),
                       pending_tile=a.astype(jnp.int8),
                       drawn=jnp.int8(-1))
    mask9 = env._claim_ask_mask(st)
    bits = ((mask9 >> _POS9) & 1).astype(bool)                    # [9]
    cs9 = (d + _OFFS9.astype(jnp.int32)) % 4                      # 9 个声明位的玩家
    obs9 = jax.vmap(obs_for_player, in_axes=(None, 0, None))(st, cs9, a)
    resp = model.apply({'params': params}, obs9)['response']      # [9, 4]
    p_claim = jax.nn.sigmoid(resp[jnp.arange(9), _CLAIM_RESP_IDX] - resp[:, 0])
    p_claim = jnp.where(bits, p_claim, 0.0)
    stacked, dead = _branch_states(st, a)
    jumpv = _jump_values(params, model, stacked, d,
                         jax.random.split(key, 7), dead, n_draws)  # [7]
    val9 = jnp.concatenate([jnp.full(3, HU_VALUE, jnp.float32), jumpv[:6]])
    # 按 env 声明顺序（胡→杠→碰，各 offset 1..3）走 walk
    rem = jnp.float32(1.0)
    acc = jnp.float32(0.0)
    for i in range(9):
        acc = acc + rem * p_claim[i] * val9[i]
        rem = rem * (1.0 - p_claim[i])
    return acc + rem * jumpv[6]


# ---------------------------------------------------------------------------
# improved policy
# ---------------------------------------------------------------------------

def improved_policy(params, model, state, key, k=8, n_draws=2, beta=8.0):
    """单个 DISCARD-phase State -> (pi_prime[34], q_top[k], v)。

    pi_prime: masked softmax(policy_logits + beta*(Q_completed - V))，非法动作概率 0。
    q_top: top-k 候选（按含噪 logits 排序）的 Q；候选不足 k 个（合法动作 < k）的槽位
    用 V 补齐。v: 根状态 value（score/3 尺度）。
    对非 DISCARD / done 状态返回无意义但有限（无 NaN）的值，调用方按 phase 过滤。
    """
    key_g, key_c = jax.random.split(key)
    obs = observe(state)
    out = model.apply({'params': params}, obs[None])
    logits = out['policy'][0]                                     # (34,)
    v = out['value'][0, 0]
    legal = env.legal_mask(state)[:34]
    u = jnp.clip(jax.random.uniform(key_g, (34,)), 1e-9, 1.0 - 1e-9)
    g = -jnp.log(-jnp.log(u))
    pert = jnp.where(legal, logits + g, NEG)
    topv, topa = jax.lax.top_k(pert, k)
    valid = topv > NEG / 2
    keys = jax.random.split(key_c, k)
    q_raw = jax.vmap(lambda aa, kk: _q_discard(params, model, state, aa, kk, n_draws))(
        topa, keys)
    q_top = jnp.where(valid, q_raw, v)
    # completed-Q：非 top-k 合法动作 Q=V（top_k 返回的索引互异，scatter 安全）
    q_full = jnp.full(34, v, jnp.float32).at[topa].set(q_top)
    adj = jnp.where(legal, logits + beta * (q_full - v), NEG)
    pi = jax.nn.softmax(adj)
    return pi, q_top, v
