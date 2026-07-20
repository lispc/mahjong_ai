# -*- coding: utf-8 -*-
"""arena BeliefExpectimaxAgent（BeliefExp-lite）的 JAX 纯函数移植，用于 PPO 对手池。

决策结构逐位镜像 `BeliefExpectimaxAgent.next_with_trace`
（algo/agents/belief_expectimax.py，arena 默认参数 max_candidates=8、
defense_margin=0.03、tenpai_min_wait=4、eval_backend='legacy'）：
1. 候选：闭手 unique 弃牌，eval0（Cython pair_coef=1.0 整数 metric）预选 top-8，
   顺序 = (score, tile id) 降序。
2. 每候选：eval2 进攻分 + opponent.tile_danger 危险分。
3. 危险信号（`context.tenpai_players - {self}` 非空，或任一对手
   player_danger_level>=1）存在时：在 offense >= best_offense - margin 的候选中
   取 danger 最小者（margin = 0.03 + 0.02*报听对手数）；否则取 offense 最大者。
   平局均保持 top-8 顺序（Python stable sort 语义）。

【关键事实：BeliefExp 的 eval2 进攻分 = 空 Context 版】
BeliefExp 想把 `context.used` 传进 algo.eval2 做剩余分布修正，但 algo.eval2 的
Cython 快路径只在 `hasattr(c, 'all_tiles_as_dict')` 时才使用 used，而
context.Context 没有该方法 → used 被静默忽略（实测同手牌 used ctx 与空 ctx
逐值相等）。parity 以 Python 实际计算为准 → 进攻分直接复用 eval2jax 的空
Context 整数分子路径（N = 123*122*eval2），margin 比较换算成整数差
（floor(margin*15006)，边界距整数 >=0.18，float 噪声不可能翻转）。

danger 表逐位镜像 algo/eval/opponent.py（纯查表 + 弃牌序列统计）：
- all_seen = 全场未声明弃牌计数（jaxenv state.discards 总和；ContextV3 的
  used/all_seen 与之等价——BeliefExp 忽略 'meld' 消息，不跟踪副露牌，本移植
  同样只用 discards；注意自对弈中无副露，混合池中别家副露的牌同样不计）。
- safety = 现物 1.0+0.1*seen；否则 base(字.35/幺九.25/二八.10/中-.10)
  + 0.05*Σ邻牌(同花色±1±2)seen。
- player_suit_weights：弃牌序列四花色计数，(n-count+0.5)/Σ 归一（n<3 均匀）。
- player_danger_level：n>=8 时 近6巡均 safety > 早期均 safety+0.15 → +1，
  近6巡中张(4-6)>=2 → +1。
- danger = max(0, (1-safety)*(1+0.3*level) + 0.5*(suit_weight-0.25))，对手取 max。

浮点口径：全部浮点用 jnp.result_type(float) —— parity 测试开 jax_enable_x64
（float64，danger 逐值容差 1e-9 可达）；PPO 训练默认 x64 关（float32，
决策级影响可忽略：整数分子/tie-break 路径不受影响，仅 danger 值本身有
~1e-7 噪声，数学严格平局的 tie 结构由相同计算路径保持）。

报听镜像 declare_tenpai 启发式：总弃牌数 <12 → 否；shanten(13)==0；
待牌（is_win 且 hand<4 且 remaining>0，remaining = 4 - all_seen - hand）非空；
总待牌 >=4 或存在现物待牌（all_seen>0）→ 是。
不变式同 eval2jax：本 agent 不碰不杠 => n_melds==0、DISCARD 时闭手恒 14 张
（Python 侧 meld_eval 恒空）。

接口镜像 jaxenv/greedy.py / eval2jax.py：belief_action(state) -> int8；批量 vmap。
"""

import numpy as np

import jax
import jax.numpy as jnp

from . import env, rules
from .eval2jax import _eval0_int, _eval2_num13

# ---------------------------------------------------------------------------
# danger 表常量（镜像 algo/eval/opponent.py）
# ---------------------------------------------------------------------------

_IDX = jnp.arange(34, dtype=jnp.int32)
_EYE = jnp.eye(34, dtype=jnp.int32)

# 位置安全度：字 0.35 / 幺九 0.25 / 二八 0.10 / 中张 -0.10
_BASE_SAFETY = np.array(
    [0.25 if (i % 9 in (0, 8)) else 0.10 if (i % 9 in (1, 7)) else -0.10
     for i in range(27)] + [0.35] * 7, np.float64)

# 同花色 ±1/±2 邻牌掩码（字牌无邻牌）
_NBR = np.zeros((34, 34), bool)
for _s in range(3):
    for _r in range(9):
        for _d in (-2, -1, 1, 2):
            if 0 <= _r + _d <= 8:
                _NBR[_s * 9 + _r, _s * 9 + _r + _d] = True

# 每 idx 的花色 id（0万 1条 2筒 3字）
_SUIT = np.array([0] * 9 + [1] * 9 + [2] * 9 + [3] * 7, np.int32)

# 中张（数牌 rank 4-6）掩码（player_danger_level 用）
_MID = np.array([i < 27 and i % 9 in (3, 4, 5) for i in range(34)], bool)

# margin 整数阈值：floor((0.03 + 0.02k) * 123*122)，k = 报听对手数
_MARGIN_INT = np.array([int((0.03 + 0.02 * k) * (123 * 122)) for k in range(4)],
                       np.int32)  # [450, 750, 1050, 1350]

_POS_BIG = jnp.int32(999)


# ---------------------------------------------------------------------------
# danger：对手特征 + tile danger
# ---------------------------------------------------------------------------

def _player_features(discard_seq, discard_len):
    """(4,64) int8 弃牌序列 + (4,) int8 长度 -> (suit_weights(4,4), level(4,) int32)。

    逐位镜像 opponent.player_suit_weights / player_danger_level（window=6）。
    """
    fx = jnp.result_type(float)
    valid = discard_seq >= 0                                  # (4,64)
    seq = jnp.clip(discard_seq, 0, 33).astype(jnp.int32)
    n = discard_len.astype(jnp.int32)                         # (4,)
    # 花色权重
    suits = jnp.asarray(_SUIT)[seq]                           # (4,64)
    counts = (jax.nn.one_hot(suits, 4, dtype=fx)
              * valid[..., None].astype(fx)).sum(axis=1)      # (4,4)
    nf = n.astype(fx)
    raw = nf[:, None] - counts + 0.5
    weights = raw / raw.sum(axis=1, keepdims=True)
    weights = jnp.where((n < 3)[:, None], jnp.full((4, 4), 0.25, fx), weights)
    # 危险等级
    pos = jnp.arange(64, dtype=jnp.int32)[None, :]
    recent = valid & (pos >= (n - 6)[:, None])
    early = valid & (pos < (n - 6)[:, None])
    bs = jnp.asarray(_BASE_SAFETY, fx)[seq]                   # (4,64)
    early_mean = (bs * early.astype(fx)).sum(1) / jnp.maximum(early.sum(1), 1).astype(fx)
    recent_mean = (bs * recent.astype(fx)).sum(1) / jnp.maximum(recent.sum(1), 1).astype(fx)
    mid = (jnp.asarray(_MID)[seq] & recent).sum(1)
    lvl = ((recent_mean > early_mean + 0.15).astype(jnp.int32)
           + (mid >= 2).astype(jnp.int32))
    lvl = jnp.where(n >= 8, lvl, jnp.int32(0))                # n < window+2 -> 0
    return weights, lvl


def _danger_matrix(state):
    """-> (34,4)：每张牌对每个座位对手的 danger（座位是否对手由调用方过滤）。"""
    fx = jnp.result_type(float)
    all_seen = state.discards.astype(jnp.int32).sum(axis=0)   # (34,) 未声明弃牌
    seen_f = all_seen.astype(fx)
    base = jnp.asarray(_BASE_SAFETY, fx)                      # (34,)
    nbr_seen = jnp.asarray(_NBR, fx) @ seen_f                 # (34,)
    safety = jnp.where(all_seen > 0, 1.0 + 0.1 * seen_f,
                       base + 0.05 * nbr_seen)                # (34,)
    weights, lvl = _player_features(state.discard_seq, state.discard_len)
    base_danger = 1.0 - safety                                # (34,)
    signal = 1.0 + 0.3 * lvl.astype(fx)                       # (4,)
    suit_w = weights[:, jnp.asarray(_SUIT)]                   # (4,34)
    danger = (base_danger[None, :] * signal[:, None]
            + 0.5 * (suit_w - 0.25))                          # (4,34)
    danger = jnp.maximum(jnp.asarray(0.0, fx), danger)
    n = state.discard_len.astype(jnp.int32)
    danger = jnp.where((n > 0)[:, None], danger, jnp.asarray(0.0, fx))
    return danger.T                                           # (34,4)


def tile_danger_vec(state):
    """-> (34,)：每张牌对 3 个对手取 max 的 danger（镜像 opponent.tile_danger）。"""
    turn = state.turn.astype(jnp.int32)
    opp = jnp.arange(4, dtype=jnp.int32) != turn
    dm = _danger_matrix(state)                                # (34,4)
    fx = jnp.result_type(float)
    return jnp.where(opp[None, :], dm, jnp.asarray(0.0, fx)).max(axis=1)


def _danger_signal(state):
    """-> (bool, int32 tenpai_opp)：是否进入安全模式 + 报听对手数。"""
    turn = state.turn.astype(jnp.int32)
    opp = jnp.arange(4, dtype=jnp.int32) != turn
    _, lvl = _player_features(state.discard_seq, state.discard_len)
    tenpai_opp = (state.locked & opp).sum().astype(jnp.int32)
    signal = (tenpai_opp > 0) | (((lvl >= 1) & opp).any())
    return signal, tenpai_opp


# ---------------------------------------------------------------------------
# DISCARD：eval0 top-8 + eval2 进攻 + danger 防守 + margin 规则
# ---------------------------------------------------------------------------

def _discard_internals(state):
    """discard 选择的全部中间量（belief_action 与 belief_labels 共享同一计算路径）。

    -> dict:
        chosen_raw  int32  选择结果（不含合法集兜底，调用方处理）
        top8        int8[8]   top-8 候选 idx（top 顺序；不足 8 个合法候选末尾 -1）
        offense     int32[8]  top-8 对应的 eval2 整数分子（填充位 -(1<<30)）
        danger8     f32/f64[8] top-8 对应的 danger 值（填充位 -1）
        signal      bool   危险信号（safe mode）
        margin_f    float  margin = 0.03 + 0.02*报听对手数（展示用；选择用整数阈值）
        best        int32  top-8 内最大进攻分子
    """
    turn = state.turn.astype(jnp.int32)
    hand = state.hands[turn].astype(jnp.int32)
    cand = hand > 0                                           # (34,)
    hands13 = hand[None, :] - _EYE                            # (34,34)
    # eval0 预选 top-8，顺序 (score, tile) 降序 => key = e0*64 + idx
    e0 = jax.vmap(_eval0_int)(hands13)                        # (34,)
    e0_key = jnp.where(cand, e0 * 64 + _IDX, jnp.int32(-1))
    thr = jnp.sort(e0_key)[-8]                                # 第 8 名（<8 候选时为 -1）
    top8_mask = cand & (e0_key >= thr)
    pos = jnp.argsort(jnp.argsort(-e0_key))                   # top 内名次（0=最优）
    # eval2 进攻分（空 Context 整数分子；arena 实际口径，见模块 docstring）
    n_score = jax.vmap(_eval2_num13)(hands13)                 # (34,)
    n_masked = jnp.where(top8_mask, n_score, jnp.int32(-(1 << 30)))
    best = n_masked.max()
    # danger + 安全模式
    danger = tile_danger_vec(state)                           # (34,)
    signal, tenpai_opp = _danger_signal(state)
    margin_int = jnp.asarray(_MARGIN_INT)[jnp.clip(tenpai_opp, 0, 3)]
    safe = top8_mask & ((best - n_score) <= margin_int)
    # 危险分支：danger 最小，平局取 top 名次小者
    fx = jnp.result_type(float)
    dmin = jnp.where(safe, danger, jnp.asarray(np.inf, fx)).min()
    pick_pos = jnp.where(safe & (danger == dmin), pos, _POS_BIG).min()
    tile_danger_branch = jnp.argmin(jnp.where(pos == pick_pos, _IDX, _POS_BIG))
    # 无危险分支：offense 最大，平局取 top 名次小者（名次经 63-pos 编入键）
    sel_key = jnp.where(top8_mask, n_score * 64 + (63 - pos), jnp.int32(-1))
    tile_offense_branch = jnp.argmax(sel_key).astype(jnp.int32)
    chosen_raw = jnp.where(signal, tile_danger_branch, tile_offense_branch)
    # top-8 数组（top 顺序；非法名次填充）
    order = jnp.argsort(-e0_key)[:8]                          # (8,)
    valid8 = top8_mask[order]
    top8_idx = jnp.where(valid8, order, jnp.int32(-1)).astype(jnp.int8)
    offense8 = jnp.where(valid8, n_score[order], jnp.int32(-(1 << 30)))
    danger8 = jnp.where(valid8, danger[order],
                        jnp.asarray(-1.0, fx)).astype(jnp.float32)
    margin_f = (0.03 + 0.02 * tenpai_opp.astype(fx)).astype(jnp.float32)
    return dict(chosen_raw=chosen_raw, top8=top8_idx, offense=offense8,
                danger8=danger8, signal=signal, margin_f=margin_f, best=best)


def discard_choice(state):
    """-> int32 弃牌 idx（不套合法掩码；调用方负责。parity 测试直接可用）。"""
    return _discard_internals(state)['chosen_raw']


def belief_labels(state):
    """单个 State -> 标签 dict（数据生产用；内部与 belief_action 完全同一计算）。

    chosen 直接取 belief_action(state)（含各 phase 行为与合法兜底，一致性由
    构造保证；XLA CSE 会合并重复子计算）。其余字段见 _discard_internals。
    """
    d = _discard_internals(state)
    return dict(
        chosen=belief_action(state),
        top8=d['top8'],
        offense=d['offense'],
        danger=d['danger8'],
        defense_flag=d['signal'],
        margin=d['margin_f'],
        best_offense=d['best'],
    )


def _discard_action(state, mask):
    act = discard_choice(state)
    # 锁手强制弃牌等：选择不在合法集时回退合法集第一个（镜像引擎强制打摸牌）
    return jnp.where(mask[act], act,
                     jnp.argmax(mask[:34].astype(jnp.int32))).astype(jnp.int32)


# ---------------------------------------------------------------------------
# TENPAI：declare_tenpai 启发式
# ---------------------------------------------------------------------------

def _tenpai_action(state):
    turn = state.turn.astype(jnp.int32)
    hand = state.hands[turn].astype(jnp.int32)                # 弃后 13 张
    all_seen = state.discards.astype(jnp.int32).sum(axis=0)   # (34,)
    n_disc = state.discard_len.astype(jnp.int32).sum()
    remaining = jnp.int32(4) - all_seen - hand                # (34,)
    wins = jax.vmap(lambda k: rules.is_win_counts(
        (hand + _EYE[k]).astype(jnp.int32), jnp.int8(0)))(_IDX)
    waits = wins & (hand < 4) & (remaining > 0)
    total_wait = jnp.where(waits, remaining, 0).sum()
    genmai = (waits & (all_seen > 0)).any()                   # 有现物待牌
    yes = ((n_disc >= 12)
           & (rules.shanten_counts(hand, jnp.int8(0)) == 0)
           & waits.any()
           & ((total_wait >= 4) | genmai))
    return jnp.where(yes, jnp.int8(env.A_TENPAI_YES), jnp.int8(env.A_TENPAI_NO))


# ---------------------------------------------------------------------------
# belief_action
# ---------------------------------------------------------------------------

def belief_action(state):
    """单个 State -> int8 action（40 动作空间）。批量请 vmap(belief_action)。"""
    mask = env.legal_mask(state)
    disc = _discard_action(state, mask)
    claim = jnp.where(state.claim_stage == jnp.int8(env.STAGE_HU),
                      jnp.int8(env.A_HU), jnp.int8(env.A_PASS))
    act = jnp.where(state.phase == jnp.int8(env.PHASE_DISCARD), disc,
          jnp.where(state.phase == jnp.int8(env.PHASE_CLAIM), claim,
                    _tenpai_action(state)))
    # 兜底：done 等 mask 全 False 时回退到合法集第一个动作（无实际影响）
    return jnp.where(mask[act.astype(jnp.int32)], act.astype(jnp.int8),
                     jnp.argmax(mask.astype(jnp.int32)).astype(jnp.int8))
