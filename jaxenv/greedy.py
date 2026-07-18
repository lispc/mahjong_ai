# -*- coding: utf-8 -*-
"""Shanten 贪心对手（JAX 纯函数，可 jit/vmap），用于 PPO 对手池。

策略（jaxenv/test_greedy.py 里有 driver/engine.py 的 Python 镜像实现）：
- DISCARD：枚举 34 种弃牌（one-hot 减法构造候选手牌，vmap rules.shanten_counts
  算弃后向听），取最小向听；平局按 字牌(2) > 幺九(1) > 中张(0) 分组、同组取
  idx 最大者。候选先与 phase 合法集（env.legal_mask，含锁手强制弃牌）取交集。
- CLAIM：当前阶段为胡（claim_stage==STAGE_HU）则胡(37)，否则 pass(34)
  （不碰不杠，保持简单）。
- TENPAI：恒 yes(38)。
- done 状态返回 0（env.step 对 done 为 no-op，值无意义但有限）。
"""

import jax
import jax.numpy as jnp

from . import env, rules

# 分组优先级：字牌 idx>=27 → 2；幺九（每花色 rank 1/9，即 idx%9 ∈ {0,8}）→ 1；中张 → 0
_IDX = jnp.arange(34, dtype=jnp.int32)
GROUP_PRIO = (( _IDX >= 27).astype(jnp.int32) * 2
              + (((_IDX < 27) & ((_IDX % 9 == 0) | (_IDX % 9 == 8)))).astype(jnp.int32))

_BIG = jnp.int32(1 << 28)


def _discard_action(state, mask):
    """DISCARD 分支：合法集内最小化弃后向听，平局按 GROUP_PRIO/idx tie-break。"""
    turn = state.turn.astype(jnp.int32)
    hand = state.hands[turn]                                    # (34,) int8
    n_melds = state.n_melds[turn]
    cands = hand[None, :] - jnp.eye(34, dtype=jnp.int8)         # (34,34) one-hot 减法
    sh = jax.vmap(lambda c: rules.shanten_counts(c, n_melds))(cands).astype(jnp.int32)
    # 排序键：向听升序为主键；组优先级降序、idx 降序为次键（键越小越优）
    key = sh * 1000 - GROUP_PRIO * 100 - _IDX
    key = jnp.where(mask[:34], key, _BIG)                       # 与合法集取交集
    return jnp.argmin(key).astype(jnp.int8)


def greedy_action(state):
    """单个 State -> int8 action（40 动作空间）。批量请 vmap(greedy_action)。"""
    mask = env.legal_mask(state)
    disc = _discard_action(state, mask)
    claim = jnp.where(state.claim_stage == jnp.int8(env.STAGE_HU),
                      jnp.int8(env.A_HU), jnp.int8(env.A_PASS))
    act = jnp.where(state.phase == jnp.int8(env.PHASE_DISCARD), disc,
          jnp.where(state.phase == jnp.int8(env.PHASE_CLAIM), claim,
                    jnp.int8(env.A_TENPAI_YES)))
    # 兜底：done 等 mask 全 False 时回退到合法集第一个动作（无实际影响）
    return jnp.where(mask[act.astype(jnp.int32)], act,
                     jnp.argmax(mask.astype(jnp.int32)).astype(jnp.int8))
