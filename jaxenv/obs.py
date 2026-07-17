# -*- coding: utf-8 -*-
"""观测函数（JAX，可 jit/vmap）：与部署特征严格对齐。

逐条对齐 ``algo/nn/features.py::extract_features``（175 维）+ 部署 agent
``algo/agents/ppo_agent.py::PPOAgent`` 的特征构造行为（含 quirk）：

- [0:34]   手牌通道 = (闭手 + 每个副露牌种 ×3) / 4。
           ×3 依据：部署 full_hand() = cur + melds 列表，Hybrid 共享 melds 列表的
           quirk 使每种副露出现 3 次（AGENTS.md 注意事项 13；杠也只算 3 张）。
           CLAIM phase 再加 pending_tile ×1（PPOAgent._response_features 把
           offered tile 并入 full_hand()+[tile_val]）。
           TENPAI phase 用弃牌后 13 张（env 中弃牌已生效；此刻必有 n_melds==0）。
- [34:68]  剩余通道 = (4 − 手牌通道分子 − 四家 discards 合计) / 4。
           副露牌不减（ContextV3.used 只记弃牌、不记副露的 quirk）；
           CLAIM 时手牌含 pending（offered tile 同时从剩余里扣掉）。
           【TENPAI quirk】报听决策时部署 PPOAgent.next() 已把刚弃的牌
           see_tile 进自己的 ctx.used / ctx.discards，因此 TENPAI phase 下
           pending_tile 额外计入 used（剩余通道）与进度。
- [68:170] 三家弃牌计数通道：按绝对座位 0..3 顺序跳过自己，各 /20。
- [170:174] 报听 flag 4 维 = locked[0..3]（座位序，含自己；TENPAI 决策当下
            自己尚未置 locked，与引擎 declare_tenpai 返回后才加锁一致）。
- [174]    进度 = min(1, (四家 discards 总数 + TENPAI?1:0) / 84)。

已知偏差（不可消除，env State 无相应字段，改 env.py 超出本阶段范围）：
锁手玩家的强制弃牌在引擎 _discard_step 锁手分支中不经 see_tile，因此部署中
该玩家自己的 ctx.used / ctx.discards 不含其报听后的弃牌，而本函数仍计入。
仅影响「锁手玩家自己被问胡（CLAIM-HU）」这一刻的特征（剩余通道与进度），
且差异上限为该玩家报听后弃牌数；其余所有决策点严格一致。
"""

import jax
import jax.numpy as jnp

from .env import PHASE_CLAIM, PHASE_TENPAI

OBS_DIM = 175


def actor_of(state):
    """当前行动玩家座位：CLAIM=(turn+claim_offset)%4，其余=turn。"""
    return jnp.where(
        state.phase == jnp.int8(PHASE_CLAIM),
        (state.turn.astype(jnp.int32) + state.claim_offset.astype(jnp.int32)) % 4,
        state.turn.astype(jnp.int32))


def observe(state):
    """单个 State -> float32[175]，当前行动玩家视角。批量请 vmap(observe)。"""
    p = actor_of(state)
    is_claim = state.phase == jnp.int8(PHASE_CLAIM)
    is_tenpai = state.phase == jnp.int8(PHASE_TENPAI)
    # pending_tile == -1 时 one_hot 全 0
    pend = jax.nn.one_hot(state.pending_tile.astype(jnp.int32), 34, dtype=jnp.float32)

    # 手牌分子：闭手 + 副露牌种×3（+ CLAIM 时 offered tile）
    hand = (state.hands[p].astype(jnp.float32)
            + 3.0 * (state.meld_counts[p] > 0).astype(jnp.float32))
    hand = hand + jnp.where(is_claim, pend, 0.0)
    hand_ch = hand / 4.0                                            # [0:34]

    disc = state.discards.astype(jnp.float32)                       # (4,34)
    disc_total = disc.sum(axis=0)                                   # (34,)
    # 部署 quirk：TENPAI 决策时 pending 已进自己 used
    used_extra = jnp.where(is_tenpai, pend, 0.0)
    rem = (4.0 - hand - disc_total - used_extra) / 4.0              # [34:68]

    # 三家弃牌：绝对座位 0..3 跳过自己。第 i 个输出座位 = i + (i >= p)
    seats = jnp.arange(3, dtype=jnp.int32) + (jnp.arange(3, dtype=jnp.int32) >= p)
    opp = (disc[seats] / 20.0).reshape(-1)                          # [68:170]

    flags = state.locked.astype(jnp.float32)                        # [170:174]

    n_disc = disc.sum() + jnp.where(is_tenpai, 1.0, 0.0)
    progress = jnp.minimum(1.0, n_disc / 84.0)                      # [174]

    return jnp.concatenate([hand_ch, rem, opp, flags,
                            progress[None]]).astype(jnp.float32)


def obs_for_player(state, p, pending):
    """指定玩家 p 视角、pending tile=pending 的 CLAIM 语义观测（搜索响应头用）。

    等价于把 state 置为 phase=CLAIM、pending_tile=pending、claim_offset=(p-turn)%4
    后调用 observe（offered tile 按 CLAIM 语义并入 p 的手牌通道与剩余通道）。
    搜索中 p 恒为某对手（offset 1..3）；p==turn 时 offset=0，仅为定义完备。
    """
    off = ((p.astype(jnp.int32) - state.turn.astype(jnp.int32)) % 4).astype(jnp.int8)
    st = state.replace(phase=jnp.int8(PHASE_CLAIM),
                       pending_tile=jnp.asarray(pending, jnp.int8),
                       claim_offset=off)
    return observe(st)
