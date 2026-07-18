# -*- coding: utf-8 -*-
"""晋北麻将（推倒胡变体）JAX 函数式对局环境（Pgx 风格，可 jit/vmap）。

规则逐条对齐 driver/engine.py（唯一事实来源）：
1. 34 种牌 ×4 = 136 张；内部索引 0-33（0-8 万, 9-17 条, 18-26 饼, 27-33 字）。
2. 4 玩家各 13 张起手，座位 0 先摸第一张。
3. 回合流程：摸牌（牌山头；杠后补牌从牌山尾）→ 自摸自动胡（无选择）→ 弃牌
   → 报听决策（见 quirk 注记）→ 其余 3 家声明。
4. 声明：逆时针 offset 1→2→3，按 胡→杠→碰 三阶段顺序询问；
   胡任一 yes 即终局（ron，dealer=弃牌者）；杠=3 张闭手+弃牌成副露并从尾部补牌；
   碰=2 张闭手+弃牌成副露，不摸牌直接弃牌。
   与引擎决策点的一个刻意差异：引擎在杠/碰阶段只询问满足条件的玩家
   （_can_gang/_can_peng 且未锁手），本环境自动跳过不可能声明的玩家，
   因此 phase=CLAIM 时「声明+pass」恒为合法动作；胡阶段引擎对不能胡的玩家
   也会调用 respond_hu（基类内部判否），此处同样自动跳过 —— 对局结果完全等价。
5. 锁手（报听后）：摸牌→自摸自动胡，否则强制打出摸到的牌；锁手玩家不能被问
   碰/杠，但仍能胡。
6. 胡牌型：14 张 = 4 面子 + 1 对子，或七对子（仅无副露）；m 副露时闭手需
   (4-m) 面子 + 1 对子（即 algo/eval/v2.py 语义；注意 Python 引擎基类 agent
   因副露记账少 2 张而有副露后无法胡牌的 quirk，本环境实现的是正确规则）。
7. 牌山摸完（head > tail）→ 流局。无计分，只有 winner/win_type/dealer。
8. 【引擎 quirk 对齐】arena 引擎只有 len(full_hand())==13（即无副露）的玩家
   才会被问报听。本环境复制此行为：仅 n_melds==0 且弃牌后向听==0 时才提供
   declare 决策点（phase=TENPAI）。

动作空间 A=40：0-33 弃某种牌, 34=pass, 35=碰, 36=杠, 37=胡, 38=报听yes, 39=报听no。
"""

from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from . import rules

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

N_ACTIONS = 40
A_PASS = 34
A_PENG = 35
A_GANG = 36
A_HU = 37
A_TENPAI_YES = 38
A_TENPAI_NO = 39

PHASE_DISCARD = 0   # 摸牌后待弃牌（含碰后 skip_draw）
PHASE_CLAIM = 1     # 待声明响应
PHASE_TENPAI = 2    # 待报听决策

STAGE_HU = 1
STAGE_GANG = 2
STAGE_PENG = 3

WIN_NONE = 0
WIN_SELF = 1
WIN_RON = 2
WIN_DRAW = 3

REWARD_SCORE = 'score'        # 自摸赢家 +3；点和赢家 +1、放炮者 -1；流局全 0
REWARD_WINLOSS = 'winloss'    # 赢家 +1、其余 -1；流局全 0
REWARD_DD = 'score_dd'        # 同 score，但流局全员 -0.25（防 from-scratch 冷启动停滞）
DEFAULT_REWARD_KIND = REWARD_SCORE

WALL_SIZE = 136
MAX_DISCARDS = 64


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@struct.dataclass
class State:
    wall: jnp.ndarray          # int8[136] 牌山（牌种 0-33），head 端摸牌，tail 端补牌
    wall_head: jnp.ndarray     # int16 下一个摸牌位置（递增）
    wall_tail: jnp.ndarray     # int16 下一个补牌位置（递减）；head > tail 即摸空
    hands: jnp.ndarray         # int8[4,34] 闭手计数
    n_melds: jnp.ndarray       # int8[4] 副露数
    meld_counts: jnp.ndarray   # int8[4,34] 副露所含牌（碰=3 张含被碰牌，杠=4 张含被杠牌）
    discards: jnp.ndarray      # int8[4,34] 弃牌堆计数（被声明的牌不进弃牌堆）
    discard_seq: jnp.ndarray   # int8[4,64] 弃牌顺序，-1 填充
    discard_len: jnp.ndarray   # int8[4]
    turn: jnp.ndarray          # int8 当前行动玩家（声明阶段=弃牌者）
    pending_tile: jnp.ndarray  # int8 待声明的弃牌（-1 无）
    drawn: jnp.ndarray         # int8 本回合摸到的牌（碰后为 -1）
    claim_stage: jnp.ndarray   # int8 0 无 / 1 胡 / 2 杠 / 3 碰
    claim_offset: jnp.ndarray  # int8 当前询问第几家（1..3）
    claim_mask: jnp.ndarray    # int16 9bit：可声明位置位图（阶段×offset），声明阶段不变
    locked: jnp.ndarray        # bool[4] 报听锁手
    phase: jnp.ndarray         # int8
    done: jnp.ndarray          # bool
    winner: jnp.ndarray        # int8（-1 无）
    win_type: jnp.ndarray      # int8
    dealer: jnp.ndarray        # int8 放炮者（-1 无）
    n_draws: jnp.ndarray       # int16 已摸牌数（诊断/局长统计）
    reward_kind: str = struct.field(pytree_node=False, default=REWARD_SCORE)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

def init(rng, reward_kind=None):
    """洗牌发牌，座位 0 摸第一张；若天胡则直接终局（引擎 add() 语义）。"""
    reward_kind = DEFAULT_REWARD_KIND if reward_kind is None else reward_kind
    perm = jax.random.permutation(rng, WALL_SIZE)
    wall = (jnp.arange(WALL_SIZE, dtype=jnp.int16) // 4).astype(jnp.int8)[perm]
    deal = wall[:52].reshape(4, 13).astype(jnp.int32)
    hands = jax.vmap(lambda w: jnp.zeros(34, jnp.int8).at[w].add(1))(deal)
    first = wall[52].astype(jnp.int32)
    hands = hands.at[0, first].add(1)
    win0 = rules.is_win_counts(hands[0], jnp.int8(0))
    return State(
        wall=wall,
        wall_head=jnp.int16(53),
        wall_tail=jnp.int16(WALL_SIZE - 1),
        hands=hands,
        n_melds=jnp.zeros(4, jnp.int8),
        meld_counts=jnp.zeros((4, 34), jnp.int8),
        discards=jnp.zeros((4, 34), jnp.int8),
        discard_seq=jnp.full((4, MAX_DISCARDS), -1, jnp.int8),
        discard_len=jnp.zeros(4, jnp.int8),
        turn=jnp.int8(0),
        pending_tile=jnp.int8(-1),
        drawn=wall[52],
        claim_stage=jnp.int8(0),
        claim_offset=jnp.int8(0),
        claim_mask=jnp.int16(0),
        locked=jnp.zeros(4, bool),
        phase=jnp.int8(PHASE_DISCARD),
        done=win0,
        winner=jnp.where(win0, jnp.int8(0), jnp.int8(-1)),
        win_type=jnp.where(win0, jnp.int8(WIN_SELF), jnp.int8(WIN_NONE)),
        dealer=jnp.int8(-1),
        n_draws=jnp.int16(1),
        reward_kind=reward_kind,
    )


# ---------------------------------------------------------------------------
# legal_mask
# ---------------------------------------------------------------------------

def legal_mask(state):
    """bool[40]：各 phase 下只开放合法动作；done 时全 False。"""
    turn = state.turn.astype(jnp.int32)
    can_discard = state.hands[turn] > 0                          # [34]
    # 锁手玩家只能打出摸到的牌（引擎 _discard_step 锁手分支）
    forced = jnp.zeros(34, bool).at[state.drawn.astype(jnp.int32)].set(True)
    disc = jnp.where(state.locked[turn], forced, can_discard)
    mask_discard = jnp.concatenate([disc, jnp.zeros(6, bool)])

    # 声明阶段：自动跳过已保证声明可行，恒为 {pass, 当前阶段动作}
    claim_action = (38 - state.claim_stage).astype(jnp.int32)    # 1->37, 2->36, 3->35
    mask_claim = jnp.zeros(N_ACTIONS, bool).at[A_PASS].set(True).at[claim_action].set(True)

    mask_tenpai = jnp.zeros(N_ACTIONS, bool).at[A_TENPAI_YES].set(True).at[A_TENPAI_NO].set(True)

    mask = jnp.where(state.phase == jnp.int8(PHASE_DISCARD), mask_discard,
           jnp.where(state.phase == jnp.int8(PHASE_CLAIM), mask_claim, mask_tenpai))
    return jnp.where(state.done, jnp.zeros(N_ACTIONS, bool), mask)


# ---------------------------------------------------------------------------
# 内部转移
# ---------------------------------------------------------------------------

def _claim_ask_mask(state):
    """计算当前 pending_tile 的 9bit 可声明位图（pos = (stage-1)*3 + offset-1）。"""
    t = state.pending_tile.astype(jnp.int32)
    d = state.turn.astype(jnp.int32)
    mask = jnp.int16(0)
    for stage in (1, 2, 3):
        for off in (1, 2, 3):
            p = (d + off) % 4
            if stage == STAGE_HU:
                # 胡：锁手玩家也可被问
                counts = state.hands[p].at[t].add(1)
                a = rules.is_win_counts(counts, state.n_melds[p])
            elif stage == STAGE_GANG:
                a = (~state.locked[p]) & (state.hands[p, t] >= 3)
            else:
                a = (~state.locked[p]) & (state.hands[p, t] >= 2)
            mask = mask | (a.astype(jnp.int16) << jnp.int16((stage - 1) * 3 + off - 1))
    return mask


def _next_claim_pos(mask, after_pos):
    """位图中 after_pos 之后的第一个可声明位置；9 表示没有。"""
    idxs = jnp.arange(9, dtype=jnp.int16)
    bits = ((mask >> idxs) & 1).astype(bool)
    cand = jnp.where(bits & (idxs > after_pos), idxs, jnp.int16(9))
    return cand.min()


def _draw_for(state, player, from_tail):
    """player 摸一张牌（from_tail=杠后补牌），自摸自动胡；摸空则流局。"""
    empty = state.wall_head > state.wall_tail
    idx = jnp.clip(jnp.where(from_tail, state.wall_tail, state.wall_head), 0, WALL_SIZE - 1)
    tile = state.wall[idx]
    p32 = player.astype(jnp.int32)
    st = state.replace(
        wall_head=state.wall_head + jnp.where(from_tail, jnp.int16(0), jnp.int16(1)),
        wall_tail=state.wall_tail - jnp.where(from_tail, jnp.int16(1), jnp.int16(0)),
        hands=state.hands.at[p32, tile.astype(jnp.int32)].add(1),
        n_draws=state.n_draws + jnp.int16(1),
        turn=player, drawn=tile, phase=jnp.int8(PHASE_DISCARD))
    win = rules.is_win_counts(st.hands[p32], st.n_melds[p32])
    st_win = st.replace(done=True, winner=player, win_type=jnp.int8(WIN_SELF))
    st_empty = state.replace(done=True, winner=jnp.int8(-1), win_type=jnp.int8(WIN_DRAW),
                             pending_tile=jnp.int8(-1), claim_stage=jnp.int8(0),
                             claim_offset=jnp.int8(0), claim_mask=jnp.int16(0))
    return jax.lax.cond(
        empty,
        lambda s: s[2],                     # 摸空 -> 流局
        lambda s: jax.lax.cond(s[3], lambda x: x[0], lambda x: x[1], s),  # 自摸 ? st_win : st
        (st_win, st, st_empty, win))


def _resolve_pass_through(state):
    """全部声明 pass：弃牌入弃牌堆，turn 传给下家并摸牌。"""
    d = state.turn.astype(jnp.int32)
    t = state.pending_tile.astype(jnp.int32)
    dl = state.discard_len[d].astype(jnp.int32)
    st = state.replace(
        discards=state.discards.at[d, t].add(1),
        discard_seq=state.discard_seq.at[d, dl].set(state.pending_tile),
        discard_len=state.discard_len.at[d].add(1),
        pending_tile=jnp.int8(-1), claim_stage=jnp.int8(0),
        claim_offset=jnp.int8(0), claim_mask=jnp.int16(0))
    return _draw_for(st, ((d + 1) % 4).astype(jnp.int8), False)


def _after_discard(state):
    """弃牌（或报听决策）后：找第一个可声明位置；没有则按全 pass 处理。"""
    mask = _claim_ask_mask(state)
    pos = _next_claim_pos(mask, jnp.int16(-1))
    has = pos < 9
    st = state.replace(claim_mask=mask)
    return jax.lax.cond(
        has,
        lambda s: s.replace(claim_stage=(pos // 3 + 1).astype(jnp.int8),
                            claim_offset=(pos % 3 + 1).astype(jnp.int8),
                            phase=jnp.int8(PHASE_CLAIM)),
        _resolve_pass_through, st)


def _step_discard(state, action):
    """phase=DISCARD：打出 action 指定的牌；可能进入报听决策或声明处理。"""
    turn = state.turn.astype(jnp.int32)
    t = action.astype(jnp.int32)
    st = state.replace(hands=state.hands.at[turn, t].add(-1),
                       pending_tile=action.astype(jnp.int8),
                       drawn=jnp.int8(-1))
    # 【引擎 quirk 对齐】arena 引擎仅对 len(full_hand())==13（即无副露）的玩家
    # 询问报听；此处同样只有 n_melds==0 且弃牌后向听==0 才提供 declare 决策点。
    offer = ((~st.locked[turn]) & (st.n_melds[turn] == 0)
             & (rules.shanten_counts(st.hands[turn], jnp.int8(0)) == 0))
    return jax.lax.cond(offer, lambda s: s.replace(phase=jnp.int8(PHASE_TENPAI)),
                        _after_discard, st)


def _step_tenpai(state, action):
    """phase=TENPAI：38=yes 锁手，39=no；然后进入声明处理。"""
    yes = action == jnp.int8(A_TENPAI_YES)
    st = state.replace(locked=state.locked.at[state.turn.astype(jnp.int32)].set(yes))
    return _after_discard(st)


def _step_claim(state, action):
    """phase=CLAIM：处理当前被询问玩家的响应（pass/碰/杠/胡）。"""
    stage = state.claim_stage.astype(jnp.int32)
    off = state.claim_offset.astype(jnp.int32)
    d = state.turn.astype(jnp.int32)
    claimer = ((d + off) % 4).astype(jnp.int8)
    c32 = claimer.astype(jnp.int32)
    t = state.pending_tile.astype(jnp.int32)
    pos = ((stage - 1) * 3 + (off - 1)).astype(jnp.int16)

    def do_pass(s):
        nxt = _next_claim_pos(s.claim_mask, pos)
        return jax.lax.cond(
            nxt < 9,
            lambda ss: ss.replace(claim_stage=(nxt // 3 + 1).astype(jnp.int8),
                                  claim_offset=(nxt % 3 + 1).astype(jnp.int8)),
            _resolve_pass_through, s)

    def do_peng(s):
        return s.replace(hands=s.hands.at[c32, t].add(-2),
                         meld_counts=s.meld_counts.at[c32, t].add(3),
                         n_melds=s.n_melds.at[c32].add(1),
                         pending_tile=jnp.int8(-1), claim_stage=jnp.int8(0),
                         claim_offset=jnp.int8(0), claim_mask=jnp.int16(0),
                         turn=claimer, drawn=jnp.int8(-1),
                         phase=jnp.int8(PHASE_DISCARD))

    def do_gang(s):
        s2 = s.replace(hands=s.hands.at[c32, t].add(-3),
                       meld_counts=s.meld_counts.at[c32, t].add(4),
                       n_melds=s.n_melds.at[c32].add(1),
                       pending_tile=jnp.int8(-1), claim_stage=jnp.int8(0),
                       claim_offset=jnp.int8(0), claim_mask=jnp.int16(0))
        return _draw_for(s2, claimer, True)   # 杠后从牌山尾补牌，自摸检查

    def do_hu(s):
        # 引擎把被胡的牌计入胡家手牌（统计用），此处同样计入以保持全牌守恒
        return s.replace(done=True, winner=claimer, win_type=jnp.int8(WIN_RON),
                         dealer=s.turn,
                         hands=s.hands.at[c32, t].add(1),
                         pending_tile=jnp.int8(-1), claim_stage=jnp.int8(0),
                         claim_offset=jnp.int8(0), claim_mask=jnp.int16(0))

    idx = jnp.clip(action.astype(jnp.int32) - A_PASS, 0, 3)
    return jax.lax.switch(idx, [do_pass, do_peng, do_gang, do_hu], state)


# ---------------------------------------------------------------------------
# reward & step
# ---------------------------------------------------------------------------

def _reward(state):
    """reward (4,)：仅终局非零。score: 自摸 +3 / 点和 +1、放炮 -1；winloss: 赢家 +1 其余 -1；
    score_dd: 同 score，另流局全员 -0.25。"""
    won = state.done & (state.winner >= 0)
    base = jnp.zeros(4, jnp.float32)
    w = state.winner.astype(jnp.int32)
    if state.reward_kind == REWARD_WINLOSS:
        r = jnp.full(4, -1.0, jnp.float32).at[w].set(1.0)
        return jnp.where(won, r, base)
    r = jnp.where(state.win_type == jnp.int8(WIN_SELF), base.at[w].add(3.0),
        jnp.where(state.win_type == jnp.int8(WIN_RON),
                  base.at[w].add(1.0).at[state.dealer.astype(jnp.int32)].add(-1.0),
                  base))
    out = jnp.where(won, r, base)
    if state.reward_kind == REWARD_DD:
        is_draw = state.done & (state.winner < 0)
        out = jnp.where(is_draw, jnp.full(4, -0.25, jnp.float32), out)
    return out


def step(state, action):
    """一步转移。返回 (State, reward[4], done)。对 done 状态调用为 no-op。"""
    st = jax.lax.cond(
        state.done,
        lambda s, a: s,
        lambda s, a: jax.lax.switch(s.phase.astype(jnp.int32),
                                    [_step_discard, _step_claim, _step_tenpai], s, a),
        state, action.astype(jnp.int8))
    return st, _reward(st), st.done
