# -*- coding: utf-8 -*-
"""jaxenv 验证套件（M2）。

用法：
    PYTHONPATH=. python3 jaxenv/test_env.py --group all
    --group rules         单测：JAX 胡牌判定 vs v2.is_win（≥100k，含副露 m=0..3）
                          + JAX 向听 vs v2.shanten（≥100k 13 张无副露）
    --group scenarios     剧本化状态机测试（自摸/荣和/碰/杠/报听锁手/流局）
    --group invariants    200 局随机策略逐步不变量断言
    --group distribution  500 局双引擎分布对比
"""

import argparse
import random
import time

import numpy as np

import jax
import jax.numpy as jnp

from jaxenv import env, rules


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def idxs_to_pyids(idxs):
    return [rules.TILE_IDS[i] for i in idxs]


def random_counts_batch(n, n_tiles, rng):
    """n 个随机手牌计数向量（每种 ≤4）。"""
    counts = np.zeros((n, 34), np.int8)
    for _ in range(n_tiles):
        t = rng.randint(0, 34, n)
        rows = np.arange(n)
        over = counts[rows, t] >= 4
        while over.any():
            t[over] = rng.randint(0, 34, over.sum())
            over = counts[rows, t] >= 4
        counts[rows, t] += 1
    return counts


def constructed_win_counts(n, m, rng):
    """构造 n 个 (4-m) 面子 + 1 对子的闭手计数（~30% 随机突变一张变成近似牌）。"""
    out = np.zeros((n, 34), np.int8)
    for i in range(n):
        while True:
            c = np.zeros(34, np.int8)
            for _ in range(4 - m):
                if rng.random() < 0.5:
                    c[rng.randint(0, 34)] += 3
                else:
                    s = rng.randint(0, 3)
                    k = rng.randint(0, 7)
                    c[s * 9 + k] += 1
                    c[s * 9 + k + 1] += 1
                    c[s * 9 + k + 2] += 1
            c[rng.randint(0, 34)] += 2
            if (c <= 4).all():
                break
        if rng.random() < 0.3:  # 突变：换一张牌（通常破坏胡牌）
            have = np.where(c > 0)[0]
            c[rng.choice(have)] -= 1
            c[rng.randint(0, 34)] += 1
            if (c > 4).any() or c.sum() != 14 - 3 * m:
                c = np.zeros(34, np.int8)  # 极少见；置空牌（必不胡）
        out[i] = c
    return out


# ---------------------------------------------------------------------------
# rules 单测
# ---------------------------------------------------------------------------

def test_is_win(n_per_m=25000, seed=11):
    from algo.eval import v2

    def ref_win(counts, m):
        closed = idxs_to_pyids([i for i in range(34) for _ in range(counts[i])])
        if m == 0:
            return bool(v2.is_win(closed))
        # m>0：闭手 (14-3m) 张需拆成 (4-m) 面子 + 1 对子（无七对子）
        from collections import Counter
        cnt = Counter(closed)
        for t, cc in cnt.items():
            if cc >= 2:
                rest = list(closed)
                rest.remove(t)
                rest.remove(t)
                if v2._split_melds(rest):
                    return True
        return False

    rng = np.random.RandomState(seed)
    total = bad = 0
    f_win = jax.jit(jax.vmap(rules.is_win_counts, in_axes=(0, 0)))
    t0 = time.time()
    for m in range(4):
        halves = n_per_m // 2
        c_rand = random_counts_batch(halves, 14 - 3 * m, rng)
        c_cons = constructed_win_counts(n_per_m - halves, m, rng)
        counts = np.concatenate([c_rand, c_cons])
        pred = np.array(f_win(jnp.asarray(counts), jnp.full(len(counts), m, jnp.int8)))
        refs = np.array([ref_win(counts[i], m) for i in range(len(counts))])
        n_bad = int((pred != refs).sum())
        bad += n_bad
        total += len(counts)
        print(f'  is_win m={m}: {len(counts)} hands, pos_rate ref={refs.mean():.3f} '
              f'jax={pred.mean():.3f}, mismatches={n_bad}', flush=True)
    dt = time.time() - t0
    print(f'[rules.is_win] total={total}, mismatches={bad} ({dt:.1f}s)')
    assert bad == 0, f'is_win mismatches: {bad}'
    return total


def test_shanten(n=100000, seed=12):
    from algo.eval import v2

    rng = np.random.RandomState(seed)
    counts = random_counts_batch(n, 13, rng)
    # 掺入结构化样本：单花色重、字牌重、对子重
    k = n // 10
    for i in range(k):
        suit = rng.randint(0, 4)
        lo, hi = (suit * 9, suit * 9 + 9) if suit < 3 else (27, 34)
        c = np.zeros(34, np.int8)
        for _ in range(13):
            choices = np.where((c < 4))[0]
            choices = choices[(choices >= lo) & (choices < hi)]
            c[rng.choice(choices)] += 1
        counts[i] = c

    t0 = time.time()
    refs = np.empty(n, np.int16)
    for i in range(n):
        refs[i] = v2.shanten(idxs_to_pyids([t for t in range(34) for _ in range(counts[i, t])]))
        if (i + 1) % 20000 == 0:
            print(f'  v2.shanten ref {i + 1}/{n} ({time.time() - t0:.1f}s)', flush=True)
    f_sh = jax.jit(jax.vmap(lambda c: rules.shanten_counts(c, jnp.int8(0))))
    pred = np.concatenate([np.array(f_sh(jnp.asarray(counts[j:j + 8192])))
                           for j in range(0, n, 8192)])
    bad = int((pred != refs).sum())
    if bad:
        idx = np.where(pred != refs)[0][:5]
        for i in idx:
            print('  MISMATCH', idxs_to_pyids([t for t in range(34) for _ in range(counts[i, t])]),
                  'jax:', pred[i], 'v2:', refs[i])
    print(f'[rules.shanten] total={n}, mismatches={bad} ({time.time() - t0:.1f}s), '
          f'dist={np.bincount(refs - refs.min(), minlength=9).tolist()}')
    assert bad == 0, f'shanten mismatches: {bad}'
    return n


# ---------------------------------------------------------------------------
# 场景测试
# ---------------------------------------------------------------------------

def _mk_state(wall, hands, **kw):
    """直接构造 State（场景测试用）。hands: 4 × list[idx] 或 counts。"""
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
        claim_stage=jnp.int8(kw.get('claim_stage', 0)),
        claim_offset=jnp.int8(kw.get('claim_offset', 0)),
        claim_mask=jnp.int16(kw.get('claim_mask', 0)),
        locked=jnp.asarray(kw.get('locked', [False] * 4)),
        phase=jnp.int8(kw.get('phase', env.PHASE_DISCARD)),
        done=bool(kw.get('done', False)),
        winner=jnp.int8(kw.get('winner', -1)),
        win_type=jnp.int8(kw.get('win_type', 0)),
        n_draws=jnp.int16(kw.get('n_draws', 1)),
        reward_kind=kw.get('reward_kind', env.REWARD_SCORE),
    )
    return st


def _wall_with(head_tiles, tail_tiles=()):
    """构造牌山：wall[53:53+len(head_tiles)] = head_tiles（默认 wall_head=53 即
    发牌+首摸后的下一个摸牌位），尾部放 tail_tiles。其余位置填 0（场景测试不依赖守恒）。"""
    wall = np.zeros(136, np.int8)
    wall[53:53 + len(head_tiles)] = head_tiles
    if tail_tiles:
        wall[136 - len(tail_tiles):] = tail_tiles
    return wall


def test_scenarios():
    stp = jax.jit(env.step)
    maskf = jax.jit(env.legal_mask)

    # --- 场景1：自摸自动胡（座位0摸即胡） ---
    # 13 张待牌: [0,0,0][1,1,1][2,2,2][3,4,5] + 7（听 7）；wall[52]=7 完成 77 对子
    hand0 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 5, 7]
    wall = _wall_with([7])
    st = _mk_state(wall, [hand0, [10] * 13, [20] * 13, [28] * 13])
    st2 = env._draw_for(st, jnp.int8(0), False)
    assert bool(st2.done) and int(st2.winner) == 0 and int(st2.win_type) == env.WIN_SELF, \
        f'self-draw win failed: done={st2.done} winner={st2.winner} wt={st2.win_type}'
    print('  scenario 1 ok: self-draw auto win')

    # --- 场景2：荣和（offset 顺序：先问 offset1） ---
    # 玩家0 弃 8(idx)；玩家2 能胡。claim_stage=hu, offset=2
    hand2 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 5, 7]  # +8? no: 需 7 成对 -> 胡 7
    wall = _wall_with([9])
    st = _mk_state(wall, [[9] * 13, [10] * 13, hand2, [28] * 13],
                   turn=0, pending_tile=7, phase=env.PHASE_CLAIM,
                   claim_stage=env.STAGE_HU, claim_offset=2,
                   claim_mask=env._claim_ask_mask(
                       _mk_state(wall, [[9] * 13, [10] * 13, hand2, [28] * 13],
                                 turn=0, pending_tile=7)))
    m = np.array(maskf(st))
    assert m[37] and m[34], f'hu mask wrong: {np.where(m)[0]}'
    st2, r, d = stp(st, jnp.int8(env.A_HU))
    assert bool(d) and int(st2.winner) == 2 and int(st2.win_type) == env.WIN_RON \
        and int(st2.dealer) == 0, f'ron failed: {st2.winner} {st2.win_type} {st2.dealer}'
    assert int(st2.hands[2, 7]) == 2, '被胡的牌应计入胡家手牌'
    assert float(r[2]) == 1.0 and float(r[0]) == -1.0, f'score reward wrong: {r}'
    print('  scenario 2 ok: ron + dealer + score reward')

    # --- 场景3：碰 -> 不摸牌直接弃牌 ---
    hand1 = [7, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    wall = _wall_with([9])
    st = _mk_state(wall, [[0] * 13, hand1, [20] * 13, [28] * 13],
                   turn=0, pending_tile=7, phase=env.PHASE_CLAIM,
                   claim_stage=env.STAGE_PENG, claim_offset=1, claim_mask=0)
    st2, r, d = stp(st, jnp.int8(env.A_PENG))
    assert not bool(d) and int(st2.turn) == 1 and int(st2.phase) == env.PHASE_DISCARD
    assert int(st2.drawn) == -1, '碰后无摸牌'
    assert int(st2.n_melds[1]) == 1 and int(st2.meld_counts[1, 7]) == 3
    assert int(st2.hands[1, 7]) == 0 and int(st2.hands[1].sum()) == 11
    print('  scenario 3 ok: peng (no draw, meld=3)')

    # --- 场景4：杠 -> 尾部补牌 ---
    hand1g = [7, 7, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    wall = _wall_with([9], tail_tiles=[30])
    st = _mk_state(wall, [[0] * 13, hand1g, [20] * 13, [28] * 13],
                   turn=0, pending_tile=7, phase=env.PHASE_CLAIM,
                   claim_stage=env.STAGE_GANG, claim_offset=1, claim_mask=0)
    st2, r, d = stp(st, jnp.int8(env.A_GANG))
    assert int(st2.n_melds[1]) == 1 and int(st2.meld_counts[1, 7]) == 4
    assert int(st2.wall_tail) == 134, f'补牌应从尾部取: tail={st2.wall_tail}'
    assert int(st2.hands[1, 30]) == 1, f'应摸到尾部的 30: {st2.hands[1, 30]}'
    assert int(st2.turn) == 1 and int(st2.phase) == env.PHASE_DISCARD
    print('  scenario 4 ok: gang (tail replacement draw)')

    # --- 场景5：报听 -> 锁手 -> 强制打出摸到的牌 ---
    # 玩家0 14 张听牌形（打 8 后听）：111 222 333 456 78 -> 打 8? 手牌 14 张需弃 1
    # 用：sets=[0,0,0],[1,1,1],[2,2,2],[3,4,5] + [6,7] 搭子 -> 13 张… 需 14 张弃 1
    # 14 张：上述 13 + 8（打出 8 后 111222333456 67 听 5/8）
    hand0t = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 7]
    wall = _wall_with([31])
    st = _mk_state(wall, [hand0t, [10] * 13, [20] * 13, [28] * 13], turn=0, drawn=6)
    # 验证向听：弃 7 -> [0,0,0,1,1,1,2,2,2,3,4,5,6] 听? 需成 4sets+pair… 没有对子!
    # 改用听牌形: 111 222 333 45 66 -> 13 张弃前 14 张 = +7: [...,4,5,6,6,7]
    hand0t = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 6]
    # 弃 idx3: 剩 [0,0,0,1,1,1,2,2,2,4,5,6,6] = 3sets + 456顺 + 66对 -> 听 3/6，向听 0
    hand_after = [0, 0, 0, 1, 1, 1, 2, 2, 2, 4, 5, 6, 6]
    c = rules.tiles_to_counts(idxs_to_pyids(hand_after))
    assert int(rules.shanten_counts(jnp.asarray(c), jnp.int8(0))) == 0
    st2 = env._step_discard(st.replace(hands=st.hands.at[0, 3].add(0)), jnp.int8(3))
    assert int(st2.phase) == env.PHASE_TENPAI, f'应进入报听决策: phase={st2.phase}'
    m = np.array(maskf(st2))
    assert m[38] and m[39] and m.sum() == 2
    st3, _, _ = stp(st2, jnp.int8(env.A_TENPAI_YES))
    assert bool(st3.locked[0]), '报听后应锁手'
    # 锁手玩家摸牌后只能打出摸到的牌（直接构造锁手+该摸牌的状态）
    wall2 = _wall_with([31])
    stL = _mk_state(wall2, [hand_after, [10] * 13, [20] * 13, [28] * 13],
                    turn=0, locked=[True, False, False, False])
    st4 = env._draw_for(stL, jnp.int8(0), False)   # 摸 wall[52]=31
    assert int(st4.phase) == env.PHASE_DISCARD and int(st4.drawn) == 31
    m = np.array(maskf(st4))
    assert m.sum() == 1 and m[31], f'锁手强制打出摸到的牌: {np.where(m)[0]}'
    print('  scenario 5 ok: tenpai declare -> locked forced discard')

    # --- 场景6：锁手玩家仍可胡牌，但不能碰/杠 ---
    hand_locked = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 5, 7]
    wall = _wall_with([9])
    st = _mk_state(wall, [[9] * 13, hand_locked, [20] * 13, [28] * 13],
                   turn=0, pending_tile=7, locked=[False, True, False, False])
    ask = env._claim_ask_mask(st)
    # pos: stage hu offset1 -> 0; stage gang offset1 -> 3; stage peng offset1 -> 6
    assert bool(np.array((ask >> 0) & 1)), '锁手玩家应能被问胡'
    assert not bool(np.array((ask >> 3) & 1)), '锁手玩家不应被问杠'
    assert not bool(np.array((ask >> 6) & 1)), '锁手玩家不应被问碰'
    print('  scenario 6 ok: locked can hu, cannot peng/gang')

    # --- 场景7：流局（摸空） ---
    wall = _wall_with([9])
    st = _mk_state(wall, [[0] * 13] * 4, wall_head=136, wall_tail=135)
    st2 = env._draw_for(st, jnp.int8(0), False)
    assert bool(st2.done) and int(st2.win_type) == env.WIN_DRAW and int(st2.winner) == -1
    # 杠后补牌摸空同样流局
    st3 = env._draw_for(st, jnp.int8(0), True)
    assert bool(st3.done) and int(st3.win_type) == env.WIN_DRAW
    print('  scenario 7 ok: wall exhaustion -> draw (head & tail)')

    # --- 场景8：声明自动跳过（无人能声明时直接过） ---
    wall = _wall_with([9])
    st = _mk_state(wall, [[0] * 13, [10] * 13, [20] * 13, [28] * 13],
                   turn=0, pending_tile=6)
    st2 = env._after_discard(st)
    # 所有家都没有 idx 6（手牌为占位用的大量同种牌）-> 直接 pass-through，玩家1 摸到 wall[52]=9
    assert int(st2.phase) == env.PHASE_DISCARD and int(st2.turn) == 1
    assert int(st2.discards[0, 6]) == 1 and int(st2.hands[1, 9]) == 1
    print('  scenario 8 ok: claim auto-skip -> pass-through draw')

    # --- 场景9：winloss reward 变体（赢家 +1、其余 -1） ---
    hand2w = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 5, 7]
    wall = _wall_with([9])
    st = _mk_state(wall, [[9] * 13, [10] * 13, hand2w, [28] * 13],
                   turn=0, pending_tile=7, phase=env.PHASE_CLAIM,
                   claim_stage=env.STAGE_HU, claim_offset=2)
    st = st.replace(reward_kind=env.REWARD_WINLOSS)
    st2, r, d = stp(st, jnp.int8(env.A_HU))
    assert bool(d) and np.allclose(np.array(r), [-1.0, -1.0, 1.0, -1.0]), \
        f'winloss reward wrong: {r}'
    # 流局 reward 全 0
    st = _mk_state(wall, [[0] * 13] * 4, wall_head=136, wall_tail=135,
                   reward_kind=env.REWARD_WINLOSS)
    st2 = env._draw_for(st, jnp.int8(0), False)
    assert bool(st2.done) and int(st2.win_type) == env.WIN_DRAW
    print('  scenario 9 ok: winloss reward variant')

    print('[scenarios] all passed')


# ---------------------------------------------------------------------------
# 不变量测试
# ---------------------------------------------------------------------------

def _check_step_invariants(states, dones_prev, it):
    """逐局不变量断言（host 侧）。dones_prev=True 的局已终局不再检查。"""
    hands = np.array(states.hands)
    melds = np.array(states.meld_counts)
    disc = np.array(states.discards)
    wall = np.array(states.wall)
    head = np.array(states.wall_head)
    tail = np.array(states.wall_tail)
    pend = np.array(states.pending_tile)
    dlen = np.array(states.discard_len)
    dseq = np.array(states.discard_seq)
    for i in range(hands.shape[0]):
        if dones_prev[i]:
            continue
        # 每种牌守恒：闭手 + 副露 + 弃牌 + pending + 牌山剩余 = 4
        rem = np.bincount(wall[i][head[i]:tail[i] + 1], minlength=34).astype(np.int64)
        tot = hands[i].sum(0) + melds[i].sum(0) + disc[i].sum(0) + rem
        if pend[i] >= 0:
            tot[pend[i]] += 1
        assert (tot == 4).all(), \
            f'iter {it} game {i}: conservation broken: {np.where(tot != 4)[0]} -> {tot[tot != 4]}'
        assert (hands[i] >= 0).all() and (hands[i] <= 4).all(), f'game {i}: hand counts'
        # 弃牌序列与计数一致（每玩家）
        for p in range(4):
            seq = dseq[i, p, :dlen[i, p]]
            assert (seq >= 0).all() and \
                (np.bincount(seq, minlength=34) == disc[i, p]).all(), \
                f'game {i} player {p}: discard seq'
        assert head[i] <= tail[i] + 1


def _run_invariant_games(states, rng, policy, step_v, mask_v, counters, max_iter=1500):
    """公共不变量循环。policy(mask_row, st_row, rng) -> action；counters 收集覆盖统计。"""
    sh_v = jax.jit(jax.vmap(rules.shanten_counts, in_axes=(0, 0)))
    n_games = states.hands.shape[0]
    for it in range(max_iter):
        masks = np.array(mask_v(states))
        dones = np.array(states.done)
        if dones.all():
            break
        phases = np.array(states.phase)
        turns = np.array(states.turn)
        locked = np.array(states.locked)
        drawn = np.array(states.drawn)
        hands = np.array(states.hands)
        n_melds = np.array(states.n_melds)

        # 贪心弃牌候选（policy='greedy' 时用）：所有 phase0 未锁手游戏的全部候选
        greedy_choice = {}
        if policy == 'greedy':
            cands, owners = [], []
            for i in range(n_games):
                if dones[i] or phases[i] != 0 or locked[i, turns[i]]:
                    continue
                cnt = hands[i, turns[i]]
                for t in np.where(cnt > 0)[0]:
                    c = cnt.copy()
                    c[t] -= 1
                    cands.append(c)
                    owners.append((i, t))
            if cands:
                pad = n_games * 14 - len(cands)
                batch = np.array(cands + [np.zeros(34, np.int8)] * pad, np.int8)
                meld_batch = np.array([n_melds[i, turns[i]] for (i, t) in owners]
                                      + [0] * pad, np.int8)
                sh = np.array(sh_v(jnp.asarray(batch), jnp.asarray(meld_batch)))[:len(cands)]
                best = {}
                for (i, t), s in zip(owners, sh):
                    if i not in best or s < best[i][0]:
                        best[i] = (s, [t])
                    elif s == best[i][0]:
                        best[i][1].append(t)
                greedy_choice = {i: int(rng.choice(ts)) for i, (s, ts) in best.items()}

        acts = np.full(n_games, 34, np.int8)
        for i in range(n_games):
            if dones[i]:
                continue
            legal = np.where(masks[i])[0]
            if phases[i] == 0 and locked[i, turns[i]]:
                # 锁手强制弃牌断言：唯一合法动作 == drawn
                assert legal.tolist() == [int(drawn[i])], \
                    f'game {i}: locked legal mask {legal} != drawn {drawn[i]}'
                a = int(drawn[i])
                counters['locked_discards'] += 1
            elif phases[i] == 2 and policy == 'greedy':
                a = 38                      # 报听必 yes（覆盖锁手路径）
            elif phases[i] == 0 and i in greedy_choice:
                a = greedy_choice[i]        # 贪心最小向听弃牌
            else:
                a = int(rng.choice(legal))  # 声明等其余决策：均匀随机
            acts[i] = a
            if a in (35, 36, 37):
                counters['claims'][a] += 1
            if a == 38:
                counters['tenpai_yes'] += 1

        states, rews, dn = step_v(states, jnp.array(acts))
        _check_step_invariants(states, dones, it)
    else:
        raise AssertionError(f'games did not finish in {max_iter} iters')

    wt = np.array(states.win_type)
    assert (wt != env.WIN_NONE).all(), '所有局都应有结果'
    # 终局 reward 精确检查（score-proxy：自摸赢家 +3；点和赢家 +1 放炮者 -1；流局全 0）
    rews = np.array(rews)
    winners = np.array(states.winner)
    dealers = np.array(states.dealer)
    for i in range(n_games):
        exp = np.zeros(4)
        if wt[i] == env.WIN_SELF:
            exp[winners[i]] = 3.0
        elif wt[i] == env.WIN_RON:
            exp[winners[i]] = 1.0
            exp[dealers[i]] = -1.0
        assert np.allclose(rews[i], exp), f'game {i}: reward {rews[i]} != expected {exp}'
    return states, it


def test_invariants(n_games=200, seed=13):
    rng = np.random.RandomState(seed)
    step_v = jax.jit(jax.vmap(env.step))
    mask_v = jax.jit(jax.vmap(env.legal_mask))

    # ---- 阶段 A：均匀随机策略 200 局（覆盖碰/杠/胡声明路径） ----
    t0 = time.time()
    states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(seed), n_games))
    counters = {'locked_discards': 0, 'tenpai_yes': 0, 'claims': {35: 0, 36: 0, 37: 0}}
    states, it = _run_invariant_games(states, rng, 'random', step_v, mask_v, counters)
    wt = np.array(states.win_type)
    nd = np.array(states.n_draws)
    print(f'[invariants A: random] {n_games} games OK ({time.time() - t0:.1f}s, {it} iters): '
          f'win_types(self/ron/draw)={np.bincount(wt, minlength=4)[1:].tolist()}, '
          f'n_draws mean={nd.mean():.1f} max={nd.max()}, '
          f'claims peng/gang/hu={counters["claims"][35]}/{counters["claims"][36]}/{counters["claims"][37]}',
          flush=True)
    assert counters['claims'][35] > 0 and counters['claims'][36] > 0 and counters['claims'][37] > 0, \
        '随机策略 200 局应覆盖碰/杠/胡路径'

    # ---- 阶段 B：贪心向听策略 60 局（报听必 yes，覆盖锁手/强制弃牌路径） ----
    t0 = time.time()
    n2 = 60
    states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(seed + 1), n2))
    counters = {'locked_discards': 0, 'tenpai_yes': 0, 'claims': {35: 0, 36: 0, 37: 0}}
    states, it = _run_invariant_games(states, rng, 'greedy', step_v, mask_v, counters)
    wt = np.array(states.win_type)
    nd = np.array(states.n_draws)
    print(f'[invariants B: greedy+tenpai] {n2} games OK ({time.time() - t0:.1f}s, {it} iters): '
          f'win_types(self/ron/draw)={np.bincount(wt, minlength=4)[1:].tolist()}, '
          f'n_draws mean={nd.mean():.1f} max={nd.max()}, '
          f'tenpai_yes={counters["tenpai_yes"]}, locked_discards={counters["locked_discards"]}',
          flush=True)
    assert counters['tenpai_yes'] > 0, '贪心策略应产生报听'
    assert counters['locked_discards'] > 0, '锁手强制弃牌路径应被覆盖'
    return n_games + n2


# ---------------------------------------------------------------------------
# 分布对比
# ---------------------------------------------------------------------------

def _jax_noclaim_games(n, seed):
    """JAX 侧：从不碰/杠/报听、按张数加权随机弃牌、能胡必胡。全 jit 循环。"""
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    states = jax.vmap(env.init)(keys)

    @jax.jit
    def policy_step(states, key):
        masks = jax.vmap(env.legal_mask)(states)
        turn_oh = jax.nn.one_hot(states.turn, 4, dtype=jnp.float32)
        counts = jnp.einsum('bp,bpt->bt', turn_oh, states.hands.astype(jnp.float32))
        logits = jnp.where(masks[:, :34], jnp.log(jnp.maximum(counts, 1e-9)), -1e30)
        key, sub = jax.random.split(key)
        disc = jax.random.categorical(sub, logits).astype(jnp.int8)
        act = jnp.where(states.phase == jnp.int8(0), disc,
              jnp.where(states.phase == jnp.int8(1),
                        jnp.where(masks[:, 37], jnp.int8(37), jnp.int8(34)),
                        jnp.int8(39)))
        states, _, _ = jax.vmap(env.step)(states, act)
        return states, key

    @jax.jit
    def play_all(states, key):
        def cond(c):
            return ~jnp.all(c[0].done)
        def body(c):
            return policy_step(c[0], c[1])
        return jax.lax.while_loop(cond, body, (states, key))[0]

    return play_all(states, jax.random.PRNGKey(seed + 1))


class _NoClaimRandomAgent:
    """Python 引擎侧镜像策略：从不碰/杠/报听、随机弃牌、能胡必胡（基类 respond_hu）。"""

    @staticmethod
    def make(name, seed):
        from agent import Agent

        class _A(Agent):
            def __init__(self):
                super().__init__(name, verbose=False)
                self._rng = random.Random(seed)

            def next(self):
                t = self._rng.choice(self.cur)
                self.cur.remove(t)
                return t

            def declare_tenpai(self, hand, context):
                return False

            def respond_peng(self, tile_val, context=None):
                return False

            def respond_gang(self, tile_val, context=None):
                return False

        return _A()


def test_distribution(n_games=500, seed=14):
    from driver import engine

    t0 = time.time()
    jstates = _jax_noclaim_games(n_games, seed)
    jwt = np.array(jstates.win_type)
    jlen = np.array(jstates.n_draws)
    print(f'  JAX side done ({time.time() - t0:.1f}s)', flush=True)

    t0 = time.time()
    pwt, plen = [], []
    for i in range(n_games):
        agents = [_NoClaimRandomAgent.make(f'p{k}', seed * 1000003 + i * 10 + k)
                  for k in range(4)]
        res = engine.play_game(agents, seed=seed * 1000003 + i, record_log=True)
        pwt.append({'self': 1, 'ron': 2, 'draw': 3}[res['win_type']])
        plen.append(sum(1 for ev in res['event_log'] if ev['type'] == 'draw'))
        if (i + 1) % 100 == 0:
            print(f'  PY side {i + 1}/{n_games} ({time.time() - t0:.1f}s)', flush=True)
    pwt = np.array(pwt)
    plen = np.array(plen)

    def stats(wt, ln, name):
        n_dec = (wt != 3).sum()
        print(f'  {name}: n={len(wt)} len_mean={ln.mean():.2f} draw={100 * (wt == 3).mean():.1f}% '
              f'self={100 * (wt == 1).mean():.1f}% ron={100 * (wt == 2).mean():.1f}% '
              f'(self share of decided: {100 * (wt == 1).sum() / max(n_dec, 1):.1f}%)')
        return ln.mean(), (wt == 3).mean(), (wt == 1).mean(), (wt == 2).mean()

    print(f'[distribution] {n_games} games/side, 策略=不碰/不杠/不报听+随机弃牌+能胡必胡')
    jl, jd, js, jr = stats(jwt, jlen, 'JAX ')
    pl, pd, ps, pr = stats(pwt, plen, 'PY  ')
    len_ok = abs(jl - pl) / max(pl, 1e-9) <= 0.05
    draw_ok = abs(jd - pd) <= 0.02
    print(f'  局长差 {100 * (jl - pl) / max(pl, 1e-9):+.2f}% (容差 ±5%), '
          f'流局率差 {100 * (jd - pd):+.2f}pp (容差 ±2pp)')
    # 注意：JAX 胡牌含七对子（v2 语义），Python 引擎 is_succ 不含；随机打法下差异 << 容差
    assert len_ok, '平均局长差异超容差'
    assert draw_ok, '流局率差异超容差'
    print('[distribution] passed')
    return n_games


# ---------------------------------------------------------------------------

GROUPS = {
    'rules': [test_is_win, test_shanten],
    'scenarios': [test_scenarios],
    'invariants': [test_invariants],
    'distribution': [test_distribution],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', default='all', choices=list(GROUPS) + ['all'])
    args = ap.parse_args()
    groups = list(GROUPS) if args.group == 'all' else [args.group]
    for g in groups:
        print(f'===== group: {g} =====', flush=True)
        for fn in GROUPS[g]:
            fn()
        print(f'===== {g}: PASS =====', flush=True)


if __name__ == '__main__':
    main()
