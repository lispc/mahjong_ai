# -*- coding: utf-8 -*-
"""生成 Perfect-Info Rollout Oracle 数据（fast shanten rollout 版）。

由于 conv-BC rollout 太慢（每局 ~70s），本脚本使用更快的 shanten-minimizing 策略做
rollout：每步选择使手牌上听距离最小的弃牌。Oracle 本身仍用完美信息（知道对手手牌和
牌山），对每个合法弃牌跑 N 次随机 wall 顺序的 rollout，取当前玩家平均 outcome 最高者。

输出 .npz：Xn, Xo, y, v（与 gen_oracle_data.py 兼容）。

用法：
    PYTHONPATH=. python3 scripts/rl/gen_rollout_oracle_fast_data.py \
        output/nn_teacher_rollout_fast_200.npz 200 32 2 60000000
"""

import sys
import os
import random
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tile_pool
import tile
import algo.eval.v2 as eval_v2
import algo.context.v3 as context_v3
from algo.nn.features import extract_features, extract_features_oracle, _TILE_TO_IDX


def _shanten_discard(hand14):
    """选择弃牌后 shanten 最小的牌（tie 时选 ukeire 大的）。"""
    best = None
    best_s = 999
    best_u = -1
    seen = set()
    for t in hand14:
        if t in seen:
            continue
        seen.add(t)
        h = list(hand14)
        h.remove(t)
        s = eval_v2.shanten(h)
        u = len(eval_v2.winning_tiles(h, {}))  # 粗略：不考虑 wall
        if s < best_s or (s == best_s and u > best_u):
            best_s = s
            best_u = u
            best = t
    return best


def _simulate_from(hands, wall, ctx, locked, turn, current_name, max_steps=200):
    """从 turn 玩家已摸牌但未弃牌的状态开始，用 shanten-minimizing 策略模拟到终局。"""
    names = sorted(hands.keys(), key=lambda n: int(n.split('@')[-1]))
    hands = {n: list(hands[n]) for n in names}
    wall = list(wall)
    ctx = ctx.copy()
    locked = set(locked)
    steps = 0
    while wall and steps < max_steps:
        pname = names[turn]
        drawn = wall.pop(0)
        hands[pname].append(drawn)

        if eval_v2.is_win(hands[pname]):
            return 1.0 if pname == current_name else -1.0

        if pname in locked:
            discarded = drawn
        else:
            discarded = _shanten_discard(hands[pname])
            if (pname not in locked and len(hands[pname]) == 13 and
                    eval_v2.shanten(hands[pname]) == 0):
                rem = ctx.remaining_wall(hands[pname])
                waits = eval_v2.winning_tiles(hands[pname], rem)
                if sum(rem.get(t, 0) for t in waits) >= 3:
                    locked.add(pname)
                    ctx.declare_tenpai(pname)
        hands[pname].remove(discarded)

        for j, other in enumerate(names):
            if j == turn:
                continue
            if eval_v2.is_win(hands[other] + [discarded]):
                return -1.0 if pname == current_name else (1.0 if other == current_name else -1.0)

        ctx.see_tile(discarded, pname)
        turn = (turn + 1) % 4
        steps += 1

    return 0.0


def _evaluate_candidate(hand14, all_hands, wall, ctx, locked, turn, current_name, cand,
                        n_rollouts):
    player_names = sorted(all_hands.keys(), key=lambda n: int(n.split('@')[-1]))
    opponents = [p for p in player_names if p != current_name]

    for opp in opponents:
        if eval_v2.is_win(list(all_hands[opp]) + [cand]):
            return -1.0

    hands = {n: list(all_hands[n]) for n in player_names}
    hands[current_name].remove(cand)
    ctx2 = ctx.copy()
    ctx2.see_tile(cand, current_name)
    locked2 = set(locked)
    turn2 = (turn + 1) % 4

    total = 0.0
    for _ in range(n_rollouts):
        wall_copy = list(wall)
        random.shuffle(wall_copy)
        total += _simulate_from(hands, wall_copy, ctx2, locked2, turn2, current_name)
    return total / n_rollouts


def _oracle_discard(hand14, all_hands, wall, ctx, locked, turn, current_name, n_rollouts):
    unique = sorted(set(hand14))
    best = None
    best_score = -float('inf')
    for cand in unique:
        score = _evaluate_candidate(hand14, all_hands, wall, ctx, locked, turn,
                                    current_name, cand, n_rollouts)
        if score > best_score:
            best_score = score
            best = cand
    return best, best_score


def _play_one_game(seed, n_rollouts):
    random.seed(seed)
    np.random.seed(seed % (2 ** 31 - 1))

    pool = tile_pool.Pool()
    names = [f'P@{i}' for i in range(4)]
    hands = {n: pool.next_n(13) for n in names}
    ctx = context_v3.ContextV3()
    locked = set()
    turn = 0
    wall = list(pool.tiles[pool.idx:])

    samples = []
    outcome = {n: 0.0 for n in names}

    while True:
        if not wall:
            for n in names:
                outcome[n] = 0.0
            break
        pname = names[turn]
        drawn = wall.pop(0)
        hands[pname].append(drawn)

        if eval_v2.is_win(hands[pname]):
            for n in names:
                outcome[n] = 1.0 if n == pname else -1.0
            full_samples = []
            for xn, xo, y, pname in samples:
                full_samples.append((xn, xo, y, outcome.get(pname, 0.0)))
            return full_samples, outcome

        if pname in locked:
            discarded = drawn
            hands[pname].remove(discarded)
        else:
            xn = extract_features(ctx, hands[pname], pname)
            xo = extract_features_oracle(ctx, hands[pname], pname, hands, wall)
            discarded, score = _oracle_discard(
                hands[pname], hands, wall, ctx, locked, turn, pname, n_rollouts)
            samples.append((xn, xo, int(_TILE_TO_IDX[discarded]), pname))
            hands[pname].remove(discarded)

        for j, other in enumerate(names):
            if j == turn:
                continue
            if eval_v2.is_win(hands[other] + [discarded]):
                for n in names:
                    if n == other:
                        outcome[n] = 1.0
                    else:
                        outcome[n] = -1.0
                full_samples = []
                for xn, xo, y, pname in samples:
                    full_samples.append((xn, xo, y, outcome.get(pname, 0.0)))
                return full_samples, outcome

        ctx.see_tile(discarded, pname)

        if (pname not in locked and len(hands[pname]) == 13 and
                eval_v2.shanten(hands[pname]) == 0):
            rem = ctx.remaining_wall(hands[pname])
            waits = eval_v2.winning_tiles(hands[pname], rem)
            if sum(rem.get(t, 0) for t in waits) >= 3:
                locked.add(pname)
                ctx.declare_tenpai(pname)

        turn = (turn + 1) % 4

    full_samples = []
    for xn, xo, y, pname in samples:
        full_samples.append((xn, xo, y, outcome.get(pname, 0.0)))
    return full_samples, outcome


def _worker(args):
    i, s, n_rollouts = args
    return _play_one_game(s, n_rollouts)[0]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'output/nn_teacher_rollout_fast.npz'
    total_games = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    n_rollouts = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    seed_base = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    print(f'Fast rollout oracle data: {total_games} games, {workers} workers, n_rollouts={n_rollouts}')
    t0 = time.time()
    all_samples = []
    if workers <= 1:
        for i in range(total_games):
            all_samples.extend(_play_one_game(seed_base + i, n_rollouts)[0])
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_worker,
                                        [(i, seed_base + i, n_rollouts) for i in range(total_games)]))
        for r in results:
            all_samples.extend(r)
    dt = time.time() - t0
    print(f'Generated {len(all_samples)} samples from {total_games} games in {dt:.1f}s')

    if not all_samples:
        print('No samples generated')
        return

    Xn = np.stack([s[0] for s in all_samples])
    Xo = np.stack([s[1] for s in all_samples])
    y = np.array([s[2] for s in all_samples], dtype=np.int64)
    v = np.array([s[3] for s in all_samples], dtype=np.float32)
    np.savez(out_path, Xn=Xn, Xo=Xo, y=y, v=v)
    print(f'Saved {out_path}: Xn={Xn.shape}, Xo={Xo.shape}, y={y.shape}, v={v.shape}')


if __name__ == '__main__':
    main()
