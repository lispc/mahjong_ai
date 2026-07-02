# -*- coding: utf-8 -*-
"""用 Hybrid-dealin07 当教师生成带 deal-in 标签的数据。

4 个座位全是 Hybrid-dealin07（平时 NN，对手报听/终盘切 BeliefExp）。
对每个弃牌决策记录：
- X：175 维公开信息特征
- dealin：34 维即时点炮标签（1=点炮，0=安全，-1=不在手牌）
- y：Hybrid 选择的动作
- v：该局最终 outcome

输出 .npz：X, dealin, y, v。

用法：
    PYTHONPATH=. python3 scripts/rl/gen_hybrid_dealin_data.py \
        output/nn_teacher_hybrid_dealin_1000.npz 1000 16 \
        output/nn_conv_bc_dealin_2000_l07.pt beliefexp 28 10000000

参数：out_path n_games n_workers nn_model_path belief_kind tenpai_threshold seed_base
"""

import sys
import os
import random
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tile_pool
import tile
import agent
import algo.eval.v2 as eval_v2
import algo.context.v3 as context_v3
from algo.nn.features import extract_features, _TILE_TO_IDX


def _dealin_label(hand14, all_hands, current_name):
    player_names = sorted(all_hands.keys(), key=lambda n: int(n.split('@')[-1]))
    opponents = [p for p in player_names if p != current_name]
    label = np.full(34, -1.0, dtype=np.float32)
    seen = set()
    for t in hand14:
        if t in seen:
            continue
        seen.add(t)
        idx = int(_TILE_TO_IDX[t])
        deals_in = False
        for opp in opponents:
            if eval_v2.is_win(list(all_hands[opp]) + [t]):
                deals_in = True
                break
        label[idx] = 1.0 if deals_in else 0.0
    return label


def _play_one_game(seed, nn_path, belief_kind, threshold):
    import torch
    torch.set_num_threads(1)
    from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent

    random.seed(seed)
    np.random.seed(seed % (2 ** 31 - 1))

    pool = tile_pool.Pool()
    names = [f'P@{i}' for i in range(4)]
    hands = {n: pool.next_n(13) for n in names}
    ctx = context_v3.ContextV3()
    locked = set()
    turn = 0
    wall = list(pool.tiles[pool.idx:])

    agents = {n: HybridNNBeliefAgent(
        n, nn_model_path=nn_path, belief_kind=belief_kind,
        tenpai_threshold=threshold, device='cpu', temperature=0.0,
        verbose=False) for n in names}
    for n in names:
        agents[n].init_tiles(list(hands[n]))

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
        agents[pname].add(drawn)

        if eval_v2.is_win(hands[pname]):
            for n in names:
                outcome[n] = 1.0 if n == pname else -1.0
            full_samples = []
            for x, d, y, pname in samples:
                full_samples.append((x, d, y, outcome.get(pname, 0.0)))
            return full_samples, outcome

        if pname in locked:
            discarded = drawn
            hands[pname].remove(discarded)
            agents[pname].cur.remove(discarded)
        else:
            x = extract_features(ctx, hands[pname], pname)
            d = _dealin_label(hands[pname], hands, pname)
            discarded = agents[pname].next()
            y = int(_TILE_TO_IDX[discarded])
            samples.append((x, d, y, pname))
            hands[pname].remove(discarded)

        # 同步其他 agent 的 context（当前玩家自己的 next() 已经更新过）
        msg = agent.Message(pname, 'put', discarded)
        for n in names:
            if n == pname:
                continue
            agents[n].handle_msg(msg)

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
                for x, d, y, pname in samples:
                    full_samples.append((x, d, y, outcome.get(pname, 0.0)))
                return full_samples, outcome

        ctx.see_tile(discarded, pname)

        if (pname not in locked and len(hands[pname]) == 13 and
                eval_v2.shanten(hands[pname]) == 0):
            rem = ctx.remaining_wall(hands[pname])
            waits = eval_v2.winning_tiles(hands[pname], rem)
            if sum(rem.get(t, 0) for t in waits) >= 3:
                locked.add(pname)
                ctx.declare_tenpai(pname)
                # 同步 tenpai
                msg_t = agent.Message(pname, 'tenpai', None)
                for n in names:
                    agents[n].handle_msg(msg_t)

        turn = (turn + 1) % 4

    full_samples = []
    for x, d, y, pname in samples:
        full_samples.append((x, d, y, outcome.get(pname, 0.0)))
    return full_samples, outcome


def _thread_worker(args):
    i, s, nn_path, belief_kind, threshold = args
    return _play_one_game(s, nn_path, belief_kind, threshold)[0]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'output/nn_teacher_hybrid_dealin.npz'
    total_games = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    nn_path = sys.argv[4] if len(sys.argv) > 4 else 'output/nn_conv_bc_dealin_2000_l07.pt'
    belief_kind = sys.argv[5] if len(sys.argv) > 5 else 'beliefexp'
    threshold = int(sys.argv[6]) if len(sys.argv) > 6 else 28
    seed_base = int(sys.argv[7]) if len(sys.argv) > 7 else 0

    print(f'Hybrid-dealin07 teacher data: {total_games} games, {workers} workers, '
          f'nn={nn_path}, belief={belief_kind}, threshold={threshold}')

    t0 = time.time()
    args_list = [(i, seed_base + i, nn_path, belief_kind, threshold)
                 for i in range(total_games)]

    # 用 spawn 避免 torch+fork 死锁；每个 worker 独立加载模型
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_thread_worker, args_list))

    all_samples = []
    for r in results:
        all_samples.extend(r)
    dt = time.time() - t0
    print(f'Generated {len(all_samples)} samples from {total_games} games in {dt:.1f}s')

    if not all_samples:
        print('No samples generated')
        return

    X = np.stack([s[0] for s in all_samples])
    D = np.stack([s[1] for s in all_samples])
    y = np.array([s[2] for s in all_samples], dtype=np.int64)
    v = np.array([s[3] for s in all_samples], dtype=np.float32)
    np.savez(out_path, X=X, dealin=D, y=y, v=v)
    print(f'Saved {out_path}: X={X.shape}, dealin={D.shape}, y={y.shape}, v={v.shape}')


if __name__ == '__main__':
    main()
