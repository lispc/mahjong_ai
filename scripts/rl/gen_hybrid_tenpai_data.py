# -*- coding: utf-8 -*-
"""用 Hybrid 教师生成带 tenpai 决策标签的数据。

4 个座位全是 Hybrid（平时 NN，对手报听/终盘切 BeliefExp）。
对每个弃牌决策记录：
- X：175 维公开信息特征（手牌 14 张）
- dealin：34 维即时点炮标签（1=点炮，0=安全，-1=不在手牌）
- y：Hybrid 选择的动作
- v：该局最终 outcome

 additionally，对每次弃牌后进入听牌的 13 张手牌状态记录：
- X_tenpai：175 维特征（手牌 13 张）
- t：教师是否选择报听（1=报听，0=不报）
- v_tenpai：该局最终 outcome

教师自身的报听决策由 agent.declare_tenpai() 给出（使用其内部启发式或 tenpai_head），
从而保证轨迹与标签一致。

用法：
    PYTHONPATH=. python3 scripts/rl/gen_hybrid_tenpai_data.py \
        output/nn_teacher_hybrid_tenpai_1000.npz 1000 16 \
        output/nn_conv_bc_hybrid_2000.pt beliefexp 28 20000000

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

    discard_samples = []
    tenpai_samples = []
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
            return _attach_outcome(discard_samples, tenpai_samples, outcome)

        if pname in locked:
            discarded = drawn
            hands[pname].remove(discarded)
            agents[pname].cur.remove(discarded)
        else:
            x14 = extract_features(ctx, hands[pname], pname)
            d = _dealin_label(hands[pname], hands, pname)
            discarded = agents[pname].next()
            y = int(_TILE_TO_IDX[discarded])
            discard_samples.append((x14, d, y, pname))
            hands[pname].remove(discarded)

        # 同步其他 agent 的 context
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
                return _attach_outcome(discard_samples, tenpai_samples, outcome)

        ctx.see_tile(discarded, pname)

        # 使用教师真实报听决策（与 agent 内部一致）
        if (pname not in locked and len(hands[pname]) == 13 and
                eval_v2.shanten(hands[pname]) == 0):
            teacher_ctx = agents[pname].nn_agent.context
            declared = agents[pname].declare_tenpai(list(hands[pname]), teacher_ctx)
            x13 = extract_features(ctx, hands[pname], pname)
            tenpai_samples.append((x13, 1.0 if declared else 0.0, pname))
            if declared:
                locked.add(pname)
                ctx.declare_tenpai(pname)
                msg_t = agent.Message(pname, 'tenpai', None)
                for n in names:
                    agents[n].handle_msg(msg_t)

        turn = (turn + 1) % 4

    return _attach_outcome(discard_samples, tenpai_samples, outcome)


def _attach_outcome(discard_samples, tenpai_samples, outcome):
    discard_full = [(x, d, y, outcome.get(pname, 0.0)) for x, d, y, pname in discard_samples]
    tenpai_full = [(x, t, outcome.get(pname, 0.0)) for x, t, pname in tenpai_samples]
    return discard_full, tenpai_full, outcome


def _thread_worker(args):
    i, s, nn_path, belief_kind, threshold = args
    return _play_one_game(s, nn_path, belief_kind, threshold)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'output/nn_teacher_hybrid_tenpai.npz'
    total_games = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    nn_path = sys.argv[4] if len(sys.argv) > 4 else 'output/nn_conv_bc_hybrid_2000.pt'
    belief_kind = sys.argv[5] if len(sys.argv) > 5 else 'beliefexp'
    threshold = int(sys.argv[6]) if len(sys.argv) > 6 else 28
    seed_base = int(sys.argv[7]) if len(sys.argv) > 7 else 0

    print(f'Hybrid tenpai teacher data: {total_games} games, {workers} workers, '
          f'nn={nn_path}, belief={belief_kind}, threshold={threshold}')

    t0 = time.time()
    args_list = [(i, seed_base + i, nn_path, belief_kind, threshold)
                 for i in range(total_games)]

    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_thread_worker, args_list))

    discard_all = []
    tenpai_all = []
    for d_samples, t_samples, _ in results:
        discard_all.extend(d_samples)
        tenpai_all.extend(t_samples)
    dt = time.time() - t0
    print(f'Generated {len(discard_all)} discard + {len(tenpai_all)} tenpai samples '
          f'from {total_games} games in {dt:.1f}s')

    data = {}
    if discard_all:
        data['X'] = np.stack([s[0] for s in discard_all])
        data['dealin'] = np.stack([s[1] for s in discard_all])
        data['y'] = np.array([s[2] for s in discard_all], dtype=np.int64)
        data['v'] = np.array([s[3] for s in discard_all], dtype=np.float32)
    if tenpai_all:
        data['X_tenpai'] = np.stack([s[0] for s in tenpai_all])
        data['t'] = np.array([s[1] for s in tenpai_all], dtype=np.float32)
        data['v_tenpai'] = np.array([s[2] for s in tenpai_all], dtype=np.float32)

    if not data:
        print('No samples generated')
        return

    np.savez(out_path, **data)
    shapes = ', '.join(f'{k}={v.shape}' for k, v in data.items())
    print(f'Saved {out_path}: {shapes}')


if __name__ == '__main__':
    main()
