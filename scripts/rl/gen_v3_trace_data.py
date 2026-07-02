# -*- coding: utf-8 -*-
"""用 V3 Expectimax 当教师生成带搜索轨迹（search trace）的数据。

4 个座位全是 V3 Expectimax（默认 expectimax depth=1, candidate_policy='nn', leaf='nn'，
可通过参数改为更快/不同的教师）。
对每个弃牌决策记录：
- X：175 维公开信息特征（手牌 14 张）
- y：教师最终选择的动作
- scores：34 维候选评分（expectimax offense value），非候选填 -1e9
- selected_value：被选中动作的评分
- v：该局最终 outcome

这些评分可作为 soft policy target（KL loss）和 dense value target，
把 conv-BC 从 hard-label 模仿升级到搜索轨迹蒸馏。

用法：
    PYTHONPATH=. python3 scripts/rl/gen_v3_trace_data.py \
        output/nn_teacher_v3_trace_500.npz 500 16 50000000 \
        --leaf eval0 --candidate nn

参数：out_path n_games n_workers seed_base [--leaf nn|eval0] [--candidate nn|...]
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


SCORE_PAD = -1e9


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


def _play_one_game(seed, leaf_evaluator, candidate_policy):
    import torch
    torch.set_num_threads(1)
    from algo.agents.belief_expectimax_v3 import BeliefExpectimaxV3Agent

    random.seed(seed)
    np.random.seed(seed % (2 ** 31 - 1))

    pool = tile_pool.Pool()
    names = [f'P@{i}' for i in range(4)]
    hands = {n: pool.next_n(13) for n in names}
    ctx = context_v3.ContextV3()
    locked = set()
    turn = 0
    wall = list(pool.tiles[pool.idx:])

    agents = {n: BeliefExpectimaxV3Agent(
        n, expectimax_depth=1, max_candidates=5,
        leaf_evaluator=leaf_evaluator, candidate_policy=candidate_policy,
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
            return [(x, y, s, sv, d, outcome.get(p, 0.0))
                    for x, y, s, sv, d, p in samples]

        if pname in locked:
            discarded = drawn
            hands[pname].remove(discarded)
            agents[pname].cur.remove(discarded)
        else:
            x = extract_features(ctx, hands[pname], pname)
            d = _dealin_label(hands[pname], hands, pname)
            discarded, trace = agents[pname].next_with_trace()
            y = int(_TILE_TO_IDX[discarded])
            scores = np.full(34, SCORE_PAD, dtype=np.float32)
            for t, sc in trace['scores'].items():
                scores[int(_TILE_TO_IDX[t])] = float(sc)
            sv = float(trace['selected_value'])
            samples.append((x, y, scores, sv, d, pname))
            hands[pname].remove(discarded)

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
                return [(x, y, s, sv, d, outcome.get(p, 0.0))
                        for x, y, s, sv, d, p in samples]

        ctx.see_tile(discarded, pname)

        if (pname not in locked and len(hands[pname]) == 13 and
                eval_v2.shanten(hands[pname]) == 0):
            teacher_ctx = agents[pname].context
            declared = agents[pname].declare_tenpai(list(hands[pname]), teacher_ctx)
            if declared:
                locked.add(pname)
                ctx.declare_tenpai(pname)
                msg_t = agent.Message(pname, 'tenpai', None)
                for n in names:
                    agents[n].handle_msg(msg_t)

        turn = (turn + 1) % 4

    return [(x, y, s, sv, d, outcome.get(p, 0.0))
            for x, y, s, sv, d, p in samples]


def _thread_worker(args):
    i, s, leaf_evaluator, candidate_policy = args
    return _play_one_game(s, leaf_evaluator, candidate_policy)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'output/nn_teacher_v3_trace.npz'
    total_games = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    seed_base = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    leaf_evaluator = 'nn'
    candidate_policy = 'nn'
    if len(sys.argv) > 5:
        leaf_evaluator = sys.argv[5]
    if len(sys.argv) > 6:
        candidate_policy = sys.argv[6]

    print(f'V3 Expectimax trace teacher data: {total_games} games, {workers} workers, '
          f'leaf={leaf_evaluator}, candidate={candidate_policy}')

    t0 = time.time()
    args_list = [(i, seed_base + i, leaf_evaluator, candidate_policy)
                 for i in range(total_games)]

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
    y = np.array([s[1] for s in all_samples], dtype=np.int64)
    S = np.stack([s[2] for s in all_samples])
    SV = np.array([s[3] for s in all_samples], dtype=np.float32)
    D = np.stack([s[4] for s in all_samples])
    v = np.array([s[5] for s in all_samples], dtype=np.float32)

    np.savez(out_path, X=X, y=y, scores=S, selected_value=SV, dealin=D, v=v)
    print(f'Saved {out_path}: X={X.shape}, scores={S.shape}, y={y.shape}, v={v.shape}')


if __name__ == '__main__':
    main()
