# -*- coding: utf-8 -*-
"""用 Hybrid-hybridBase 当教师生成带 critical-trace 的数据。

4 个座位全是 Hybrid-hybridBase（平时 NN，对手报听/终盘切 BeliefExp）。
只在 critical 状态（agent 实际使用 BeliefExp 搜索）记录搜索轨迹；
非 critical 状态只记录普通 hard label，scores 全 -1e9、has_trace=0。

输出 .npz 包含：
- X：175 维特征
- y：最终动作
- scores：34 维候选评分（非 critical 全 -1e9）
- selected_value：被选中动作评分（非 critical 0）
- has_trace：是否含轨迹（1/0）
- dealin：34 维即时点炮标签
- v：最终 outcome

用法：
    PYTHONPATH=. python3 scripts/rl/gen_hybrid_trace_data.py \
        output/nn_teacher_hybrid_trace_1000.npz 1000 16 \
        output/nn_conv_bc_hybrid_2000.pt beliefexp 28 13000000

参数：out_path n_games n_workers nn_model_path belief_kind tenpai_threshold seed_base
"""

import sys
import os
import random
import numpy as np
import time
import threading

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
            return [(x, y, s, sv, ht, d, outcome.get(p, 0.0))
                    for x, y, s, sv, ht, d, p in samples]

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
            sv = 0.0
            ht = 0.0
            if trace is not None:
                ht = 1.0
                sv = float(trace['selected_value'])
                for t, sc in trace['scores'].items():
                    scores[int(_TILE_TO_IDX[t])] = float(sc)
            samples.append((x, y, scores, sv, ht, d, pname))
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
                return [(x, y, s, sv, ht, d, outcome.get(p, 0.0))
                        for x, y, s, sv, ht, d, p in samples]

        ctx.see_tile(discarded, pname)

        if (pname not in locked and len(hands[pname]) == 13 and
                eval_v2.shanten(hands[pname]) == 0):
            teacher_ctx = agents[pname].nn_agent.context
            declared = agents[pname].declare_tenpai(list(hands[pname]), teacher_ctx)
            if declared:
                locked.add(pname)
                ctx.declare_tenpai(pname)
                msg_t = agent.Message(pname, 'tenpai', None)
                for n in names:
                    agents[n].handle_msg(msg_t)

        turn = (turn + 1) % 4

    return [(x, y, s, sv, ht, d, outcome.get(p, 0.0))
            for x, y, s, sv, ht, d, p in samples]


def _thread_worker(args):
    i, s, nn_path, belief_kind, threshold, completed = args
    result = _play_one_game(s, nn_path, belief_kind, threshold)
    if completed is not None:
        completed.value += 1
    return result


def _progress_reporter(completed, total, interval=30):
    while True:
        time.sleep(interval)
        if completed is None:
            break
        n = completed.value
        if n >= total:
            break
        pct = 100.0 * n / total if total > 0 else 0.0
        print(f'[progress] {n}/{total} games completed ({pct:.1f}%)', flush=True)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'output/nn_teacher_hybrid_trace.npz'
    total_games = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    nn_path = sys.argv[4] if len(sys.argv) > 4 else 'output/nn_conv_bc_hybrid_2000.pt'
    belief_kind = sys.argv[5] if len(sys.argv) > 5 else 'beliefexp'
    threshold = int(sys.argv[6]) if len(sys.argv) > 6 else 28
    seed_base = int(sys.argv[7]) if len(sys.argv) > 7 else 0

    print(f'Hybrid trace teacher data: {total_games} games, {workers} workers, '
          f'nn={nn_path}, belief={belief_kind}, threshold={threshold}')

    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    completed = manager.Value('i', 0)

    args_list = [(i, seed_base + i, nn_path, belief_kind, threshold, completed)
                 for i in range(total_games)]

    reporter = threading.Thread(target=_progress_reporter, args=(completed, total_games), daemon=True)
    reporter.start()

    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_thread_worker, args_list))

    dt = time.time() - t0
    print(f'Generated {len(results)} game results in {dt:.1f}s')

    all_samples = []
    for r in results:
        all_samples.extend(r)
    n_trace = sum(1 for s in all_samples if s[4] > 0.5)
    print(f'Generated {len(all_samples)} samples ({n_trace} with trace) '
          f'from {total_games} games in {dt:.1f}s')

    if not all_samples:
        print('No samples generated')
        return

    X = np.stack([s[0] for s in all_samples])
    y = np.array([s[1] for s in all_samples], dtype=np.int64)
    S = np.stack([s[2] for s in all_samples])
    SV = np.array([s[3] for s in all_samples], dtype=np.float32)
    HT = np.array([s[4] for s in all_samples], dtype=np.float32)
    D = np.stack([s[5] for s in all_samples])
    v = np.array([s[6] for s in all_samples], dtype=np.float32)

    np.savez(out_path, X=X, y=y, scores=S, selected_value=SV, has_trace=HT,
             dealin=D, v=v)
    print(f'Saved {out_path}: X={X.shape}, trace={n_trace}, y={y.shape}, v={v.shape}')


if __name__ == '__main__':
    main()
