# -*- coding: utf-8 -*-
"""用 Hybrid 教师生成带 critical-trace 的数据。

4 个座位全是同一个 Hybrid 教师（平时 NN，对手报听/终盘切 BeliefExp）。
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
        output/nn_conv_bc_hybrid_2000.pt beliefexp 28 13000000 --save-every 250

参数：out_path n_games n_workers nn_model_path belief_kind tenpai_threshold seed_base [--save-every N] [--resume]
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


def _empty_arrays():
    return {'X': None, 'y': None, 'scores': None, 'selected_value': None,
            'has_trace': None, 'dealin': None, 'v': None}


def _stack_chunk(chunk):
    if not chunk:
        return _empty_arrays()
    X = np.stack([s[0] for s in chunk])
    y = np.array([s[1] for s in chunk], dtype=np.int64)
    S = np.stack([s[2] for s in chunk])
    SV = np.array([s[3] for s in chunk], dtype=np.float32)
    HT = np.array([s[4] for s in chunk], dtype=np.float32)
    D = np.stack([s[5] for s in chunk])
    v = np.array([s[6] for s in chunk], dtype=np.float32)
    return {'X': X, 'y': y, 'scores': S, 'selected_value': SV,
            'has_trace': HT, 'dealin': D, 'v': v}


def _merge_arrays(acc, chunk_arrays):
    for k in acc:
        a = chunk_arrays.get(k)
        if a is None:
            continue
        if acc[k] is None:
            acc[k] = a
        else:
            acc[k] = np.concatenate([acc[k], a], axis=0)
    return acc


def _save_checkpoint(acc, completed, path):
    if acc['X'] is None:
        return
    save = dict(acc)
    save['completed'] = np.array(completed, dtype=np.int64)
    np.savez(path, **save)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('out_path', default='output/nn_teacher_hybrid_trace.npz')
    parser.add_argument('total_games', type=int, default=1000)
    parser.add_argument('workers', type=int, default=8)
    parser.add_argument('nn_path', default='output/nn_conv_bc_hybrid_2000.pt')
    parser.add_argument('belief_kind', default='beliefexp')
    parser.add_argument('threshold', type=int, default=28)
    parser.add_argument('seed_base', type=int, default=0)
    parser.add_argument('--save-every', type=int, default=250,
                        help='每完成 N 局保存一次 checkpoint（0 表示不保存）')
    parser.add_argument('--resume', action='store_true',
                        help='从 .checkpoint.npz 断点续跑')
    args = parser.parse_args()

    out_path = args.out_path
    total_games = args.total_games
    workers = args.workers
    nn_path = args.nn_path
    belief_kind = args.belief_kind
    threshold = args.threshold
    seed_base = args.seed_base
    save_every = args.save_every
    checkpoint_path = out_path + '.checkpoint.npz'

    print(f'Hybrid trace teacher data: {total_games} games, {workers} workers, '
          f'nn={nn_path}, belief={belief_kind}, threshold={threshold}, '
          f'save_every={save_every}, resume={args.resume}')

    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    completed = manager.Value('i', 0)

    acc = _empty_arrays()
    start_i = 0
    if args.resume and os.path.exists(checkpoint_path):
        d = np.load(checkpoint_path)
        start_i = int(d['completed'])
        completed.value = start_i
        for k in acc:
            if k in d:
                acc[k] = d[k]
        print(f'Resumed from checkpoint: {start_i}/{total_games} games already done, '
              f'{acc["X"].shape[0] if acc["X"] is not None else 0} samples')

    args_list = [(i, seed_base + i, nn_path, belief_kind, threshold, completed)
                 for i in range(start_i, total_games)]

    reporter = threading.Thread(target=_progress_reporter, args=(completed, total_games), daemon=True)
    reporter.start()

    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor
    chunk = []
    processed = start_i
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(_thread_worker, args_list):
            chunk.extend(result)
            processed += 1
            if save_every > 0 and (processed % save_every == 0 or processed == total_games):
                chunk_arrays = _stack_chunk(chunk)
                acc = _merge_arrays(acc, chunk_arrays)
                _save_checkpoint(acc, processed, checkpoint_path)
                print(f'[checkpoint] {processed}/{total_games} games, '
                      f'{acc["X"].shape[0] if acc["X"] is not None else 0} samples', flush=True)
                chunk = []

    dt = time.time() - t0
    if chunk:
        chunk_arrays = _stack_chunk(chunk)
        acc = _merge_arrays(acc, chunk_arrays)

    print(f'Generated {total_games - start_i} new game results in {dt:.1f}s')
    n_trace = int((acc['has_trace'] > 0.5).sum()) if acc['has_trace'] is not None else 0
    print(f'Total {acc["X"].shape[0] if acc["X"] is not None else 0} samples ({n_trace} with trace) '
          f'from {total_games} games')

    if acc['X'] is None:
        print('No samples generated')
        return

    np.savez(out_path, **acc)
    print(f'Saved {out_path}: X={acc["X"].shape}, trace={n_trace}, y={acc["y"].shape}, v={acc["v"].shape}')

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f'Removed checkpoint {checkpoint_path}')


if __name__ == '__main__':
    main()
