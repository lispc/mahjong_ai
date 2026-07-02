# -*- coding: utf-8 -*-
"""用纯 BeliefExpectimaxAgent 当教师生成带 search trace 的数据。

4 个座位全是 BeliefExpectimaxAgent（每步都搜索），记录每个决策的 trace。
输出 .npz 包含：
- X：175 维特征
- y：最终动作
- scores：34 维候选 offense 评分
- dangers：34 维候选 danger 评分
- selected_value：被选中动作 offense 评分
- has_trace：是否含轨迹（恒为 1）
- dealin：34 维即时点炮标签
- v：最终 outcome

用法：
    PYTHONPATH=. python3 scripts/rl/gen_beliefexp_trace_data.py \
        output/nn_teacher_beliefexp_trace_500.npz 500 16 15000000

参数：out_path n_games n_workers seed_base
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
from algo.agents.belief_expectimax import BeliefExpectimaxAgent
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


def _play_one_game(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 31 - 1))

    pool = tile_pool.Pool()
    names = [f'P@{i}' for i in range(4)]
    hands = {n: pool.next_n(13) for n in names}
    ctx = context_v3.ContextV3()
    locked = set()
    turn = 0
    wall = list(pool.tiles[pool.idx:])

    agents = {n: BeliefExpectimaxAgent(n, verbose=False) for n in names}
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
            return [(x, y, s, dang, sv, ht, d, v)
                    for x, y, s, dang, sv, ht, d, v, p in samples]

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
            dangers = np.full(34, SCORE_PAD, dtype=np.float32)
            sv = 0.0
            ht = 1.0
            if trace is not None:
                sv = float(trace['selected_value'])
                for t, sc in trace['scores'].items():
                    scores[int(_TILE_TO_IDX[t])] = float(sc)
                for t, dg in trace.get('dangers', {}).items():
                    dangers[int(_TILE_TO_IDX[t])] = float(dg)
            samples.append((x, y, scores, dangers, sv, ht, d,
                            outcome.get(pname, 0.0), pname))
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
                return [(x, y, s, dang, sv, ht, d, v)
                        for x, y, s, dang, sv, ht, d, v, p in samples]

        ctx.see_tile(discarded, pname)

        if (pname not in locked and len(hands[pname]) == 13 and
                eval_v2.shanten(hands[pname]) == 0):
            declared = agents[pname].declare_tenpai(list(hands[pname]), ctx)
            if declared:
                locked.add(pname)
                ctx.declare_tenpai(pname)
                msg_t = agent.Message(pname, 'tenpai', None)
                for n in names:
                    agents[n].handle_msg(msg_t)

        turn = (turn + 1) % 4

    return [(x, y, s, dang, sv, ht, d, v)
            for x, y, s, dang, sv, ht, d, v, p in samples]


def _thread_worker(args):
    i, s, completed = args
    result = _play_one_game(s)
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
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'output/nn_teacher_beliefexp_trace.npz'
    total_games = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    seed_base = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    print(f'BeliefExp trace teacher data: {total_games} games, {workers} workers')

    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    completed = manager.Value('i', 0)

    args_list = [(i, seed_base + i, completed) for i in range(total_games)]

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
    print(f'Generated {len(all_samples)} samples from {total_games} games in {dt:.1f}s')

    if not all_samples:
        print('No samples generated')
        return

    X = np.stack([s[0] for s in all_samples])
    y = np.array([s[1] for s in all_samples], dtype=np.int64)
    S = np.stack([s[2] for s in all_samples])
    DANG = np.stack([s[3] for s in all_samples])
    SV = np.array([s[4] for s in all_samples], dtype=np.float32)
    HT = np.array([s[5] for s in all_samples], dtype=np.float32)
    D = np.stack([s[6] for s in all_samples])
    v = np.array([s[7] for s in all_samples], dtype=np.float32)

    np.savez(out_path, X=X, y=y, scores=S, dangers=DANG, selected_value=SV,
             has_trace=HT, dealin=D, v=v)
    print(f'Saved {out_path}: X={X.shape}, y={y.shape}, v={v.shape}')


if __name__ == '__main__':
    main()
