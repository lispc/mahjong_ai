# -*- coding: utf-8 -*-
"""用 AlphaZeroMCTSAgent 生成 search trace 数据。

每个 sample 包含：
    features: 175-dim 当前玩家视角特征
    visit_dist: 34-dim MCTS 根节点访问分布（policy target）
    value: value target，默认用该局最终 outcome（P0 赢 +1 / 输 -1 / 流局 0）

用法：
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python3 scripts/rl/gen_alphazero_data.py \
        output/nn_full_action_best.pt output/alphazero_trace_100.npz 100 4 \
        --n-worlds 4 --n-sims 16 --max-depth 2 --device cuda --resume
"""
import os
import sys
import time
import argparse
import numpy as np
import multiprocessing as mp

# 必须在任何 CUDA 初始化前设置 spawn
mp.set_start_method('spawn', force=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from driver.engine import play_game
from agent import Agent
from algo.agents.alphazero_mcts_agent import AlphaZeroMCTSAgent


def play_one_game(seed, model_path, n_worlds, n_sims, max_depth, device):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed % 2**32)
    mcts = AlphaZeroMCTSAgent('P0', model_path=model_path,
                              n_worlds=n_worlds, n_sims=n_sims, max_depth=max_depth,
                              device=device, temperature=0.0, verbose=False)
    agents = [mcts] + [Agent(f'P{i}', verbose=False) for i in range(1, 4)]
    result = play_game(agents, verbose=False, record_time=False)
    traces = mcts.all_traces()
    # 游戏结局：P0 赢 +1，流局 0，输 -1
    winner = result.get('winner')
    if winner == 'P0':
        outcome = 1.0
    elif winner is None:
        outcome = 0.0
    else:
        outcome = -1.0
    return result, traces, outcome


def _save_checkpoint(out_path, traces, seed_offsets, outcome=None):
    ckpt = out_path + '.checkpoint.npz'
    meta = out_path + '.checkpoint_meta.json'
    if not traces:
        return
    X = np.stack([t['features'] for t in traces], axis=0).astype(np.float32)
    visits = np.stack([t['visit_dist'] for t in traces], axis=0).astype(np.float32)
    values = np.array([t['value'] for t in traces], dtype=np.float32)
    seed_offsets = np.array(seed_offsets, dtype=np.int64)
    kw = {'X': X, 'visit_dist': visits, 'value': values, 'seed_offsets': seed_offsets}
    if outcome is not None:
        kw['outcome'] = np.float32(outcome)
    np.savez(ckpt, **kw)
    import json
    with open(meta, 'w') as f:
        json.dump({'n_traces': len(traces), 'n_games': len(seed_offsets),
                   'outcome': float(outcome) if outcome is not None else None}, f)


def _load_checkpoint(out_path):
    ckpt = out_path + '.checkpoint.npz'
    if not os.path.exists(ckpt):
        return [], [], None
    d = np.load(ckpt)
    traces = []
    for i in range(len(d['X'])):
        traces.append({
            'features': d['X'][i],
            'visit_dist': d['visit_dist'][i],
            'value': float(d['value'][i]),
        })
    seed_offsets = d['seed_offsets'].tolist()
    outcome = float(d['outcome']) if 'outcome' in d else None
    print(f'Resumed {len(traces)} traces from {ckpt}')
    return traces, seed_offsets, outcome


def _save_final(out_path, traces):
    X = np.stack([t['features'] for t in traces], axis=0).astype(np.float32)
    visits = np.stack([t['visit_dist'] for t in traces], axis=0).astype(np.float32)
    values = np.array([t['value'] for t in traces], dtype=np.float32)
    np.savez(out_path, X=X, visit_dist=visits, value=values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model_path')
    ap.add_argument('out_path')
    ap.add_argument('n_games', type=int)
    ap.add_argument('n_workers', type=int)
    ap.add_argument('--n-worlds', type=int, default=4)
    ap.add_argument('--n-sims', type=int, default=16)
    ap.add_argument('--max-depth', type=int, default=2)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--seed-base', type=int, default=700000)
    ap.add_argument('--save-every', type=int, default=50,
                    help='每完成多少局保存一次 checkpoint')
    ap.add_argument('--resume', action='store_true',
                    help='若存在 checkpoint 则续跑')
    ap.add_argument('--timeout', type=int, default=3600,
                    help='单局最大等待秒数')
    ap.add_argument('--value-target', type=str, default='outcome',
                    choices=['outcome', 'mcts'],
                    help='value target：outcome 用该局最终结果，mcts 用搜索根节点值')
    args = ap.parse_args()

    all_traces, done_offsets, _ = _load_checkpoint(args.out_path) if args.resume else ([], [], None)
    done_set = set(done_offsets)
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=mp.get_context('spawn')) as pool:
        futures = []
        offsets = []
        for i in range(args.n_games):
            if i in done_set:
                continue
            offsets.append(i)
            futures.append(pool.submit(play_one_game, args.seed_base + i, args.model_path,
                                       args.n_worlds, args.n_sims, args.max_depth, args.device))
        print(f'Submitted {len(futures)} new games ({args.n_games - len(futures)} resumed)')
        for offset, fut in zip(offsets, futures):
            try:
                result, traces, outcome = fut.result(timeout=args.timeout)
                # 根据目标替换 value
                if args.value_target == 'outcome':
                    for tr in traces:
                        tr['value'] = outcome
                all_traces.extend(traces)
                done_offsets.append(offset)
                if len(done_offsets) % 10 == 0:
                    print(f'  {len(done_offsets)}/{args.n_games} done, traces={len(all_traces)}, time={time.time()-t0:.1f}s')
                if len(done_offsets) % args.save_every == 0:
                    _save_checkpoint(args.out_path, all_traces, done_offsets,
                                     outcome=outcome)
                    print(f'  checkpoint saved ({len(all_traces)} traces)')
            except Exception as e:
                print(f'  game {offset} failed: {e}')

    # 删除 checkpoint，保存最终文件
    ckpt = args.out_path + '.checkpoint.npz'
    meta = args.out_path + '.checkpoint_meta.json'
    if os.path.exists(ckpt):
        os.remove(ckpt)
    if os.path.exists(meta):
        os.remove(meta)

    if not all_traces:
        print('No traces collected')
        return
    _save_final(args.out_path, all_traces)
    print(f'Saved {args.out_path}: {len(all_traces)} traces in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
