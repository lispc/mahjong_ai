# -*- coding: utf-8 -*-
"""用 AlphaZeroMCTSAgent 生成 search trace 数据。

每个 sample 包含：
    features: 175-dim 当前玩家视角特征
    visit_dist: 34-dim MCTS 根节点访问分布（policy target）
    value: MCTS 估计的当前玩家期望价值（value target）

用法：
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python3 scripts/rl/gen_alphazero_data.py \
        output/nn_full_action_best.pt output/alphazero_trace_100.npz 100 4 \
        --n-worlds 4 --n-sims 16 --max-depth 2 --device cuda
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
    return result, traces


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
    args = ap.parse_args()

    all_traces = []
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(play_one_game, args.seed_base + i, args.model_path,
                               args.n_worlds, args.n_sims, args.max_depth, args.device)
                   for i in range(args.n_games)]
        for i, fut in enumerate(futures):
            try:
                result, traces = fut.result(timeout=600)
                all_traces.extend(traces)
                if (i + 1) % 10 == 0:
                    print(f'  {i+1}/{args.n_games} done, traces={len(all_traces)}, time={time.time()-t0:.1f}s')
            except Exception as e:
                print(f'  game {i} failed: {e}')

    if not all_traces:
        print('No traces collected')
        return
    X = np.stack([t['features'] for t in all_traces], axis=0).astype(np.float32)
    visits = np.stack([t['visit_dist'] for t in all_traces], axis=0).astype(np.float32)
    values = np.array([t['value'] for t in all_traces], dtype=np.float32)
    np.savez(args.out_path, X=X, visit_dist=visits, value=values)
    print(f'Saved {args.out_path}: {len(all_traces)} traces in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
