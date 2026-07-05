# -*- coding: utf-8 -*-
"""当前 best policy 自对弈，只保留赢家 trajectory 做 BC。

用法：
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python3 scripts/rl/gen_selfplay_winner_data.py \
        output/nn_full_action_best.pt output/nn_selfplay_winner_5000.npz 5000 16
"""
import os
import sys
import time
import argparse
import multiprocessing as mp
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from driver.engine import play_game
from algo.agents.ppo_agent import PPOAgent
from algo.nn.features import extract_features, tile_to_index
from algo.context.v3 import ContextV3


def _tile_indices(tiles):
    return [tile_to_index(t) for t in tiles]


class RecordingPPOAgent(PPOAgent):
    def __init__(self, name, model_path, device='cpu'):
        super().__init__(name, model_path=model_path, device=device, temperature=0.0, verbose=False)
        self.discard_steps = []

    def init_tiles(self, l):
        super().init_tiles(l)
        self.discard_steps = []

    def next(self):
        # 在 super().next() 修改状态前记录当前特征
        feats = extract_features(self.context, self.cur, self.name)
        tile = super().next()
        self.discard_steps.append((np.asarray(feats, dtype=np.float32), tile_to_index(tile)))
        return tile


def _worker(args):
    model_path, n_games, seed_base, device = args
    torch.set_num_threads(1)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    import random
    random.seed(seed_base)
    np.random.seed(seed_base % 2**32)
    torch.manual_seed(seed_base % 2**32)

    Xd, yd, vd = [], [], []
    for g in range(n_games):
        random.seed(seed_base + g)
        np.random.seed((seed_base + g) % 2**32)
        agents = [RecordingPPOAgent(f'P{i}', model_path, device=device) for i in range(4)]
        result = play_game(agents, verbose=False, record_time=False)
        winner = result.get('winner')
        if winner is None:
            continue
        for a in agents:
            if a.name != winner:
                continue
            for feats, act in a.discard_steps:
                Xd.append(feats); yd.append(act); vd.append(1.0)
    if not Xd:
        return (np.zeros((0, 175), np.float32), np.zeros((0,), np.int64), np.zeros((0,), np.float32))
    return (np.stack(Xd), np.asarray(yd, np.int64), np.asarray(vd, np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model_path')
    ap.add_argument('out_path')
    ap.add_argument('n_games', type=int)
    ap.add_argument('n_workers', type=int)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--seed-base', type=int, default=1000000)
    ap.add_argument('--games-per-task', type=int, default=5)
    args = ap.parse_args()

    mp.set_start_method('spawn', force=True)
    tasks = []
    remaining = args.n_games
    g0 = args.seed_base
    while remaining > 0:
        g = min(args.games_per_task, remaining)
        tasks.append((args.model_path, g, g0, args.device))
        g0 += g
        remaining -= g

    print(f'Self-play winner data: model={args.model_path} games={args.n_games} workers={args.n_workers} tasks={len(tasks)}')
    t0 = time.time()
    Xd, yd, vd = [], [], []
    with mp.Pool(args.n_workers) as pool:
        for i, (Xd_, yd_, vd_) in enumerate(pool.imap_unordered(_worker, tasks)):
            if len(Xd_):
                Xd.append(Xd_); yd.append(yd_); vd.append(vd_)
            if (i + 1) % max(1, args.n_workers) == 0 or (i + 1) == len(tasks):
                print(f'  [{sum(len(x) for x in Xd)} discard samples]')

    # 混合现有 response 数据，保持 response head 可用
    rd = np.load('output/nn_full_action_data_128000.npz')
    n_resp = min(len(rd['X_response']), 2_000_000)
    idx = np.random.choice(len(rd['X_response']), n_resp, replace=False)

    np.savez(args.out_path,
             X_discard=np.concatenate(Xd), y_discard=np.concatenate(yd), v_discard=np.concatenate(vd),
             tenpai_discard=np.zeros(len(np.concatenate(yd)), dtype=np.float32),
             X_response=rd['X_response'][idx], y_response=rd['y_response'][idx],
             legal_response=rd['legal_response'][idx], v_response=rd['v_response'][idx])
    print(f'Saved {args.out_path}: {len(np.concatenate(yd))} discard + {n_resp} response samples in {(time.time()-t0)/60:.1f}min')


if __name__ == '__main__':
    main()
