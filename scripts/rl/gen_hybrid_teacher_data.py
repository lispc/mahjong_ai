# -*- coding: utf-8 -*-
"""用 Hybrid NN + BeliefExp（soup 模型）当教师自对弈，记录所有玩家的 discard trajectory。

与 gen_selfplay_winner_data.py 的区别：
- 教师是 HybridNNBeliefAgent（NN policy + BeliefExp  fallback），不是纯 PPOAgent；
- 记录所有玩家（不只是赢家），获得更多样本；
- discard value label 用该局最终 seat reward（+1 赢 / -1 输 / 0 流局）；
- response 数据直接复用 nn_full_action_data_128000.npz 的子集。

用法：
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python3 scripts/rl/gen_hybrid_teacher_data.py \
        output/nn_full_action_soup_best_epoch7.pt output/nn_hybrid_soup_teacher_8000.npz 8000 32
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
from algo.agents.hybrid_nn_belief_agent import HybridNNBeliefAgent
from algo.nn.features import extract_features, tile_to_index


def _outcome(result, name):
    if result.get('win_type') == 'draw':
        return 0.0
    return 1.0 if result.get('winner') == name else -1.0


class RecordingHybridAgent(HybridNNBeliefAgent):
    def __init__(self, name, model_path, device='cpu', belief_kind='beliefexp', tenpai_threshold=28):
        super().__init__(name, nn_model_path=model_path, belief_kind=belief_kind,
                         tenpai_threshold=tenpai_threshold, device=device, temperature=0.0, verbose=False)
        self.discard_steps = []

    def init_tiles(self, l):
        super().init_tiles(l)
        self.discard_steps = []

    def next(self):
        # 记录当前状态特征（在 BeliefExp / NN 决策前）
        feats = extract_features(self.nn_agent.context, self.cur, self.name)
        tile = super().next()
        self.discard_steps.append((np.asarray(feats, dtype=np.float32), tile_to_index(tile)))
        return tile


def _worker(args):
    model_path, n_games, seed_base, device, belief_kind, tenpai_threshold = args
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
        agents = [RecordingHybridAgent(f'H{i}', model_path, device=device,
                                       belief_kind=belief_kind, tenpai_threshold=tenpai_threshold)
                  for i in range(4)]
        result = play_game(agents, verbose=False, record_time=False)
        for a in agents:
            o = _outcome(result, a.name)
            for feats, act in a.discard_steps:
                Xd.append(feats); yd.append(act); vd.append(o)
    if not Xd:
        return (np.zeros((0, 175), np.float32), np.zeros((0,), np.int64), np.zeros((0,), np.float32))
    return (np.stack(Xd), np.asarray(yd, np.int64), np.asarray(vd, np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model_path')
    ap.add_argument('out_path')
    ap.add_argument('n_games', type=int)
    ap.add_argument('n_workers', type=int)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--belief-kind', type=str, default='beliefexp')
    ap.add_argument('--tenpai-threshold', type=int, default=28)
    ap.add_argument('--seed-base', type=int, default=2000000)
    ap.add_argument('--games-per-task', type=int, default=5)
    args = ap.parse_args()

    mp.set_start_method('spawn', force=True)
    tasks = []
    remaining = args.n_games
    g0 = args.seed_base
    while remaining > 0:
        g = min(args.games_per_task, remaining)
        tasks.append((args.model_path, g, g0, args.device, args.belief_kind, args.tenpai_threshold))
        g0 += g
        remaining -= g

    print(f'Hybrid teacher data: model={args.model_path} games={args.n_games} workers={args.n_workers} tasks={len(tasks)}')
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
