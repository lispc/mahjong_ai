# -*- coding: utf-8 -*-
"""从对手建模数据生成 "tile danger" 标签。

对每个 sample（当前玩家视角），三个对手各有一张 13 张手牌的 multi-hot。
若某对手已听牌（shanten==0），且待牌包含 tile t，则 tile t 对该对手是危险的。
最终 danger[t] = 任一对手能荣和 t 的指示（0/1）。

输出 .npz：
    X: (N, 175) 公开特征（与 opponent_model_data 相同）
    danger: (N, 34) 二值 danger map

用法：
    PYTHONPATH=. python3 scripts/rl/compute_opponent_danger.py \
        output/opponent_model_data_16000.npz \
        output/opponent_danger_data_16000.npz \
        32
"""
import os
import sys
import time
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algo.eval.v2 import shanten_fast, is_win, VALID_TILES
from algo.nn.features import tile_to_index


def _compute_for_sample(opp_hand_3x34):
    """opp_hand_3x34: (3, 34) multi-hot，顺序与 VALID_TILES 一致。返回 34-dim danger map。"""
    danger = np.zeros(34, dtype=np.float32)
    tile_list = list(VALID_TILES)
    for opp in range(3):
        hand = []
        for idx, cnt in enumerate(opp_hand_3x34[opp]):
            if cnt:
                tile_val = tile_list[idx]
                hand.extend([tile_val] * int(cnt))
        if len(hand) != 13:
            continue
        if shanten_fast(hand) != 0:
            continue
        counts = np.zeros(34, dtype=np.int32)
        for t in hand:
            counts[tile_to_index(t)] += 1
        for idx, tile_val in enumerate(tile_list):
            if counts[idx] >= 4:
                continue
            if is_win(hand + [tile_val]):
                danger[idx] = 1.0
    return danger


def _worker_chunk(args):
    start, end, data_path = args
    data = np.load(data_path, mmap_mode='r')
    opp_hands = data['opp_hand'][start:end]
    dangers = np.empty((end - start, 34), dtype=np.float32)
    for i in range(len(opp_hands)):
        dangers[i] = _compute_for_sample(opp_hands[i])
    return start, dangers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('in_path')
    ap.add_argument('out_path')
    ap.add_argument('--workers', type=int, default=min(32, cpu_count()))
    args = ap.parse_args()

    data = np.load(args.in_path, mmap_mode='r')
    X = data['X']
    n = X.shape[0]
    print(f'Loaded {n} samples, computing danger labels with {args.workers} workers ...')

    chunk = (n + args.workers - 1) // args.workers
    tasks = [(i, min(i + chunk, n), args.in_path) for i in range(0, n, chunk)]

    t0 = time.time()
    dangers = np.empty((n, 34), dtype=np.float32)
    with Pool(processes=args.workers) as pool:
        for start, arr in pool.imap_unordered(_worker_chunk, tasks):
            dangers[start:start + len(arr)] = arr
            print(f'  chunk {start}/{n} done')
    dt = time.time() - t0
    print(f'Computed in {dt:.1f}s, positive rate per tile: {dangers.mean(axis=0).mean():.4f}')

    np.savez(args.out_path, X=X[:], danger=dangers)
    print(f'Saved {args.out_path}')


if __name__ == '__main__':
    main()
