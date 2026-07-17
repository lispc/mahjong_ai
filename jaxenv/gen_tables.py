# -*- coding: utf-8 -*-
"""离线生成胡牌/向听查找表 -> jaxenv/tables.npz。

用法：
    PYTHONPATH=. python3 jaxenv/gen_tables.py [--workers 32] [--out jaxenv/tables.npz]

表内容（详见 rules.py  docstring）：
    suit_win   uint16 [5^9]      数牌花色 (g,p) 全划分位掩码
    suit_smax  int8   [5^9, 5]   数牌花色每 g 的最大 对子+搭子
    honor_win  uint16 [5^7]      字牌 (g,p) 全划分位掩码
    honor_smax int8   [5^7, 5]   字牌每 g 的最大 对子+搭子
"""

import argparse
import multiprocessing as mp
import time

import numpy as np

from jaxenv import rules


def _build(kind_count, suited, workers, label):
    total = 5 ** kind_count
    n_chunks = max(workers * 4, 16)
    bounds = [round(i * total / n_chunks) for i in range(n_chunks + 1)]
    tasks = [(kind_count, suited, bounds[i], bounds[i + 1]) for i in range(n_chunks)
             if bounds[i + 1] > bounds[i]]
    win = np.zeros(total, np.uint16)
    smax = np.full((total, 5), -1, np.int8)
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for j, (lo, hi, w, s) in enumerate(pool.imap_unordered(_worker, tasks)):
            win[lo:hi] = w
            smax[lo:hi] = s
            if (j + 1) % 8 == 0 or j + 1 == len(tasks):
                dt = time.time() - t0
                print(f'[{label}] {j + 1}/{len(tasks)} chunks, {dt:.1f}s', flush=True)
    return win, smax


def _worker(task):
    kind_count, suited, lo, hi = task
    return rules.compute_table_chunk(kind_count, suited, lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--out', type=str, default=rules.TABLES_PATH)
    args = ap.parse_args()

    t0 = time.time()
    suit_win, suit_smax = _build(9, True, args.workers, 'suited 5^9')
    honor_win, honor_smax = _build(7, False, args.workers, 'honors 5^7')

    np.savez(args.out,
             suit_win=suit_win, suit_smax=suit_smax,
             honor_win=honor_win, honor_smax=honor_smax)
    print(f'saved {args.out} in {time.time() - t0:.1f}s')

    # 简单自检：空向量、刻子+对子、顺子划分
    def show(win, smax, idx, name):
        print(f'  {name}: win_mask={int(win[idx]):#05x} smax={smax[idx].tolist()}')

    show(suit_win, suit_smax, 0, 'zeros')
    idx = 2 + 3 * 5 + 3 * 25 + 3 * 125  # counts [2,3,3,3]: 1对+3刻 -> (3,1)
    show(suit_win, suit_smax, idx, '[2,3,3,3]')
    idx = sum(5 ** i for i in range(9))  # 1-9 各一张 -> 3 顺子
    show(suit_win, suit_smax, idx, '1..9 seq')


if __name__ == '__main__':
    main()
