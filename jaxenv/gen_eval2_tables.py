# -*- coding: utf-8 -*-
"""离线生成 eval2（arena Baseline 度量）查找表 -> jaxenv/tables_eval2.npz。

用法：
    PYTHONPATH=. python3 jaxenv/gen_eval2_tables.py [--workers 32] [--out jaxenv/tables_eval2.npz]

与 jaxenv/tables.npz（胡牌/向听，g<=4 全划分位掩码）不同，本表面向
algo.eval.legacy eval0 的「最大面子数」语义（允许留孤张）：
    suit_a  int8 [5^9]  数牌花色：可提取的最大面子数（不含对子的分解）
    suit_b  int8 [5^9]  数牌花色：含至少一个对子的分解中可提取的最大面子数（不可达 -1）
    honor_a int8 [5^7]  字牌同上
    honor_b int8 [5^7]  字牌同上

为什么不复用 tables.npz：eval2 内层 eval0 会看到 15 张手牌（弃后 13 + 两层
摸牌各 +1），一个花色可达 5 面子；tables.npz 的 win 掩码 g 封顶 4（胡牌/向听
只需 <=4），直接复用会在 5 面子可达的 15 张手上少算 1 个面子。

a/b 由递归分解直接计算（孤张/对子/刻子/顺子选项，与 algo/eval/_fast_eval0.pyx
的 _search_suit 选项集一致；字牌等价于其对子/刻子贪心点的 Pareto 闭包）。
"""

import argparse
import multiprocessing as mp
import time
from functools import lru_cache

import numpy as np

TABLES_EVAL2_PATH = __file__.rsplit('/', 1)[0] + '/tables_eval2.npz'

# worker 进程内缓存的 solver（跨 chunk 复用 lru_cache，避免重复展开子问题）
_SOLVERS = {}


def _make_ab_solver(kind_count, suited):
    P5 = [5 ** i for i in range(kind_count)]

    def _digits(idx):
        ds = []
        for _ in range(kind_count):
            ds.append(idx % 5)
            idx //= 5
        return ds

    @lru_cache(maxsize=None)
    def ab(idx):
        """-> (a, b)：a = 无对子分解的最大面子数；b = 含对子分解的最大面子数（-1 不可达）。"""
        ds = _digits(idx)
        if not any(ds):
            return (0, -1)
        i = next(j for j, d in enumerate(ds) if d > 0)
        # 孤张（这张牌不用）
        best_a, best_b = ab(idx - P5[i])
        # 对子（此后分解含对子；子问题的 a/b 路径都变为含对子）
        if ds[i] >= 2:
            a, b = ab(idx - 2 * P5[i])
            best_b = max(best_b, a, b)
        # 刻子
        if ds[i] >= 3:
            a, b = ab(idx - 3 * P5[i])
            best_a = max(best_a, a + 1)
            if b >= 0:
                best_b = max(best_b, b + 1)
        # 顺子
        if suited and i <= 6 and ds[i + 1] > 0 and ds[i + 2] > 0:
            a, b = ab(idx - P5[i] - P5[i + 1] - P5[i + 2])
            best_a = max(best_a, a + 1)
            if b >= 0:
                best_b = max(best_b, b + 1)
        return (best_a, best_b)

    return ab


def _worker(task):
    kind_count, suited, lo, hi = task
    key = (kind_count, suited)
    if key not in _SOLVERS:
        _SOLVERS[key] = _make_ab_solver(kind_count, suited)
    ab = _SOLVERS[key]
    n = hi - lo
    a = np.zeros(n, np.int8)
    b = np.full(n, -1, np.int8)
    for idx in range(lo, hi):
        av, bv = ab(idx)
        a[idx - lo] = av
        b[idx - lo] = bv
    return lo, hi, a, b


def _build(kind_count, suited, workers, label):
    total = 5 ** kind_count
    n_chunks = max(workers * 4, 16)
    bounds = [round(i * total / n_chunks) for i in range(n_chunks + 1)]
    tasks = [(kind_count, suited, bounds[i], bounds[i + 1]) for i in range(n_chunks)
             if bounds[i + 1] > bounds[i]]
    a = np.zeros(total, np.int8)
    b = np.full(total, -1, np.int8)
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for j, (lo, hi, ca, cb) in enumerate(pool.imap_unordered(_worker, tasks)):
            a[lo:hi] = ca
            b[lo:hi] = cb
            if (j + 1) % 8 == 0 or j + 1 == len(tasks):
                dt = time.time() - t0
                print(f'[{label}] {j + 1}/{len(tasks)} chunks, {dt:.1f}s', flush=True)
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--out', type=str, default=TABLES_EVAL2_PATH)
    args = ap.parse_args()

    t0 = time.time()
    suit_a, suit_b = _build(9, True, args.workers, 'suited 5^9')
    honor_a, honor_b = _build(7, False, args.workers, 'honors 5^7')

    np.savez(args.out,
             suit_a=suit_a, suit_b=suit_b,
             honor_a=honor_a, honor_b=honor_b)
    print(f'saved {args.out} in {time.time() - t0:.1f}s')

    # 简单自检
    def show(a, b, idx, name):
        print(f'  {name}: a={int(a[idx])} b={int(b[idx])}')

    show(suit_a, suit_b, 0, 'zeros')                       # 空 -> (0,-1)
    idx = 2 + 3 * 5 + 3 * 25 + 3 * 125                     # counts [2,3,3,3]
    show(suit_a, suit_b, idx, '[2,3,3,3]')                 # 3刻+对 -> (3,3)
    idx = sum(5 ** i for i in range(9))                    # 1-9 各一张
    show(suit_a, suit_b, idx, '1..9 seq')                  # 3顺无对 -> (3,-1)
    idx = 4                                                # counts [4]
    show(suit_a, suit_b, idx, '[4]')                       # 刻+孤 或 对 -> (1,0)
    show(honor_a, honor_b, 3, 'honor [3]')                 # 刻 或 对 -> (1,0)


if __name__ == '__main__':
    main()
