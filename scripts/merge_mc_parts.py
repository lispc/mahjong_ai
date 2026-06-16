#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并多个 MC value label npz part 文件。"""
import sys
import os
import numpy as np


def main():
    parts = sys.argv[1:-1]
    out_path = sys.argv[-1]
    if not parts:
        print('Usage: python scripts/merge_mc_parts.py part0.npz part1.npz ... out.npz')
        sys.exit(1)

    Xs, ys, vs, qs = [], [], [], []
    for p in parts:
        print(f'Loading {p} ...')
        d = np.load(p)
        Xs.append(d['X'])
        ys.append(d['y'])
        vs.append(d['v'])
        if 'q' in d:
            qs.append(d['q'])

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    v = np.concatenate(vs, axis=0)
    print(f'Merged: X{X.shape}, y{y.shape}, v{v.shape}')

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if qs:
        q = np.concatenate(qs, axis=0)
        np.savez_compressed(out_path, X=X, y=y, v=v, q=q)
        print(f'Saved to {out_path} with quality flags')
    else:
        np.savez_compressed(out_path, X=X, y=y, v=v)
        print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()
