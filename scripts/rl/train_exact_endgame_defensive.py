# -*- coding: utf-8 -*-
"""用 exact endgame 防守标签训练 defensive head。

输入 `output/exact_endgame_labels_*.npz`，样本为 pickle 序列化的 dict：
- features: 175-dim 当前玩家视角特征
- evs: dict tile_value -> exact EV
- best_exact: int tile

训练目标：对每个 tile 输出 predicted EV（34-dim 回归），只计算候选弃牌 tile 的 loss。

用法:
    PYTHONPATH=. python3 scripts/rl/train_exact_endgame_defensive.py \
        output/exact_endgame_labels_1000.npz \
        output/nn_full_action_best.pt \
        output/nn_exact_endgame_defensive.pt \
        --epochs 60 --batch 128 --lr 5e-5
"""

import os
import sys
import json
import time
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algo.nn.model import build_model
from algo.nn.features import tile_to_index


def _load_samples(npz_path):
    d = np.load(npz_path)
    raw = pickle.loads(d['samples'])
    return raw


def _build_tensors(samples):
    """把样本列表转成 X (N,175), ev (N,34), mask (N,34), best (N,)。"""
    n = len(samples)
    dim = len(samples[0]['features'])
    X = np.zeros((n, dim), dtype=np.float32)
    ev = np.zeros((n, 34), dtype=np.float32)
    mask = np.zeros((n, 34), dtype=np.float32)
    best = np.zeros(n, dtype=np.int64)
    for i, s in enumerate(samples):
        X[i] = s['features']
        best[i] = tile_to_index(s['best_exact'])
        for t, v in s['evs'].items():
            idx = tile_to_index(int(t))
            ev[i, idx] = float(v)
            mask[i, idx] = 1.0
    return torch.from_numpy(X), torch.from_numpy(ev), torch.from_numpy(mask), torch.from_numpy(best)


def evaluate(model, X, ev, mask, device, bs=2048):
    model.eval()
    mse = nn.MSELoss(reduction='sum')
    n = X.shape[0]
    total_loss = 0.0
    total_mask = 0.0
    correct = 0
    with torch.no_grad():
        for s in range(0, n, bs):
            e = min(s + bs, n)
            xb = X[s:e].to(device)
            evb = ev[s:e].to(device)
            mb = mask[s:e].to(device)
            out = model(xb)
            # defensive head 是最后一个 34-dim 输出
            defensive_ev = out[-1]
            if defensive_ev.shape[-1] != 34:
                raise RuntimeError('model has no defensive_head')
            diff = (defensive_ev - evb) * mb
            total_loss += float((diff * diff).sum())
            total_mask += float(mb.sum())
            # best tile accuracy（只在候选里 argmax）
            pred_masked = defensive_ev - (1.0 - mb) * 1e9
            correct += int((pred_masked.argmax(1) == evb.argmax(1)).sum())
    return total_loss / (total_mask + 1e-8), correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='exact endgame labels .npz')
    parser.add_argument('backbone', help='backbone .pt path')
    parser.add_argument('out', help='output .pt path')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--freeze-backbone', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f'Loading samples from {args.data} ...')
    samples = _load_samples(args.data)
    print(f'Loaded {len(samples)} samples')

    X, ev, mask, best = _build_tensors(samples)
    n = X.shape[0]
    print(f'feature dim={X.shape[1]}, ev range=[{ev.min():.3f}, {ev.max():.3f}]')

    # 划分 train/val
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g)
    n_val = min(1000, n // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, evtr, mtr, btr = X[tr_idx], ev[tr_idx], mask[tr_idx], best[tr_idx]
    Xval, evval, mval, bval = X[val_idx], ev[val_idx], mask[val_idx], best[val_idx]

    # 加载 backbone
    ckpt = torch.load(args.backbone, map_location='cpu')
    if 'config' in ckpt:
        config = ckpt['config']
        state = ckpt.get('model_state', ckpt)
    else:
        config = json.load(open(args.backbone.replace('.pt', '_config.json')))
        state = ckpt

    config['defensive_head'] = True
    print(f'Building model with config: {config}')
    model = build_model(config).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'missing keys: {missing[:10]} ...' if len(missing) > 10 else f'missing keys: {missing}')
    if unexpected:
        print(f'unexpected keys: {unexpected[:10]} ...' if len(unexpected) > 10 else f'unexpected keys: {unexpected}')

    if args.freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for name in ['defensive_conv', 'defensive_fc', 'defensive_head']:
            m = getattr(model, name, None)
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = True
        print('Backbone frozen; only defensive head trainable')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    mse = nn.MSELoss(reduction='none')

    best_loss = float('inf')
    ntr = Xtr.shape[0]

    for ep in range(args.epochs):
        model.train()
        perm_tr = torch.randperm(ntr)
        t0 = time.time()
        train_loss = 0.0
        train_mask = 0.0
        for s in range(0, ntr, args.batch):
            idx = perm_tr[s:s + args.batch]
            xb = Xtr[idx].to(device)
            evb = evtr[idx].to(device)
            mb = mtr[idx].to(device)

            out = model(xb)
            # defensive head 是最后一个 34-dim 输出
            defensive_ev = out[-1]
            if defensive_ev.shape[-1] != 34:
                raise RuntimeError('model has no defensive_head')

            loss = (mse(defensive_ev, evb) * mb).sum() / (mb.sum() + 1e-8)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += float(loss) * float(mb.sum())
            train_mask += float(mb.sum())

        sched.step()
        val_loss, val_acc = evaluate(model, Xval, evval, mval, device)
        dt = time.time() - t0

        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
            torch.save({'model_state': model.state_dict(), 'config': config}, args.out)
            json.dump(config, open(args.out.replace('.pt', '_config.json'), 'w'))

        print(f'ep {ep:2d} | train_mse {train_loss/(train_mask+1e-8):.4f} '
              f'val_mse {val_loss:.4f} val_best_acc {val_acc:.3f} '
              f'{"*" if improved else " "} | {dt:.1f}s')

    print(f'Done. Best val MSE = {best_loss:.4f}; saved to {args.out}')


if __name__ == '__main__':
    main()
