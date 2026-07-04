# -*- coding: utf-8 -*-
"""训练对手建模网络：从公开特征预测每个对手是否听牌。

输入：当前玩家视角的 175-dim 公开特征（含自己手牌）。
输出：3 个 logits，分别对应下家/对家/上家是否听牌。

用法：
    PYTHONPATH=. python3 scripts/rl/train_opponent_model.py \
        output/opponent_model_data_16000.npz \
        output/opponent_model.pt \
        --epochs 30 --batch 512 --lr 1e-3 --hidden 256,128
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split


def build_mlp(input_dim, hidden_dims, output_dim, dropout=0.2):
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_pos = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        total_loss += loss.item()
        pred = (torch.sigmoid(logits) > 0.5).float()
        total_acc += (pred == y).float().mean().item()
        total_pos += y.mean().item()
        n += 1
    return {
        'loss': total_loss / max(n, 1),
        'acc': total_acc / max(n, 1),
        'pos_rate': total_pos / max(n, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_path')
    ap.add_argument('out_model')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--hidden', type=str, default='256,128')
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    hidden_dims = [int(x) for x in args.hidden.split(',') if x]

    print(f'Loading data from {args.data_path}')
    data = np.load(args.data_path)
    X = torch.tensor(data['X'], dtype=torch.float32)
    Y = torch.tensor(data['opp_tenpai'], dtype=torch.float32)
    print(f'X: {tuple(X.shape)}, Y: {tuple(Y.shape)}, positive rate: {Y.mean(dim=0).numpy()}')

    ds = TensorDataset(X, Y)
    n_val = int(len(ds) * args.val_ratio)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_mlp(X.size(1), hidden_dims, Y.size(1), dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    best_state = None
    cfg = {
        'input_dim': X.size(1),
        'output_dim': Y.size(1),
        'hidden_dims': hidden_dims,
        'dropout': args.dropout,
        'framework': 'pytorch',
    }

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n += 1
        scheduler.step()

        val_m = evaluate(model, val_loader, device)
        print(f'Epoch {epoch:3d} | train_loss={train_loss/max(n,1):.4f} | '
              f'val_loss={val_m["loss"]:.4f} val_acc={val_m["acc"]:.3f} val_pos={val_m["pos_rate"]:.3f}')

        if val_m['loss'] < best_val_loss:
            best_val_loss = val_m['loss']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    out_cfg_path = args.out_model.replace('.pt', '_config.json')
    with open(out_cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    torch.save({'model_state': model.state_dict(), 'config': cfg}, args.out_model)
    print(f'Training finished in {time.time() - t0:.1f}s, best val_loss {best_val_loss:.4f}')
    print(f'Saved {args.out_model} + {out_cfg_path}')


if __name__ == '__main__':
    main()
