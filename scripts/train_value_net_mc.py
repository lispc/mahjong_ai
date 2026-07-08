#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 MC rollout 胜率标签训练一个更深的独立价值网络（PyTorch）。"""

import sys
import os
import time
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from algo.nn.value_model import MahjongValueNetDeep


def evaluate(model, X_val, yv_val, batch_size=1024):
    model.eval()
    criterion = nn.MSELoss(reduction='sum')
    total_loss = 0.0
    total = 0
    n = X_val.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            Xb = X_val[start:end]
            yv_b = yv_val[start:end]
            pred = model(Xb)
            total_loss += float(criterion(pred, yv_b))
            total += end - start
    return total_loss / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', nargs='?', default='output/nn_training_data_mc.npz')
    parser.add_argument('epochs', nargs='?', type=int, default=80)
    parser.add_argument('batch_size', nargs='?', type=int, default=256)
    parser.add_argument('lr', nargs='?', type=float, default=1e-3)
    parser.add_argument('hidden_dims', nargs='?', default='512,256,128')
    parser.add_argument('wd', nargs='?', type=float, default=0.0)
    parser.add_argument('--out', default='output/nn_value_model_mc',
                        help='output prefix for .pt and .json (default: output/nn_value_model_mc)')
    args = parser.parse_args()

    data_path = args.data_path
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    wd = args.wd
    out_prefix = args.out

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print(f'Loading MC data from {data_path} ...')
    data = np.load(data_path)
    X = data['X']
    y_value = data['v'].astype(np.float32)

    # 如果数据带 quality flag，过滤掉 timeout/exception/truncated 样本
    if 'q' in data:
        q = data['q']
        mask = q == 0
        n_total = len(q)
        n_bad = n_total - int(mask.sum())
        if n_bad > 0:
            print(f'Filtering {n_bad}/{n_total} bad samples (timeout/exception/truncated)')
            X = X[mask]
            y_value = y_value[mask]

    n_total = X.shape[0]
    n_val = min(5000, n_total // 10)
    n_train = n_total - n_val

    X_train = torch.tensor(X[:n_train], dtype=torch.float32, device=device)
    X_val = torch.tensor(X[n_train:], dtype=torch.float32, device=device)
    yv_train = torch.tensor(y_value[:n_train], dtype=torch.float32, device=device)
    yv_val = torch.tensor(y_value[n_train:], dtype=torch.float32, device=device)

    print(f'Train: {n_train}, Val: {n_val}, features: {X.shape[1]}')

    model = MahjongValueNetDeep(input_dim=X.shape[1], hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    out_dir = os.path.dirname(out_prefix) or 'output'
    os.makedirs(out_dir, exist_ok=True)
    out_pt = out_prefix + '.pt'
    out_json = out_prefix + '_config.json'

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        perm = torch.randperm(n_train, device=device)
        train_loss_sum = 0.0
        batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            Xb = X_train[idx]
            yv_b = yv_train[idx]

            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yv_b)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss)
            batches += 1

        val_loss = evaluate(model, X_val, yv_val, batch_size=batch_size)
        elapsed = time.time() - start
        print(f'Epoch {epoch:2d}/{epochs}  '
              f'train_loss={train_loss_sum/batches:.4f}  '
              f'val_loss={val_loss:.4f}  '
              f'time={elapsed:.1f}s')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_pt)
            with open(out_json, 'w') as f:
                json.dump({'input_dim': int(X.shape[1]), 'arch': 'deep',
                           'hidden_dims': hidden_dims, 'framework': 'pytorch'}, f)
            print(f'  -> saved best MC value model to {out_pt}')

    print(f'Training complete. Best MC value model at {out_pt}')


if __name__ == '__main__':
    main()
