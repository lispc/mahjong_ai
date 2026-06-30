#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 TD(λ) target 训练 value net（PyTorch）。

与 train_value_net_mc.py 区别：
  1. 支持 warm start（默认从 best_1581 加载）
  2. TD target 已 clip 到 [-1, 1]，loss 用 MSE
  3. 每 epoch 保存 checkpoint，支持断点续训
  4. Early stopping：val_loss 连续 patience epoch 不降就停
  5. 输出到独立文件 nn_value_model_mc_td.pt，不覆盖 best_1581
"""
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
            total_loss += float(criterion(pred.squeeze(-1), yv_b))
            total += end - start
    return total_loss / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='TD target .npz file')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--hidden_dims', default='512,256,128')
    parser.add_argument('--init_from', default='output/nn_value_model_mc_best_1581.pt',
                        help='warm start weights')
    parser.add_argument('--init_config', default='output/nn_value_model_mc_config_best_1581.json')
    parser.add_argument('--out', default='output/nn_value_model_mc_td.pt')
    parser.add_argument('--out_config', default='output/nn_value_model_mc_td_config.json')
    parser.add_argument('--patience', type=int, default=10,
                        help='early stopping patience')
    parser.add_argument('--resume', action='store_true',
                        help='resume from checkpoint if exists')
    args = parser.parse_args()

    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)

    print(f'Loading TD data from {args.data} ...', flush=True)
    data = np.load(args.data)
    X = data['X']
    y_value = data['v'].astype(np.float32)
    # 额外 clip 一次（理论上 compute_td 已 clip，但保险）
    y_value = np.clip(y_value, -1.0, 1.0)

    n_total = X.shape[0]
    n_val = min(5000, n_total // 10)
    n_train = n_total - n_val

    X_train = torch.tensor(X[:n_train], dtype=torch.float32, device=device)
    X_val = torch.tensor(X[n_train:], dtype=torch.float32, device=device)
    yv_train = torch.tensor(y_value[:n_train], dtype=torch.float32, device=device)
    yv_val = torch.tensor(y_value[n_train:], dtype=torch.float32, device=device)

    print(f'Train: {n_train}, Val: {n_val}, features: {X.shape[1]}', flush=True)
    print(f'Target stats: mean={y_value.mean():.3f}, std={y_value.std():.3f}', flush=True)

    model = MahjongValueNetDeep(input_dim=X.shape[1], hidden_dims=hidden_dims).to(device)

    # Warm start
    start_epoch = 1
    best_val_loss = float('inf')
    ckpt_path = args.out + '.checkpoint.pt'

    if args.resume and os.path.exists(ckpt_path):
        print(f'Resuming from {ckpt_path} ...', flush=True)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        print(f'  resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}', flush=True)
    elif os.path.exists(args.init_from):
        print(f'Warm start from {args.init_from} ...', flush=True)
        # 用 init config 的 hidden_dims 加载，但训练用 args.hidden_dims
        with open(args.init_config) as f:
            init_cfg = json.load(f)
        init_hidden = init_cfg.get('hidden_dims', [512, 256, 128])
        if init_hidden == hidden_dims and init_cfg['input_dim'] == X.shape[1]:
            model.load_state_dict(torch.load(args.init_from, map_location=device))
            print(f'  loaded warm start weights', flush=True)
        else:
            # 结构不同，尝试 strict=False 加载共享层
            try:
                init_model = MahjongValueNetDeep(
                    input_dim=init_cfg['input_dim'], hidden_dims=init_hidden).to(device)
                init_model.load_state_dict(torch.load(args.init_from, map_location=device))
                # 如果结构不同就从头训
                print(f'  WARNING: init model shape mismatch, training from scratch', flush=True)
            except Exception as e:
                print(f'  WARNING: warm start failed ({e}), training from scratch', flush=True)
    else:
        print(f'No init weights found at {args.init_from}, training from scratch', flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.MSELoss()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    no_improve = 0

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        model.train()
        perm = torch.randperm(n_train, device=device)
        train_loss_sum = 0.0
        batches = 0

        for i in range(0, n_train, args.batch_size):
            idx = perm[i:i + args.batch_size]
            Xb = X_train[idx]
            yv_b = yv_train[idx]

            optimizer.zero_grad()
            pred = model(Xb).squeeze(-1)
            loss = criterion(pred, yv_b)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss)
            batches += 1

        val_loss = evaluate(model, X_val, yv_val, batch_size=args.batch_size)
        elapsed = time.time() - start
        print(f'Epoch {epoch:2d}/{args.epochs}  '
              f'train_loss={train_loss_sum/batches:.4f}  '
              f'val_loss={val_loss:.4f}  '
              f'time={elapsed:.1f}s', flush=True)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.out)
            with open(args.out_config, 'w') as f:
                json.dump({'input_dim': int(X.shape[1]), 'arch': 'deep',
                           'hidden_dims': hidden_dims, 'framework': 'pytorch',
                           'td_lambda': True}, f)
            print(f'  -> saved best TD value model (val_loss={val_loss:.4f})', flush=True)
            no_improve = 0
        else:
            no_improve += 1
            print(f'  no improvement ({no_improve}/{args.patience})', flush=True)

        # checkpoint（无论是否 improved 都存，用于 resume）
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss,
        }, ckpt_path)

        if no_improve >= args.patience:
            print(f'Early stopping at epoch {epoch}', flush=True)
            break

    print(f'Training complete. Best TD value model at {args.out}', flush=True)
    print(f'Best val_loss: {best_val_loss:.4f}', flush=True)
    print(f'For reference, MC best val_loss ~ 0.199', flush=True)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


if __name__ == '__main__':
    main()
