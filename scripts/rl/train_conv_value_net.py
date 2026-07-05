# -*- coding: utf-8 -*-
"""从已有 policy-value conv 网络热启，单独精调 value head，用于 expectimax leaf。

数据格式沿用 nn_training_data_*.npz：X(175-dim), y(policy), v(value target)。
只使用 v 做 MSE loss，policy/dealin/tenpai/response 头均不参与训练。
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algo.nn.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_npz')
    ap.add_argument('init_model')
    ap.add_argument('out_pt')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-5)
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--unfreeze-trunk', action='store_true',
                    help='默认只训练 value head；设置此项则同时放开 trunk')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 加载数据
    d = np.load(args.data_npz)
    X = d['X'].astype(np.float32)
    v = d['v'].astype(np.float32)
    print(f'Data: {len(X)} samples, v mean={v.mean():.3f}, std={v.std():.3f}')

    X_train, X_val, v_train, v_val = train_test_split(
        X, v, test_size=args.val_ratio, random_state=args.seed)

    # 加载初始化模型
    cfg_path = args.init_model.replace('.pt', '_config.json')
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(args.init_model), 'nn_model_config.json')
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    print('Init config:', cfg)

    model = build_model(cfg)
    sd = torch.load(args.init_model, map_location='cpu')
    if isinstance(sd, dict):
        if 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        elif 'model_state' in sd:
            sd = sd['model_state']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print('Missing keys:', missing[:10])
    if unexpected:
        print('Unexpected keys:', unexpected[:10])

    # 默认冻结除 value 头外的所有参数
    for name, p in model.named_parameters():
        p.requires_grad = args.unfreeze_trunk or ('value_fc' in name or 'value_head' in name)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'Trainable params: {n_train}/{n_total} ({n_train/n_total:.1%})')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    X_train_t = torch.from_numpy(X_train).to(device)
    v_train_t = torch.from_numpy(v_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    v_val_t = torch.from_numpy(v_val).to(device)

    best_mse = float('inf')
    best_state = None
    N = len(X_train)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(N)
        total_loss = 0.0
        for i in range(0, N, args.batch):
            idx = perm[i:i + args.batch]
            xb, vb = X_train_t[idx], v_train_t[idx]
            optimizer.zero_grad()
            out = model(xb)
            pred_v = out[1].squeeze(-1)
            loss = nn.functional.mse_loss(pred_v, vb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val_t)[1].squeeze(-1)
            val_mse = nn.functional.mse_loss(pred_val, v_val_t).item()
            val_mae = torch.abs(pred_val - v_val_t).mean().item()
        if val_mse < best_mse:
            best_mse = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f'Epoch {epoch:3d} | train_mse {total_loss/N:.4f} | '
              f'val_mse {val_mse:.4f} val_mae {val_mae:.4f} (best {best_mse:.4f})')

    # 保存 best val 模型
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval().cpu()
    torch.save(model.state_dict(), args.out_pt)
    out_cfg_path = args.out_pt.replace('.pt', '_config.json')
    json.dump(cfg, open(out_cfg_path, 'w'), indent=2)
    print(f'Saved {args.out_pt} + {out_cfg_path}; best val_mse={best_mse:.4f}')


if __name__ == '__main__':
    main()
