# -*- coding: utf-8 -*-
"""用 MCTS trace value 训练一个 conv value net。

用法：
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/rl/train_value_net_from_trace.py \
        output/alphazero_trace_500_mctsvalue.npz \
        output/nn_full_action_best.pt \
        output/nn_value_conv_mctsvalue.pt \
        --epochs 60 --batch 512 --lr 1e-4 --device cuda
"""
import os
import sys
import json
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algo.nn.model import build_model


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg_path = path.replace('.pt', '_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    elif isinstance(checkpoint, dict) and 'config' in checkpoint:
        cfg = checkpoint['config']
    else:
        raise ValueError(f'No config found for {path}')

    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state = checkpoint['model_state']
    else:
        state = checkpoint
    model = build_model(cfg)
    model.load_state_dict(state, strict=False)
    model.to(device)
    return model, cfg


def split_outputs(model, cfg, out):
    """解析 forward 元组，返回 value（下标 1）。"""
    idx = 0
    d_logit = out[idx]; idx += 1
    val = out[idx]; idx += 1
    if cfg.get('dealin_head', False):
        idx += 1
    if cfg.get('candidate_value_head', False):
        idx += 1
    if cfg.get('response_head', False):
        idx += 1
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace')
    ap.add_argument('init_model')
    ap.add_argument('out_model')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--freeze_trunk', action='store_true')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    data = np.load(args.trace)
    X = torch.from_numpy(data['X']).float()
    y = torch.from_numpy(data['value']).float()
    print(f'Loaded trace: {len(X)} samples, value mean={y.mean():.3f}, std={y.std():.3f}')

    n_val = max(1, int(len(X) * args.val_ratio))
    n_train = len(X) - n_val
    train_ds, val_ds = random_split(TensorDataset(X, y), [n_train, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    model, cfg = load_model(args.init_model, device)
    print('Init model cfg:', cfg)

    if args.freeze_trunk:
        # 只训练 value head
        for name, param in model.named_parameters():
            if 'value_head' not in name:
                param.requires_grad = False
        params = [p for p in model.parameters() if p.requires_grad]
        print('Trunk frozen, only training value head')
    else:
        params = model.parameters()

    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_mse = float('inf')
    best_state = None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            val = split_outputs(model, cfg, out)
            loss = F.mse_loss(val.squeeze(1), yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_mse = 0.0
            val_n = 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val = split_outputs(model, cfg, out)
                val_mse += F.mse_loss(val.squeeze(1), yb, reduction='sum').item()
                val_n += xb.size(0)
            val_mse /= val_n

        print(f'Epoch {epoch:3d} | train_mse {train_loss/n:.4f} | val_mse {val_mse:.4f}')
        if val_mse < best_mse:
            best_mse = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(args.out_model) or '.', exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'config': cfg}, args.out_model)
    with open(args.out_model.replace('.pt', '_config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'Saved {args.out_model} best val_mse {best_mse:.4f} in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
