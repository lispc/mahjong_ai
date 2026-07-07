# -*- coding: utf-8 -*-
"""训练 34 维终局防守 EV 头（defensive_head）。

从 exact endgame 标签加载 (features, evs, best_exact) 样本，在 current best backbone
上新增 defensive_head 并只训练该 head。

用法：
    PYTHONPATH=. python3 scripts/rl/train_defensive_head.py \
        output/exact_endgame_labels_1000.npz \
        output/nn_full_action_best.pt \
        output/nn_defensive_1000.pt \
        --epochs 60 --batch 512 --lr 1e-3
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.nn.model import build_model
from algo.nn.features import _TILE_TO_IDX


def _load_data(path):
    with np.load(path) as data:
        samples = pickle.loads(bytes(data['samples']))
    X = []
    Y = []
    best = []
    legal_masks = []
    for s in samples:
        X.append(s['features'])
        ev = np.zeros(34, dtype=np.float32)
        mask = np.zeros(34, dtype=np.float32)
        for t, v in s['evs'].items():
            ev[int(_TILE_TO_IDX[t])] = float(v)
            mask[int(_TILE_TO_IDX[t])] = 1.0
        Y.append(ev)
        best.append(int(_TILE_TO_IDX[s['best_exact']]))
        legal_masks.append(mask)
    return np.stack(X).astype(np.float32), np.stack(Y).astype(np.float32), np.array(best, dtype=np.int64), np.stack(legal_masks).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='.npz with pickled samples')
    parser.add_argument('base_model', help='current best .pt')
    parser.add_argument('output_model', help='output .pt path')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--train-ratio', type=float, default=0.9)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    cfg_path = args.base_model.replace('.pt', '_config.json')
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(args.base_model), 'nn_model_config.json')
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg['defensive_head'] = True

    model = build_model(cfg).to(device)
    sd = torch.load(args.base_model, map_location=device)
    if isinstance(sd, dict):
        if 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        elif 'model_state' in sd:
            sd = sd['model_state']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print('Missing keys (expected for new defensive_head):', missing[:10])
    if unexpected:
        print('Unexpected keys:', unexpected[:10])

    # Freeze backbone, only train defensive head
    for name, param in model.named_parameters():
        if 'defensive' not in name:
            param.requires_grad = False
        else:
            print('Trainable:', name)

    X, Y, best, legal_mask = _load_data(args.data)
    n = len(X)
    n_train = int(n * args.train_ratio)
    perm = np.random.permutation(n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    best_val_idx = best[val_idx]
    mask_val = legal_mask[val_idx]
    print(f'Data: {n} total, {len(train_idx)} train, {len(val_idx)} val')
    print(f'EV range: [{Y.min():.3f}, {Y.max():.3f}], mean={Y.mean():.3f}')

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(Y_val),
        torch.from_numpy(best_val_idx), torch.from_numpy(mask_val))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.wd)
    criterion = nn.MSELoss()

    best_val = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            defensive_ev = out[-1]
            loss = criterion(defensive_ev, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        train_loss = total_loss / len(train_idx)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb, best_b, mask_b in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                defensive_ev = out[-1]
                loss = criterion(defensive_ev, yb)
                val_loss += loss.item() * len(xb)
                # best tile accuracy: among legal tiles, argmax predicted EV == exact best
                for i in range(len(yb)):
                    legal = mask_b[i].numpy() > 0
                    if legal.sum() == 0:
                        continue
                    pred_best = np.where(legal, defensive_ev[i].cpu().numpy(), -np.inf).argmax()
                    if pred_best == best_b[i].item():
                        correct += 1
                    total += 1
        val_loss /= len(val_idx)
        acc = correct / total if total > 0 else 0.0

        print(f'Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f} '
              f'val_loss={val_loss:.4f} best_acc={acc:.3f}')

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': cfg,
            }, args.output_model)
            print('  saved best')

    # Save config next to model
    out_cfg_path = args.output_model.replace('.pt', '_config.json')
    with open(out_cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'Done. Best val loss {best_val:.4f}. Model saved to {args.output_model}')


if __name__ == '__main__':
    main()
